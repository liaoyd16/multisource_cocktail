import __init__

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import pickle
import os
import json
import numpy as np
import gc
import cv2

import psutil


#=============================================
#        path
#=============================================

from dir_utils import *
from dataset_meta import *

import test_multisource
from test_multisource import test_multisource


# 20 in train dir, 4 in test dir
# 20 = 1+19, 4 = 1+3

#=============================================
#        Hyperparameters
#=============================================

epoch = 1
lr = 0.02
mom = 0.9
BS = 10

reuse = True

#=============================================
#        Define Dataloader
#=============================================

from FAB_Dataset import FAB_DataSet

mixset = FAB_DataSet(TRAIN_DIR, list_json_in_dir(TRAIN_DIR))
mixloader = torch.utils.data.DataLoader(dataset = mixset,
    batch_size = BS,
    shuffle = False)

#=============================================
#        Model
#=============================================
from featureNet import featureNet
featurenet = featureNet()
try:
    featurenet.load_state_dict(torch.load(os.path.join(ROOT_DIR, 'multisource_cocktail/featureNet/FeatureNet.pkl')))
except Exception as e:
    print(e, "F-model not available")


from ANet import ANet
A_model = ANet()
if reuse:
    try:
        A_model.load_state_dict(torch.load(os.path.join(ROOT_DIR, 'multisource_cocktail/ANet/ANet_multi_2_trained.pkl')))
    except Exception as e:
        print(e, "A-model not available")
else:
    try:
        A_model.load_state_dict(torch.load(os.path.join(ROOT_DIR, 'multisource_cocktail/ANet/ANet_raw_2.pkl')))
    except Exception as e:
        print(e, "A-model not available")
# print(A_model)


from conv_fc import ResDAE
Res_model = ResDAE()
if reuse:
    try:
        Res_model.load_state_dict(torch.load(os.path.join(ROOT_DIR, 'multisource_cocktail/DAE/DAE_multi_2.pkl')))
    except Exception as e:
        print(e, "Res-model not available")
else:
    try:
        Res_model.load_state_dict(torch.load(os.path.join(ROOT_DIR, 'multisource_cocktail/DAE/DAE_multi_2.pkl')))
    except Exception as e:
        print(e, "Res-model not available")
# print(Res_model)



#=============================================
#        Optimizer
#=============================================

criterion = nn.MSELoss()
#feat_optimizer = torch.optim.SGD(featurenet.parameters(), lr = lr, momentum=mom)
anet_optimizer = torch.optim.SGD(A_model.parameters(), lr = lr, momentum=mom)
res_optimizer = torch.optim.SGD(Res_model.parameters(), lr = lr, momentum=mom)



#=============================================
#        Loss Record
#=============================================

loss_record = []

#=============================================
#        Train
#=============================================

#input_sum = []
#output_sum = []

from mel import mel, norm, mix, mel_norm
zeros = torch.zeros(BS, 256, 128)

for epo in range(epoch):
    # train
    for i, data in enumerate(mixloader, 0):

        Res_model.train()

        # get mix spec & label        
        feat_data, a_specs, b_specs = data


        feat_data = feat_data.squeeze()
        #a_specs = norm(a_specs.squeeze())
        #b_specs = norm(b_specs.squeeze())


        mix_specs = mel_norm(mix(a_specs, b_specs))
        target_specs = mel_norm(a_specs)

        #feat_optimizer.zero_grad()
        anet_optimizer.zero_grad()
        res_optimizer.zero_grad()

        # get feature
        feats = featurenet.feature(feat_data)

        # feed in feature to ANet
        a7, a6, a5, a4, a3, a2 = A_model(feats)

        # Res_model
        tops = Res_model.upward(mix_specs, a7, a6, a5, a4, a3, a2)

        outputs = Res_model.downward(tops, shortcut = True).squeeze()

        loss_train = criterion(outputs, target_specs)

        loss_train.backward()
        
        res_optimizer.step()
        anet_optimizer.step()
        #feat_optimizer.step()

        loss_record.append(loss_train.item())

        print ('[%d, %5d] loss: %.3f, input: %.3f, output: %.3f'\
         % (epo, i, loss_train.item(), criterion(target_specs, zeros).item(), criterion(outputs, zeros).item()))
        
#        input_sum.append(np.array(criterion(target_specs, zeros).item()))
#        output_sum.append(np.array(criterion(outputs, zeros).item()))
        # print("\ttrainDataSet: probe", psutil.virtual_memory().percent)

        if i % 5 == 0:
            # print images: mix, target, attention, separated
            tarr = target_specs[0].view(256, 128).detach().numpy() * 255
            cv2.imwrite(os.path.join(ROOT_DIR, 'results/combinemodel/' + str(i)  + "_tar.png"), tarr)

            mask = (b_specs[0]).view(256, 128).detach().numpy() * 255
            cv2.imwrite(os.path.join(ROOT_DIR, 'results/combinemodel/' + str(i)  + "_mask.png"), mask)

            mixx = mix_specs[0].view(256, 128).detach().numpy() * 255
            cv2.imwrite(os.path.join(ROOT_DIR, 'results/combinemodel/' + str(i)  + "_mix.png"), mixx)

            outt = outputs[0].view(256, 128).detach().numpy() * 255
            cv2.imwrite(os.path.join(ROOT_DIR, 'results/combinemodel/' + str(i)  + "_sep.png"), outt)

            # a7.detach().numpy() * 255

        if i % 50 == 0:
            # test
            loss_test = test_multisource(Res_model, A_model, featurenet)
            print(loss_test)


        plt.figure(figsize = (20, 10))
        plt.plot(loss_record)
        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.savefig(os.path.join(ROOT_DIR, 'results/combinemodel/loss_training_epoch_{}.png'.format(epo)))
        gc.collect()
        plt.close("all")

    train_average_loss = np.average(loss_record)

    print ("train epoch #{} finish, loss average {}".format(epo, train_average_loss))

    loss_record = []



#=============================================
#        Save Model & Loss
#=============================================

torch.save(Res_model.state_dict(), os.path.join(ROOT_DIR, 'multisource_cocktail/DAE/DAE_multi_2.pkl'))
torch.save(A_model.state_dict(), os.path.join(ROOT_DIR, 'multisource_cocktail/ANet/ANet_multi_2_trained.pkl'))

#with open('input_sum.txt', 'w') as f:
#    f.write(input_sum)

#with open('output_sum.txt', 'w') as f:
#    f.write(output_sum)
