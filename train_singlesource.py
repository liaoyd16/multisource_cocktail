import __init__

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import torch.nn.init as init

import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import pickle
import os
import json
import numpy as np
import gc
import cv2

#=============================================
#        path
#=============================================

from dir_utils import list_json_in_dir, TRAIN_DIR, TEST_DIR, ROOT_DIR
from dataset_meta import *



# 20 in train dir, 4 in test dir
# 20 = 1+19, 4 = 1+3

#=============================================
#        Hyperparameters
#=============================================

epoch = 1
lr = 0.02
mom = 0.9
BS = 10

#=============================================
#        Define Dataloader
#=============================================

from FAB_Dataset import FAB_DataSet

mixset = FAB_DataSet(TRAIN_DIR, list_json_in_dir(TRAIN_DIR)[:])
mixloader = torch.utils.data.DataLoader(dataset = mixset,
    batch_size = BS,
    shuffle = False)

#=============================================
#        Model
#=============================================
reuse = True

from featureNet import featureNet

featurenet = featureNet()
try:
    featurenet.load_state_dict(torch.load(os.path.join(ROOT_DIR, 'multisource_cocktail/featureNet/FeatureNet.pkl')))
except Exception as e:
    print(e, "F-model not available")


from ANet import ANet

A_model = ANet()
try:
    if reuse:
        A_model.load_state_dict(torch.load(os.path.join(ROOT_DIR, 'multisource_cocktail/ANet/ANet_raw_2.pkl')))
except Exception as e:
    print(e, "A-model not available")


from conv_fc import ResDAE

Res_model = ResDAE()
try:
    Res_model.load_state_dict(torch.load(os.path.join(ROOT_DIR, 'multisource_cocktail/DAE/DAE_multi_2.pkl')))
except Exception as e:
    print(e, "Res-model not available")


#=============================================
#        Optimizer
#=============================================

criterion = nn.MSELoss()

# feat_optimizer = torch.optim.SGD(featurenet.parameters(), lr = lr, momentum=mom)
anet_optimizer = torch.optim.SGD(A_model.parameters(), lr = lr, momentum=mom)
res_optimizer = torch.optim.SGD(Res_model.parameters(), lr = lr, momentum=mom)




#=============================================
#        Loss Record
#=============================================

loss_record = []

#=============================================
#        Train
#=============================================

from mel import mel, norm, mel_norm
zeros = torch.zeros(BS, 256, 128)

Res_model.train()
for epo in range(epoch):
    # train
    for i, data in enumerate(mixloader, 0):
        # get mix spec & label        
        feat_data, a_specs, _ = data

        feat_data = feat_data.squeeze()

        target_specs = mel_norm(a_specs.squeeze())

#        a_specs = a_specs.squeeze()

#        a_specs = norm(a_specs)
#        target_specs = a_specs


#       feat_optimizer.zero_grad()
        anet_optimizer.zero_grad()
        res_optimizer.zero_grad()

        # get feature
        feats = featurenet.feature(feat_data)

        # feed in feature to ANet
        a7, a6, a5, a4, a3, a2 = A_model(feats)

        # Res_model
        tops = Res_model.upward(a_specs, a7, a6, a5, a4, a3, a2)

        outputs = Res_model.downward(tops, shortcut = True).squeeze()

        loss_train = criterion(outputs, target_specs)
        loss_train.backward()
        res_optimizer.step()
        anet_optimizer.step()
#        feat_optimizer.step()

#       loss_record.append(loss_train.item())
        print ('[%d, %5d] loss: %.3f, input: %.3f, output: %.3f'\
         % (epo, i, loss_train.item(), criterion(target_specs, zeros).item(), criterion(outputs, zeros).item()))
         
        # print("\ttrainDataSet: probe", psutil.virtual_memory().percent)
        

        if i % 5 == 0:
            # print images: mix, target, attention, separated

#            inn = a_specs[0].view(256, 128).detach().numpy() * 255
#            np.clip(inn, np.min(inn), 1)
#            cv2.imwrite(os.path.join(ROOT_DIR, 'results/single_source/' + str(i)  + "_mix.png"), inn)

            tarr = target_specs[0].view(256, 128).detach().numpy() * 255
            np.clip(tarr, np.min(tarr), 1)
            cv2.imwrite(os.path.join(ROOT_DIR, 'results/single_source/' + str(i)  + "_tar.png"), tarr)

            outt = outputs[0].view(256, 128).detach().numpy() * 255
            np.clip(outt, np.min(outt), 1)
            cv2.imwrite(os.path.join(ROOT_DIR, 'results/single_source/' + str(i)  + "_sep.png"), outt)

            # a7.detach().numpy() * 255

            gc.collect()
            plt.close("all")

#    plt.figure(figsize = (20, 10))
#    plt.plot(loss_record)
#    plt.xlabel('iterations')
#    plt.ylabel('loss')
#    plt.savefig(os.path.join(ROOT_DIR, 'results/combinemodel/loss_training_epoch_{}.png'.format(epo)))

#    train_average_loss = np.average(loss_record)

#    print ("train epoch #{} finish, loss average {}".format(epo, train_average_loss))

#    loss_record = []


#=============================================
#        Save Model & Loss
#=============================================

torch.save(Res_model.state_dict(), os.path.join(ROOT_DIR, 'multisource_cocktail/DAE/DAE_multi_2.pkl'))
torch.save(A_model.state_dict(), os.path.join(ROOT_DIR, 'multisource_cocktail/ANet/ANet_raw_2.pkl'))
#torch.save(featurenet.state_dict(), os.path.join(ROOT_DIR, 'multisource_cocktail/featureNet/FeatureNet_multi_2.pkl'))