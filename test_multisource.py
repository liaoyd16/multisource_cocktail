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

from DAE.aux import *
from DAE.resblock import ResBlock as ResBlock
from DAE.resblock import ResTranspose as ResTranspose


#=============================================
#        path
#=============================================

from utils.dir_utils import *
from utils.dataset_meta import *


#=============================================
#        Hyperparameters
#=============================================

BS = 10

ATTEND = True

#=============================================
#        Define Dataloader
#=============================================

from FMA_Dataset import FMA_DataSet

testset = FMA_DataSet(FEAT_DIR_TEST, list_json_in_dir(FEAT_DIR_TEST)[0], 
                      NOISE_DIR_TEST, list_json_in_dir(NOISE_DIR_TEST),
                      TEST_DIR, list_json_in_dir(TEST_DIR),
                      0)
testloader = torch.utils.data.DataLoader(dataset = testset,
    batch_size = BS,
    shuffle = False)


#=============================================
#        Model
#=============================================
from featureNet import featureNet

featurenet = featureNet()
try:
    if ATTEND:
        featurenet.load_state_dict(torch.load(os.path.join(ROOT_DIR, 'multisource_cocktail/featureNet/FeatureNet_multi.pkl')))
    else:
        featurenet.load_state_dict(torch.load(os.path.join(ROOT_DIR, 'multisource_cocktail/featureNet/FeatureNet.pkl')))
except Exception as e:
    print(e, "F-model not available")



from ANet import ANet

A_model = ANet()
try:
    A_model.load_state_dict(torch.load(os.path.join(ROOT_DIR, 'multisource_cocktail/ANet/ANet_multi.pkl')))
except Exception as e:
    print(e, "A-model not available")
# print(A_model)



from conv_fc import ResDAE

Res_model = ResDAE()
try:
    if ATTEND:
        Res_model.load_state_dict(torch.load(os.path.join(ROOT_DIR, 'multisource_cocktail/DAE/DAE_multi.pkl')))
    else:
        Res_model.load_state_dict(torch.load(os.path.join(ROOT_DIR, 'multisource_cocktail/DAE/DAE.pkl')))
except Exception as e:
    print(e, "Res-model not available")
# print(Res_model)





#=============================================
#        Optimizer
#=============================================

criterion = nn.MSELoss()

#=============================================
#        Loss Record
#=============================================

test_record = []
top_record = []


from mel import mel


Res_model.eval()

for i, data in enumerate(testloader, 0):

    # get mix spec & label        
    feat_data, mix_specs, a_specs = data

    target_specs = mel(a_specs)

    if ATTEND:
        # get feature
        feats = featurenet.feature(mel(feat_data))
        # feed in feature to ANet
        a7, a6, a5, a4, a3, a2 = A_model(feats)
        # Res_model
        tops = Res_model.upward(mel(mix_specs), a7, a6, a5, a4, a3, a2)
    else:
        tops = Res_model.upward(mel(mix_specs))

    top_record.append(tops.detach().numpy().tolist())
    outputs = Res_model.downward(tops, shortcut = True)
    loss_test = criterion(outputs, target_specs)

    print ('[%d] loss_test: %.3f' % (i, loss_test.item()))
    test_record.append(loss_test.item)

    if i % 5 == 0:
        # print images: mix, target, attention, separated
        mixx = (mix_specs[0]).view(256, 128).detach().numpy() * 255
        np.clip(mixx, np.min(mixx), 1)
        cv2.imwrite(os.path.join(ROOT_DIR, 'results/noise_test/' + str(i)  + "_mix.png"), mixx)

        tarr = (target_specs[0]).view(256, 128).detach().numpy() * 255
        np.clip(tarr, np.min(tarr), 1)
        cv2.imwrite(os.path.join(ROOT_DIR, 'results/noise_test/' + str(i)  + "_tar.png"), tarr)

        outt = (outputs[0]).view(256, 128).detach().numpy() * 255
        np.clip(outt, np.min(outt), 1)
        cv2.imwrite(os.path.join(ROOT_DIR, 'results/noise_test/' + str(i)  + "_sep.png"), outt)

with open(os.path.join(ROOT_DIR, 'results/tops/top_atten_{}.json'.format(ATTEND)), "w") as f:
    json.dump(top_record,f)

test_average_loss = np.average(test_record)
print ("loss average {}".format(test_average_loss))

