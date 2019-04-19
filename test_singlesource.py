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
import random
random.seed(7)

from utils.dir_utils import ROOT_DIR, TRAIN_DIR, TEST_DIR, list_json_in_dir


#=============================================
#        Hyperparameters
#============================================
bs = 1

#=============================================
#        Define Dataloader
#=============================================
from FMA_Dataset import FMA_DataSet

testset = FMA_DataSet(TEST_DIR, ['captaincook_white.json'], \
                      TEST_DIR, ["captaincook_clean.json"])
#testset = FAB_DataSet(TRAIN_DIR, list_json_in_dir(TRAIN_DIR)[:2])
testloader = torch.utils.data.DataLoader(dataset = testset,
                                      batch_size = bs,
                                      shuffle = False)


#=============================================
#        Model
#=============================================
import featureNet
from featureNet import featureNet

featurenet = featureNet()
try:
    featurenet.load_state_dict(torch.load(os.path.join(ROOT_DIR, 'multisource_cocktail/featureNet/FeatureNet.pkl')))
except Exception as e:
    print(e, "F-model not available")


import ANet
from ANet import ANet

A_model = ANet()
try:
    A_model.load_state_dict(torch.load(os.path.join(ROOT_DIR, 'multisource_cocktail/ANet/ANet_raw_2.pkl')))
except Exception as e:
    print(e, "A-model not available")
# print(A_model)


import conv_fc
from conv_fc import ResDAE

Res_model = ResDAE()
try:
    Res_model.load_state_dict(torch.load(os.path.join(ROOT_DIR, 'multisource_cocktail/DAE/DAE_multi_2.pkl')))
except Exception as e:
    print(e, "Res-model not available")
# print(Res_model)


# ============================================
# optimizer
# ============================================

criterion = nn.MSELoss()


# ============================================
# test
# ============================================

from mel import mel_norm, mel
zeros = torch.zeros(bs, 256, 128)

Res_model.eval()
for epo in range(1):
    # train
    for i, data in enumerate(testloader, 0):
        # get mix spec & label
        feat_data, _, a_specs = data

        feat_data = feat_data.squeeze()
        a_specs = a_specs.squeeze()

        a_specs = mel_norm(a_specs)
        target_specs = a_specs

        # get feature
        feats = featurenet.feature(feat_data)

        # feed in feature to ANet
        a7, a6, a5, a4, a3, a2 = A_model(feats)

        # Res_model
        tops = Res_model.upward(a_specs, a7, a6, a5, a4, a3, a2)

        outputs = Res_model.downward(tops, shortcut = True).squeeze()

        loss_train = criterion(outputs, target_specs)

        print ('[%d, %5d] loss: %.3f, input: %.3f, output: %.3f'\
         % (epo, i, loss_train.item(), criterion(target_specs, zeros).item(), criterion(outputs, zeros).item()))
        
        if i % 1 == 0:
            # print images: mix, target, attention, separated

#            inn = a_specs[0].view(256, 128).detach().numpy() * 255
#            np.clip(inn, np.min(inn), 1)
#            cv2.imwrite(os.path.join(ROOT_DIR, 'results/single_source/' + str(i)  + "_mix.png"), inn)

            tarr = target_specs.view(256, 128).detach().numpy() * 255
            cv2.imwrite(os.path.join(ROOT_DIR, 'results/single_source/test_singlesource/' + str(i)  + "_tar.png"), tarr)

            outt = outputs.view(256, 128).detach().numpy() * 255
            cv2.imwrite(os.path.join(ROOT_DIR, 'results/single_source/test_singlesource/' + str(i)  + "_sep.png"), outt)

 

            gc.collect()
            plt.close("all")
