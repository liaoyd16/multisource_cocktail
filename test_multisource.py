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

import acoustics
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

from utils.dir_utils import *
from utils.dataset_meta import *

#=============================================
#        functions
#=============================================


#=============================================
#        Hyperparameters
#=============================================

BS = 1

ATTEND = True

#=============================================
#        Define Dataloader
#=============================================

from FAB_Dataset import FAB_DataSet
from FMA_Dataset import FMA_DataSet


#=============================================
#        Optimizer
#=============================================

criterion = nn.MSELoss()

#=============================================
#        Loss Record
#=============================================

top_record = []


from mel import mel, norm, mix, mel_norm
zeros = torch.zeros(BS, 256, 128)

input_sum = []
output_sum = []

def inv_mel(spec):
    return (10 ** (spec) - 1) * 4

def test_multisource(Res_model, A_model, featurenet, main=False):
    Res_model.eval()
    test_record = []

#    testset = FAB_DataSet(TEST_DIR, ['people_to_people.json']) # use this for multispeaker
    testset = FMA_DataSet(TEST_DIR, ['captaincook_white.json'], TEST_DIR, ['captaincook_clean.json'])
    if main:
        testloader = torch.utils.data.DataLoader(dataset = testset, \
            batch_size = BS, \
            shuffle = False)
    else:
        testloader = torch.utils.data.DataLoader(dataset = testset, \
            batch_size = BS, \
            shuffle = False)

    for i, data in enumerate(testloader, 0):
        # get mix spec & label
        feat_data, mix_specs, a_specs = data # use this for denoise test

        print ("mix_specs = ", np.sum(np.array(mix_specs)))
        mix_specs = mel_norm(mix_specs)/10      # use this for denoise test
#        feat_data, a_specs, b_specs = data  # use this for multispeaker test
#        mix_specs = mel_norm(mix(a_specs, b_specs)) # use this for multispeaker test
        print ("mix_specs = ", np.sum(np.array(mix_specs)))

        target_specs = mel_norm(a_specs)/10
#        b_specs = mel_norm(b_specs) # use this for multispeaker test
#        target_specs = a_specs       # use this for multispeaker test

        if ATTEND:
            # get feature
            feats = featurenet.feature(feat_data)
            # feed in feature to ANet
            a7, a6, a5, a4, a3, a2 = A_model(feats)
            # Res_model
            tops = Res_model.upward(mix_specs, a7, a6, a5, a4, a3, a2)
        else:
            tops = Res_model.upward(mix_specs)

        #top_record.append(tops.detach().numpy().tolist())
        outputs = Res_model.downward(tops, shortcut = True)
        loss_test = criterion(outputs, target_specs)

        if (not main) or (main and i % 1 == 0):
            # print images: mix, target, attention, separated
            tarr = (target_specs[0]).view(256, 128).detach().numpy()
            #mask = (b_specs[0]).view(256, 128).detach().numpy()
            mixx = (mix_specs[0]).view(256, 128).detach().numpy()
            outt = (outputs[0]).view(256, 128).detach().numpy()
            
            cv2.imwrite(os.path.join(ROOT_DIR, 'results/people_to_people/' + str(i)  + "_tar.png"), tarr)
#            cv2.imwrite(os.path.join(ROOT_DIR, 'results/people_to_people/' + str(i)  + "_mask.png"), mask)
            cv2.imwrite(os.path.join(ROOT_DIR, 'results/people_to_people/' + str(i)  + "_mix.png"), mixx)
            cv2.imwrite(os.path.join(ROOT_DIR, 'results/people_to_people/' + str(i)  + "_sep.png"), outt)
            
            print(np.mean(inv_mel(tarr[-10:])), 
                  np.mean(inv_mel(mixx[-10:])))

    return np.average(test_record)

#with open(os.path.join(ROOT_DIR, 'results/tops/top_atten_{}.json'.format(ATTEND)), "w") as f:
#    json.dump(top_record,f)

if __name__=="__main__":
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
    try:
        if ATTEND:
            A_model.load_state_dict(torch.load(os.path.join(ROOT_DIR, 'multisource_cocktail/ANet/ANet_multi_2_trained.pkl')))
        else:
            A_model.load_state_dict(torch.load(os.path.join(ROOT_DIR, 'multisource_cocktail/ANet/ANet_raw_2.pkl')))
    except Exception as e:
        print(e, "A-model not available")
    # print(A_model)



    from conv_fc import ResDAE

    Res_model = ResDAE()
    try:
        if ATTEND:
            Res_model.load_state_dict(torch.load(os.path.join(ROOT_DIR, 'multisource_cocktail/DAE/DAE_multi_2.pkl')))
        else:
            Res_model.load_state_dict(torch.load(os.path.join(ROOT_DIR, 'multisource_cocktail/DAE/DAE_raw_2.pkl')))
    except Exception as e:
        print(e, "Res-model not available")
    # print(Res_model)

    test_average_loss = test_multisource(Res_model, A_model, featurenet, main=True)
    print ("loss average {}".format(test_average_loss))
