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
from FAB_Dataset import FAB_DataSet

testset = FAB_DataSet(TRAIN_DIR, list_json_in_dir(TRAIN_DIR)[:2])
testloader = torch.utils.data.DataLoader(dataset = testset,
                                      batch_size = bs,
                                      shuffle = False)

#=============================================
#        Model
#=============================================
import featureNet
from featureNet import featureNet

featureNet = featureNet()
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
# test
# ============================================


W0 = 256
H0 = 128
LAYER_NO_MIN = 2
LAYER_NO_MAX = 7

def get_strf(layerno):
    if layerno < LAYER_NO_MIN or layerno > LAYER_NO_MAX:
        raise ValueError("target_layer not in {}~{}".format(LAYER_NO_MIN, LAYER_NO_MAX))

    Res_model.eval()
    A_model.eval()
    featureNet.eval()

    zero_input = torch.Tensor(np.zeros(1,1,W0,H0))
    Res_model.do_strf(zero_input)
    zero_output = Res_model.xs[layerno - LAYER_NO_MIN]
    layer_shape = zero_output.shape

    if len(layer_shape) == 4: # required layer is a conv layer
        chan, wid, hei = layer_shape[1], layer_shape[2], layer_shape[3]
        layer_strfs = np.zeros((chan, wid, hei, W0, H0))

        N = 0
        for i, data in enumerate(testloader, 0):
            N += 1
            feat_data, _, a_specs = data
        
            a_specs = mel_norm(a_specs)
        
            feats = featurenet.feature(feat_data)
        
            a7, a6, a5, a4, a3, a2 = A_model(feats)

            Res_model.do_strf(a_specs, a7, a6, a5, a4, a3, a2)

            response = Res_model.xs[layerno - LAYER_NO_MIN][0] # use [0] because BS = 1: feeding one sample a time
            if len(layer_shape) == 4:
                layer_strfs[:,:,:] += response.reshape(chan, wid, hei, 1, 1) * a_specs.detach().numpy()
            else:
                layer_strfs[:] += response.reshape(wid, 1, 1) * a_specs.detach().numpy()

        layer_strfs /= N

    return layer_strfs

if __name__ == '__main__':
    strf = get_strf(7)

    import json
    json.dump(strf.tolist(), open("strf_clean.json", "w"))