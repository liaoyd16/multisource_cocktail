import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import pickle
import os
import json
import numpy as np
import random 
random.seed(0)


#=============================================
#        Hyperparameters
#=============================================

epoch = 2
bs = 10

#========================================

#========================================
from Spec_Label_Dataset import Spec_Label_Dataset as Spec_Label_Dataset

#=================================================    
#           Dataloader 
#=================================================
from utils.dir_utils import TEST_DIR, TRAIN_DIR, ROOT_DIR
featureset = Spec_Label_Dataset(TRAIN_DIR)
testloader = torch.utils.data.DataLoader(dataset = featureset,
                                                batch_size = bs,
                                                shuffle = False)

#============================================
#              testing
#============================================
from featureNet import featureNet as featureNet

def test(model):
    criterion = torch.nn.NLLLoss()

    loss_record = []
    every_loss = []
    epoch_loss = []
    correct = 0
    total = 0
    blank = []
    
    model.eval()
    
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):

            print("testing i = {}".format(i))

            inputs, labels = data
            outputs = model(inputs)
            labels = labels.to(dtype=torch.long)
    
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            
            correct += (predicted == labels).sum()

            loss_record.append(loss.item())
            every_loss.append(loss.item())
            
        epoch_loss.append(np.mean(every_loss))
        every_loss = []

        plt.figure(figsize = (20, 10))
        plt.plot(loss_record)
        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.savefig(os.path.join(ROOT_DIR, 'results/featurenet/test_loss.png'))
        plt.close()

    return correct, total

if __name__ == '__main__':
    model_for_test = featureNet()
    model_for_test.load_state_dict(torch.load('/home/tk/Documents/FeatureNet.pkl'))
    print(model_for_test)
    test(model)
