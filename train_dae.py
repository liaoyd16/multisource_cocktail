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

from utils.dir_utils import ROOT_DIR, TRAIN_DIR

#=============================================
#        Hyperparameters
#=============================================

epoch = 1
lr = 0.005
mom = 0.9
bs = 10

#=============================================
#        Define Dataloader
#=============================================
from Specgram_Dataset import MSourceDataSet

trainset = MSourceDataSet(TRAIN_DIR)
trainloader = torch.utils.data.DataLoader(dataset = trainset,
                                                batch_size = bs,
                                                shuffle = False)

#=============================================
#        Model
#=============================================
from DAE.conv_fc import ResDAE as ResDAE
from DAE.aux import white

model = ResDAE()
try:
    model.load_state_dict(torch.load(os.path.join(ROOT_DIR, 'multisource_cocktail/DAE/DAE.pkl')))
except:
    print("model not available")

#=============================================
#        Optimizer
#=============================================

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = mom)

#=============================================
#        Loss Record
#=============================================

loss_record = []

#=============================================
#        Train
#=============================================

model.train()
for epo in range(epoch):
    for i, data in enumerate(trainloader, 0):

        top = model.upward(data)
        outputs = model.downward(top, shortcut = True)

        targets = data.view(bs, 1, 256, 128)
        outputs = outputs.view(bs, 1, 256, 128)
        loss = criterion(outputs, targets)

        loss_record.append(loss.item())
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        print ('[%d, %5d] loss: %.3f' % (epo, i, loss.item()))

        if i % 5 == 0:
            inn = data[0].view(256, 128).detach().numpy() * 255
            cv2.imwrite(os.path.join(ROOT_DIR, 'results/autoencoder/' + str(epo) + "_" + str(i) + "_clean.png"), inn)
            
            out = outputs[0].view(256, 128).detach().numpy() * 255
            cv2.imwrite(os.path.join(ROOT_DIR, 'results/autoencoder/' + str(epo) + "_" + str(i) + "_re.png"), out)    

            plt.figure(figsize = (20, 10))
            plt.plot(loss_record)
            plt.xlabel('iterations')
            plt.ylabel('loss')
            plt.savefig(os.path.join(ROOT_DIR, 'results/autoencoder/DAE_loss.png'))
            plt.close("all")
            gc.collect()


#=============================================
#        Save Model & Loss
#=============================================
torch.save(model.state_dict(), os.path.join(ROOT_DIR, 'multisource_cocktail/DAE/DAE.pkl'))