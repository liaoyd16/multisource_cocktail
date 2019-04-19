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
import random
random.seed(7)

from utils.dir_utils import ROOT_DIR

#=============================================
#        Hyperparameters
#=============================================

epoch = 10
lr = 0.002
mom = 0.9
bs = 10
reuse = True

#=================================================    
#               Dataloader 
#=================================================
from Spec_Label_Dataset import Spec_Label_Dataset as Spec_Label_Dataset
from utils.dir_utils import TRAIN_DIR
featureset  = Spec_Label_Dataset(TRAIN_DIR)
trainloader = torch.utils.data.DataLoader(dataset = featureset,
                                                batch_size = bs,
                                                shuffle = False) # must be False for efficiency

#=================================================    
#               load
#=================================================
from featureNet import featureNet as featureNet

model = featureNet()
try:
    if reuse:
        model.load_state_dict(torch.load(os.path.join(ROOT_DIR, "multisource_cocktail/featureNet/FeatureNet.pkl")))
except:
    print("reused model not available")
print (model)


#============================================
#              optimizer
#============================================
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = mom)


#============================================
#              training
#============================================
import test_featurenet
from test_featurenet import test as test

loss_record = []
every_loss = []
epoch_loss = []
epoch_accu = []

#from mel import mel, norm

model.train()
for epo in range(epoch):

    for i, data in enumerate(trainloader, 0):
        
        inputs, labels = data
        
        optimizer.zero_grad()
        outputs = model(inputs)
        labels = labels.to(dtype=torch.long)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_record.append(loss.item())
        every_loss.append(loss.item())
        print ('[%d, %5d] loss: %.3f hits: %d/%d' % 
            (
                epo, i, loss.item(), 
                np.sum( np.argmax(outputs.detach().numpy(), axis=1) == labels.detach().numpy()),
                bs
            )
        )

    epoch_loss.append(np.mean(every_loss))
    every_loss = []

    print("testing")
    corr, total = test(model)
    accuracy = (float)(corr) / total
    epoch_accu.append(accuracy)
    print('test: [%d] accuracy: %.4f' % (epo, accuracy))

            
torch.save(model.state_dict(), os.path.join(ROOT_DIR, "multisource_cocktail/featureNet/FeatureNet.pkl"))


plt.figure(figsize = (20, 10))
plt.plot(loss_record)
plt.xlabel('iterations')
plt.ylabel('loss')
plt.savefig(os.path.join(ROOT_DIR, 'results/featurenet/train_batch_loss.png'))
plt.show()

plt.figure(figsize = (20, 10))
plt.plot(epoch_loss)
plt.xlabel('iterations')
plt.ylabel('epoch_loss')
plt.savefig(os.path.join(ROOT_DIR, 'results/featurenet/train_epoch_loss.png'))
plt.show()

#plt.figure(figsize = (20, 10))
#plt.plot(epoch_accu)
#plt.xlabel('iterations')
#plt.ylabel('accu')
#plt.savefig('accuracy.png')
#plt.show()
