'''imports'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import numpy as np

from aux import initialize
from resblock import ResBlock as ResBlock, ResTranspose as ResTranspose

class ResDAE(nn.Module):
    def __init__(self):
        super(ResDAE, self).__init__()

        # 256x128x1 -> 256x128x8
        self.upward_net1 = nn.Sequential(
            ResBlock(1, 8),
            ResBlock(8, 8),
            ResBlock(8, 8),
            nn.BatchNorm2d(8),
        )

        # 256x128x8 -> 128x64x16
        self.upward_net2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(2,2), stride=2),
            nn.ReLU(),
            ResBlock(8, 8),
            ResBlock(8, 16),
            ResBlock(16, 16),
            nn.BatchNorm2d(16),
        )

        # 128x64x16 -> 64x32x32
        self.upward_net3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2,2), stride=2),
            nn.ReLU(),
            ResBlock(16, 16),
            ResBlock(16, 32),
            ResBlock(32, 32),
            nn.BatchNorm2d(32),
        )

        # 64x32x32 -> 32x16x64
        self.upward_net4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2,2), stride=2),
            nn.ReLU(),
            ResBlock(32, 32),
            ResBlock(32, 64),
            ResBlock(64, 64),
            nn.BatchNorm2d(64),
        )

        # 32x16x64 -> 16x8x128
        self.upward_net5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2,2), stride=2),
            nn.ReLU(),
            ResBlock(64, 64),
            ResBlock(64, 128),
            ResBlock(128, 128),
            nn.BatchNorm2d(128),
        )

        # 16x8x128 -> 8x4x128
        self.upward_net6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(2,2), stride=2),
            nn.ReLU(),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 128),
            nn.BatchNorm2d(128),
        )

        self.fc1 = nn.Linear(4096, 256)
        self.fc2 = nn.Linear(256, 4096)

        # 8x4x128 -> 16x8x128
        self.downward_net6 = nn.Sequential(
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResTranspose(128, 128),
            nn.BatchNorm2d(128),
        )

        # 16x8x128 -> 32x16x64
        # (cat -> 16x8x256 -> 16x8x128)
        self.uconv5 = ResBlock(256, 128)
        self.downward_net5 = nn.Sequential(
            ResBlock(128, 128),
            ResBlock(128, 64),
            ResBlock(64, 64),
            ResTranspose(64, 64),
            nn.BatchNorm2d(64),
        )

        # 32x16x64 -> 64x32x32
        # (cat -> 32x16x128 -> 32x16x64)
        self.uconv4 = ResBlock(128, 64)
        self.downward_net4 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 32),
            ResBlock(32, 32),
            ResTranspose(32, 32),
            nn.BatchNorm2d(32),
        )

        # 64x32x32 -> 128x64x16
        # (cat -> 64x32x64 -> 64x32x32)
        self.uconv3 = ResBlock(64, 32)
        self.downward_net3 = nn.Sequential(
            ResBlock(32, 32),
            ResBlock(32, 16),
            ResBlock(16, 16),
            ResTranspose(16, 16),
            nn.BatchNorm2d(16),
        )

        # 128x64x16 -> 256x128x8
        # (cat -> 128x64x32 -> 128x64x16)
        self.uconv2 = ResBlock(32, 16)
        self.downward_net2 = nn.Sequential(
            ResBlock(16, 16),
            ResBlock(16, 8),
            ResBlock(8, 8),
            ResTranspose(8, 8),
            nn.BatchNorm2d(8),
        )

        # 256x128x8 -> 256x128x1
        self.downward_net1 = nn.Sequential(
            ResBlock(8, 8),
            ResBlock(8, 4),
            ResBlock(4, 1),
            ResBlock(1, 1),
            nn.BatchNorm2d(1),
        )
        
        self.apply(initialize)


    def upward(self, x, a7=None, a6=None, a5=None, a4=None, a3=None, a2=None):
        x = x.view(-1, 1, 256, 128)

        x = self.upward_net1(x)

        x = self.upward_net2(x)
        if a2 is not None: x = x * a2
        self.x2 = x

        x = self.upward_net3(x)
        if a3 is not None: x = x * a3
        self.x3 = x

        x = self.upward_net4(x)
        if a4 is not None: x = x * a4
        self.x4 = x

        x = self.upward_net5(x)
        if a5 is not None: x = x * a5
        self.x5 = x

        x = self.upward_net6(x)
        if a6 is not None: x = x * a6

        x = x.view(-1, 4096)
        x = self.fc1(x)
        if a7 is not None: x = x * a7.squeeze()

        self.top = x

        return x

    def do_strf(self, x, a7=None, a6=None, a5=None, a4=None, a3=None, a2=None):
        self.xs = []

        x = x.view(-1, 1, 256, 128)

        x = self.upward_net1(x)
        x = self.upward_net2(x)
        if a2 is not None: x = x * a2
        self.xs.append(x.detach().numpy())
        x = self.upward_net3(x)
        if a3 is not None: x = x * a3
        self.xs.append(x.detach().numpy())
        x = self.upward_net4(x)
        if a4 is not None: x = x * a4
        self.xs.append(x.detach().numpy())
        x = self.upward_net5(x)
        if a5 is not None: x = x * a5
        self.xs.append(x.detach().numpy())
        x = self.upward_net6(x)
        if a6 is not None: x = x * a6
        self.xs.append(x.detach().numpy())
        x = self.fc1(x.view(-1, 4096))
        if a7 is not None: x = x * a7.squeeze()
        self.xs.append(x.detach().numpy())


    def downward(self, y, shortcut= True):
        
        y = self.fc2(y)

        y = y.view(-1, 128, 8, 4)

        y = self.downward_net6(y)

        if shortcut:
            y = torch.cat((y, self.x5), 1)
            y = F.relu(self.uconv5(y))
        y = self.downward_net5(y)

        if shortcut:
            y = torch.cat((y, self.x4), 1)
            y = F.relu(self.uconv4(y))
        y = self.downward_net4(y)

        if shortcut:
            y = torch.cat((y, self.x3), 1)
            y = F.relu(self.uconv3(y))
        y = self.downward_net3(y)

        if shortcut:
            y = torch.cat((y, self.x2), 1)
            y = F.relu(self.uconv2(y))
        y = self.downward_net2(y)

        y = self.downward_net1(y)

        return y.reshape(-1, 256, 128)
