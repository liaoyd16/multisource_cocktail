import torch
import torch.nn as nn
import torch.nn.functional as F

'''ANet'''
class ANet(nn.Module):
    
    def __init__(self):
        super(ANet, self).__init__()

        self.linear7 = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
        )
        self.linear6 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.linear5 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.linear4 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.linear3 = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(256, 16),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.view(-1, 1, 256)

        a7 = self.linear7(x).view(-1, 512, 1, 1)
        a6 = self.linear6(x).view(-1, 256, 1, 1)
        a5 = self.linear5(x).view(-1, 128, 1, 1)
        a4 = self.linear4(x).view(-1, 64, 1, 1)
        a3 = self.linear3(x).view(-1, 32, 1, 1)
        a2 = self.linear2(x).view(-1, 16, 1, 1)

        return a7, a6, a5, a4, a3, a2
