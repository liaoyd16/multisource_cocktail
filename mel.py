import numpy as np
import torch

EPS = 1e-5

def lg(x):
    x[x <= 0] = EPS
    return torch.log(x) / torch.log(torch.Tensor([np.e]))

def mel(specgram):
    return lg(1 + specgram/ 4) / 10