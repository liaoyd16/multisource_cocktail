import numpy as np
import torch

EPS = 1e-5

def lg(x):
    x[x <= 0] = EPS
    return torch.log(x) / torch.log(torch.Tensor([10]))

def mel(specgram):
    specgram[specgram < 0] = 0
    return lg(1 + specgram/ 4)

def norm(specgram):
	specgram[specgram < 0] = 0
	specgram[specgram > 255] = 255
	specgram /= 255
	return specgram

def mel_norm(specgram):
	return mel(specgram) + norm(specgram)

def mix(a_spec, b_spec):
    return a_spec + b_spec