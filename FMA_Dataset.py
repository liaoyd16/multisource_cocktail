import torch
from torch.utils.data import Dataset, DataLoader

from utils.dataset_meta import *

import numpy as np
import json
import os

from utils.dataset_meta import *


class FMA_DataSet():
    
    def __init__(self, feat_dir, feat_block, mix_dir, mix_blocks, a_dir, a_blocks, attendee):
        self.feat_block = np.array( 
            json.load(open(os.path.join(feat_dir, feat_block), "r"))
        ).transpose(1,0,2,3)

        self.feat_dir = feat_dir
        self.mix_dir = mix_dir
        self.a_dir = a_dir

        self.mix_block = np.array(json.load(open(os.path.join(mix_dir, mix_blocks[0]), "r")))
        self.a_block = np.array(json.load(open(os.path.join(a_dir, a_blocks[0]), "r")))

        self.curr_json_index = 0
        self.mix_blocks = mix_blocks
        self.a_blocks = a_blocks

        self.attendee = 0

    def __len__(self):
        return ENTRIES_PER_JSON * len(self.mix_blocks)

    def __getitem__(self, index):
        new_json_index = index % ENTRIES_PER_JSON
        print("index = {}, new_json_index = {}".format(index, new_json_index))
        if not new_json_index == self.curr_json_index:
            self.curr_json_index = new_json_index
            self.mix_block = np.array(json.load(open(os.path.join(self.mix_dir, self.mix_blocks[self.curr_json_index]), "r")))
            self.a_block = np.array(json.load(open(os.path.join(self.a_dir, self.a_blocks[self.curr_json_index]), "r")))

        return torch.Tensor(self.feat_block[index, self.attendee]), \
               torch.Tensor(self.mix_block[index]), \
               torch.Tensor(self.a_block[index])