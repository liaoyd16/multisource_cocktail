import torch
from torch.utils.data import Dataset, DataLoader

from utils.dataset_meta import *

import numpy as np
import json
import os

from utils.dataset_meta import *
from utils.dir_utils import *

CLASSES_FMA = 1

class FMA_DataSet():
    
    def __init__(self, mix_dir, mix_blocks, a_dir, a_blocks):
        self.feat_block = np.array(
            json.load(open(os.path.join(FEAT_DIR, list_json_in_dir(FEAT_DIR)[0]), "r"))
        ).transpose(1,0,2,3)

        self.mix_dir = mix_dir
        self.a_dir = a_dir

        self.mix_block = np.array(
            json.load(open(os.path.join(mix_dir, mix_blocks[0]), "r"))
        ).transpose(1,0,2,3)
        self.a_block = np.array(
            json.load(open(os.path.join(a_dir, a_blocks[0]), "r"))
        ).transpose(1,0,2,3)

        self.curr_json_index = 0
        self.mix_blocks = mix_blocks
        self.a_blocks = a_blocks

    def __len__(self):
        return ENTRIES_PER_JSON * len(self.mix_blocks) * CLASSES_FMA

    def __getitem__(self, index):
        new_json_index = index // ENTRIES_PER_JSON
        index = index % ENTRIES_PER_JSON
        entry_index = index // CLASSES_FMA
        class_index = index % CLASSES_FMA

        if not new_json_index == self.curr_json_index:
            self.curr_json_index = new_json_index
            self.mix_block = np.array(
                json.load(open(os.path.join(self.mix_dir, self.mix_blocks[self.curr_json_index]), "r"))
            ).transpose(1,0,2,3)
            self.a_block = np.array(
                json.load(open(os.path.join(self.a_dir, self.a_blocks[self.curr_json_index]), "r"))
            ).transpose(1,0,2,3)

        return torch.Tensor(self.feat_block[0, class_index]), \
               torch.Tensor(self.mix_block[entry_index, class_index]), \
               torch.Tensor(self.a_block[entry_index, class_index])
