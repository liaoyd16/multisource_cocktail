import torch
from torch.utils.data import Dataset, DataLoader

from utils.dir_utils import *

import numpy as np
import json
import os

from utils.dataset_meta import *


class denoiseDataSet():

    '''
    参数中的 xxx_block/xxx_blocks 为路径/路径list
    self.xxx_block/xxx_blocks 为np.array
    '''
    def __init__(self, block_dir, feat_block, noise_block, spec_blocks):
        self.feat_block = np.array( 
            json.load(open(os.path.join(block_dir, feat_block), "r"))
        ).transpose(1,0,2,3)

        self.noise_block = np.array( json.load(open(os.path.join(NOISE_DIR, noise_block), "r")))
        
        spec_block = np.array(
            json.load(open(os.path.join(block_dir, spec_blocks[0]), "r"))
        ).transpose(1,0,2,3)
        self.f_n_c = gen_f_n_c(self.feat_block, self.noise_block, spec_block)

        self.curr_json_index = 0
        self.curr_fnc_index = 0

        print("FNC-DataSet: feature block: {}, noise block: {}".format(feat_block, noise_block) )
        self.block_dir = block_dir
        self.spec_blocks = spec_blocks

    def __len__(self):
        return ENTRIES_PER_JSON * len(self.spec_blocks) * ALL_SAMPLES_PER_ENTRY

    def __getitem__(self, index):

        # block号
        newest_json_index = index // (ENTRIES_PER_JSON * ALL_SAMPLES_PER_ENTRY)
        newest_fnc_index = index % (ENTRIES_PER_JSON * ALL_SAMPLES_PER_ENTRY)

        #print(index, newest_json_index, newest_fab_index)

        if not (self.curr_json_index == newest_json_index):
            print("load new block")
            self.curr_json_index = newest_json_index
            spec_block = np.array(json.load(open(self.block_dir + '{}'.format(self.spec_test_blocks[newest_json_index])))).transpose(1,0,2,3)
            self.f_n_c = gen_f_n_c(self.feat_block, self.noise_block, spec_block)

        f = torch.Tensor(self.f_n_c[newest_fnc_index, 0])
        n = torch.Tensor(self.f_n_c[newest_fnc_index, 1])
        c = torch.Tensor(self.f_n_c[newest_fnc_index, 2])

        return f, n, c


def gen_f_n_c(feat_block, noise_block, spec_block):
    fnc = []
    for cl in range(CLASSES):
        feats = np.array(
            [feat_block[0, cl] for _ in range(ENTRIES_PER_JSON)]
        ).reshape(ENTRIES_PER_JSON, 1, 256, 128)
        
        noise = np.array( 
            noise_block[ np.random.choice(ENTRIES_PER_JSON, ENTRIES_PER_JSON, replace=True) ] 
        ).reshape(ENTRIES_PER_JSON, 1, 256, 128)
        
        specs = spec_block[:, cl].reshape(ENTRIES_PER_JSON, 1, 256, 128)
        
        one_class_f_n_c = np.concatenate((feats, noise+specs, specs), axis=1)
        fnc.append(one_class_f_n_c)

    ans = np.concatenate(fnc, axis=0)
    return ans
