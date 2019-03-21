import torch
from torch.utils.data import Dataset, DataLoader

from utils.dataset_meta import *

import numpy as np
import json
import os

from utils.dataset_meta import *


class FAB_DataSet():
    '''
    基于 __CLASSES__ENTRIES_PER_JSON__256__128__ 格式数据块
    的 dataloader
    load的结果为一个f(feat)-a(speaker-A)-b(speaker-B)

    构造方法：
        - 提供 block_dir ，feat_list 和 spec_list ，并指示是否 generate_fab_randomly

    类变量：
        - self.feat_block：读取到进程内存中，用于作为feat分量
        - 读取标号：
         - self.curr_json_index：trainset中的block号
         - self.curr_entry_index：block中的entry号
         - self.curr_fab_index：entry对应的所有(ALL_SAMPLES_PER_ENTRY) fab的编号
    '''
    def __init__(self, block_dir, feat_block, spec_blocks):
        self.feat_block = np.array( 
            json.load(open(os.path.join(block_dir, feat_block), "r"))
        ).transpose(1,0,2,3)
        
        spec_block = np.array(
            json.load(open(os.path.join(block_dir, spec_blocks[0]), "r"))
        ).transpose(1,0,2,3)
        self.f_a_b = gen_f_a_b(spec_block, self.feat_block)

        self.curr_json_index = 0
        self.curr_fab_index = 0

        print("FAB-DataSet: feature blocks: ", feat_block)
        self.block_dir = block_dir
        self.spec_blocks = spec_blocks

    def __len__(self):
        return ENTRIES_PER_JSON * len(self.spec_blocks) * ALL_SAMPLES_PER_ENTRY

    def __getitem__(self, index):

        # block号
        newest_json_index = index // (ENTRIES_PER_JSON * ALL_SAMPLES_PER_ENTRY)
        newest_fab_index = index % (ENTRIES_PER_JSON * ALL_SAMPLES_PER_ENTRY)

        if not (self.curr_json_index == newest_json_index):
            print("loading new block")
            self.curr_json_index = newest_json_index
            f = open(self.block_dir + '{}'.format(self.spec_blocks[newest_json_index]))
            spec_block = np.array(json.load(f)).transpose(1,0,2,3)
            f.close()
            self.f_a_b = gen_f_a_b(spec_block, self.feat_block)

        f = torch.Tensor(self.f_a_b[0, newest_fab_index])
        a = torch.Tensor(self.f_a_b[1, newest_fab_index])
        b = torch.Tensor(self.f_a_b[2, newest_fab_index])

        return f, a, b


'''
tools:
to gen f-a-b block of a block
'''
def gen_all_pairs():
    all_pairs = []
    for i in range(CLASSES):
        for j in range(CLASSES):
            if(i==j): continue
            all_pairs.append([i, j])
    return np.array(all_pairs)

all_combinations = gen_all_pairs()

def gen_f_a_b(spec_block, feat_block):
    fab = []
    for entry_index in range(spec_block.shape[0]):
        a_b_indexes = all_combinations.transpose()
        a_index_list, b_index_list = a_b_indexes[0], a_b_indexes[1]

        a_b = np.array([
            spec_block[entry_index, a_index_list], 
            spec_block[entry_index, b_index_list]
        ])
        feats = feat_block[
                    0, #np.random.randint(feat_block.shape[0]),
                    a_index_list
                ].reshape(1, ALL_SAMPLES_PER_ENTRY, 256, 128)
        temp = np.concatenate((feats, a_b), axis=0)
        # print(temp.shape)
        fab.append(temp)

    ans = np.concatenate(fab, axis=1)

    print("gen fab finish")
    return ans
