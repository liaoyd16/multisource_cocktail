import torch
from torch.utils.data import Dataset, DataLoader

from utils.dataset_meta import *
from utils.dir_utils import TRAIN_DIR, TEST_DIR

import numpy as np
import json
import os

from utils.dataset_meta import *

import gc

class BlockBasedDataSet(Dataset):
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
    def __init__(self, block_dir, feat_block_list, spec_block_list, gen_fab_random_mode):
        self.feat_block = []
        for block in feat_block_list: self.feat_block.append( json.load(open(os.path.join(block_dir, block), "r")) )
        self.feat_block = np.concatenate( np.array(self.feat_block), axis=1 ).transpose(1,0,2,3)

        self.curr_json_index = 0
        self.curr_entry_index = 0

        self.spec_block = np.array(json.load(open(os.path.join(block_dir, spec_block_list[0]), "r"))).transpose(1,0,2,3)
        self.f_a_b = gen_f_a_b(self.spec_block, self.feat_block, random_mode=gen_fab_random_mode)

        print(self.f_a_b.shape)

        self.curr_fab_index = 0

    def __len__(self):
        return 0

    def __getitem__(self, index):
        return None


class trainDataSet(BlockBasedDataSet):
    # 不用考虑batch了，直接一个一个读取
    # 从block中，取出entry
    # 从entry中，取出一系列f-a-b
    def __init__(self, bs, feat_train_blocks, spec_train_blocks):
        print("trainDataSet: feature blocks: ", feat_train_blocks)
        super(trainDataSet, self).__init__(TRAIN_DIR, feat_train_blocks, spec_train_blocks, gen_fab_random_mode=False)
        self.spec_train_blocks = spec_train_blocks

    def __len__(self):
        return ENTRIES_PER_JSON * len(self.spec_train_blocks) * ALL_SAMPLES_PER_ENTRY

    def __getitem__(self, index):

        # block号
        newest_json_index = index // (ENTRIES_PER_JSON * ALL_SAMPLES_PER_ENTRY)
        newest_fab_index = index % (ENTRIES_PER_JSON * ALL_SAMPLES_PER_ENTRY)

        if not (self.curr_json_index == newest_json_index):
            print("load new block")
            self.curr_json_index = newest_json_index
            f = open(clean_dir + '{}'.format(cleanfolder[newest_json_index]))
            self.spec_block = np.array(json.load(f)).transpose(1,0,2,3)
            self.f_a_b = gen_f_a_b(self.spec_block, self.feat_block, random_mode=False)

        f = torch.Tensor(self.f_a_b[0, newest_fab_index])
        a = torch.Tensor(self.f_a_b[1, newest_fab_index])
        b = torch.Tensor(self.f_a_b[2, newest_fab_index])

        return f, a, b


class testDataSet(BlockBasedDataSet):
    # 不用考虑batch了，直接一个一个读取
    # 从block中，取出entry
    # 从entry中，取出一系列f-a-b
    def __init__(self, bs, feat_test_blocks, spec_test_blocks):
        print("testDataSet: feature blocks: ", feat_test_blocks)
        super(testDataSet, self).__init__(TEST_DIR, feat_test_blocks, spec_test_blocks, gen_fab_random_mode=False)
        self.spec_test_blocks = spec_test_blocks

    def __len__(self):
        return ENTRIES_PER_JSON * len(self.spec_test_blocks) * ALL_SAMPLES_PER_ENTRY

    def __getitem__(self, index):

        # block号
        newest_json_index = index // (ENTRIES_PER_JSON * ALL_SAMPLES_PER_ENTRY)
        newest_fab_offset = index % (ENTRIES_PER_JSON * ALL_SAMPLES_PER_ENTRY)
        # newest_entry_index = entry_offset // ALL_SAMPLES_PER_ENTRY
        # newest_fab_index = entry_offset % ALL_SAMPLES_PER_ENTRY

        if not (self.curr_json_index == newest_json_index):
            print("load new block")
            self.curr_json_index = newest_json_index
            f = open(clean_dir + '{}'.format(cleanfolder[newest_json_index]))
            self.spec_block = np.array(json.load(f)).transpose(1,0,2,3)
            self.f_a_b = gen_f_a_b(self.spec_block, self.curr_entry_index, self.feat_block, random_mode=False)

        f = torch.Tensor(self.f_a_b[0, newest_fab_index])
        a = torch.Tensor(self.f_a_b[1, newest_fab_index])
        b = torch.Tensor(self.f_a_b[2, newest_fab_index])

        return f, a, b


def gen_all_pairs():
    all_pairs = []
    for i in range(CLASSES):
        for j in range(CLASSES):
            if(i==j): continue
            all_pairs.append([i, j])
    return np.array(all_pairs)

all_combinations = gen_all_pairs()
all_combination_indices = np.arange(CLASSES * (CLASSES-1))

def gen_rand_pairs(num_pairs):
    ''' 至多C(10,2)对组合 '''
    assert(num_pairs <= CLASSES * (CLASSES - 1))
    ''' 长为 num_pairs 的 list ，为 [0,CLASSES-1]x[0,CLASSES-1] 中的序偶 '''
    chosen = all_combinations[ 
        np.array( np.random.choice(all_combination_indices, num_pairs, replace=False) ) 
    ]
    return chosen

# def gen_f_a_b(spec_block, entry_index, feat_block, random_mode=True):
def gen_f_a_b(spec_block, feat_block, random_mode=True):
    fab = []
    for entry_index in range(spec_block.shape[0]):
        if random_mode: 
            samples_selected = RANDOM_SAMPLES_PER_ENTRY
        else:
            samples_selected = ALL_SAMPLES_PER_ENTRY
        a_b_indexes = gen_rand_pairs(samples_selected).transpose()
        a_index_list, b_index_list = a_b_indexes[0], a_b_indexes[1]

        a_b = np.array([
            spec_block[entry_index, a_index_list], 
            spec_block[entry_index, b_index_list]
        ])
        feats = feat_block[
                    0, #np.random.randint(feat_block.shape[0]),
                    a_index_list
                ].reshape(1, samples_selected, 256, 128)
        temp = np.concatenate((feats, a_b), axis=0)
        # print(temp.shape)
        fab.append(temp)

    ans = np.concatenate(fab, axis=1)

    print("gen fab finish")
    return ans
