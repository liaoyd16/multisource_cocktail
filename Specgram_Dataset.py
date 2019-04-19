import __init__

import torch
from torch.utils.data import Dataset, DataLoader
from utils.dataset_meta import *
from utils.dir_utils import list_json_in_dir
import os

import json
import numpy as np

class MSourceDataSet(Dataset):
    
    def __init__(self, clean_dir):
        self.curr_json_index = -1
        self.spec_block = None
        self.clean_dir = clean_dir
        self.clean_blocks = list_json_in_dir(clean_dir)
        
    def __len__(self):
        return ENTRIES_PER_JSON * CLASSES * len(self.clean_blocks)
                
    def __getitem__(self, index): 
        # index = json_index * ENTRIES_PER_JSON * CLASSES + entry_index * CLASSES + class_index
        json_index = index // (ENTRIES_PER_JSON * CLASSES)
        entry_index = (index % (ENTRIES_PER_JSON * CLASSES)) // CLASSES
        class_index = index % CLASSES

        if not self.curr_json_index == json_index:
            self.curr_json_index = json_index
            print("json loaded: ", os.path.join(self.clean_dir, self.clean_blocks[json_index]))
            self.spec_block = np.array(
                json.load(open(os.path.join(self.clean_dir, self.clean_blocks[json_index]), "r"))
            ).transpose(1,0,2,3)

        return torch.Tensor(self.spec_block[entry_index, class_index])
