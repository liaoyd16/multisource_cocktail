import torch
from torch.utils.data import Dataset, DataLoader
from utils.dataset_meta import *
import os

class MSourceDataSet(Dataset):
    
    def __init__(self, clean_dir):
        self.curr_json_index = -1
        self.spec_block = None
        self.clean_dir = clean_dir
        self.clean_blocks = os.listdir(clean_dir)
        
    def __len__(self):
        return ENTRIES_PER_JSON * CLASSES * len(self.clean_blocks)
                
    def __getitem__(self, index): 
        # index = json_index * ENTRIES_PER_JSON * CLASSES + entry_index * CLASSES + class_index
        json_index = index // (ENTRIES_PER_JSON * CLASSES)
        entry_index = (index % (ENTRIES_PER_JSON * CLASSES)) // CLASSES
        class_index = index % CLASSES

        if not self.curr_json_index == json_index:
            self.spec_block = json.load(open(os.join(self.clean_dir, self.clean_blocks[json_index]), "r")).transpose(1,0,2,3)

        return self.spec_block[entry_index, class_index]