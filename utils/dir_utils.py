
import os

def list_json_in_dir(dir):
    temp = os.listdir(dir)
    ans = []
    for t in temp:
        if '.json' == t[-5:]:
            ans.append(t)
    ans.sort()
    return ans

from utils.config_local import LOCAL

if LOCAL:
    ROOT_DIR = "/Users/liaoyuanda/Desktop/multisource_cocktail/"
else:
    ROOT_DIR = '/home/tk/Desktop/multisource_cocktail/'

FEAT_DIR_TEST = os.path.join(ROOT_DIR, 'feat_block_test/')
NOISE_DIR_TEST  = os.path.join(ROOT_DIR, 'noise_blocks/')
TEST_DIR  = os.path.join(ROOT_DIR, 'clean_test/')

TRAIN_DIR = os.path.join(ROOT_DIR, 'cleanblock/')