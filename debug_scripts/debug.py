
import json
import sys
sys.path.append("..")
import utils.dir_utils
from utils.dir_utils import *
import numpy as np
import os


test = np.array(json.load(open(TEST_DIR + "captaincook_clean.json")))
