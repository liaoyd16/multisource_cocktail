
import json
import sys
sys.path.append("..")
from utils.dir_utils import *
import numpy as np
import os

ENTRIES_PER_JSON = 900

noise = np.random.randn(ENTRIES_PER_JSON, 256, 128)
json.dump(noise.tolist(), open(NOISE_DIR+"white_noise.json", "w"))