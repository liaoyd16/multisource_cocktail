
import json
import sys
sys.path.append("..")
from utils.dir_utils import TEST_DIR
import numpy as np

data = json.load(open(TEST_DIR + "25.json", "r"))
data = np.array(data)
print(data.shape)