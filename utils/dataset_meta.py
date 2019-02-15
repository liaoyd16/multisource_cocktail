'''
dataset constants
'''
from utils.config_local import LOCAL

CLASSES = 6
RANDOM_SAMPLES_PER_ENTRY = 10
ALL_SAMPLES_PER_ENTRY = CLASSES * (CLASSES - 1) // 2
ENTRIES_PER_JSON = 100