'''
dataset constants
'''
from utils.config_local import LOCAL

CLASSES = 2
ENTRIES_PER_JSON = 250

DS_LIST = {'birdstudybook':0,
           'captaincook':1,
           'cloudstudies_02_clayden_12':2, 
           'constructivebeekeeping':3,
           'discoursesbiologicalgeological_16_huxley_12':4, 
           'natureguide':5, 
           'pioneersoftheoldsouth':6, 
           'pioneerworkalps_02_harper_12':7, 
           'romancecommonplace':8, 
           'travelstoriesretold':9
          }
RANDOM_SAMPLES_PER_ENTRY = 2
ALL_SAMPLES_PER_ENTRY = CLASSES * (CLASSES - 1)