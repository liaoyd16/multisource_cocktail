
import json
import sys
sys.path.append("..")
from utils.dir_utils import *
import numpy as np
import os

'''
_0 = np.array(json.load(open(TEST_DIR + "0.json", "r")))
print(_0.shape)

_1 = np.array(json.load(open(TEST_DIR + "1.json", "r")))
print(_1.shape)
'''

'''
people_people = json.load(open(TEST_DIR + "person_person.json", "r"))
people_people = np.array(people_people)

feature_block = open(TEST_DIR + "0.json", "w")
spec_block = open(TEST_DIR + "1.json", "w")

l = people_people.shape[0] // 2
A = people_people[:l]
B = people_people[l:]
print(A.shape, B.shape)

A_B = np.array([A, B]).squeeze()
print(A_B.shape)

feature_part = A_B[:,:10]
spec_part = A_B[:,10:]

feature_part = feature_part.tolist()
spec_part = spec_part.tolist()

json.dump(feature_part, feature_block)
json.dump(spec_part, spec_block)
'''

for file in os.listdir(TRAIN_DIR):
	fh = open(TRAIN_DIR + file, "r")
	array = np.array(json.load(fh))
	print(file, array.shape)