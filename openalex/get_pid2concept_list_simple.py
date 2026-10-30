import os
import pickle
from collections import defaultdict
from tqdm import tqdm
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)               # utils.py is in this same directory
from utils import META_DIRE


meta_dire = META_DIRE
with open(os.path.join(meta_dire, 'concept2pid_list.pkl'), 'rb') as f:
    concept2pid_list = pickle.load(f)
print("load concept2pid_list finished")

pid2concept_list = defaultdict(list)
for concept, pid_list in tqdm(concept2pid_list.items(), desc = 'process pid2concept_list s1'):
    for pid in tqdm(pid_list, position = 1, leave = False):
        pid2concept_list[pid].append(concept)


with open(os.path.join(meta_dire, 'pid2concept_list_simple.pkl'), 'wb') as f:
    pickle.dump(pid2concept_list, f, pickle.HIGHEST_PROTOCOL)
print("finished")