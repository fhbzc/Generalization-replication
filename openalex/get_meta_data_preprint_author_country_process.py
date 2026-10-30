import json
import os
import gzip
from tqdm import tqdm
import pickle
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import pandas as pd
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)               # utils.py is in this same directory
from utils import META_DIRE

save_dire = META_DIRE
os.makedirs(save_dire, exist_ok = True)

print("save_dire",save_dire)

with open(os.path.join(save_dire, 'pid2author_country_list.pkl'), 'rb') as f:
    pid2author_list_M = pickle.load(f)

pid2author_list = {}
for pid, author_country_item in tqdm(pid2author_list_M.items()):

    pid2author_list[pid] = [item[0] for item in author_country_item]
with open(os.path.join(save_dire, 'pid2author_list.pkl'), 'wb') as f:
    pickle.dump(pid2author_list, f, pickle.HIGHEST_PROTOCOL)

print("size of pid2author_list", len(pid2author_list))
print("finished")