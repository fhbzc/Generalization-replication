import os
import random
from collections import defaultdict, OrderedDict
from tqdm import tqdm
import numpy as np
import pickle
import time
import pandas as pd
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)             # utils.py is in the parent directory
from utils import META_DIRE

meta_dire = META_DIRE
print("meta_dire",meta_dire)


year_before = 2025
print("year_before", year_before)
with open(os.path.join(meta_dire, 'citing_pid2cited_pids_list.pkl'), 'rb') as f:
    citing2up_list = pickle.load(f)
print("load citing2up_list finished")



with open(os.path.join(meta_dire, 'pid2pub_year.pkl'), 'rb') as f:
    pid2year = pickle.load(f)
print("pid2year load finished")

with open(os.path.join(meta_dire, 'pid2primay_location.pkl'), 'rb') as f:
    pid2p_l_id_main = pickle.load(f)
print("load pid2p_l_id_main finished")



citing2location_list = {citing: [pid2p_l_id_main[up] for up in up_list if up in pid2p_l_id_main] for citing, up_list in tqdm(citing2up_list.items()) if citing in pid2year and pid2year[citing] < year_before}

del citing2up_list
del pid2p_l_id_main

with open(os.path.join(meta_dire, 'citing_pid2cited_locations_list_before_%s.pkl'%year_before), 'wb') as f:
    pickle.dump(citing2location_list, f, pickle.HIGHEST_PROTOCOL)
print("size of citing2location_list", len(citing2location_list))
print("finished")