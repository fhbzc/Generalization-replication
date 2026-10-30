import json
import os
import sys
import gzip
from tqdm import tqdm
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict, OrderedDict
import pandas as pd
from pathlib import Path
import re
from nltk.corpus import stopwords
import numpy as np
import networkx as nx
import spacy
import math

# utils.py lives in the parent (semantic_scholar) directory; make it importable regardless of CWD
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
from utils import META_DIRE

n_cpu = 16
batch_size = 1000
print("process count", n_cpu)
print("batch_size", batch_size)

print("V1")
# stop_words_set = set(stopwords.words('english'))
meta_dire = META_DIRE
load_dire = os.path.join(meta_dire, 'global_tribigram_eng')

inspect_save_dire = './bi_trigam_inspect'
os.makedirs(inspect_save_dire, exist_ok = True)

start_year = 1951
end_year = 2026 # 2026
PMI_bigram = 0
print("start_year",start_year)
print("end_year",end_year)
print("PMI_bigram",PMI_bigram)

# make sure EVERY year finished stage 1 before we start aggregating (fail fast)
for year in range(start_year, end_year):
    finish_dire = os.path.join(load_dire,f'finished_y_{year}.pkl')
    assert os.path.exists(finish_dire), finish_dire

total_sigular_count = 0
token2fre_single_M = defaultdict(int)
token2fre_bigram_M = defaultdict(int)
for year in tqdm(range(start_year, end_year)):
    with open(os.path.join(load_dire,f'single_token2fre_y_{year}.pkl'),'rb') as f:
        token2fre_single = pickle.load(f)

    for token, fre in tqdm(token2fre_single.items(), position = 1, leave = False, desc = 'process single'):
        token2fre_single_M[token] += fre
        total_sigular_count += fre
    with open(os.path.join(load_dire,f'bigram_token2fre_y_{year}.pkl'),'rb') as f:
        token2fre_bigram = pickle.load(f)
    for token, fre in tqdm(token2fre_bigram.items(), position = 1, leave = False, desc = 'process bigram'):
        token2fre_bigram_M[token] += fre

df_res = []


valid_bigram2pmi = {}
for bigram, fre in tqdm(token2fre_bigram_M.items()):

    item1, item2 = bigram
    fre1 = token2fre_single_M[item1]
    fre2 = token2fre_single_M[item2]


    if fre1 > 10 and fre2 > 10 and fre > 10:
        PMI = math.log2(fre * total_sigular_count / (fre1 * fre2))
        if PMI > PMI_bigram:
            valid_bigram2pmi[bigram] = PMI
            od = OrderedDict()
            od['item1'] = item1
            od['item2'] = item2
            od['item1_fre'] = fre1
            od['item2_fre'] = fre2
            od['overlap'] = fre
            od['total_sigular_count'] = total_sigular_count

            od['PMI'] = PMI
            df_res.append(od)

df_res = pd.DataFrame(df_res)

df_res.to_csv(os.path.join(inspect_save_dire, f'bigram_PMI_t_{PMI_bigram}_b_{start_year}_e_{end_year}.csv'), index = False)


with open(os.path.join(load_dire,f'valid_bigram2pmi_t_{PMI_bigram}_b_{start_year}_e_{end_year}.pkl'),'wb') as f:
    pickle.dump(valid_bigram2pmi, f, pickle.HIGHEST_PROTOCOL)





print("total bigram", len(token2fre_bigram_M))
print("size of valid_bigram2pmi", len(valid_bigram2pmi))

print("finished")
