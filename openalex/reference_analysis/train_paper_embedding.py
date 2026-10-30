import os
import pickle
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import pandas as pd
import numpy as np
import random
import time
from gensim.models import Word2Vec
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)             # utils.py is in the parent directory
from utils import META_DIRE

meta_dire = META_DIRE
print("meta_dire",meta_dire)

year_before = 2025
embed_size = 128
seed = 42
window_size = 100
epochs = 2
min_count = 0
end_year_start = 1951
end_year_finish = 2025
period_length = 5
print("embed_size", embed_size)
print("seed", seed)
print("window_size", window_size)
print("epochs", epochs)
print("min_count", min_count)
print("year_before", year_before)


print("period_length", period_length)
print("end_year_start", end_year_start)
print("end_year_finish", end_year_finish)


token_size_threshold = 2 * window_size
print("token_size_threshold", token_size_threshold)

with open(os.path.join(meta_dire, 'pid2pub_year.pkl'), 'rb') as f:
    pid2pub_year = pickle.load(f)
print("load pid2pub_year finished")
pub_year2pid_list = defaultdict(list)

for pid, pub_year in tqdm(pid2pub_year.items(), desc = 'process pub_year'):
    pub_year2pid_list[pub_year].append(pid)
del pid2pub_year
with open(os.path.join(meta_dire, 'citing_pid2cited_pids_list.pkl'), 'rb') as f:
    pid2token_list = pickle.load(f)
print("load pid2token_list finished")


save_directory = os.path.join(meta_dire, 'reference_atypicality')
os.makedirs(save_directory, exist_ok = True)

for embedding_year_before in tqdm(range(end_year_start, end_year_finish)):


    if period_length > 0:
        embedding_year_after = embedding_year_before - period_length - 1 # extra year

        save_file_name = os.path.join(save_directory, 'paper_embedding_before_%s_after_%s_dim_%s_min_count_%s_epoch_%s_seed_%s.model'%(embedding_year_before, embedding_year_after, embed_size, min_count, epochs, seed))

    else:
        embedding_year_after = -1
        save_file_name = os.path.join(save_directory, 'paper_embedding_before_%s_dim_%s_min_count_%s_epoch_%s_seed_%s.model'%(embedding_year_before, embed_size, min_count, epochs, seed))


    if os.path.exists(save_file_name) == True:
        continue


    list_of_token_list = []

    for year in tqdm(range(embedding_year_after, embedding_year_before), desc = 'process tokens', leave = False, position = 1):
        if year in pub_year2pid_list:
            list_of_token_list.extend([pid2token_list[pid] for pid in pub_year2pid_list[year] if pid in pid2token_list and len(pid2token_list[pid]) < token_size_threshold and len(pid2token_list[pid]) > 1])




    model = Word2Vec(sentences=list_of_token_list, 
                    vector_size=embed_size, 
                    window=window_size, 
                    min_count=min_count, 
                    sg=1, 
                    seed = seed,
                    workers=16,
                    epochs = epochs)

    model.save(save_file_name)


print("finished")