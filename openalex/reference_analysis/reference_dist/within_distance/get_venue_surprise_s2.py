import os
import pickle
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import random
import time
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_distances
from datetime import datetime
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils import META_DIRE

meta_dire = META_DIRE
print("meta_dire",meta_dire)
embed_size = 128
seed = 42
window_size = 100
epochs = 5
min_count = 0
max_year = 2025
min_year = 1950

print("embed_size", embed_size)
print("seed", seed)
print("window_size", window_size)
print("epochs", epochs)
print("min_count", min_count)
print("max_year", max_year)
print("min_year", min_year)
period_length = 5
print("period_length", period_length)

start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
unique_location_flag = False

unique_post = '_unique' if unique_location_flag else ''
norm_flag = False
print("norm_flag",norm_flag)

print("unique_location_flag",unique_location_flag)

load_dire = os.path.join(meta_dire, 'pid_reference_surprise', 'yearly')
save_dire = os.path.join(meta_dire, 'pid_reference_surprise')
os.makedirs(save_dire, exist_ok = True)

all_dfs = []

for embedding_year_before in tqdm(range(min_year, max_year), desc = 'iterate over years'):


    if norm_flag:
        file_save_name = 'normed_reference_dist_creation_p_l_%s_y_%s_s_%s_embed_%s%s.csv'%(period_length, embedding_year_before, seed, embed_size, unique_post)
    else:
        file_save_name = 'reference_dist_creation_p_l_%s_y_%s_s_%s_embed_%s%s.csv'%(period_length, embedding_year_before, seed, embed_size, unique_post)



    df_res = pd.read_csv(os.path.join(load_dire, file_save_name))
    all_dfs.append(df_res)

merged_df = pd.concat(all_dfs, ignore_index=True)
del all_dfs
print("merge finished")
if norm_flag:
    merged_df.to_csv(os.path.join(save_dire, 'normed_reference_dist_creation_p_l_%s_min_y_%s_max_y_%s_s_%s_embed_%s%s.csv'%(period_length, min_year, max_year, seed, embed_size, unique_post)), index=False)
else:
    merged_df.to_csv(os.path.join(save_dire, 'reference_dist_creation_p_l_%s_min_y_%s_max_y_%s_s_%s_embed_%s%s.csv'%(period_length, min_year, max_year, seed, embed_size, unique_post)), index=False)

print("size of merged_df", len(merged_df))
print("finished")