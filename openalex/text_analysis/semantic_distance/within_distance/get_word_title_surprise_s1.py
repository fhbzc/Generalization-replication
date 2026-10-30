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
# utils.py lives four levels up (openalex/), not in this within_distance/ directory;
# add it to the path so the import resolves regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils import META_DIRE

meta_dire = META_DIRE
print("meta_dire",meta_dire)


embed_size = 128
seed = 100
epochs = 5
min_count = 0
min_count = 0
parse_method = 'lemma'
year_before = 2025
max_year = 2025
min_year = 1950

print("embed_size", embed_size)
print("seed", seed)
print("epochs", epochs)
print("min_count", min_count)
print("max_year", max_year)
print("min_year", min_year)
period_length = 5
print("period_length", period_length)
print("parse_method",parse_method)

start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

log_dire = './log/%s/'%start_time
os.makedirs(log_dire, exist_ok = True)
norm_flag = False
print("norm_flag",norm_flag)
def normalize_vector(vec):

    return vec / np.linalg.norm(vec)



load_dire = os.path.join(meta_dire, 'efg_computation')
save_dire = os.path.join(meta_dire, 'pid_semantic_surprise', 'yearly')
os.makedirs(save_dire, exist_ok = True)

# must match get_efg.py / get_efg_s2_merge.py
year_offset = 5
division_count = 20
# normal mode only (reshuffle variant lives in atypicality/get_efg_reshuffle.py)
print("year_offset", year_offset)

# only the focal pid set and its pub_year are needed here; concat the per-division
# F/E/G tables from get_efg.py (the merged file from get_efg_s2_merge.py is
# year-filtered and would drop the recent focal years selected here).
df_list = []
for division_index in tqdm(range(division_count), desc='load divisions'):
    csv_path = os.path.join(load_dire, f'gen_V3_y{year_offset}_d_{division_index}_max_{division_count}.csv')
    df_list.append(pd.read_csv(csv_path, usecols=['pid', 'pub_year']))
df_generalization = pd.concat(df_list, ignore_index=True)


with open(os.path.join(meta_dire, 'pid2eng_token_list_%s_before_%s.pkl'%(parse_method, year_before)), 'rb') as f:
    pid2token_list = pickle.load(f)
print("load pid2token_list finished")


def return_mean_token(pid):

    pid_tokens = [text2embedding[token] for token in pid2token_list[pid] if token in text2embedding]
    l = len(pid_tokens)
    if l < 1:
        return None, None, l
    pid_tokens = np.array(pid_tokens, dtype = np.float32)
    pid_center_token = np.mean(pid_tokens, axis=0)
    return pid_center_token, pid_tokens, l




for embedding_year_before in tqdm(range(min_year, max_year), desc = 'iterate over years'):


    if norm_flag:
        file_save_name = 'normed_semantic_dist_creation_p_l_%s_y_%s_s_%s_embed_%s.csv'%(period_length, embedding_year_before, seed, embed_size)
    else:
        file_save_name = 'semantic_dist_creation_p_l_%s_y_%s_s_%s_embed_%s.csv'%(period_length, embedding_year_before, seed, embed_size)

    if os.path.exists(os.path.join(save_dire, file_save_name))== True:
        continue
    
    if period_length > 0:
        embedding_year_after = embedding_year_before - period_length - 1 # extra year
        embedding_directory = os.path.join(meta_dire, 'text_embedding', 'token2embed_before_%s_after_%s_dim_%s_min_count_%s_epoch_%s_seed_%s.pkl'%(embedding_year_before, embedding_year_after, embed_size, min_count, epochs, seed))
    else:
        embedding_directory = os.path.join(meta_dire, 'text_embedding', 'token2embed_before_%s_dim_%s_min_count_%s_epoch_%s_seed_%s.pkl'%(embedding_year_before, embed_size, min_count, epochs, seed))



    with open(embedding_directory, 'rb') as f:
        text2embedding = pickle.load(f)

    if norm_flag == True:
        text2embedding = {text: normalize_vector(embedding) for text, embedding in text2embedding.items()}
    df_generalization_select = df_generalization[df_generalization['pub_year'] == embedding_year_before]


    df_res = []
    for pid in tqdm(set(df_generalization_select['pid']), desc = 'iterate over pids', position = 1, leave = False):
        if pid in pid2token_list:

            center_token, all_tokens, token_count = return_mean_token(pid)
            if center_token is not None:


                center_token = center_token.reshape(1, -1)
                distances = cosine_distances(all_tokens, center_token) #[N, 1]
                

                distance_list = [distances[i, 0] for i in range(len(distances))]

                orderdict = OrderedDict()
                orderdict['pid'] = pid
                orderdict['token_count'] = token_count
                orderdict['distance_mean'] = np.mean(distance_list)
                orderdict['distance_median'] = np.median(distance_list)

                percentiles = np.percentile(distance_list, [10, 25, 75, 90])
                orderdict['distance_90th'] = percentiles[3]
                orderdict['distance_75th'] = percentiles[2]
                orderdict['distance_25th'] = percentiles[1]
                orderdict['distance_10th'] = percentiles[0]
                df_res.append(orderdict)

    result = f"Year {embedding_year_before} completed."

    # Get current timestamp
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    with open(os.path.join(log_dire, file_save_name.split('.')[0] + '.txt'), "a", encoding="utf-8") as f:
        f.write(f"[{current_time}] {result}\n")




    df_res = pd.DataFrame(df_res)

    df_res.to_csv(os.path.join(save_dire, file_save_name), index = False)

print("finished")