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

log_dire = './log/%s/'%start_time
os.makedirs(log_dire, exist_ok = True)
norm_flag = False
print("norm_flag",norm_flag)
print("unique_location_flag", unique_location_flag)

unique_post = '_unique' if unique_location_flag else ''
def normalize_vector(vec):

    return vec / np.linalg.norm(vec)



load_dire = os.path.join(meta_dire, 'efg_computation')
save_dire = os.path.join(meta_dire, 'pid_reference_surprise', 'yearly')
os.makedirs(save_dire, exist_ok = True)

# focal pid set + pub_year from get_efg_s2_merge.py; year_offset must match get_efg.py
year_offset = 5
print("year_offset", year_offset)


df_generalization = pd.read_csv(os.path.join(load_dire, f'gen_V3_y{year_offset}.csv'),
                                usecols = ['pid', 'pub_year'])


with open(os.path.join(meta_dire, 'citing_pid2cited_locations_list_before_%s.pkl'%max_year), 'rb') as f:
    pid2token_list = pickle.load(f)

print("load pid2token_list finished")

if unique_location_flag:
    pid2token_list = {pid: set(token_list) for pid, token_list in tqdm(pid2token_list.items(), desc = 'get unique location')}
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
        file_save_name = 'normed_reference_dist_creation_p_l_%s_y_%s_s_%s_embed_%s%s.csv'%(period_length, embedding_year_before, seed, embed_size, unique_post)
    else:
        file_save_name = 'reference_dist_creation_p_l_%s_y_%s_s_%s_embed_%s%s.csv'%(period_length, embedding_year_before, seed, embed_size, unique_post)

    if os.path.exists(os.path.join(save_dire, file_save_name))== True:
        continue
    
    if period_length > 0:
        embedding_year_after = embedding_year_before - period_length - 1 # extra year
        embedding_directory = os.path.join(meta_dire, 'reference_atypicality', 'location2embedding_before_%s_after_%s_dim_%s_min_count_%s_epoch_%s_seed_%s.model'%(embedding_year_before, embedding_year_after, embed_size, min_count, epochs, seed))


    else:
        embedding_directory = os.path.join(meta_dire, 'reference_atypicality', 'location2embedding_before_%s_dim_%s_min_count_%s_epoch_%s_seed_%s.model'%(embedding_year_before, embed_size, min_count, epochs, seed))



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

                orderdict['valid_paper_with_journal_count'] = len(pid2token_list[pid])
                orderdict['token_count'] = token_count
                orderdict['distance_mean'] = np.mean(distance_list)

                percentiles = np.percentile(distance_list, [5, 10, 25, 50, 75, 90, 95])
                orderdict['distance_median'] = percentiles[3]
                orderdict['distance_95th'] = percentiles[6]
                orderdict['distance_90th'] = percentiles[5]
                orderdict['distance_75th'] = percentiles[4]
                orderdict['distance_25th'] = percentiles[2]
                orderdict['distance_10th'] = percentiles[1]
                orderdict['distance_5th'] = percentiles[0]

                df_res.append(orderdict)

    result = f"Year {embedding_year_before} completed."

    # Get current timestamp
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    with open(os.path.join(log_dire, file_save_name.split('.')[0] + '.txt'), "a", encoding="utf-8") as f:
        f.write(f"[{current_time}] {result}\n")




    df_res = pd.DataFrame(df_res)

    df_res.to_csv(os.path.join(save_dire, file_save_name), index = False)

print("finished")