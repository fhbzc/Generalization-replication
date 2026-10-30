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
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_distances
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils import META_DIRE

meta_dire = META_DIRE
print("meta_dire",meta_dire)
file_directory = os.path.join(meta_dire, 'cited_reference_text_distance_citing_t')

log_dir = os.path.join(file_directory, "logs_cited_paper_dist")
os.makedirs(log_dir, exist_ok=True)

os.makedirs(file_directory, exist_ok = True)



def write_log(year):
    """Append a log message for the given embedding_year_before."""
    log_file = os.path.join(log_dir, f"log_{year}.txt")
    with open(log_file, "a") as f:
        f.write("finished")



embed_size = 128
seed = 42
epochs = 2
min_count = 0
token_length_threshold = 1


print("embed_size",embed_size)
print("seed",seed)
print("epochs",epochs)
print("min_count",min_count)
print("token_length_threshold",token_length_threshold)

year_before = 2025
max_year = 2025
min_year = 1951

print("max_year", max_year)
print("min_year", min_year)
print("year_before", year_before)

period_length = 5
print("period_length", period_length)


load_dire = os.path.join(meta_dire, 'efg_computation')

# must match get_efg.py / get_efg_s2_merge.py
year_offset = 5
# normal mode only (reshuffle variant lives in atypicality/get_efg_reshuffle.py)
print("year_offset", year_offset)

# F = foundation (pillar), E = extension, G = generalization (normalized shares)
foundation_col = f'f_y{year_offset}'
extension_col = f'e_y{year_offset}'
generalization_col = f'g_y{year_offset}'
citation_col = f'citation_count_y{year_offset}'



def return_mean_token(pid, item2embedding, pid2item_list, token_threshold):

    pid_tokens = [item2embedding[token] for token in pid2item_list[pid] if token in item2embedding]
    if len(pid_tokens) < token_threshold:
        return None, None
    pid_tokens = np.array(pid_tokens, dtype = np.float32)
    pid_center_token = np.mean(pid_tokens, axis=0)
    return pid_center_token, pid_tokens



# F/E/G table from get_efg_s2_merge.py (merged output; pub_year < 2025 - year_offset)
gen_file_load_dire = os.path.join(load_dire, f'gen_V3_y{year_offset}.csv')
df_generalization = pd.read_csv(gen_file_load_dire,
                                usecols = ['pid', 'pub_year', 'up_refer_count', citation_col,
                                           foundation_col, extension_col, generalization_col])

df_generalization = df_generalization[df_generalization['pub_year'] < max_year]
df_generalization = df_generalization[df_generalization['pub_year'] >= min_year]
df_generalization = df_generalization.rename(columns={foundation_col: 'pillar_ratio',
                                    extension_col: 'extension_ratio',
                                    generalization_col: 'generalization_ratio',
                                    citation_col: 'citation_count'})





with open(os.path.join(meta_dire, 'citing_pid2cited_pids_list.pkl'), 'rb') as f: 
    # ordered version for optimization
    citing_pid2cited_works = pickle.load(f)
print("citing_pid2cited_works load finished")

pid2token_list = citing_pid2cited_works
with open(os.path.join(meta_dire, 'pid2pub_year.pkl'), 'rb') as f:
    pid2year = pickle.load(f)
print("pid2year load finished")

pid2citing_text_distance_list = defaultdict(list)

for embedding_year_before in tqdm(range(min_year, max_year), desc = 'iterate over years'):

    if period_length > 0:
        embedding_year_after = embedding_year_before - period_length - 1 # extra year

        embedding_text_directory = os.path.join(meta_dire, 'reference_atypicality', 'paper2embedding_before_%s_after_%s_dim_%s_min_count_%s_epoch_%s_seed_%s.model'%(embedding_year_before, embedding_year_after, embed_size, min_count, epochs, seed))

    else:
        embedding_text_directory = os.path.join(meta_dire, 'reference_atypicality', 'paper2embedding_before_%s_dim_%s_min_count_%s_epoch_%s_seed_%s.model'%(embedding_year_before, embed_size, min_count, epochs, seed))
        

    



    with open(embedding_text_directory, 'rb') as f:
        text2embedding = pickle.load(f)


    df_generalization_select = df_generalization[df_generalization['pub_year'] == embedding_year_before]

    pid2text_center_token = {}



    pid_list = list(set(df_generalization_select['pid']))
    for pid in tqdm(pid_list, desc = 'iterate over pids', position = 1, leave = False):
        if pid in pid2token_list:
            if pid not in citing_pid2cited_works:
                continue
            if pid not in pid2text_center_token:
                center_token_text, all_tokens = return_mean_token(pid, text2embedding, pid2token_list, token_length_threshold)
                pid2text_center_token[pid] = center_token_text
            else:
                center_token_text = pid2text_center_token[pid]


            if center_token_text is not None:
                pub_year = pid2year[pid]
                assert pub_year == embedding_year_before
                for cited_work in citing_pid2cited_works[pid]:

                    cited_pub_year = pid2year[cited_work]

                    dif_year = pub_year - cited_pub_year
                    assert dif_year >= 0
                    if dif_year > 5:
                        continue
                    if cited_work in pid2token_list:
                        
                        if cited_work not in pid2text_center_token:
                            center_citing, _ = return_mean_token(cited_work, text2embedding, pid2token_list, token_length_threshold)
                            pid2text_center_token[cited_work] = center_citing
                        else:
                            center_citing = pid2text_center_token[cited_work]

                        if center_citing is not None:
                            d = cosine_distances(center_citing.reshape(1, -1), center_token_text.reshape(1, -1))[0, 0]  # shape: (m, 1)
                            pid2citing_text_distance_list[cited_work].append(d)

    write_log(embedding_year_before)



valid_pid_set = set(pid2citing_text_distance_list)

pid2citing_text_distance_count = {pid: len(citing_dist_list) for pid, citing_dist_list in pid2citing_text_distance_list.items()}

pid2citing_text_distance_mean = {pid: np.mean(citing_dist_list) for pid, citing_dist_list in pid2citing_text_distance_list.items()}
pid2citing_text_distance_median = {pid: np.median(citing_dist_list) for pid, citing_dist_list in pid2citing_text_distance_list.items()}

df_generalization = df_generalization[df_generalization['pid'].isin(valid_pid_set)].copy()

df_generalization['citation_has_refer_count'] = df_generalization['pid'].map(pid2citing_text_distance_count)
df_generalization['mean_citing_refer_dist'] = df_generalization['pid'].map(pid2citing_text_distance_mean)
df_generalization['median_citing_refer_dist'] = df_generalization['pid'].map(pid2citing_text_distance_median)

df_generalization.to_csv(os.path.join(file_directory, f'citing_refer_paper_distance_min_y_{min_year}_max_y_{max_year}_p_l_{period_length}_t_l_{token_length_threshold}.csv'), index = False)
print("size of df_generalization", len(df_generalization))
print("finished")
