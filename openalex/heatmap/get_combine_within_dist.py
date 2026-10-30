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
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)             # utils.py is in the parent directory
from utils import META_DIRE

meta_dire = META_DIRE
print("meta_dire",meta_dire)

by_domain_type = None # [all, top, None]
print("by_domain_type",by_domain_type)

if by_domain_type == 'all':
    with open(os.path.join(meta_dire, 'concept2pid_list.pkl') , 'rb') as f:
        concept2pid_list = pickle.load(f)

elif by_domain_type == 'top':
    with open(os.path.join(meta_dire, 'top_concept2pid_list_level_0.pkl') , 'rb') as f:
        concept2pid_list = pickle.load(f)
elif by_domain_type is None:
    pass
else:
    assert False


abstract_distance_dire = os.path.join(meta_dire, 'pid_semantic_surprise_abstract', 'yearly')
reference_distance_dire = os.path.join(meta_dire, 'pid_reference_surprise')
save_dire_main = os.path.join(meta_dire, 'merged_stats')
save_dire_main_by_domain = os.path.join(save_dire_main, 'by_domain')
save_dire = os.path.join(meta_dire, 'merged_stats', 'yearly')
os.makedirs(save_dire, exist_ok = True)
os.makedirs(save_dire_main, exist_ok = True)
os.makedirs(save_dire_main_by_domain, exist_ok = True)

embed_size_refer = 128
seed_refer = 42
abstract_lower_year = 1986
print("abstract_lower_year", abstract_lower_year)
print("embed_size_refer", embed_size_refer)
print("seed_refer", seed_refer)

embed_size_text = 128
seed_abs_text = 42

seed_title_text = 100
print("embed_size_text", embed_size_text)
print("seed_abs_text", seed_abs_text)
print("seed_title_text", seed_title_text)

parse_method = 'raw'
max_year = 2025
min_year = 1951

print("min_year", min_year)

print("max_year", max_year)
period_length = 5
print("period_length", period_length)
print("parse_method",parse_method)

# F/E/G table from get_efg_s2_merge.py (gen_V3_y5.csv)

assert min_year >= 1950

refer_threshold = 1
citation_threshold = 5

print("refer_threshold",refer_threshold)
print("citation_threshold",citation_threshold)

nrow_count = None
print("nrow_count", nrow_count)
df_generalization = pd.read_csv(os.path.join(meta_dire, 'efg_computation', 'gen_V3_y5.csv')
                                    , usecols = ['pid', 'pub_year', 'up_refer_count', 'citation_count_y5',
                                            'f_y5', 'e_y5', 'g_y5'],
                                            nrows = nrow_count)

df_generalization = df_generalization[df_generalization['pub_year'] >= min_year]
df_generalization = df_generalization[df_generalization['pub_year'] < max_year]

df_generalization = df_generalization[df_generalization['up_refer_count'] >= refer_threshold]
df_generalization = df_generalization[df_generalization['citation_count_y5'] >= citation_threshold]

valid_pid_set = set(df_generalization['pid'])
print("size of valid_pid_set", len(valid_pid_set))

df_generalization = df_generalization.rename(columns={'f_y5': 'pillar_ratio',
                                    'e_y5': 'extension_ratio',
                                    'g_y5': 'generalization_ratio',
                                    'citation_count_y5': 'citation_count'})


df_refer_dist = pd.read_csv(os.path.join(reference_distance_dire, 'reference_dist_creation_p_l_%s_min_y_1950_max_y_%s_s_%s_embed_%s.csv'%(period_length, max_year, seed_refer, embed_size_refer)),
                            usecols = ['pid', 'token_count', 'distance_mean', 'distance_95th', 'distance_median', 'distance_90th'],
                            nrows = nrow_count)
df_refer_dist = df_refer_dist.rename(columns={'token_count': 'journal_refer_count', 'distance_mean': 'refer_dist_mean', 'distance_95th': 'refer_dist_95', 'distance_median': 'refer_dist_median', 'distance_90th': 'refer_dist_90'})
df_refer_dist = df_refer_dist[df_refer_dist['pid'].isin(valid_pid_set)]



df_refer_paper_dist = pd.read_csv(os.path.join(reference_distance_dire, 'reference_dist_paper_creation_p_l_%s_min_y_1951_max_y_%s_s_%s_embed_%s.csv'%(period_length, max_year, seed_refer, embed_size_refer)),
                            usecols = ['pid', 'token_count', 'distance_mean', 'distance_95th', 'distance_median', 'distance_90th'],
                            nrows = nrow_count)
df_refer_paper_dist = df_refer_paper_dist.rename(columns={'token_count': 'valid_paper_refer_count', 'distance_mean': 'refer_paper_dist_mean', 'distance_95th': 'refer_paper_dist_95', 'distance_median': 'refer_paper_dist_median', 'distance_90th': 'refer_paper_dist_90'})
df_refer_paper_dist = df_refer_paper_dist[df_refer_paper_dist['pid'].isin(valid_pid_set)]



df_title_dist = pd.read_csv(os.path.join(meta_dire, 'pid_semantic_surprise','semantic_dist_creation_p_l_%s_min_y_1950_max_y_%s_s_%s_embed_%s.csv'%(period_length, max_year, seed_title_text, embed_size_text)),
                usecols = ['pid', 'token_count', 'distance_mean', 'distance_90th'],
                nrows = nrow_count)
df_title_dist = df_title_dist[df_title_dist['pid'].isin(valid_pid_set)]
df_title_dist = df_title_dist.rename(columns={'token_count': 'title_token_count', 'distance_mean': 'title_dist_mean', 'distance_90th': 'title_dist_90'})




df_merged = df_generalization.merge(df_title_dist, on='pid', how='inner').merge(df_refer_dist, on='pid', how='inner').merge(df_refer_paper_dist, on='pid', how='inner')

print("size of df_merged", len(df_merged))
df_merged.to_csv(os.path.join(save_dire_main, 'merged_dist_min_y_%s_max_y_%s_p_l_%s_r_t_%s_c_t_%s_no_abs.csv'%(min_year, max_year, period_length, refer_threshold, citation_threshold)), index = False)

if by_domain_type is not None:
    for concept, pid_list in tqdm(concept2pid_list.items(), desc = 'process domain title'):

        df_merged_domain = df_merged[df_merged['pid'].isin(set(pid_list))]
        df_merged_domain.to_csv(os.path.join(save_dire_main_by_domain, '%s_%s_merged_dist_min_y_%s_max_y_%s_p_l_%s_r_t_%s_c_t_%s_no_abs.csv'%(by_domain_type, concept, min_year, max_year, period_length, refer_threshold, citation_threshold)), index = False)

df_res_list = []
for year in tqdm(range(min_year, max_year)):
    if year >= abstract_lower_year:
        df_abs_dist = pd.read_csv(os.path.join(abstract_distance_dire, 'semantic_dist_parse_%s_creation_p_l_%s_y_%s_s_%s_embed_%s.csv'%(parse_method, period_length, year, seed_abs_text, embed_size_text)),
                        usecols = ['pid', 'token_count', 'distance_mean'])

        df_abs_dist = df_abs_dist.rename(columns={'token_count': 'abs_token_count', 'distance_mean': 'abs_dist_mean'})

        df_res = df_merged.merge(df_abs_dist, on='pid', how='inner')

        df_res.to_csv(os.path.join(save_dire, 'merged_dist_y_%s_p_l_%s_r_t_%s_c_t_%s.csv'%(year, period_length, refer_threshold, citation_threshold)), index = False)
        df_res_list.append(df_res)
true_min_year = int(max(min_year, abstract_lower_year))
df_res_merge = pd.concat(df_res_list, ignore_index=True)
df_res_merge.to_csv(os.path.join(save_dire_main, 'merged_dist_min_y_%s_max_y_%s_p_l_%s_r_t_%s_c_t_%s.csv'%(true_min_year, max_year, period_length, refer_threshold, citation_threshold)), index = False)

if by_domain_type is not None:

    for concept, pid_list in tqdm(concept2pid_list.items(), desc = 'process domain abstract'):

        df_merged_domain = df_res_merge[df_res_merge['pid'].isin(set(pid_list))]
        df_merged_domain.to_csv(os.path.join(save_dire_main_by_domain, '%s_%s_merged_dist_min_y_%s_max_y_%s_p_l_%s_r_t_%s_c_t_%s.csv'%(by_domain_type, concept, true_min_year, max_year, period_length, refer_threshold, citation_threshold)), index = False)


print("finished")