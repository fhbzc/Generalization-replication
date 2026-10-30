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
cited_dist_dire = os.path.join(meta_dire, 'cited_reference_text_distance_citing_t')
by_domain_dire = os.path.join(cited_dist_dire, 'by_domain')
os.makedirs(by_domain_dire, exist_ok = True)
by_domain_type = 'all' # [all, top]
print("by_domain_type",by_domain_type)

if by_domain_type == 'all':
    with open(os.path.join(meta_dire, 'concept2pid_list.pkl') , 'rb') as f:
        concept2pid_list = pickle.load(f)

elif by_domain_type == 'top':
    with open(os.path.join(meta_dire, 'top_concept2pid_list_level_0.pkl') , 'rb') as f:
        concept2pid_list = pickle.load(f)
else:
    assert False


# F/E/G table from get_efg_s2_merge.py (gen_V3_y5.csv)

max_year = 2020
min_year = 1951
period_length = 5
print("max_year",max_year)
print("min_year",min_year)
print("period_length",period_length)

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


df_refer_dist = pd.read_csv(os.path.join(cited_dist_dire, f'citing_refer_distance_min_y_1951_max_y_2025_p_l_{period_length}_t_l_1.csv'),
                            usecols = ['pid', 'mean_citing_refer_dist'],
                            nrows = nrow_count)
df_refer_dist = df_refer_dist.rename(columns={'mean_citing_refer_dist': 'refer_dist_mean'})
df_refer_dist = df_refer_dist[df_refer_dist['pid'].isin(valid_pid_set)]




df_refer_paper_dist = pd.read_csv(os.path.join(cited_dist_dire, f'citing_refer_paper_distance_min_y_1951_max_y_2025_p_l_{period_length}_t_l_1.csv'),
                            usecols = ['pid', 'mean_citing_refer_dist'],
                            nrows = nrow_count)
df_refer_paper_dist = df_refer_paper_dist.rename(columns={'mean_citing_refer_dist': 'refer_paper_dist_mean'})
df_refer_paper_dist = df_refer_paper_dist[df_refer_paper_dist['pid'].isin(valid_pid_set)]


df_merged_refer = df_generalization.merge(df_refer_dist, on='pid', how='inner').merge(df_refer_paper_dist, on='pid', how='inner')



df_title_dist = pd.read_csv(os.path.join(cited_dist_dire, f'citing_title_text_distance_min_y_1951_max_y_2025_p_l_{period_length}_t_l_1.csv'),
                usecols = ['pid', 'mean_citing_text_dist'],
                nrows = nrow_count)
df_title_dist = df_title_dist[df_title_dist['pid'].isin(valid_pid_set)]
df_title_dist = df_title_dist.rename(columns={'mean_citing_text_dist': 'title_dist_mean'})




df_merged = df_merged_refer.merge(df_title_dist, on='pid', how='inner')

print("size of df_merged", len(df_merged))

actual_min = max(min_year, 1951)
actual_max = min(max_year, 2025)

df_merged.to_csv(os.path.join(cited_dist_dire, f'merged_dist_min_y_{actual_min}_max_y_{actual_max}_p_l_{period_length}_r_t_{refer_threshold}_c_t_{citation_threshold}_title.csv'), index = False)

for concept, pid_list in tqdm(concept2pid_list.items(), desc = 'process domain title'):

    df_merged_domain = df_merged[df_merged['pid'].isin(set(pid_list))]
    df_merged_domain.to_csv(os.path.join(by_domain_dire, f'{by_domain_type}_{concept}_merged_dist_min_y_{actual_min}_max_y_{actual_max}_p_l_{period_length}_r_t_{refer_threshold}_c_t_{citation_threshold}_title.csv'), index = False)


df_abs_dist = pd.read_csv(os.path.join(cited_dist_dire, f'citing_abstract_text_distance_min_y_1986_max_y_2025_p_l_{period_length}_t_l_1.csv'),
                usecols = ['pid', 'mean_citing_text_dist'],
                nrows = nrow_count)
df_abs_dist = df_abs_dist[df_abs_dist['pid'].isin(valid_pid_set)]
df_abs_dist = df_abs_dist.rename(columns={'mean_citing_text_dist': 'abs_dist_mean'})
actual_min = max(min_year, 1986)
actual_max = min(max_year, 2025)

df_merged = df_merged_refer.merge(df_abs_dist, on='pid', how='inner')
df_merged.to_csv(os.path.join(cited_dist_dire, f'merged_dist_min_y_{actual_min}_max_y_{actual_max}_p_l_{period_length}_r_t_{refer_threshold}_c_t_{citation_threshold}_abs.csv'), index = False)
print("size of df_merged", len(df_merged))
print("finished")