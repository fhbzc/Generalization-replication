import os
import pickle
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import json
from scipy.spatial.distance import cosine
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import META_DIRE
embed_size = 128
seed = 42
window_size = 2
epochs = 2
min_count = 0
parse_method = 'raw'
year_before = 2020
print("embed_size", embed_size)
print("seed", seed)
print("window_size", window_size)
print("epochs", epochs)
print("min_count", min_count)
period_length = 5
print("period_length", period_length)
print("parse_method",parse_method)

min_year = 1986

print("min_year", min_year)

# When True, an edge (pid -> up) is only kept if the edge's CITED paper `up` itself
# has at least one reference (i.e. up cites >= 1 paper in the network). Edges whose
# cited paper has no outgoing reference are dropped before e/f/g classification.
require_cited_has_reference = True
print("require_cited_has_reference", require_cited_has_reference)
# keep the two modes' outputs in separate files so they never overwrite each other
mode_suffix = '_cited_has_ref' if require_cited_has_reference else ''

from scipy.stats import norm



def mean_ci_normal(data, ci=0.95):
    arr = np.asarray(list(data), dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return np.nan, np.nan, np.nan

    mean = arr.mean()
    # 用无偏样本标准差
    s = arr.std(ddof=1)
    z = norm.ppf(0.5 * (1.0 + ci))
    half = z * s / np.sqrt(n)
    return mean, mean - half, mean + half

def estimate_mean_group_mean_ci(group2list,ci=0.95, expected_keys=(0, 1)):
    res = {}
    for k in expected_keys:
        data = group2list.get(k, [])
        mean_est, lo, hi = mean_ci_normal(data, ci=ci)
        res[k] = {
            'group': k,
            'mean': mean_est,
            'ci_lower': lo,
            'ci_upper': hi,
            'n': len(data)
        }
    return res

# --- helper to save results ---
def save_group_results(group2list, save_path):
    res = estimate_mean_group_mean_ci(group2list, ci=0.95)
    df = pd.DataFrame.from_dict(res, orient='index')
    df.to_csv(save_path, index=False)



meta_dire = META_DIRE
print("meta_dire",meta_dire)

type_list = ['gen','fou_ext']
print("type_list",type_list)
meta_save_dire = os.path.join(meta_dire, 'reference_text_distance_g_e_f/two_group')
os.makedirs(meta_save_dire, exist_ok = True)
data_loaded_flag = False

# Each pid's up-references are classified e/f/g directly from the citation network
# (same up/down comparison as get_efg.py applied to each (reference, pid) pair), so
# no precomputed key-citation weights are needed. Difference from get_efg.py: a tie
# (up_count == down_count > 0) is classified as foundation here, not split 0.5/0.5.




def return_mean_token(pid):

    pid_tokens = [text2embedding[token] for token in pid2token_list_M[pid] if token in text2embedding]
    if len(pid_tokens) == 0:
        return None
    pid_tokens = np.array(pid_tokens, dtype = np.float32)
    pid_center_token = np.mean(pid_tokens, axis=0)
    return pid_center_token


def classify_reference_counts(pid):
    """For pid (the citing paper), classify each of its up-references (papers it
    cites) as e/f/g, restricted to references in valid_pid_set. Applies get_efg.py's
    up/down comparison to the (cited=up, citing=pid) pair:
      up_count   = #references shared between the reference `up` and pid
      down_count = #of `up`'s citers that pid also cites
    Returns {up: (up_count, down_count)}."""
    pid_references = citing_pid2cited_works.get(pid, set())
    counts = {}
    for up in pid_references:
        if up not in valid_pid_set:
            continue
        up_references = citing_pid2cited_works.get(up, set())
        # optional mode: require the cited paper `up` to have >= 1 reference
        if require_cited_has_reference and len(up_references) == 0:
            continue
        up_citers = cited_pid2citing_works.get(up, set())
        up_count = len(up_references & pid_references)
        down_count = len(up_citers & pid_references)
        counts[up] = (up_count, down_count)
    return counts


for type in tqdm(type_list):

    if data_loaded_flag == False:

        data_disp = pd.read_csv(os.path.join(meta_dire, 'disruption', 'disp_full.csv'), usecols = ['pid', "pub_year"])

        data_disp = data_disp[data_disp['pub_year'] < year_before]



        year2pid_list = data_disp.groupby('pub_year')['pid'].apply(list).to_dict()


        valid_pid_set = set(data_disp['pid'])
        del data_disp


        pid2token_list_M = {}

        for y in tqdm(range(min_year, year_before), desc = 'load years data'):
            with open(os.path.join(meta_dire, 'abstract_yearly', 'pid2abstract_eng_token_list_%s_y_%s.pkl'%(parse_method, y)), 'rb') as f:
                pid2token_list = pickle.load(f)
            pid2token_list_M.update(pid2token_list)

            



        valid_pid_set &= set(pid2token_list_M)

        # citation network for g/e/f classification (replaces the key-citation weight files)
        with open(os.path.join(meta_dire, 'citing_pid2cited_pids_list.pkl'), 'rb') as f:
            citing_pid2cited_works = pickle.load(f)
        citing_pid2cited_works = {pid: set(lst) for pid, lst in tqdm(citing_pid2cited_works.items(), desc='process citing_pid2cited_works')}
        with open(os.path.join(meta_dire, 'cited_pid2citing_pids_list.pkl'), 'rb') as f:
            cited_pid2citing_works = pickle.load(f)
        cited_pid2citing_works = {pid: set(lst) for pid, lst in tqdm(cited_pid2citing_works.items(), desc='process cited_pid2citing_works')}
        print("load citation network finished")

        pid2reference_counts = {
            pid: classify_reference_counts(pid)
            for pid in tqdm(valid_pid_set, desc='classify up-references from network')
        }

        data_loaded_flag = True


    if type == 'gen':
        gen_group2distance_list = defaultdict(list)
        pid2up2gen_group = defaultdict(dict)

        for pid, counts in pid2reference_counts.items():
            for up, (up_count, down_count) in counts.items():
                # generalization reference (isolated): pid shares none of up's
                # references and cites none of up's other citers
                if up_count == 0 and down_count == 0:
                    pid2up2gen_group[pid][up] = 1
                else:
                    pid2up2gen_group[pid][up] = 0

    elif type == 'fou_ext':

        fou_group2distance_list = defaultdict(list)
        pid2up2fou_group = defaultdict(dict)

        ext_group2distance_list = defaultdict(list)
        pid2up2ext_group = defaultdict(dict)

        for pid, counts in pid2reference_counts.items():
            for up, (up_count, down_count) in counts.items():
                if up_count == 0 and down_count == 0:
                    # isolated / generalization: neither extension nor foundation
                    ext_int = 0
                    fou_int = 0
                elif up_count > down_count:
                    ext_int = 1
                    fou_int = 0
                else:
                    # down_count > up_count, or tie (up_count == down_count > 0) -> foundation
                    ext_int = 0
                    fou_int = 1
                pid2up2ext_group[pid][up] = ext_int
                pid2up2fou_group[pid][up] = fou_int



    for embedding_year_before in tqdm(range(min_year, year_before), desc = 'process plot each period'):

        if period_length > 0:
            embedding_year_after = embedding_year_before - period_length - 1 # extra year
            embedding_directory = os.path.join(meta_dire, 'text_embedding_abs', 'token2embed_parse_%s_before_%s_after_%s_dim_%s_min_count_%s_epoch_%s_seed_%s.pkl'%(parse_method, embedding_year_before, embedding_year_after, embed_size, min_count, epochs, seed))
        else:
            embedding_directory = os.path.join(meta_dire, 'text_embedding_abs', 'token2embed_parse_%s_before_%s_dim_%s_min_count_%s_epoch_%s_seed_%s.pkl'%(parse_method, embedding_year_before, embed_size, min_count, epochs, seed))


        pid2center = {}
        with open(embedding_directory, 'rb') as f:
            text2embedding = pickle.load(f)

        for pid in year2pid_list[embedding_year_before]:
            # should be at the time of citation
            if pid not in valid_pid_set:
                continue
            if type == 'gen':
                if pid not in pid2up2gen_group:
                    continue

                up_list = list(pid2up2gen_group[pid])
            elif type == 'fou_ext':
                if pid not in pid2up2ext_group:
                    continue

                up_list = list(pid2up2ext_group[pid])

            if pid in pid2center:
                pid_center_token = pid2center[pid]
            else:
                pid_center_token = return_mean_token(pid)
                pid2center[pid] = pid_center_token
            if pid_center_token is None:
                continue
            for up in up_list:
                if up not in valid_pid_set:
                    continue
                up_center_token = return_mean_token(up)

                if up in pid2center:
                    up_center_token = pid2center[up]
                else:
                    up_center_token = return_mean_token(up)
                    pid2center[pid] = up_center_token

                if up_center_token is None:
                    continue
                cosine_dist = cosine(pid_center_token, up_center_token)
                if type == 'gen':
                    gen_group = pid2up2gen_group[pid][up]
                    gen_group2distance_list[gen_group].append(cosine_dist)
                elif type == 'fou_ext':
                    ext_group = pid2up2ext_group[pid][up]
                    fou_group = pid2up2fou_group[pid][up]

                    ext_group2distance_list[ext_group].append(cosine_dist)
                    fou_group2distance_list[fou_group].append(cosine_dist)
    if type == 'gen':
        save_group_results(gen_group2distance_list, os.path.join(meta_save_dire, "gen_abs_dist_p_l_%s%s.csv"%(period_length, mode_suffix)))

    elif type == 'fou_ext':
        save_group_results(fou_group2distance_list, os.path.join(meta_save_dire, "fou_abs_dist_p_l_%s%s.csv"%(period_length, mode_suffix)))

        save_group_results(ext_group2distance_list, os.path.join(meta_save_dire, "ext_abs_dist_p_l_%s%s.csv"%(period_length, mode_suffix)))

print("finish")