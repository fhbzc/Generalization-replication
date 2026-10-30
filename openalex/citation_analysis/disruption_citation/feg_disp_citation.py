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
from scipy.stats import norm
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import META_DIRE



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

# When True, an edge (pid -> up) is only kept if the edge's CITED paper `up` itself
# has at least one reference (i.e. up cites >= 1 paper in the network). Edges whose
# cited paper has no outgoing reference are dropped before e/f/g classification.
require_cited_has_reference = True
print("require_cited_has_reference", require_cited_has_reference)
# keep the two modes' outputs in separate files so they never overwrite each other
mode_suffix = '_cited_has_ref' if require_cited_has_reference else ''

type_list = ['gen', 'fou_ext']

meta_save_dire = os.path.join(meta_dire, 'reference_disp_g_e_f/two_group')
os.makedirs(meta_save_dire, exist_ok = True)
data_loaded_flag = False

disp_pair_dire = os.path.join(meta_dire, 'disruption_pair_division')

# g/e/f classification of each pid's up-references is computed directly from the
# citation network (same up/down comparison as get_efg.py applied to each
# (reference, pid) pair), so no precomputed key-citation weights are needed.
# Difference from get_efg.py: a tie (up_count == down_count > 0) is classified as
# foundation here, not split 0.5/0.5.


def classify_reference_counts(pid):
    """For pid, return {up: (up_count, down_count)} for each of pid's up-references
    in valid_pid_set. up_count = #references shared between up and pid; down_count =
    #of up's citers that pid also cites (get_efg.py's comparison on (cited=up, citing=pid))."""
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

        data_disp = data_disp[data_disp['pub_year'] < 2020]






        valid_pid_set = set(data_disp['pid'])
        del data_disp

        # citation network for g/e/f classification (replaces the key-citation weight files)
        with open(os.path.join(meta_dire, 'citing_pid2cited_pids_list.pkl'), 'rb') as f:
            citing_pid2cited_works = pickle.load(f)
        citing_pid2cited_works = {p: set(lst) for p, lst in tqdm(citing_pid2cited_works.items(), desc='process citing_pid2cited_works')}
        with open(os.path.join(meta_dire, 'cited_pid2citing_pids_list.pkl'), 'rb') as f:
            cited_pid2citing_works = pickle.load(f)
        cited_pid2citing_works = {p: set(lst) for p, lst in tqdm(cited_pid2citing_works.items(), desc='process cited_pid2citing_works')}
        print("load citation network finished")

        pid2reference_counts = {
            pid: classify_reference_counts(pid)
            for pid in tqdm(valid_pid_set, desc='classify up-references from network')
        }

        data_loaded_flag = True


    division_count = 20

    if type == 'gen':
        gen_group2disp_raw_list = defaultdict(list)
        gen_group2disp_i_d_i_j_list = defaultdict(list)
        gen_group2disp_i_d_i_j_k_list = defaultdict(list)

    elif type == 'fou_ext':

        fou_group2disp_raw_list = defaultdict(list)
        fou_group2disp_i_d_i_j_list = defaultdict(list)
        fou_group2disp_i_d_i_j_k_list = defaultdict(list)

        ext_group2disp_raw_list = defaultdict(list)
        ext_group2disp_i_d_i_j_list = defaultdict(list)
        ext_group2disp_i_d_i_j_k_list = defaultdict(list)

    for division_index in tqdm(range(division_count), position = 1, leave = False):
        with open(os.path.join(disp_pair_dire, 'pid2up2i_count_d_%s_max_%s.pkl'%(division_index, division_count)), 'rb') as f:
            pid2up_refer2i_count = pickle.load(f)

        with open(os.path.join(disp_pair_dire, 'pid2up2j_count_d_%s_max_%s.pkl'%(division_index, division_count)), 'rb') as f:
            pid2up_refer2j_count = pickle.load(f)

        with open(os.path.join(disp_pair_dire, 'pid2up2i_j_k_count_d_%s_max_%s.pkl'%(division_index, division_count)), 'rb') as f:
            pid2up_refer2i_j_k_count = pickle.load(f)

        valid_pid_single = set(pid2up_refer2i_count) & valid_pid_set

        for pid in tqdm(valid_pid_single, position = 2, leave = False):

            up2i_count = pid2up_refer2i_count[pid]
            up2j_count = pid2up_refer2j_count[pid]
            up2_i_j_k = pid2up_refer2i_j_k_count[pid]

            up2counts = pid2reference_counts.get(pid, {})

            for up, i_count in up2i_count.items():
                if up not in valid_pid_set:
                    continue
                if up not in up2counts:
                    continue
                j_count = up2j_count[up]
                i_j_k = up2_i_j_k[up]

                if i_j_k == 0 or (i_count + j_count) == 0:
                    continue
                disp_raw = (i_count - j_count) / i_j_k
                disp_i_d_i_j = i_count / (i_count + j_count)
                disp_i_d_i_j_k = i_count / i_j_k

                up_count, down_count = up2counts[up]

                if type == 'gen':
                    # generalization reference (isolated)
                    group_int = 1 if (up_count == 0 and down_count == 0) else 0

                    gen_group2disp_raw_list[group_int].append(disp_raw)
                    gen_group2disp_i_d_i_j_list[group_int].append(disp_i_d_i_j)
                    gen_group2disp_i_d_i_j_k_list[group_int].append(disp_i_d_i_j_k)
                elif type == 'fou_ext':
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

                    fou_group2disp_raw_list[fou_int].append(disp_raw)
                    fou_group2disp_i_d_i_j_list[fou_int].append(disp_i_d_i_j)
                    fou_group2disp_i_d_i_j_k_list[fou_int].append(disp_i_d_i_j_k)

                    ext_group2disp_raw_list[ext_int].append(disp_raw)
                    ext_group2disp_i_d_i_j_list[ext_int].append(disp_i_d_i_j)
                    ext_group2disp_i_d_i_j_k_list[ext_int].append(disp_i_d_i_j_k)
    if type == 'gen':
        save_group_results(gen_group2disp_raw_list, os.path.join(meta_save_dire, "gen_disp_raw%s.csv"%mode_suffix))
        save_group_results(gen_group2disp_i_d_i_j_list, os.path.join(meta_save_dire, "gen_disp_i_j%s.csv"%mode_suffix))
        save_group_results(gen_group2disp_i_d_i_j_k_list, os.path.join(meta_save_dire, "gen_disp_i_j_k%s.csv"%mode_suffix))
    elif type == 'fou_ext':
        save_group_results(fou_group2disp_raw_list, os.path.join(meta_save_dire, "fou_disp_raw%s.csv"%mode_suffix))
        save_group_results(fou_group2disp_i_d_i_j_list, os.path.join(meta_save_dire, "fou_disp_i_j%s.csv"%mode_suffix))
        save_group_results(fou_group2disp_i_d_i_j_k_list, os.path.join(meta_save_dire, "fou_disp_i_j_k%s.csv"%mode_suffix))

        save_group_results(ext_group2disp_raw_list, os.path.join(meta_save_dire, "ext_disp_raw%s.csv"%mode_suffix))
        save_group_results(ext_group2disp_i_d_i_j_list, os.path.join(meta_save_dire, "ext_disp_i_j%s.csv"%mode_suffix))
        save_group_results(ext_group2disp_i_d_i_j_k_list, os.path.join(meta_save_dire, "ext_disp_i_j_k%s.csv"%mode_suffix))

print("finish")