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
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import META_DIRE

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

    # Welch's t-test between group 0 and group 1
    arr0 = np.asarray(group2list.get(0, []), dtype=float)
    arr1 = np.asarray(group2list.get(1, []), dtype=float)
    arr0 = arr0[np.isfinite(arr0)]
    arr1 = arr1[np.isfinite(arr1)]
    if len(arr0) > 1 and len(arr1) > 1:
        ttest_result = stats.ttest_ind(arr0, arr1, equal_var=False)
        t_stat = ttest_result.statistic
        p_val = ttest_result.pvalue
        # Welch-Satterthwaite degrees of freedom
        v0, v1 = arr0.var(ddof=1) / len(arr0), arr1.var(ddof=1) / len(arr1)
        df_val = (v0 + v1) ** 2 / (v0 ** 2 / (len(arr0) - 1) + v1 ** 2 / (len(arr1) - 1))
    else:
        t_stat, p_val, df_val = np.nan, np.nan, np.nan

    df['t'] = np.nan
    df['p'] = np.nan
    df['df'] = np.nan
    ttest_row = pd.DataFrame([{
        'group': 'welch_ttest', 'mean': np.nan, 'ci_lower': np.nan,
        'ci_upper': np.nan, 'n': np.nan, 't': t_stat, 'p': p_val, 'df': df_val
    }])
    df = pd.concat([df, ttest_row], ignore_index=True)
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

meta_save_dire = os.path.join(meta_dire, 'reference_interdis_g_e_f/two_group')
os.makedirs(meta_save_dire, exist_ok = True)

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


data_disp = pd.read_csv(os.path.join(meta_dire, 'disruption', 'disp_full.csv'), usecols = ['pid', "pub_year"])

data_disp = data_disp[data_disp['pub_year'] < 2020]





with open(os.path.join(meta_dire, 'pid2concept_list_simple.pkl'), 'rb') as f:
    pid2concept_list = pickle.load(f)
print("load pid2concept_list finished")

with open(os.path.join(meta_dire, 'pid2root_concepts_level_0.pkl'), 'rb') as f:
    pid2root_concept_list = pickle.load(f)
print("load pid2root_concept_list finished")

valid_pid_set = set([pid for pid, root_concept in pid2root_concept_list.items() if root_concept is not None])

valid_pid_set &= set(data_disp['pid'])
del data_disp

pid2concept_list = {pid: set(pid2concept_list[pid]) for pid in tqdm(valid_pid_set, desc = 'process pid2concept_list', position = 1, leave = False)}
pid2root_concept_list = {pid: set(pid2root_concept_list[pid]) for pid in tqdm(valid_pid_set, desc = 'process pid2root_concept_list', position = 1, leave = False)}



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

gen_group2inter_all = defaultdict(list)
gen_group2inter_main = defaultdict(list)


fou_group2inter_all = defaultdict(list)
fou_group2inter_main = defaultdict(list)

ext_group2inter_all = defaultdict(list)
ext_group2inter_main = defaultdict(list)

for pid in tqdm(valid_pid_set, position = 1, leave = False):

    pid_concept = pid2concept_list[pid]
    pid_root_concept = pid2root_concept_list[pid]

    for up, (up_count, down_count) in pid2reference_counts[pid].items():
        # up is guaranteed to be in valid_pid_set (classify_reference_counts filters)

        if pid_concept.isdisjoint(pid2concept_list[up]):
            interall = 1
        else:
            interall = 0

        if pid_root_concept.isdisjoint(pid2root_concept_list[up]):
            intermain = 1
        else:
            intermain = 0

        if up_count == 0 and down_count == 0:
            # generalization reference (isolated)
            gen_int = 1
            ext_int = 0
            fou_int = 0
        else:
            gen_int = 0
            if up_count > down_count:
                ext_int = 1
                fou_int = 0
            else:
                # down_count > up_count, or tie (up_count == down_count > 0) -> foundation
                ext_int = 0
                fou_int = 1

        gen_group2inter_all[gen_int].append(interall)
        gen_group2inter_main[gen_int].append(intermain)


        fou_group2inter_all[fou_int].append(interall)
        fou_group2inter_main[fou_int].append(intermain)


        ext_group2inter_all[ext_int].append(interall)
        ext_group2inter_main[ext_int].append(intermain)

save_group_results(gen_group2inter_all, os.path.join(meta_save_dire, "gen_inter_all%s.csv"%mode_suffix))
save_group_results(gen_group2inter_main, os.path.join(meta_save_dire, "gen_inter_main%s.csv"%mode_suffix))

save_group_results(fou_group2inter_all, os.path.join(meta_save_dire, "fou_inter_all%s.csv"%mode_suffix))
save_group_results(fou_group2inter_main, os.path.join(meta_save_dire, "fou_inter_main%s.csv"%mode_suffix))

save_group_results(ext_group2inter_all, os.path.join(meta_save_dire, "ext_inter_all%s.csv"%mode_suffix))
save_group_results(ext_group2inter_main, os.path.join(meta_save_dire, "ext_inter_main%s.csv"%mode_suffix))

print("finish")