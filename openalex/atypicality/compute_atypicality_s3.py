"""
Atypicality step 3 (Uzzi et al., Science 2013): turn the per-source-pair z-scores
from compute_atypicality_s2.py into a per-paper atypicality profile.

For each upper_year cutoff and each citing paper published in [upper_year, ...), map
its references to their primary source/journal, count how many references land in each
source, then weight every (source_i, source_j) co-citation pair (including the i==i
self-pair) by its co-occurrence count and pull that pair's z-score. The weighted bag of
z-scores is summarized per paper as mean / median / 5,10,25,75,90th percentiles
(conventional/novel tails). Papers with no resolvable pair get sentinel -999.

upper_year_list is processed one cutoff at a time (each is independent), writing one CSV
per cutoff.

Input  : META_DIRE/source_min2source_max2z_no_more_than_{upper_year}.pkl   (compute_atypicality_s2.py)
         META_DIRE/citing_pid2cited_pids_list.pkl                          (observed network)
         META_DIRE/pid2pub_year.pkl
         META_DIRE/pid2primay_location.pkl
Output : META_DIRE/atypicality/atypicality_all_after_{upper_year}.csv
         columns: pid, pub_year, up_refer_count, up_refer_has_location_count,
                  location_count, atypicality_valid_weight, atypicality_invalid_weight,
                  Z_mean, Z_median, Z_90th, Z_75th, Z_25th, Z_10th, Z_5th
"""

import os
import sys
import random
from collections import defaultdict, OrderedDict
from tqdm import tqdm
import numpy as np
import pickle
import time
import pandas as pd

# utils.py is in the parent directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
from utils import META_DIRE  # noqa: E402

meta_dire = META_DIRE
print("meta_dire",meta_dire)


save_dire = os.path.join(meta_dire, 'atypicality')
os.makedirs(save_dire, exist_ok = True)

upper_year_list = [1960, 1970, 1980, 1990, 2000, 2010, 2020]
print("upper_year_list",upper_year_list)

# ---------------------------------------------------------------------------
# Two versions of the atypicality profile are supported:
#   * "full"    : use every reference's source            -> atypicality_all_after_{upper_year}.csv
#   * "removed" : drop references whose source is excluded -> atypicality_all_after_{upper_year}_removed.csv
# Set DO_REMOVE = True to produce the "removed" version (excludes REMOVE_SOURCE_IDS),
# or DO_REMOVE = False to produce the "full" version (uses every source).
# ---------------------------------------------------------------------------
DO_REMOVE = True
REMOVE_SOURCE_IDS = set({
    "S4306525036", # PubMed
    "S4306400194", # arXiv (Cornell University)
    "S4306402512", # HAL (Le Centre pour la Communication Scientifique Directe)
    "S4306401280", # DOAJ (DOAJ: Directory of Open Access Journals)
    "S4210172589", # SSRN Electronic Journal
    "S4306401271", # RePEc: Research Papers in Economics
    "S41354064", # ChemInform
    "S4317411217", # Deleted Journal
    "S4306400562", # Zenodo (CERN European Organization for Nuclear Research)
    "S4306402450", # Research Square (Research Square)
    "S4210187579", # Chemischer Informationsdienst
    "S4306402567", # bioRxiv (Cold Spring Harbor Laboratory)
})
VERSION_TAG = 'removed' if DO_REMOVE else 'full'
print("VERSION_TAG", VERSION_TAG, "DO_REMOVE", DO_REMOVE, "remove_source_count", len(REMOVE_SOURCE_IDS))

with open(os.path.join(meta_dire, 'pid2pub_year.pkl'), 'rb') as f:
    pid2year = pickle.load(f)
print("pid2year load finished")


with open(os.path.join(meta_dire, 'citing_pid2cited_pids_list.pkl'), 'rb') as f:
    citing2up_list = pickle.load(f)
print("load citing2up_list finished")


with open(os.path.join(meta_dire, 'pid2primay_location.pkl'), 'rb') as f:
    pid2p_l_id_main = pickle.load(f)
print("load pid2p_l_id_main finished")


pid2s_id2count = {}
pid2has_location_up_count = {}
removed_refer_count = 0  # how many references were dropped because their source is in REMOVE_SOURCE_IDS
for upper_year in tqdm(upper_year_list):
    if VERSION_TAG == 'removed':
        out_name = 'atypicality_all_after_%s_removed.csv' % upper_year
    else:
        out_name = 'atypicality_all_after_%s.csv' % upper_year
    out_path = os.path.join(save_dire, out_name)
    if os.path.exists(out_path):
        print("skip (already exists):", out_path)
        continue

    with open(os.path.join(meta_dire, 'source_min2source_max2z_no_more_than_%s.pkl'%upper_year), 'rb') as f:
        s_min2s_max2z = pickle.load(f)

    df_res = []
    for citing, up_list in tqdm(citing2up_list.items(), position = 1, leave = False):
        if pid2year[citing] < upper_year:
            continue
        


        if citing not in pid2s_id2count:
            s_id2count = defaultdict(int)
            has_location_up_count = 0
            for up in up_list:
                if up in pid2p_l_id_main:
                    s_id = pid2p_l_id_main[up]
                    if DO_REMOVE and s_id in REMOVE_SOURCE_IDS:
                        removed_refer_count += 1
                        continue
                    s_id2count[s_id] += 1
                    has_location_up_count += 1

            pid2s_id2count[citing] = s_id2count
            pid2has_location_up_count[citing] = has_location_up_count
        else:
            s_id2count = pid2s_id2count[citing]
            has_location_up_count = pid2has_location_up_count[citing]
        s_id_list = list(s_id2count)

        atypicality_valid_norm = 0
        atypicality_invalid_norm = 0
        atypicality_list = []

        for s_id_index1 in range(len(s_id_list)):
            s_id1 = s_id_list[s_id_index1]
            # self citation
            s1_fre =s_id2count[s_id1]
            n = s1_fre
            w = (n * (n-1)// 2)
            if s_id1 in s_min2s_max2z and s_id1 in s_min2s_max2z[s_id1]:
                atypicality_valid_norm += w

                atypicality_list.extend([s_min2s_max2z[s_id1][s_id1]] * w)

            else:
                atypicality_invalid_norm += w

            for s_id_index2 in range(s_id_index1 + 1, len(s_id_list)):
                s_id2 = s_id_list[s_id_index2]
                s2_fre =s_id2count[s_id2]

                s_min = min(s_id1, s_id2)
                s_max = max(s_id1, s_id2)
                w = s1_fre * s2_fre
                if s_min in s_min2s_max2z and s_max in s_min2s_max2z[s_min]:
                    atypicality_valid_norm += w

                    atypicality_list.extend([s_min2s_max2z[s_min][s_max]] * w)

                else:
                    atypicality_invalid_norm += w

        orderdict = OrderedDict()
        orderdict['pid'] = citing
        orderdict['pub_year'] = pid2year[citing]
        orderdict['up_refer_count'] = len(up_list)
        orderdict['up_refer_has_location_count'] = has_location_up_count
        orderdict['location_count'] = len(s_id_list)

        orderdict['atypicality_valid_weight'] = atypicality_valid_norm
        orderdict['atypicality_invalid_weight'] = atypicality_invalid_norm

        if len(atypicality_list) > 0:

            percentiles = np.percentile(atypicality_list, [5, 10, 25, 50, 75, 90])
            orderdict['Z_mean'] = np.mean(atypicality_list)
            orderdict['Z_median'] = percentiles[3]
            orderdict['Z_90th'] = percentiles[5]
            orderdict['Z_75th'] = percentiles[4]
            orderdict['Z_25th'] = percentiles[2]
            orderdict['Z_10th'] = percentiles[1]
            orderdict['Z_5th'] = percentiles[0]

        else:

            orderdict['Z_mean'] = -999
            orderdict['Z_median'] = -999
            orderdict['Z_90th'] = -999
            orderdict['Z_75th'] = -999
            orderdict['Z_25th'] = -999
            orderdict['Z_10th'] = -999
            orderdict['Z_5th'] = -999

        df_res.append(orderdict)

    df_res = pd.DataFrame(df_res)
    df_res.to_csv(out_path, index = False)
    print("saved:", out_path, "| removed_refer_count so far:", removed_refer_count)
print("finished")