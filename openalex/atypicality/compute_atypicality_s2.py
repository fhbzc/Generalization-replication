"""
Atypicality step 2 (Uzzi et al., Science 2013): turn the per-source-pair co-citation
frequencies from compute_atypicality_s1.py into z-scores.

For each source pair (s_min, s_max), collect its frequency across the reshuffled
null-model networks (iterations where the pair never appears count as 0), compute the
null mean/std, and z-score the OBSERVED frequency against that null distribution:
    z = (obs - mean_null) / std_null
Pairs whose null frequencies are all identical (std 0) are skipped.

upper_year_list is processed one year at a time (each is independent), so multiple
cutoffs can be z-scored in a single run.

Input  : null models  : META_DIRE/random_shuffle_citation/{iter}/constructed_network/
                           source_min2source_max2fre_no_more_than_{upper_year}.pkl
         observed      : META_DIRE/source_min2source_max2fre_no_more_than_{upper_year}.pkl
         (both from compute_atypicality_s1.py; each upper_year must be one of its
          upper_year_list values)
Output : META_DIRE/source_min2source_max2z_no_more_than_{upper_year}.pkl
"""

import os
import sys
from collections import defaultdict
import pickle

import numpy as np
from tqdm import tqdm

# utils.py is in the parent directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
from utils import META_DIRE  # noqa: E402

meta_dire = META_DIRE
print("meta_dire", meta_dire)


upper_year_list = [2020]   # each must be one of compute_atypicality_s1.py's upper_year_list
print("upper_year_list", upper_year_list)


# the reshuffled null-model iterations (the observed network is read separately below)
random_iteration_list = [0,1,2,3,4,5,6,7,8,9]
random_batch_size = len(random_iteration_list)
print("random_iteration_list", random_iteration_list)
print("random_batch_size", random_batch_size)


def compute_z_for_upper_year(upper_year):
    """Z-score every source pair's observed co-citation frequency against the null
    distribution from the reshuffled iterations, for a single upper_year cutoff."""
    s_min2s_max2fre_list_main = defaultdict(lambda: defaultdict(list))

    for random_iteration in tqdm(random_iteration_list, desc='load file (upper_year %s)' % upper_year):

        load_dire = os.path.join(meta_dire, 'random_shuffle_citation', str(random_iteration), 'constructed_network') # random iteration

        with open(os.path.join(load_dire, 'source_min2source_max2fre_no_more_than_%s.pkl'%upper_year), 'rb') as f:
            s_min2s_max2fre = pickle.load(f)

        for s_min, s_max2fre in s_min2s_max2fre.items():
            for s_max, fre in s_max2fre.items():
                s_min2s_max2fre_list_main[s_min][s_max].append(fre)


    # observed fre

    with open(os.path.join(meta_dire, 'source_min2source_max2fre_no_more_than_%s.pkl'%upper_year), 'rb') as f:
        s_min2s_max2fre_obs = pickle.load(f)

    s_min2s_max2z = {}
    invalid_item_count = 0
    valid_item_count = 0
    for s_min, s_max2fre_list in tqdm(s_min2s_max2fre_list_main.items(), desc='compute z-score (upper_year %s)' % upper_year):
        for s_max, fre_list in s_max2fre_list.items():
            l_fre = len(fre_list)
            if l_fre < 1:
                invalid_item_count += 1
            else:

                if l_fre < random_batch_size:
                    new_fre_list = fre_list + [0 for _ in range(random_batch_size - l_fre)]
                else:
                    new_fre_list = fre_list

                all_same_flag = (len(set(new_fre_list)) == 1)
                if all_same_flag:
                    invalid_item_count += 1
                    continue

                valid_item_count += 1

                std = np.std(new_fre_list)
                mean = np.mean(new_fre_list)
                obs_count = s_min2s_max2fre_obs[s_min][s_max] if s_min in s_min2s_max2fre_obs and s_max in s_min2s_max2fre_obs[s_min] else 0
                z = (obs_count - mean) / std

                s_min2s_max2z.setdefault(s_min, {})
                s_min2s_max2z[s_min][s_max] = z



    with open(os.path.join(meta_dire, 'source_min2source_max2z_no_more_than_%s.pkl'%upper_year), 'wb') as f:
        pickle.dump(s_min2s_max2z, f, pickle.HIGHEST_PROTOCOL)

    print("invalid_item_count",invalid_item_count)
    print("valid_item_count",valid_item_count)


    print("size of s_min2s_max2z", len(s_min2s_max2z))


# process each upper_year in turn (each cutoff is independent)
for upper_year in upper_year_list:
    print("==== upper_year", upper_year, "====")
    compute_z_for_upper_year(upper_year)

print("finished")