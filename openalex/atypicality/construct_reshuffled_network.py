"""
Merge the per-source-year reshuffled citation files written by
network_reshuffle_efficient.py into a single reshuffled network, per random-shuffle
iteration. For each target_iteration, reads the per-source-year files
citing_pid2cited_works_list_shuffled_source_<year>.pkl, keeps only those whose source
year is strictly before upper_year, and accumulates both directions of the network
(citing -> cited and cited -> citing). The merged maps are written with upper_year in
the filename. Resumable via a per-iteration finished-flag.

Input  : META_DIRE/random_shuffle_citation/{target_iteration}/
           citing_pid2cited_works_list_shuffled_source_{source_year}.pkl
           (from network_reshuffle_efficient.py)
Output : META_DIRE/random_shuffle_citation/{target_iteration}/constructed_network/
           shuffled_citing_pid2cited_pids_list_before_{upper_year}.pkl
           shuffled_cited_pid2citing_pids_list_before_{upper_year}.pkl
           reshuffled_network_constructed_before_{upper_year}.pkl   (finished marker)
"""

import os
import sys
import time
import pickle
from collections import defaultdict

from tqdm import tqdm

# utils.py is in the parent directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
from utils import META_DIRE  # noqa: E402

meta_dire = META_DIRE
print("meta_dire",meta_dire)

# only keep the reshuffled per-source-year files whose source year is strictly BEFORE this year
upper_year = 2020
print("upper_year", upper_year)

for target_iteration in [4,5,6,7,8,9]:

    print("target_iteration",target_iteration)

    load_dire = os.path.join(meta_dire, 'random_shuffle_citation', str(target_iteration))
    save_dire = os.path.join(load_dire, 'constructed_network')
    os.makedirs(save_dire, exist_ok = True)

    citing_pid2cited_pid_set = defaultdict(set)
    save_citing_pid2cited_pid_set = os.path.join(save_dire, f'shuffled_citing_pid2cited_pids_list_before_{upper_year}.pkl')


    cited_pid2citing_pid_set = defaultdict(set)
    save_cited_pid2citing_pid_set = os.path.join(save_dire, f'shuffled_cited_pid2citing_pids_list_before_{upper_year}.pkl')

    finished_save_dire = os.path.join(save_dire, f'reshuffled_network_constructed_before_{upper_year}.pkl')
    if os.path.exists(finished_save_dire) == False:
        file_list = os.listdir(load_dire)
        file_list = sorted(file_list, reverse = True)
        for file in tqdm(file_list, desc = 'process files for citing_pid2cited_pid_set'):

            if 'reshuffle_source' in file or 'constructed_network' in file:
                continue

            # the per-source-year reshuffle files are named
            # citing_pid2cited_works_list_shuffled_source_<year>.pkl; keep only the
            # files whose source year is strictly before upper_year.
            source_year = int(file.replace('.pkl', '').split('_')[-1])
            if source_year >= upper_year:
                continue

            full_dire = os.path.join(load_dire, file)


            start_time = time.time()  # Record start time
            with open(full_dire, 'rb') as f:
                source2target_list = pickle.load(f) # source is the citing pid, target is the cited pid, in our specification
            end_time = time.time()  # Record end time
            execution_time = end_time - start_time
            print(f"Execution Time: {execution_time:.4f} seconds", full_dire)

            for source, target_list in tqdm(source2target_list.items(), desc = 'compact citing_pid2cited_pid_set', position = 1, leave = False):
                citing_pid2cited_pid_set[source] |= set(target_list)
                for target in target_list:
                    cited_pid2citing_pid_set[target].add(source)

        citing_pid2cited_pid_set = {citing_pid: list(cited_pid_set) for citing_pid, cited_pid_set in tqdm(citing_pid2cited_pid_set.items(), desc = 'process citing_pid2cited_pid_set')}
        with open(save_citing_pid2cited_pid_set, 'wb') as f:
            pickle.dump(citing_pid2cited_pid_set, f, pickle.HIGHEST_PROTOCOL)


        del citing_pid2cited_pid_set

        cited_pid2citing_pid_set = {cited_pid: list(citing_pid_set) for cited_pid, citing_pid_set in tqdm(cited_pid2citing_pid_set.items(), desc = 'process cited_pid2citing_pid_set')}
        with open(save_cited_pid2citing_pid_set, 'wb') as f:
            pickle.dump(cited_pid2citing_pid_set, f, pickle.HIGHEST_PROTOCOL)

        with open(finished_save_dire, 'wb') as f:
            pickle.dump([], f, pickle.HIGHEST_PROTOCOL)
        del cited_pid2citing_pid_set

print("finished")

