"""
Merge the per-source-year topic-controlled reshuffled citation files written by
network_reshuffle_efficient_control_topic.py into a single reshuffled network, per
random-shuffle iteration.

This is the topic-hierarchy twin of
atypicality_control_venue/construct_reshuffled_network_control_venue.py: same merge logic,
but it reads from / writes to the random_shuffle_citation_control_<level>/ tree (level =
CONTROL_LEVEL, one of topic/subfield/field/domain) so it never collides with the venue or
uncontrolled reshuffles.

For each target_iteration, reads the per-source-year files
citing_pid2cited_works_list_shuffled_source_<year>.pkl, keeps only those whose source year
is strictly before upper_year, and accumulates both directions of the network
(citing -> cited and cited -> citing). The merged maps are written with upper_year in the
filename. Resumable via a per-iteration finished-flag.

Input  : META_DIRE_2026/random_shuffle_citation_control_<level>/{target_iteration}/
           citing_pid2cited_works_list_shuffled_source_{source_year}.pkl
           (from network_reshuffle_efficient_control_topic.py)
Output : META_DIRE_2026/random_shuffle_citation_control_<level>/{target_iteration}/constructed_network/
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

# utils.py is in openalex/, two levels up from this file
# (openalex/2026_version/atypicality_control_topic/<this file>).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OPENALEX_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.insert(0, OPENALEX_DIR)
from utils import META_DIRE_2026  # noqa: E402

# ---- which topic level to control for --------------------------------------------------
# one of: 'topic', 'subfield', 'field', 'domain'; must match the reshuffle step.
CONTROL_LEVEL = 'topic'
assert CONTROL_LEVEL in ('topic', 'subfield', 'field', 'domain')
print("CONTROL_LEVEL", CONTROL_LEVEL)

meta_dire = META_DIRE_2026
print("meta_dire", meta_dire)

# only keep the reshuffled per-source-year files whose source year is strictly BEFORE this year
upper_year = 2025
print("upper_year", upper_year)
# each observed network is reshuffled only ONCE -> a single iteration (0)
target_iteration_list = [0]
print("target_iteration_list", target_iteration_list)
for target_iteration in target_iteration_list:

    print("target_iteration", target_iteration)

    load_dire = os.path.join(meta_dire, f'random_shuffle_citation_control_{CONTROL_LEVEL}', str(target_iteration))
    save_dire = os.path.join(load_dire, 'constructed_network')
    os.makedirs(save_dire, exist_ok=True)

    citing_pid2cited_pid_set = defaultdict(set)
    save_citing_pid2cited_pid_set = os.path.join(save_dire, f'shuffled_citing_pid2cited_pids_list_before_{upper_year}.pkl')

    cited_pid2citing_pid_set = defaultdict(set)
    save_cited_pid2citing_pid_set = os.path.join(save_dire, f'shuffled_cited_pid2citing_pids_list_before_{upper_year}.pkl')

    finished_save_dire = os.path.join(save_dire, f'reshuffled_network_constructed_before_{upper_year}.pkl')
    if os.path.exists(finished_save_dire) == False:
        # the source year immediately before upper_year (upper_year - 1) MUST have a
        # finished reshuffle file present; otherwise the reshuffle step has not caught
        # up to the latest year and the merged network would silently miss it.
        found_upper_minus_one = False

        file_list = os.listdir(load_dire)
        file_list = sorted(file_list, reverse=True)
        for file in tqdm(file_list, desc='process files for citing_pid2cited_pid_set'):

            if 'reshuffle_source' in file or 'constructed_network' in file:
                continue

            # the per-source-year reshuffle files are named
            # citing_pid2cited_works_list_shuffled_source_<year>.pkl; keep only the
            # files whose source year is strictly before upper_year.
            source_year = int(file.replace('.pkl', '').split('_')[-1])
            if source_year == upper_year - 1:
                found_upper_minus_one = True
            if source_year >= upper_year:
                continue

            full_dire = os.path.join(load_dire, file)

            start_time = time.time()  # Record start time
            with open(full_dire, 'rb') as f:
                source2target_list = pickle.load(f)  # source is the citing pid, target is the cited pid, in our specification
            end_time = time.time()  # Record end time
            execution_time = end_time - start_time
            print(f"Execution Time: {execution_time:.4f} seconds", full_dire)

            for source, target_list in tqdm(source2target_list.items(), desc='compact citing_pid2cited_pid_set', position=1, leave=False):
                citing_pid2cited_pid_set[source] |= set(target_list)
                for target in target_list:
                    cited_pid2citing_pid_set[target].add(source)

        # the year exactly one less than upper_year must be finished/present,
        # otherwise abort instead of writing a network that misses the latest year.
        assert found_upper_minus_one, (
            f"source year {upper_year - 1} (upper_year - 1) not finished: no reshuffle "
            f"file for it in {load_dire}"
        )

        citing_pid2cited_pid_set = {citing_pid: list(cited_pid_set) for citing_pid, cited_pid_set in tqdm(citing_pid2cited_pid_set.items(), desc='process citing_pid2cited_pid_set')}
        with open(save_citing_pid2cited_pid_set, 'wb') as f:
            pickle.dump(citing_pid2cited_pid_set, f, pickle.HIGHEST_PROTOCOL)

        del citing_pid2cited_pid_set

        cited_pid2citing_pid_set = {cited_pid: list(citing_pid_set) for cited_pid, citing_pid_set in tqdm(cited_pid2citing_pid_set.items(), desc='process cited_pid2citing_pid_set')}
        with open(save_cited_pid2citing_pid_set, 'wb') as f:
            pickle.dump(cited_pid2citing_pid_set, f, pickle.HIGHEST_PROTOCOL)

        with open(finished_save_dire, 'wb') as f:
            pickle.dump([], f, pickle.HIGHEST_PROTOCOL)
        del cited_pid2citing_pid_set

print("finished")
