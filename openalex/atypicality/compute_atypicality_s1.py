"""
Atypicality step 1 (Uzzi et al., Science 2013): accumulate the OBSERVED co-citation
source-pair frequencies on the reshuffled (null-model) citation networks.

For each random iteration and each upper_year, iterate over every citing paper
published before upper_year, map its cited papers to their primary source/journal
(pid2primay_location), count co-occurrences of every source pair within that paper's
reference list, and accumulate them into s_min2s_max2fre[s_min][s_max] (s_min<=s_max;
the diagonal s->s counts within-source pairs as n*(n-1)/2).

Input  : META_DIRE/pid2pub_year.pkl
         META_DIRE/pid2primay_location.pkl   (pid -> primary source id; the 'primay'
                                               spelling is the upstream filename, kept as-is)
         random_iteration is None (OBSERVED / real network):
           META_DIRE/citing_pid2cited_pids_list.pkl  (full network, cut in memory)
         random_iteration 0..9 (reshuffled null models):
           META_DIRE/random_shuffle_citation/{iter}/constructed_network/
             shuffled_citing_pid2cited_pids_list_before_{main_upper_year}.pkl
           (construct_reshuffled_network.py output; the network is built truncated
            before main_upper_year, then sub-thresholded here by upper_year_list)
Output : <load_dire>/source_min2source_max2fre_no_more_than_{upper_year}.pkl
         (observed -> META_DIRE/ ; shuffles -> the iteration's constructed_network/)
"""

import os
import sys
import random
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pickle
import time

# utils.py is in the parent directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
from utils import META_DIRE  # noqa: E402

meta_dire = META_DIRE
print("meta_dire", meta_dire)
upper_year_list = [2020]
# None = the OBSERVED (real, un-shuffled) network; 0..9 = the reshuffled null models
random_iteration_list = [4, 5]
# the reshuffled network is truncated to citing papers BEFORE this year when it is
# built (construct_reshuffled_network.py writes ..._before_{upper_year}.pkl); load
# that single network and then sub-threshold it in memory with upper_year_list.
# Must be >= max(upper_year_list) so every sub-threshold is fully covered.
main_upper_year = 2025
assert main_upper_year >= max(upper_year_list), (main_upper_year, upper_year_list)

print("random_iteration_list",random_iteration_list)
print("upper_year_list",upper_year_list)
print("main_upper_year", main_upper_year)


with open(os.path.join(meta_dire, 'pid2pub_year.pkl'), 'rb') as f:
    pid2year = pickle.load(f)
print("pid2year load finished")



with open(os.path.join(meta_dire, 'pid2primay_location.pkl'), 'rb') as f:
    pid2p_l_id_main = pickle.load(f)
print("load pid2p_l_id_main finished")

def dump_pair_fre(pair_fre, load_dire, upper_year):
    # rebuild the nested s_min -> s_max -> fre output format from the flat
    # (s_min, s_max) -> fre accumulator, then pickle it (format unchanged)
    nested = {}
    for (s_min, s_max), fre in pair_fre.items():
        nested.setdefault(s_min, {})[s_max] = fre
    out_path = os.path.join(load_dire, 'source_min2source_max2fre_no_more_than_%s.pkl' % upper_year)
    with open(out_path, 'wb') as f:
        pickle.dump(nested, f, pickle.HIGHEST_PROTOCOL)
    print("size of s_min2s_max2fre", len(nested), "| upper_year", upper_year)


for random_iteration in tqdm(random_iteration_list, desc = 'iterate over random iterations'):

    if random_iteration is None:
        # observed (real, un-shuffled) network: the full citation network, with the
        # upper_year cutoff applied in memory below
        load_dire = meta_dire
        file_dire = os.path.join(meta_dire, 'citing_pid2cited_pids_list.pkl')
    else:
        load_dire = os.path.join(meta_dire, 'random_shuffle_citation', str(random_iteration), 'constructed_network')
        file_dire = os.path.join(load_dire, f'shuffled_citing_pid2cited_pids_list_before_{main_upper_year}.pkl')

    # ---- only (re)process the upper_years that have NOT been written yet ----
    # The result for cutoff U is the co-citation frequency over all papers with
    # pub_year < U, and it is ADDITIVE over papers, so:
    #   result(U) = result(seed) + contributions from papers in [seed, U)
    # for any earlier cutoff `seed` < U. We therefore (a) skip the iteration if every
    # cutoff is already on disk, and (b) seed the accumulator from the largest already-
    # computed cutoff below the smallest missing one, so papers already baked into that
    # result are not rescanned.
    sorted_upper_years = sorted(upper_year_list)

    def output_path(uy):
        return os.path.join(load_dire, 'source_min2source_max2fre_no_more_than_%s.pkl' % uy)

    existing = {uy for uy in sorted_upper_years if os.path.exists(output_path(uy))}
    missing = [uy for uy in sorted_upper_years if uy not in existing]
    if not missing:
        print(f"skip random_iteration {random_iteration}: all outputs already exist")
        continue

    smallest_missing = missing[0]
    seed_candidates = [uy for uy in existing if uy < smallest_missing]
    seed_year = max(seed_candidates) if seed_candidates else None
    max_missing = missing[-1]

    print(f"random_iteration {random_iteration}: existing={sorted(existing)} "
          f"missing={missing} seed={seed_year}")

    with open(file_dire, 'rb') as f:
        citing2up_list = pickle.load(f)
    print("load citing2up_list finished")

    # ---- (#2) process each remaining paper ONCE across the missing thresholds ----
    # Keep only papers that still affect a missing cutoff: pub_year < max_missing, and
    # (when seeding) pub_year >= seed_year, since earlier papers are already in the seed.
    # One growing accumulator, dump a snapshot each time a MISSING cutoff is crossed.
    papers = [(pid2year[citing], up_list)
              for citing, up_list in citing2up_list.items()
              if pid2year[citing] < max_missing
              and (seed_year is None or pid2year[citing] >= seed_year)]
    papers.sort(key=lambda yu: yu[0])           # ascending by pub_year
    del citing2up_list                          # value lists still referenced by `papers`
    print(f"papers in [{seed_year}, {max_missing}): {len(papers)}")

    pair_fre = defaultdict(int)                 # (s_min, s_max) -> frequency
    if seed_year is not None:
        # rehydrate the flat accumulator from the nested result(seed_year) on disk
        with open(output_path(seed_year), 'rb') as f:
            seed_nested = pickle.load(f)
        for s_min, inner in seed_nested.items():
            for s_max, fre in inner.items():
                pair_fre[(s_min, s_max)] = fre
        del seed_nested
        print(f"seeded accumulator from upper_year {seed_year}")

    dump_points = missing                       # ascending; all strictly > seed_year
    ti = 0                                       # next missing cutoff to dump

    for year, up_list in tqdm(papers, desc='iterate over papers', position=1, leave=False):
        # this paper's year is >= some pending missing cutoff(s): the accumulator now
        # holds exactly the papers below each such cutoff, so dump them before adding it
        while ti < len(dump_points) and year >= dump_points[ti]:
            dump_pair_fre(pair_fre, load_dire, dump_points[ti])
            ti += 1

        # ---- (#3) count sources, sort once so each pair is already (s_min, s_max) ----
        s_id2count = defaultdict(int)
        for up in up_list:
            s = pid2p_l_id_main.get(up)
            if s is not None:
                s_id2count[s] += 1
        if not s_id2count:
            continue
        items = sorted(s_id2count.items())      # by source id ascending -> s1 < s2

        for a in range(len(items)):
            s1, f1 = items[a]
            # within-source pairs (diagonal); key created even when f1 == 1 (adds 0),
            # matching the original output
            pair_fre[(s1, s1)] += f1 * (f1 - 1) // 2
            for b in range(a + 1, len(items)):
                s2, f2 = items[b]
                pair_fre[(s1, s2)] += f1 * f2    # s1 < s2 because items is sorted

    # papers ran out before crossing the remaining (largest) missing cutoffs -> dump them
    while ti < len(dump_points):
        dump_pair_fre(pair_fre, load_dire, dump_points[ti])
        ti += 1

print("finished")