"""
Semantic Scholar (S2AG) version of the OpenAlex
network_reshuffle_efficient_control_venue.py.

Build a degree-preserving reshuffled (null-model) citation network that ALSO controls
for venue, following Uzzi et al. (Science, 2013) but with an extra stratification
dimension. The node id is the S2AG corpusid (cid).

Edges are reshuffled with degree-preserving target swaps: two edges (s1, t1) and
(s2, t2) become (s1, t2) and (s2, t1). The base reshuffle stratifies edges into
(source_year, target_year) blocks and only swaps within a block, so each paper's
citation YEARS are preserved.

Here we add the venue control: a swap must keep, in addition to the source/target cids
acting as source/target, the VENUE of both the source and target papers unchanged. We
achieve this by extending the block key to

    (source_year, source_venue, target_year, target_venue)

and only ever swapping targets WITHIN one such block. Because swaps only exchange targets
inside a single block:
  - every source cid keeps its out-degree and (trivially, it never moves) its venue;
  - every target cid keeps its in-degree, and any target it is swapped with shares the
    same target_year AND target venue;
  - the count of edges between each (source_venue -> target_venue) venue pair, within
    each (source_year, target_year) cell, is exactly preserved.

Input  : META_DIRE/cid2pub_year.pkl                          (get_s2ag_meta.py)
         META_DIRE/cid2venue.pkl                             (cid -> venue; get_s2ag_meta_venue.py)
         META_DIRE/citing2cited_cid_list_remove_circle_valid_venue.pkl
             (the venue-filtered network from filter_network_valid_venue.py;
              every cid in it already has a valid, non-excluded venue)
Output : META_DIRE/random_shuffle_citation_control_venue/{target_iteration}/
           citing_pid2cited_works_list_shuffled_source_{source_year}.pkl
           reshuffle_source_{source_year}_finished.pkl           (per-source-year marker)
"""

import os
import sys
import math
import random
import pickle
from collections import defaultdict

from tqdm import tqdm

# utils.py is in the parent (semantic_scholar/) directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
from utils import META_DIRE  # noqa: E402

meta_dire = META_DIRE
print("meta_dire", meta_dire)

# input network: the venue-filtered one written by filter_network_valid_venue.py
INPUT_NETWORK = 'citing2cited_cid_list_remove_circle_valid_venue.pkl'

target_iteration = 0
print("target_iteration", target_iteration)

save_dire = os.path.join(meta_dire, 'random_shuffle_citation_control_venue', str(target_iteration))
os.makedirs(save_dire, exist_ok=True)

lower_year = -1
upper_year = 2024
print("lower_year", lower_year)
print("upper_year", upper_year)


def reshuffle_network_approx(source_target_list, N):
    """
    Degree-preserving random reshuffle (Uzzi et al., Science 2013). Repeatedly picks two
    edges and swaps their targets. Caller guarantees every edge in source_target_list
    belongs to the SAME (source_year, source_venue, target_year, target_venue) block, so
    the swap preserves both endpoints' year and venue.

    Input : source_target_list -- list of mutable [source, target] pairs
            N                   -- number of swap attempts
    Output: the same list, reshuffled in place
    """
    index_list = [i for i in range(len(source_target_list))]

    choice1_select = random.choices(index_list, k=N)
    choice2_select = random.choices(index_list, k=N)

    for chose_index in tqdm(range(N), desc='conduct network reshuffle', leave=False, position=1):
        choice_index1 = choice1_select[chose_index]
        choice_index2 = choice2_select[chose_index]
        if choice_index1 == choice_index2:
            continue

        source_target_1 = source_target_list[choice_index1]
        source_target_2 = source_target_list[choice_index2]

        source1, target1 = source_target_1
        source2, target2 = source_target_2

        if source1 != target2 and source2 != target1 and source1 != source2:
            source_target_1[1] = target2
            source_target_2[1] = target1

    return source_target_list


_finished_cache = {}


def source_year_finished(source_year):
    """True if this source year has already been fully reshuffled & dumped (memoized so
    the per-edge check in process_yearly_network does not stat the filesystem repeatedly)."""
    cached = _finished_cache.get(source_year)
    if cached is None:
        cached = os.path.exists(os.path.join(save_dire, 'reshuffle_source_%s_finished.pkl' % source_year))
        _finished_cache[source_year] = cached
    return cached


def process_yearly_network(cid2cited_cid_list, cid2year, cid2venue, lower_bound, upper_bound):
    """
    Bucket every kept edge into a (source_year, source_venue, target_year, target_venue) block.

    Returns:
        block2source2targets : {block_key: {source_cid: set(target_cids)}}
        source_year2blocks    : {source_year: set(block_key)}   (iteration order per year)
    where block_key = (source_year, source_venue, target_year, target_venue).
    Only source papers with lower_bound <= source_year <= upper_bound are kept, and
    source years already finished (reshuffle_source_<year>_finished.pkl exists) are
    skipped entirely -- there is no point bucketing edges we will not reshuffle.
    """
    block2source2targets = defaultdict(lambda: defaultdict(set))
    source_year2blocks = defaultdict(set)

    for cid, cited_cid_list in tqdm(cid2cited_cid_list.items(), desc='process citation network'):
        if not cited_cid_list:
            continue

        source_year = cid2year.get(cid)
        if source_year is None:
            continue
        if upper_bound >= 0 and source_year > upper_bound:
            continue
        if lower_bound >= 0 and source_year < lower_bound:
            continue
        if source_year_finished(source_year):       # already done on a previous run
            continue

        source_venue = cid2venue.get(cid)
        if source_venue is None:                     # should not happen on the filtered net
            continue

        for cited_cid in cited_cid_list:
            target_year = cid2year.get(cited_cid)
            if target_year is None:
                continue
            target_venue = cid2venue.get(cited_cid)
            if target_venue is None:
                continue

            block_key = (source_year, source_venue, target_year, target_venue)
            block2source2targets[block_key][cid].add(cited_cid)
            source_year2blocks[source_year].add(block_key)

    return block2source2targets, source_year2blocks


def get_source2target_list(source_target_list):
    """Compact a flat [source, target] edge list back into {source: list(targets)}."""
    citing_cid2cited_cid_set = defaultdict(set)
    for source, target in tqdm(source_target_list, desc='compact citing_cid2cited_cid_set', leave=False):
        citing_cid2cited_cid_set[source].add(target)
    return {citing_cid: list(cited_cid_set)
            for citing_cid, cited_cid_set in tqdm(citing_cid2cited_cid_set.items())}


def reshuffle_network_in_groups(block2source2targets, source_year2blocks):
    """
    For each source year, reshuffle every (source_venue, target_year, target_venue) block
    that belongs to it, then dump the merged result for that source year. Resumable per
    source year via reshuffle_source_<year>_finished.pkl.
    """
    for source_year in tqdm(sorted(source_year2blocks), desc='process reshuffle all years'):

        finished_marker = os.path.join(save_dire, 'reshuffle_source_%s_finished.pkl' % source_year)
        if os.path.exists(finished_marker):
            continue

        source_year_link_list_M = []

        for block_key in source_year2blocks[source_year]:
            source2targets = block2source2targets[block_key]

            # flatten the block into mutable [source, target] edges
            input_source_target_list = [
                [source, target]
                for source, targets in source2targets.items()
                for target in targets
            ]

            E = len(input_source_target_list)
            # ~2 * E * ln(E) swap attempts (matches the reference); no swaps if E <= 1
            shuffle_count = int(2 * math.log(E) * E) if E > 1 else 0

            output_source_target_list = reshuffle_network_approx(input_source_target_list, shuffle_count)
            source_year_link_list_M.extend(output_source_target_list)

        with open(os.path.join(save_dire, 'citing_pid2cited_works_list_shuffled_source_%s.pkl' % source_year), 'wb') as f:
            pickle.dump(get_source2target_list(source_year_link_list_M), f, pickle.HIGHEST_PROTOCOL)

        with open(finished_marker, 'wb') as f:
            pickle.dump([], f, pickle.HIGHEST_PROTOCOL)


with open(os.path.join(meta_dire, 'cid2pub_year.pkl'), 'rb') as f:
    cid2year = pickle.load(f)
print("cid2pub_year load finished")

with open(os.path.join(meta_dire, 'cid2venue.pkl'), 'rb') as f:
    cid2venue = pickle.load(f)
print("cid2venue load finished")

with open(os.path.join(meta_dire, INPUT_NETWORK), 'rb') as f:
    citing_cid2cited_works = pickle.load(f)
print(f"{INPUT_NETWORK} load finished")

for lower_bound, upper_bound in [(lower_year, upper_year)]:

    block2source2targets, source_year2blocks = process_yearly_network(
        citing_cid2cited_works, cid2year, cid2venue, lower_bound, upper_bound)

    reshuffle_network_in_groups(block2source2targets, source_year2blocks)

print("finished")
