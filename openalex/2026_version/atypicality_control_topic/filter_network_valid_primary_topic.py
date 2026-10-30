"""
Drop every citation edge that involves a pid WITHOUT a valid primary SUBFIELD.

This is the topic-hierarchy twin of atypicality_control_venue/filter_network_valid_primary_location.py.
Instead of controlling for venue (primary_location), the topic-hierarchy pipeline controls
for one level of the OpenAlex topic hierarchy. But the FILTER produces a SINGLE shared
observed network: a pid is valid iff it has a non-None primary SUBFIELD
(pid2primary_subfield.pkl, from get_meta_data_preprint_2026_s2_topic.py).

Why subfield, and why a single version: a paper's primary_topic carries its topic,
subfield, field AND domain ids together -- they are present or absent as a unit -- so
"has a subfield" is equivalent to "has a topic / field / domain". One filtered network
therefore serves every control level: the downstream reshuffle steps
(network_reshuffle_efficient_control_topic.py) all start from THIS same observed network
and only differ in which level they stratify the swaps by.

For each of the two adjacency dicts (pid -> list of pids), an edge (k, n) is kept only
when BOTH endpoints k and n are valid. Invalid keys are removed, invalid neighbours are
filtered out of the kept lists, and keys whose neighbour list becomes empty are dropped.

Input  : META_DIRE_2026/pid2primary_subfield.pkl
         META_DIRE_2026/cited_pid2citing_pids_list.pkl
         META_DIRE_2026/citing_pid2cited_pids_list.pkl
Output : META_DIRE_2026/cited_pid2citing_pids_list_valid_primary_subfield.pkl
         META_DIRE_2026/citing_pid2cited_pids_list_valid_primary_subfield.pkl
"""

import os
import sys
import pickle
from tqdm import tqdm

# utils.py lives in openalex/, two levels up from this file
# (openalex/2026_version/atypicality_control_topic/<this file>).
OPENALEX_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, OPENALEX_DIR)
from utils import META_DIRE_2026  # noqa: E402

# Validity for the single shared observed network is defined by the SUBFIELD level:
# having a primary subfield is equivalent to having a primary topic/field/domain, so this
# one filtered network is reused by every control level downstream.
FILTER_LEVEL = 'subfield'
print("FILTER_LEVEL", FILTER_LEVEL)

meta_dire = META_DIRE_2026
print("meta_dire", meta_dire)


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# ---- valid pid set = pids with a non-None primary-subfield value ----------------------
pid2group = load_pkl(os.path.join(meta_dire, f'pid2primary_{FILTER_LEVEL}.pkl'))
valid_pids = {pid for pid, group in pid2group.items() if group is not None}
del pid2group
print(f"number of valid pids (non-None primary_{FILTER_LEVEL}):", len(valid_pids))


def filter_network(in_name, out_name):
    out_path = os.path.join(meta_dire, out_name)
    if os.path.exists(out_path):
        print(f"skip {out_name}: already exists")
        return

    net = load_pkl(os.path.join(meta_dire, in_name))
    print(f"{in_name}: {len(net)} keys loaded")

    edges_before = 0
    edges_after = 0
    filtered = {}
    for k, neighbours in tqdm(net.items(), desc=f'filter {in_name}'):
        edges_before += len(neighbours)
        if k not in valid_pids:
            continue
        kept = [n for n in neighbours if n in valid_pids]
        if not kept:                     # no edges remain for this node
            continue
        filtered[k] = kept
        edges_after += len(kept)

    del net
    print(f"  keys:  {len(filtered)} kept")
    print(f"  edges: {edges_before} -> {edges_after}")

    with open(out_path, 'wb') as f:
        pickle.dump(filtered, f, pickle.HIGHEST_PROTOCOL)
    print(f"  wrote {out_path}")


filter_network(
    'citing_pid2cited_pids_list.pkl',
    f'citing_pid2cited_pids_list_valid_primary_{FILTER_LEVEL}.pkl',
)
filter_network(
    'cited_pid2citing_pids_list.pkl',
    f'cited_pid2citing_pids_list_valid_primary_{FILTER_LEVEL}.pkl',
)

print("finished")
