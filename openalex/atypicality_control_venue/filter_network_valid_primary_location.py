"""
Drop every citation edge that involves a pid WITHOUT a valid primary_location, or
whose primary_location source is in the excluded-source filter list.

A pid is considered VALID iff:
  - it is a key of pid2primay_location.pkl with a non-None value (the upstream 'primay'
    spelling is kept as-is; that file stores pid -> primary source id), AND
  - its primary source id is NOT in FILTER_SOURCE_IDS.
Pids whose primary location falls in FILTER_SOURCE_IDS are treated exactly like pids
with no valid primary_location: they (and every citation edge touching them) are removed.

For each of the two adjacency dicts (pid -> list of pids), an edge (k, n) is kept
only when BOTH endpoints k and n are valid. Concretely: invalid keys are removed,
invalid neighbours are filtered out of the kept lists, and keys whose neighbour list
becomes empty are dropped (no edges left).

Input  : META_DIRE/pid2primay_location.pkl
         META_DIRE/cited_pid2citing_pids_list.pkl
         META_DIRE/citing_pid2cited_pids_list.pkl
Output : META_DIRE/cited_pid2citing_pids_list_valid_primary_location.pkl
         META_DIRE/citing_pid2cited_pids_list_valid_primary_location.pkl
"""

import os
import sys
import pickle
from tqdm import tqdm

# utils.py lives in the parent openalex/ directory
OPENALEX_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, OPENALEX_DIR)
from utils import META_DIRE  # noqa: E402

meta_dire = META_DIRE
print("meta_dire", meta_dire)

# ---- excluded sources (venues) ----------------------------------------------------
# Source ids (sids) whose papers should be removed from the network, together with all
# of their citation edges. Ids are in the same stripped form stored in
# pid2primay_location (no 'https://openalex.org/' prefix), e.g. 'S4306402512'.
FILTER_SOURCE_IDS = {
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
}
print("number of excluded source ids:", len(FILTER_SOURCE_IDS))


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# ---- valid pid set = pids with a valid primary_location NOT in the excluded list ----
pid2primary_location = load_pkl(os.path.join(meta_dire, 'pid2primay_location.pkl'))
valid_pids = {
    pid
    for pid, loc in pid2primary_location.items()
    if loc is not None and loc not in FILTER_SOURCE_IDS
}
del pid2primary_location
print("number of valid pids (valid primary_location, not excluded):", len(valid_pids))


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
    'citing_pid2cited_pids_list_valid_primary_location.pkl',
)
filter_network(
    'cited_pid2citing_pids_list.pkl',
    'cited_pid2citing_pids_list_valid_primary_location.pkl',
)

print("finished")
