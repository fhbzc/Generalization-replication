"""
Semantic Scholar (S2AG) version of the OpenAlex
filter_network_valid_primary_location.py.

Drop every citation edge that involves a cid WITHOUT a valid venue, or whose venue is in
the excluded-venue filter list.

The node id is the S2AG corpusid (cid). The venue is the string stored in cid2venue.pkl
(from get_s2ag_meta_venue.py). A cid is considered VALID iff:
  - it is a key of cid2venue.pkl with a non-None / non-empty venue, AND
  - its venue is NOT in FILTER_VENUES.
Cids whose venue is missing/empty (or in the excluded list) are treated exactly like cids
with no valid venue: they (and every citation edge touching them) are removed.

The network filtered here is the temporally-consistent '_remove_circle' network produced
by get_s2ag_citation_s2_remove_circle.py -- the same network get_efg.py consumes and the
OpenAlex analog of get_meta_data_preprint.py's (already temporally filtered) base network.

NOTE on FILTER_VENUES: empty by default. S2AG venue strings are free text; add exact
venue strings here to drop a venue (e.g. preprint servers) from the network.

For each of the two adjacency dicts (cid -> list of cids), an edge (k, n) is kept only
when BOTH endpoints k and n are valid. Concretely: invalid keys are removed, invalid
neighbours are filtered out of the kept lists, and keys whose neighbour list becomes empty
are dropped (no edges left).

Input  : META_DIRE/cid2venue.pkl                              (get_s2ag_meta_venue.py)
         META_DIRE/citing2cited_cid_list_remove_circle.pkl    (get_s2ag_citation_s2_remove_circle.py)
         META_DIRE/cited2citing_cid_list_remove_circle.pkl    (get_s2ag_citation_s2_remove_circle.py)
Output : ./venue2paper_count.csv   (per-venue paper counts; written first, to the CWD)
         META_DIRE/citing2cited_cid_list_remove_circle_valid_venue.pkl
         META_DIRE/cited2citing_cid_list_remove_circle_valid_venue.pkl
"""

import os
import sys
import csv
import pickle
from collections import Counter
from tqdm import tqdm

# utils.py lives in the parent semantic_scholar/ directory
S2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, S2_DIR)
from utils import META_DIRE  # noqa: E402

meta_dire = META_DIRE
print("meta_dire", meta_dire)

# ---- excluded venues ----------------------------------------------------------------
# Venue strings whose papers should be removed from the network, together with all of
# their citation edges. Empty by default. Add exact venue strings here to drop a venue,
# e.g. FILTER_VENUES = {"arXiv", "bioRxiv"}.
FILTER_VENUES = set(['arXiv.org', 'Social Science Research Network', 'bioRxiv',
        'medRxiv', 'Radiopaedia.org', 'Other Conferences'])
print("number of excluded venues:", len(FILTER_VENUES))


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


cid2venue = load_pkl(os.path.join(meta_dire, 'cid2venue.pkl'))

# ---- per-venue paper counts: how many papers each venue has --------------------------
# Counted over every paper with a non-empty venue (BEFORE the exclusion filter), so the
# table can be inspected to decide what to drop via FILTER_VENUES. Saved to the current
# directory (./) as a CSV sorted by descending paper count.
venue2paper_count = Counter(
    venue for venue in cid2venue.values() if venue is not None and venue != ''
)
count_csv_path = os.path.join('.', 'venue2paper_count.csv')
with open(count_csv_path, 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f)
    w.writerow(['venue', 'paper_count'])
    for venue, cnt in venue2paper_count.most_common():
        w.writerow([venue, cnt])
print("distinct venues:", len(venue2paper_count))
print("wrote", os.path.abspath(count_csv_path))

# ---- valid cid set = cids with a valid venue NOT in the excluded list ----------------
valid_cids = {
    cid
    for cid, venue in cid2venue.items()
    if venue is not None and venue != '' and venue not in FILTER_VENUES
}
del cid2venue
print("number of valid cids (valid venue, not excluded):", len(valid_cids))


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
        if k not in valid_cids:
            continue
        kept = [n for n in neighbours if n in valid_cids]
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
    'citing2cited_cid_list_remove_circle.pkl',
    'citing2cited_cid_list_remove_circle_valid_venue.pkl',
)
filter_network(
    'cited2citing_cid_list_remove_circle.pkl',
    'cited2citing_cid_list_remove_circle_valid_venue.pkl',
)

print("finished")
