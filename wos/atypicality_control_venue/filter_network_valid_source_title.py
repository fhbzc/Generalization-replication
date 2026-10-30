"""
Web of Science (WoS) version of the OpenAlex
filter_network_valid_primary_location.py.

Drop every citation edge that involves a pid WITHOUT a valid venue (source_title), or
whose source_title is in the excluded-venue filter list.

WoS has no OpenAlex-style primary_location source id; the venue is the journal name
string stored in pid2source_title.pkl (from venue_validation/get_meta_data_get_venue.py,
run with use_with_review=False so the output is the plain pid2source_title.pkl). A pid
is considered VALID iff:
  - it is a key of pid2source_title.pkl with a non-None / non-empty source_title, AND
  - its source_title is NOT in FILTER_SOURCE_TITLES.
Pids whose source_title is missing/empty (or in the excluded list) are treated exactly
like pids with no valid venue: they (and every citation edge touching them) are removed.

NOTE on FILTER_SOURCE_TITLES: WoS get_meta_data.py already restricts the network to
journal articles (pubtype == 'Journal' AND doctypes == 'article'), so the preprint /
repository venues the OpenAlex pipeline excludes (arXiv, bioRxiv, SSRN, PubMed, ...) do
not appear here. The exclusion mechanism is kept (empty by default) only so a venue can
be dropped later without changing the code.

For each of the two adjacency dicts (pid -> list of pids), an edge (k, n) is kept
only when BOTH endpoints k and n are valid. Concretely: invalid keys are removed,
invalid neighbours are filtered out of the kept lists, and keys whose neighbour list
becomes empty are dropped (no edges left).

Input  : META_DIRE/pid2source_title.pkl
         META_DIRE/cited_pid2citing_pids_list.pkl
         META_DIRE/citing_pid2cited_pids_list.pkl
Output : META_DIRE/source_title2paper_count.csv   (per-venue paper counts; written first)
         META_DIRE/cited_pid2citing_pids_list_valid_source_title.pkl
         META_DIRE/citing_pid2cited_pids_list_valid_source_title.pkl
"""

import os
import sys
import csv
import pickle
from collections import Counter
from tqdm import tqdm

# utils.py lives in the parent wos/ directory
WOS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WOS_DIR)
from utils import META_DIRE  # noqa: E402

meta_dire = META_DIRE
print("meta_dire", meta_dire)

# match the upstream get_meta_data[_with_review].py / get_meta_data_get_venue.py runs:
# when True, read the '_with_review' inputs (the doctypes == 'review'-inclusive set) and
# tag every output with the same suffix so the two variants never collide.
use_with_review = True
data_suffix = '_with_review' if use_with_review else ''
print("use_with_review", use_with_review)
print("data_suffix", repr(data_suffix))

# ---- excluded venues (source_titles) ----------------------------------------------
# Journal source_title strings whose papers should be removed from the network,
# together with all of their citation edges. Empty by default (see module docstring:
# WoS is already journal-articles-only). Add exact source_title strings here to drop a
# venue, e.g. FILTER_SOURCE_TITLES = {"SOME JOURNAL NAME"}.
FILTER_SOURCE_TITLES = set()
print("number of excluded source titles:", len(FILTER_SOURCE_TITLES))


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


pid2source_title = load_pkl(os.path.join(meta_dire, f'pid2source_title{data_suffix}.pkl'))

# ---- per-venue paper counts: how many papers each source_title has -------------------
# Counted over every paper with a non-empty source_title (BEFORE the exclusion filter),
# so the table can be inspected to decide what to drop via FILTER_SOURCE_TITLES. Saved as
# a CSV sorted by descending paper count.
source_title2paper_count = Counter(
    src for src in pid2source_title.values() if src is not None and src != ''
)
count_csv_path = os.path.join(meta_dire, f'source_title2paper_count{data_suffix}.csv')
with open(count_csv_path, 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f)
    w.writerow(['source_title', 'paper_count'])
    for src, cnt in source_title2paper_count.most_common():
        w.writerow([src, cnt])
print("distinct source_titles:", len(source_title2paper_count))
print("wrote", count_csv_path)

# ---- valid pid set = pids with a valid source_title NOT in the excluded list --------
valid_pids = {
    pid
    for pid, src in pid2source_title.items()
    if src is not None and src != '' and src not in FILTER_SOURCE_TITLES
}
del pid2source_title
print("number of valid pids (valid source_title, not excluded):", len(valid_pids))


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
    f'citing_pid2cited_pids_list{data_suffix}.pkl',
    f'citing_pid2cited_pids_list_valid_source_title{data_suffix}.pkl',
)
filter_network(
    f'cited_pid2citing_pids_list{data_suffix}.pkl',
    f'cited_pid2citing_pids_list_valid_source_title{data_suffix}.pkl',
)

print("finished")
