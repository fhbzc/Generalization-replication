"""
Rebuild the citation network with all self-citations removed.

Starting from the real citation network produced by get_meta_data_preprint.py
(citing pid -> list of cited pids, and its inversion cited pid -> citing pids),
this script drops every citing->cited edge that is a SELF-CITATION, i.e. the
citing paper and the cited paper share at least one author.

Authors come from pid2author_list.pkl (produced by
get_meta_data_preprint_author_country_process.py), which maps each pid to the
list of its OpenAlex author ids. The rule for one citing->cited edge:

    - if BOTH papers have a non-empty identified author list AND those two
      author sets intersect  -> self-citation -> drop the edge
    - otherwise (no overlap, OR either paper has no identified author list)
      -> keep the edge

So a paper with no identified authors is never treated as a self-citation; its
edges are left untouched (we simply cannot decide, so we keep them).

The filtered citing->cited mapping is then inverted to rebuild the cited->citing
mapping, so both directions stay perfectly consistent (same convention as
get_meta_data_preprint.py: cited->citing values are de-duplicated).

Input  : META_DIRE/citing_pid2cited_pids_list.pkl   (from get_meta_data_preprint.py)
         META_DIRE/cited_pid2citing_pids_list.pkl    (from get_meta_data_preprint.py; only its keys are reused implicitly via inversion)
         META_DIRE/pid2author_list.pkl               (from get_meta_data_preprint_author_country_process.py)
Output : META_DIRE/citing_pid2cited_pids_list_no_self_citation.pkl
         META_DIRE/cited_pid2citing_pids_list_no_self_citation.pkl
"""

import os
import pickle
import sys

from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)               # utils.py is in this same directory
from utils import META_DIRE

meta_dire = META_DIRE
print("meta_dire", meta_dire)

# ==================================================
# 1. load the real citation network and the author lists
# ==================================================
citing_pid2cited_pids_path = os.path.join(meta_dire, 'citing_pid2cited_pids_list.pkl')
author_list_path = os.path.join(meta_dire, 'pid2author_list.pkl')

assert os.path.exists(citing_pid2cited_pids_path), citing_pid2cited_pids_path
assert os.path.exists(author_list_path), author_list_path

with open(citing_pid2cited_pids_path, 'rb') as f:
    citing_pid2cited_works = pickle.load(f)
print("citing_pid2cited_works load finished | size:", len(citing_pid2cited_works))

with open(author_list_path, 'rb') as f:
    pid2author_list = pickle.load(f)
print("pid2author_list load finished | size:", len(pid2author_list))

# pre-convert each paper's author list to a set once (a paper with no entry, or an
# empty list, has no identified authors -> we can never call its edges self-cites)
pid2author_set = {}
for pid, author_list in tqdm(pid2author_list.items(), desc='build author sets'):
    if author_list:
        pid2author_set[pid] = set(author_list)
print("papers with >=1 identified author:", len(pid2author_set))

# ==================================================
# 2. filter citing->cited edges, dropping self-citations
# ==================================================
citing_pid2cited_works_clean = {}

n_edges_total = 0
n_edges_kept = 0
n_edges_self = 0           # dropped: citing & cited share an author
n_edges_undecidable = 0    # kept: one/both sides have no identified author list

for citing_pid, cited_list in tqdm(citing_pid2cited_works.items(), desc='remove self-citations'):
    citing_authors = pid2author_set.get(citing_pid)
    kept = []
    for cited_pid in cited_list:
        n_edges_total += 1
        cited_authors = pid2author_set.get(cited_pid)
        # only a decidable, overlapping pair counts as a self-citation
        if citing_authors is not None and cited_authors is not None:
            if citing_authors & cited_authors:
                n_edges_self += 1
                continue   # drop this self-citation edge
        else:
            n_edges_undecidable += 1
        kept.append(cited_pid)
        n_edges_kept += 1
    # keep the key even if it now has an empty reference list, to mirror the
    # original mapping's key set as closely as possible
    citing_pid2cited_works_clean[citing_pid] = kept

print("edges total:", n_edges_total)
print("edges kept:", n_edges_kept)
print("edges dropped as self-citation:", n_edges_self)
print("edges kept because author list missing on >=1 side:", n_edges_undecidable)

# ==================================================
# 3. invert the filtered citing->cited mapping to rebuild cited->citing
#    (de-duplicate the citing side, same as get_meta_data_preprint.py)
# ==================================================
cited_pid2citing_works_clean = {}
for citing_pid, cited_list in tqdm(citing_pid2cited_works_clean.items(), desc='invert to cited->citing'):
    for cited_pid in cited_list:
        if cited_pid not in cited_pid2citing_works_clean:
            cited_pid2citing_works_clean[cited_pid] = []
        cited_pid2citing_works_clean[cited_pid].append(citing_pid)

cited_pid2citing_works_clean = {
    pid: list(set(citing_list))
    for pid, citing_list in tqdm(cited_pid2citing_works_clean.items(), desc='dedup cited->citing')
}

# ==================================================
# 4. save the self-citation-removed network
# ==================================================
citing_out_path = os.path.join(meta_dire, 'citing_pid2cited_pids_list_no_self_citation.pkl')
cited_out_path = os.path.join(meta_dire, 'cited_pid2citing_pids_list_no_self_citation.pkl')

with open(citing_out_path, 'wb') as f:
    pickle.dump(citing_pid2cited_works_clean, f, pickle.HIGHEST_PROTOCOL)
print("saved:", citing_out_path, "| size:", len(citing_pid2cited_works_clean))

with open(cited_out_path, 'wb') as f:
    pickle.dump(cited_pid2citing_works_clean, f, pickle.HIGHEST_PROTOCOL)
print("saved:", cited_out_path, "| size:", len(cited_pid2citing_works_clean))

print("finished")
