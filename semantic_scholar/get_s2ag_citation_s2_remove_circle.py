"""
Remove temporally-inconsistent ("circular") edges from the S2AG citation network
produced by get_s2ag_citation.py. Using each paper's publication date (preferred)
or year, keep an edge only if its direction agrees with chronology:
  * cited2citing: a citing paper must be published on/after the cited paper.
  * citing2cited: a cited paper must be published on/before the citing paper.
Self-loops (a corpusid citing itself) are also dropped. When both endpoints have a
publication date their dates are compared; otherwise the comparison falls back to
publication year.

Input  : META_DIRE/{cid2pub_year, cid2pub_date}.pkl                 (from get_s2ag_meta.py)
         META_DIRE/{cited2citing_cid_list, citing2cited_cid_list}.pkl   (from get_s2ag_citation.py)
Outputs: META_DIRE/cited2citing_cid_list_remove_circle.pkl
         META_DIRE/citing2cited_cid_list_remove_circle.pkl
"""

import os
import sys
import pickle

from tqdm import tqdm

# utils.py is in this same directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from utils import META_DIRE  # noqa: E402

meta_dire = META_DIRE

with open(os.path.join(meta_dire,'cid2pub_year.pkl'),'rb') as f:
    cid2pub_year = pickle.load(f)
print("load cid2pub_year finished")
with open(os.path.join(meta_dire,'cid2pub_date.pkl'),'rb') as f:
    cid2pub_date = pickle.load(f)
print("load cid2pub_date finished")



with open(os.path.join(meta_dire,'cited2citing_cid_list.pkl'),'rb') as f:
    cid2citing_cid_list = pickle.load(f)
print("load cid2citing_cid_list finished")




for cid, cid_list in tqdm(cid2citing_cid_list.items(), desc='process cid2citing_cid_list'):
    cid_pub_date = cid2pub_date.get(cid)
    cid_pub_year = cid2pub_year[cid]

    if cid_pub_date is not None:
        cid2citing_cid_list[cid] = [
            citing_cid
            for citing_cid in cid_list
            if (
                (citing_cid in cid2pub_date and cid2pub_date[citing_cid] > cid_pub_date)
                or
                (citing_cid not in cid2pub_date and cid2pub_year[citing_cid] >= cid_pub_year)
            ) and (citing_cid != cid)
        ]
    else:
        cid2citing_cid_list[cid] = [
            citing_cid
            for citing_cid in cid_list
            if (cid2pub_year[citing_cid] >= cid_pub_year) and (citing_cid != cid)
        ]

with open(os.path.join(meta_dire,'cited2citing_cid_list_remove_circle.pkl'),'wb') as f:
    pickle.dump(cid2citing_cid_list, f, pickle.HIGHEST_PROTOCOL)

print("size of cid2citing_cid_list", len(cid2citing_cid_list))
del cid2citing_cid_list




with open(os.path.join(meta_dire,'citing2cited_cid_list.pkl'),'rb') as f:
    citing2cited_cid_list = pickle.load(f)
print("load citing2cited_cid_list finished")




for cid, cid_list in tqdm(citing2cited_cid_list.items(), desc='process citing2cited_cid_list'):
    cid_pub_date = cid2pub_date.get(cid)
    cid_pub_year = cid2pub_year[cid]

    if cid_pub_date is not None:
        citing2cited_cid_list[cid] = [
            citing_cid
            for citing_cid in cid_list
            if (
                (citing_cid in cid2pub_date and cid2pub_date[citing_cid] < cid_pub_date)
                or
                (citing_cid not in cid2pub_date and cid2pub_year[citing_cid] <= cid_pub_year)
            ) and (citing_cid != cid)
        ]
    else:
        citing2cited_cid_list[cid] = [
            citing_cid
            for citing_cid in cid_list
            if (cid2pub_year[citing_cid] <= cid_pub_year) and (citing_cid != cid)
        ]

with open(os.path.join(meta_dire,'citing2cited_cid_list_remove_circle.pkl'),'wb') as f:
    pickle.dump(citing2cited_cid_list, f, pickle.HIGHEST_PROTOCOL)

print("size of citing2cited_cid_list", len(citing2cited_cid_list))



print("finished")