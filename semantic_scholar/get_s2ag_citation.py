"""
Build the S2AG citation network (corpusid -> [corpusid]) from the Semantic Scholar
Academic Graph (S2AG) "citations" dump (gzipped JSONL shards, one citation edge
per line), restricted to "valid" papers: corpusids that have BOTH a publication
year and a title (intersection of cid2pub_year and cid2title from get_s2ag_meta.py).
The valid-cid set is cached to valid_cid_list.pkl and reused on later runs.

Input  : S2AG_DIRE/citations/*.gz          (S2AG citations dump; S2AG_DIRE set in utils.py)
         META_DIRE/{cid2pub_year, cid2title}.pkl   (from get_s2ag_meta.py)
Outputs: META_DIRE/valid_cid_list.pkl       (cached valid-cid set)
         META_DIRE/cited2citing_cid_list.pkl   cited corpusid -> [citing corpusids]
         META_DIRE/citing2cited_cid_list.pkl   citing corpusid -> [cited corpusids]
"""

import json
import os
import sys
import gzip
import pickle
from collections import defaultdict

from tqdm import tqdm

# utils.py is in this same directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from utils import S2AG_DIRE, META_DIRE  # noqa: E402

file_dire = os.path.join(S2AG_DIRE, 'citations')
meta_dire = META_DIRE

valid_cid_list_dire = os.path.join(meta_dire, 'valid_cid_list.pkl')

if os.path.exists(valid_cid_list_dire):
    print("load existing valid_cid_set")
    with open(valid_cid_list_dire,'rb') as f:
        valid_cid_set = set(pickle.load(f))

else:
    print("create new valid_cid_set")

    with open(os.path.join(meta_dire,'cid2pub_year.pkl'),'rb') as f:
        cid2pub_year = pickle.load(f)
    print("len cid2pub_year", len(cid2pub_year))
    with open(os.path.join(meta_dire,'cid2title.pkl'),'rb') as f:
        cid2title = pickle.load(f)
    print("len cid2title", len(cid2title))

    valid_cid_set = set(cid2pub_year.keys()) & set(cid2title.keys())

    with open(valid_cid_list_dire,'wb') as f:
        pickle.dump(list(valid_cid_set), f, pickle.HIGHEST_PROTOCOL)

print("size of valid_cid_set", len(valid_cid_set))

file_list = sorted(os.listdir(file_dire))

cid2cited_cid_list = defaultdict(list)
cid2citing_cid_list = defaultdict(list)

valid_citation_count = 0

for file in tqdm(file_list):
    file_path = os.path.join(file_dire, file)

    with gzip.open(file_path, "rt", encoding="utf-8") as f:

        for line in tqdm(f, unit="lines", position=2, leave=False):
            obj = json.loads(line)

            cited_id = obj.get('citedcorpusid', None)
            citing_id = obj.get('citingcorpusid', None)

            if cited_id is not None and citing_id is not None and \
                cited_id in valid_cid_set and \
                citing_id in valid_cid_set:

                valid_citation_count += 1
                cid2cited_cid_list[citing_id].append(cited_id)
                cid2citing_cid_list[cited_id].append(citing_id)


cid2cited_cid_list = {cid: list(set(cid_list)) for cid, cid_list in tqdm(cid2cited_cid_list.items())}
cid2citing_cid_list = {cid: list(set(cid_list)) for cid, cid_list in tqdm(cid2citing_cid_list.items())}


with open(os.path.join(meta_dire,'cited2citing_cid_list.pkl'),'wb') as f:
    pickle.dump(cid2citing_cid_list, f, pickle.HIGHEST_PROTOCOL)

with open(os.path.join(meta_dire,'citing2cited_cid_list.pkl'),'wb') as f:
    pickle.dump(cid2cited_cid_list, f, pickle.HIGHEST_PROTOCOL)

print("valid_citation_count",valid_citation_count)
print("finished")