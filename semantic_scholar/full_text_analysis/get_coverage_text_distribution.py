"""
Compute per-domain in-text citation coverage: for each cited-paper domain, how many
of its incoming citations are actually observed as in-text reference mentions in the
S2ORC full text, vs. all citations recorded in the de-circularized S2AG network.

For each cited paper (cid) with domain label(s) (cid2category_list, normalised
'And' -> 'and'):
  total_citation_count   += number of papers citing it (S2AG cited2citing network)
  covered_citation_count += number of distinct (citing document, cited paper)
                            mentions actually found in S2ORC full text (via DOI->cid)
Each domain's totals are summed across its papers.

Input  : META_DIRE/{cid2category_list, cid2DOI, cited2citing_cid_list_remove_circle}.pkl
         S2ORC_META_DIRE/DOI2refer_instance_list_cid_parsed_V2.pkl  (from identify_citation_property.py)
Output : S2ORC_META_DIRE/text_coverage_V2/coverage.csv
           category, total_citation_count, covered_citation_count
"""

import os
import sys
import pickle
from collections import defaultdict, OrderedDict

import pandas as pd
from tqdm import tqdm

# utils.py is in the parent directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
from utils import S2ORC_META_DIRE, META_DIRE  # noqa: E402

s2orc_meta_dire = S2ORC_META_DIRE

save_dire = os.path.join(s2orc_meta_dire, 'text_coverage_V2')
os.makedirs(save_dire, exist_ok = True)
with open(os.path.join(META_DIRE,'cid2category_list.pkl'),'rb') as f:
    cid2category_list = pickle.load(f)

for cid, category_list in tqdm(cid2category_list.items()):

    cid2category_list[cid] = set([category.replace('And', 'and') for category in category_list])

with open(os.path.join(META_DIRE,'cid2DOI.pkl'),'rb') as f:
    cid2DOI = pickle.load(f)

doi2pid_list = defaultdict(list)

for cid, doi in tqdm(cid2DOI.items(), desc="Building doi2pid_list"):

    doi2pid_list[doi].append(cid)

doi2pid = {doi: pid_list[0] for doi, pid_list in tqdm(doi2pid_list.items()) if len(pid_list) == 1}

with open(os.path.join(s2orc_meta_dire, 'DOI2refer_instance_list_cid_parsed_V2.pkl'), 'rb') as f:
    DOI2refer_instances = pickle.load(f)

category2citation_count = defaultdict(int)
category2citation_count_covered = defaultdict(int)
with open(os.path.join(META_DIRE, 'cited2citing_cid_list_remove_circle.pkl'), 'rb') as f: 
    # ordered version for optimization
    pid2down_list = pickle.load(f)
print("pid2down_list load finished")

for cid,down_list in tqdm(pid2down_list.items()):
    l = len(down_list)
    for category in cid2category_list.get(cid, []):
        category2citation_count[category] += l

for DOI, refer_instance_list in tqdm(DOI2refer_instances.items()):
    cited_cid_set = []
    for refer_instance in refer_instance_list:
        doi_referred = refer_instance['doi']
        if doi_referred is not None and doi_referred in doi2pid:

            cited_cid_set.append(doi2pid[doi_referred])

    cited_cid_set = set(cited_cid_set)
    for cid in cited_cid_set:
        for category in cid2category_list.get(cid, []):
            category2citation_count_covered[category] += 1

df_res = []
for category, total_citation_count in tqdm(category2citation_count.items()):
    od = OrderedDict()
    od['category'] = category
    od['total_citation_count'] = total_citation_count
    od['covered_citation_count'] = category2citation_count_covered[category]
    df_res.append(od)

df_res = pd.DataFrame(df_res)
df_res.to_csv(os.path.join(save_dire, 'coverage.csv'), index = False)
print("size of df_res", len(df_res))
print("finished")