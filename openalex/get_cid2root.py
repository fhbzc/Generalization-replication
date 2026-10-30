import os
import pandas as pd
import json
from tqdm import tqdm
from collections import defaultdict
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)               # utils.py is in this same directory
from utils import CONCEPT_DIRE, META_DIRE
def get_downstreams(df_concepts_list, root_cid):
    upper_level_concepts = set([root_cid])
    for row in tqdm(df_concepts_list, desc='process concepts', position = 1, leave = False):

        upper_concepts = row['parent_ids']
        if isinstance(upper_concepts, str):
            for up_concept in upper_concepts.split(','):
                up_concept = up_concept.strip().lower()
                if up_concept in upper_level_concepts:
                    current_concept = row['openalex_id'].strip().lower()
                    upper_level_concepts.add(current_concept)
    return upper_level_concepts

root_level = 0
print("root_level",root_level)

concept_directory = CONCEPT_DIRE
df_concepts = pd.read_excel(concept_directory)
df_concepts_list = df_concepts.to_dict('records')


concept2name = dict(zip(df_concepts['openalex_id'], df_concepts['display_name']))
concept2level = dict(zip(df_concepts['openalex_id'], df_concepts['level']))

root_cids = [open_id.strip().lower() for open_id in set(df_concepts[df_concepts['level'] == root_level]['openalex_id'])]




cid2root_cid_list = defaultdict(set)

for root_cid in tqdm(root_cids, desc = 'process root_cid', position = 0):
    upper_level_concepts = get_downstreams(df_concepts_list, root_cid)

    for lower_concept in upper_level_concepts:
        
        cid2root_cid_list[lower_concept].add(root_cid)

for cid in cid2root_cid_list:
    cid2root_cid_list[cid] = list(cid2root_cid_list[cid])
with open(os.path.join(META_DIRE, 'cid2root_cid_list_l_%s.json'%root_level), 'w') as f:
    json.dump(cid2root_cid_list, f)

print("size of cid2root_cid_list", len(cid2root_cid_list))
print("finished")