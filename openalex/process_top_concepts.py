import pandas as pd
import json
from tqdm import tqdm
from collections import defaultdict
import pickle
import os
import time
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)               # utils.py is in this same directory
from utils import META_DIRE, CONCEPT_DIRE


root_level = 0
print("root_level",root_level)
df_concept = pd.read_excel(CONCEPT_DIRE)
df_concept_level = df_concept[df_concept['level'] == root_level]
valid_root_level_set = set([cid[21:].lower() for cid in set(df_concept_level['openalex_id'])])
print("size of valid_root_level_set", len(valid_root_level_set))

concept2level = dict(zip(df_concept['openalex_id'], df_concept['level']))

concept2level_P = {cid[21:].lower(): level for cid,level in concept2level.items()}
meta_dire = META_DIRE
print("meta_dire",meta_dire)
with open(os.path.join(META_DIRE, 'cid2root_cid_list_l_%s.json'%root_level) , 'r') as f:
    cid2root_cid_list = json.load(f)

cid2root_cid_list_P = {}
for cid, root_cid_list in cid2root_cid_list.items():
    cid_P = cid[21:]
    root_cid_list_P = [root_cid[21:] for root_cid in root_cid_list]
    cid2root_cid_list_P[cid_P] = root_cid_list_P
print("size of cid2root_cid_list_P", len(cid2root_cid_list_P))

with open(os.path.join(meta_dire, 'pid2concept_list.pkl'), 'rb') as f:
    pid2concept_list = pickle.load(f)

pid2root_concepts = {}

identified_paper_count = 0
unidentified_paper_count = 0
check_count = 0

for pid, concept_list in tqdm(pid2concept_list.items()):

    level2root_cid2score_sum = defaultdict(lambda: defaultdict(int))
    # start_time = time.time()
    cid_level_set = set()
    for concept_item in concept_list:

            
        cid = concept_item['id'].lower()
        if cid not in concept2level_P:
            continue
        cid_level = concept2level_P[cid]
        if concept_item['score'] == 0:
            continue
        if cid_level >= root_level:

            if cid in cid2root_cid_list_P:
                for root_cid in cid2root_cid_list_P[cid]:
                    level2root_cid2score_sum[cid_level][root_cid] += concept_item['score']
                    cid_level_set.add(cid_level)

    sorted_cid_level = sorted(list(cid_level_set))
    candidate_list = []
    top_candidate_set = set(valid_root_level_set)
    identified_flag = False

    for cid_level in sorted_cid_level:
        max_score = -1

        for root_cid, score in level2root_cid2score_sum[cid_level].items():
            if root_cid not in top_candidate_set:
                continue
            if score > max_score:
                candidate_list = [root_cid]
                max_score = score
            elif score == max_score:
                candidate_list.append(root_cid)

        N = len(candidate_list)
        if N >= 1:
            assert max_score >= 0
            identified_flag = True


            if N == 1:
                top_candidate_set = candidate_list
                break
            top_candidate_set &= set(candidate_list)

        candidate_list = []

    if identified_flag and len(top_candidate_set) > 0:
        pid2root_concepts[pid] = list(top_candidate_set)
        identified_paper_count += 1
    else:

        pid2root_concepts[pid] = None
        unidentified_paper_count += 1
with open(os.path.join(meta_dire, 'pid2root_concepts_level_%s.pkl'%root_level), 'wb') as f:
    pickle.dump(pid2root_concepts, f, pickle.HIGHEST_PROTOCOL)


print("identified_paper_count",identified_paper_count)
print("unidentified_paper_count",unidentified_paper_count)
print("size of pid2root_concepts", len(pid2root_concepts))
print("finished")