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
from utils import CONCEPT_DIRE, META_DIRE


root_level = 0
print("root_level",root_level)
df_concept = pd.read_excel(CONCEPT_DIRE)
df_concept_level = df_concept[df_concept['level'] == root_level]

concept2name = dict(zip(df_concept['openalex_id'], df_concept['display_name']))
concept2name = {cid[21:].lower(): name for cid,name in concept2name.items()}

meta_dire = META_DIRE
print("meta_dire",meta_dire)
with open(os.path.join(meta_dire, 'pid2root_concepts_level_%s.pkl'%root_level), 'rb') as f:
    pid2root_concepts = pickle.load(f)

print("load pid2root_concepts finished")
top_concept2pid = defaultdict(list)
for pid, top_concepts_list in tqdm(pid2root_concepts.items()):
    if top_concepts_list is None:
        continue

    for top_concept in top_concepts_list:
        concept_name = concept2name[top_concept]
        top_concept2pid[concept_name].append(pid)



with open(os.path.join(meta_dire, 'top_concept2pid_list_level_%s.pkl'%root_level), 'wb') as f:
    pickle.dump(top_concept2pid, f, pickle.HIGHEST_PROTOCOL)

print("size of top_concept2pid", len(top_concept2pid))
print("finished")