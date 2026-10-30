import json
import os
import gzip
from tqdm import tqdm
import pickle
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import pandas as pd
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)               # utils.py is in this same directory
from utils import OPENALEX_DIRE, META_DIRE, CONCEPT_DIRE



path_openalex = OPENALEX_DIRE
save_dire = META_DIRE
os.makedirs(save_dire, exist_ok = True)
process_count = 1

print("save_dire",save_dire)
print("process_count",process_count)

df_concept = pd.read_excel(CONCEPT_DIRE)

df_concept = df_concept[df_concept['level'] == 0]
concept2name = dict(zip(df_concept['openalex_id'], df_concept['display_name']))



def get_valid_pid(file_path):
    pid2year_single = {}
    pid2pub_date_single = {}
    pid2title_single = {}
    pid2concept_list = {}
    concept2pid_list = defaultdict(list)
    # file-level worker: process a single .gz file (parallelism is over files)
    with gzip.open(file_path, 'rb') as f:
        for line in f:
            line = json.loads(line)

            id_ = line['id'][21:]


            if line and line['publication_year'] and line['authorships'] and line['type'] and (line['type']=='article' or line['type'] == 'preprint'):
                # try:
                pid2year_single[id_] = line['publication_year']
                title = line['title']
                pid2pub_date_single[id_] = line['publication_date']
                pid2title_single[id_] = title
                has_concept = False
                concept_list = []
                for concept in line['concepts']:
                    if concept['level'] == 0:
                        concept_name = concept2name[concept['id'].lower()]
                        has_concept = True
                        concept2pid_list[concept_name].append(id_)

                    concept_item = {"id":concept['id'][21:], 'score': concept['score']}
                    concept_list.append(concept_item)

                pid2concept_list[id_] = concept_list
                if has_concept == False:
                    concept2pid_list['Others'].append(id_)

    return pid2year_single, pid2pub_date_single, pid2title_single, concept2pid_list, pid2concept_list



def get_reference(file_path):
    pid2up_refer_single = {}
    pid2down_refer_single = defaultdict(list)
    # file-level worker: process a single .gz file (parallelism is over files)
    with gzip.open(file_path, 'rb') as f:
        for line in f:
            line = json.loads(line)

            id_ = line['id'][21:]

            if id_ in valid_pid_set:
                source_work_date = pid2pub_date[id_]
                if line['referenced_works']:
                    refs = set([r[21:] for r in line['referenced_works'] if r[21:] in valid_pid_set and pid2pub_date[r[21:]] < source_work_date])
                    # this should remove items where the reference has a size of 0
                    refs = list(refs)
                    pid2up_refer_single[id_] = refs
                    for ref in refs:
                        pid2down_refer_single[ref].append(id_)
    return pid2up_refer_single, pid2down_refer_single


path_work = os.path.join(path_openalex, 'openalex-snapshot/data/works/')
dirs = sorted(os.listdir(path_work))

# ===============================
# Collect all valid gz files (file-level parallelism, like get_meta_data_preprint_arxiv.py)
# ===============================
gz_files = []
for d in dirs:
    if '_updated_date' not in d and 'updated_date' in d:
        dpath = os.path.join(path_work, d)
        for fname in sorted(os.listdir(dpath)):
            if fname.endswith('.gz') and '_part' not in fname:
                gz_files.append(os.path.join(dpath, fname))
print("Total gz files:", len(gz_files))



if os.path.exists(os.path.join(save_dire, 'concept2pid_list.pkl')) ==  False:
    JournalArticle_Year = {}
    pid2title = {}
    concept2pid_list = {}
    pid2pub_date = {}
    pid2concept_list = {}
    with ProcessPoolExecutor(max_workers=process_count) as executor:
        with tqdm(total=len(gz_files)) as pbar:
            for pid2year_single, pid2pub_date_single, pid2title_single, concept2pid_list_single, pid2concept_list_single in executor.map(get_valid_pid, gz_files):
                pbar.update(1)
                JournalArticle_Year.update(pid2year_single)
                pid2pub_date.update(pid2pub_date_single)
                pid2title.update(pid2title_single)
                pid2concept_list.update(pid2concept_list_single)
                for concept in concept2pid_list_single:
                    if concept not in concept2pid_list:
                        concept2pid_list[concept] = []

                    concept2pid_list[concept].extend(concept2pid_list_single[concept])
    with open(os.path.join(save_dire, 'pid2pub_year.pkl'), 'wb') as f:
        pickle.dump(JournalArticle_Year, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_dire, 'pid2pub_date.pkl'), 'wb') as f:
        pickle.dump(pid2pub_date, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_dire, 'pid2title.pkl'), 'wb') as f:
        pickle.dump(pid2title, f, pickle.HIGHEST_PROTOCOL)


    with open(os.path.join(save_dire, 'pid2concept_list.pkl'), 'wb') as f:
        pickle.dump(pid2concept_list, f, pickle.HIGHEST_PROTOCOL)

    for concept in concept2pid_list:
        concept2pid_list[concept] = list(set(concept2pid_list[concept]))

    with open(os.path.join(save_dire, 'concept2pid_list.pkl'), 'wb') as f:
        pickle.dump(concept2pid_list, f, pickle.HIGHEST_PROTOCOL)


        
else:

    with open(os.path.join(save_dire, 'pid2pub_date.pkl'), 'rb') as f:
        pid2pub_date = pickle.load(f)

valid_pid_set = set(pid2pub_date)
if os.path.exists(os.path.join(save_dire, 'cited_pid2citing_pids_list.pkl')) ==  False:
    pid2up_refer = {}
    pid2down_citation = {}

    if process_count > 1:
        with ProcessPoolExecutor(max_workers=process_count) as executor:
            with tqdm(total=len(gz_files), desc = 'process reference') as pbar:
                for pid2up_refer_single, pid2down_refer_single in executor.map(get_reference, gz_files):
                    pbar.update(1)
                    pid2up_refer.update(pid2up_refer_single)
                    for pid in pid2down_refer_single:
                        if pid not in pid2down_citation:
                            pid2down_citation[pid] = []

                        pid2down_citation[pid].extend(pid2down_refer_single[pid])
    else:
        for file_path in tqdm(gz_files, desc = 'process reference'):
            pid2up_refer_single, pid2down_refer_single = get_reference(file_path)
            pid2up_refer.update(pid2up_refer_single)
            for pid in pid2down_refer_single:
                if pid not in pid2down_citation:
                    pid2down_citation[pid] = []

                pid2down_citation[pid].extend(pid2down_refer_single[pid])
    with open(os.path.join(save_dire, 'citing_pid2cited_pids_list.pkl'), 'wb') as f:
        pickle.dump(pid2up_refer, f, pickle.HIGHEST_PROTOCOL)

    pid2down_citation = {pid: list(set(pid2down_citation[pid])) for pid in tqdm(pid2down_citation, desc = 'process pid2down_citation')}
    with open(os.path.join(save_dire, 'cited_pid2citing_pids_list.pkl'), 'wb') as f:
        pickle.dump(pid2down_citation, f, pickle.HIGHEST_PROTOCOL)

    print("size of pid2up_refer", len(pid2up_refer))
    print("size of pid2down_citation", len(pid2down_citation))


print("finished")