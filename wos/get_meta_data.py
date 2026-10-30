"""
Build the Web of Science (WoS) paper-metadata and citation-network tables used by
the rest of the pipeline. Reads the WoS papers parquet shards (journal articles
only) and records each paper's title, publication year, and macro citation topic(s);
then reads the references parquet shards and builds the directed citation network,
keeping only edges between known papers and dropping temporally-inconsistent ones
(citing year >= cited year). Both stages run in parallel over the shards and are
skipped if their outputs already exist. WoS ids are stored with the 'WOS:' prefix
stripped; papers with no topic are bucketed under 'Others'.

Input  : WOS_DIRE/wos_papers.parquet/*.parquet, WOS_DIRE/wos_references.parquet/*.parquet
         (EXTERNAL WoS dataset; WOS_DIRE configured in utils.py)
Outputs (to META_DIRE/): pid2title.pkl, pid2pub_year.pkl, concept2pid_list.pkl,
         cited_pid2citing_pids_list.pkl, citing_pid2cited_pids_list.pkl
"""

import os
import sys
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

# utils.py is in this same directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from utils import META_DIRE, WOS_DIRE  # noqa: E402

max_workers = 6


dire_raw_meta = WOS_DIRE
print("dire_raw_meta",dire_raw_meta)

paper_dire_raw = os.path.join(dire_raw_meta, 'wos_papers.parquet') # we use a parquet internalled processed from the papers item in web of science
DUMMY_TOPIC = 'Others'

print("max_workers",max_workers)
output_dire = META_DIRE
print("output_dire",output_dire)


os.makedirs(output_dire, exist_ok = True)

def process_meta(file_name):
    path = os.path.join(paper_dire_raw, file_name)

    df_data = pd.read_parquet(path, columns = ['UID', 'item_title', 'pubyear', 'macro_citation_topic', 'pubtype', 'doctypes']) 
    df_data = df_data[(df_data['pubtype'] == 'Journal') & (df_data['doctypes'] == 'article')]

    id2topic = dict(zip(df_data['UID'], df_data['macro_citation_topic']))

    id2title = dict(zip(df_data['UID'], df_data['item_title']))

    id2year = dict(zip(df_data['UID'], df_data['pubyear']))

    pid2title = {}
    pid2pub_year = {}
    topics2pid = defaultdict(set)
    for pid in tqdm(id2title, desc = 'process %s'%file_name, position = 1, leave = False):


        if id2title[pid] is None or len(id2title[pid]) == 0:
            continue
        if np.isnan(id2year[pid]):
            continue


        assert pid[:4] == 'WOS:'
        short_pid = pid[4:]

        has_topic_flag = False
        if id2topic[pid] is not None:
            for topic in id2topic[pid].split(';'):
                topic = topic.strip()
                topics2pid[topic].add(short_pid)
                has_topic_flag = True

        if has_topic_flag == False:
            topics2pid[DUMMY_TOPIC].add(short_pid)


        pid2title[short_pid] = id2title[pid]
        pid2pub_year[short_pid] = int(id2year[pid])

    return pid2title, pid2pub_year, topics2pid





def process_reference(file_name):
    path = os.path.join(reference_dire_raw, file_name)

    df_data = pd.read_parquet(path, columns = ['UID_source', 'UID_target']) 
    
    cited_work2citing_pid_list = defaultdict(set)
    citing_work2cited_pid_list = defaultdict(set)
    error_link_count = 0
    correct_link_count = 0
    for index, row in df_data.iterrows():

        if row['UID_source'] is None or row['UID_target'] is None:
            continue

        if row['UID_source'][:4] != 'WOS:' or row['UID_target'][:4] != 'WOS:':
            continue

        source_uid = row['UID_source'][4:]
        target_uid = row['UID_target'][4:]

        
        if source_uid not in valid_pid_set or target_uid not in valid_pid_set:
            continue

        if pid2pub_year_M[source_uid] < pid2pub_year_M[target_uid]:
            error_link_count += 1
            continue
        else:
            correct_link_count += 1
        citing_work2cited_pid_list[source_uid].add(target_uid)
        cited_work2citing_pid_list[target_uid].add(source_uid)

        
    return cited_work2citing_pid_list, citing_work2cited_pid_list, error_link_count, correct_link_count



file_name_list = [file_name for file_name in os.listdir(paper_dire_raw) if file_name.endswith('.parquet')]


if os.path.exists(os.path.join(output_dire, 'pid2pub_year.pkl')) == False:

    pid2title_M = {}
    pid2pub_year_M = {}
    topics2pid_M = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(file_name_list), desc = 'process process_meta') as pbar:
            for pid2title, pid2pub_year, topics2pid in executor.map(process_meta, file_name_list):

                pid2title_M.update(pid2title)
                pid2pub_year_M.update(pid2pub_year)
                for topic in topics2pid:
                    topics2pid_M.setdefault(topic, set())
                    topics2pid_M[topic] |= topics2pid[topic]

                pbar.update(1)

    topics2pid_M = {topic: list(topics2pid_M[topic]) for topic in topics2pid_M}



    with open(os.path.join(output_dire, 'pid2title.pkl'), 'wb') as f:
        pickle.dump(pid2title_M, f, pickle.HIGHEST_PROTOCOL)


    with open(os.path.join(output_dire, 'concept2pid_list.pkl'), 'wb') as f:
        pickle.dump(topics2pid_M, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(output_dire, 'pid2pub_year.pkl'), 'wb') as f:
        pickle.dump(pid2pub_year_M, f, pickle.HIGHEST_PROTOCOL)

    print("size of pid2pub_year_M created", len(pid2pub_year_M))



if os.path.exists(os.path.join(output_dire, 'citing_pid2cited_pids_list.pkl')) == False:
    print("generate reference")

    with open(os.path.join(output_dire, 'pid2pub_year.pkl'), 'rb') as f:
        pid2pub_year_M = pickle.load(f)


    valid_pid_set = set(pid2pub_year_M)
    ### get reference

    reference_dire_raw = os.path.join(dire_raw_meta, 'wos_references.parquet')



    file_name_list = [file_name for file_name in os.listdir(reference_dire_raw) if file_name.endswith('.parquet')]

    cited_work2citing_pid_list_M = defaultdict(set)
    citing_work2cited_pid_list_M = defaultdict(set)
    error_link_count_M = 0
    correct_link_count_M = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(file_name_list), desc = 'process reference') as pbar:
            for cited_work2citing_pid_list, citing_work2cited_pid_list, error_link_count, correct_link_count in executor.map(process_reference, file_name_list):

                for pid in cited_work2citing_pid_list:
                    cited_work2citing_pid_list_M[pid] |= cited_work2citing_pid_list[pid]

                for pid in citing_work2cited_pid_list:
                    citing_work2cited_pid_list_M[pid] |= citing_work2cited_pid_list[pid]

                error_link_count_M += error_link_count
                correct_link_count_M += correct_link_count
                pbar.update(1)

    cited_work2citing_pid_list_M = {pid: list(citing_list) for pid, citing_list in cited_work2citing_pid_list_M.items()}
    citing_work2cited_pid_list_M = {pid: list(cited_list) for pid, cited_list in citing_work2cited_pid_list_M.items()}

    print("error_link_count_M",error_link_count_M)
    print("correct_link_count_M",correct_link_count_M)

    with open(os.path.join(output_dire, 'cited_pid2citing_pids_list.pkl'), 'wb') as f:
        pickle.dump(cited_work2citing_pid_list_M, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(output_dire, 'citing_pid2cited_pids_list.pkl'), 'wb') as f:
        pickle.dump(citing_work2cited_pid_list_M, f, pickle.HIGHEST_PROTOCOL)


    print("size of cited_work2citing_pid_list_M", len(cited_work2citing_pid_list_M))
    print("size of citing_work2cited_pid_list_M", len(citing_work2cited_pid_list_M))
print("finished")

