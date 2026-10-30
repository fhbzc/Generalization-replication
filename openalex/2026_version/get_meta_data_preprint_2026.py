import json
import os
import gzip
import pickle
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import pandas as pd
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)             # utils.py is in the parent directory
from utils import OPENALEX_DIRE_2026, META_DIRE_2026


print("OPENALEX_DIRE_2026", OPENALEX_DIRE_2026)
print("META_DIRE_2026", META_DIRE_2026)

path_openalex = OPENALEX_DIRE_2026
save_dire = META_DIRE_2026   # 2026 metadata kept separate from the main META_DIRE
os.makedirs(save_dire, exist_ok=True)

process_count = 16
print("save_dire", save_dire)
print("process_count", process_count)

# =========================
# Concept（当前没用到，但保留）
# =========================

# =========================
# 收集所有 file-level task
# =========================
path_work = os.path.join(path_openalex, 'data/works/')
dirs = sorted(os.listdir(path_work))

all_files = []
for dir_ in tqdm(dirs):
    if '_updated_date' not in dir_ and 'updated_date' in dir_:
        full_dir = os.path.join(path_work, dir_)
        for file in os.listdir(full_dir):
            if file.endswith('.gz') and '_part' not in file:
                all_files.append(os.path.join(full_dir, file))

print("Total gz files:", len(all_files))

# =========================
# worker: metadata
# =========================
def get_valid_pid_from_file(file_path):
    pid2year = {}
    pid2pub_date = {}
    pid2title = {}

    with gzip.open(file_path, 'rb') as f:
        for line in f:
            line = json.loads(line)
            if 'id' not in line:
                continue

            id_ = line['id'][21:]

            if (
                line.get('publication_year')
                and line.get('publication_date')
                and line.get('title')
                and line.get('authorships')
                and line.get('type') in {'article', 'preprint'}
            ):
                pid2year[id_] = line['publication_year']
                pid2pub_date[id_] = line['publication_date']
                pid2title[id_] = line['title']

    return pid2year, pid2pub_date, pid2title


# =========================
# metadata 主流程
# =========================
if not os.path.exists(os.path.join(save_dire, 'pid2title.pkl')):

    JournalArticle_Year = {}
    pid2pub_date = {}
    pid2title = {}

    with ProcessPoolExecutor(max_workers=process_count) as executor:
        with tqdm(total=len(all_files), desc='process works (file-level)') as pbar:
            for y, d, t in executor.map(get_valid_pid_from_file, all_files):
                JournalArticle_Year.update(y)
                pid2pub_date.update(d)
                pid2title.update(t)
                pbar.update(1)

    with open(os.path.join(save_dire, 'pid2pub_year.pkl'), 'wb') as f:
        pickle.dump(JournalArticle_Year, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_dire, 'pid2pub_date.pkl'), 'wb') as f:
        pickle.dump(pid2pub_date, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_dire, 'pid2title.pkl'), 'wb') as f:
        pickle.dump(pid2title, f, pickle.HIGHEST_PROTOCOL)

else:
    with open(os.path.join(save_dire, 'pid2pub_date.pkl'), 'rb') as f:
        pid2pub_date = pickle.load(f)

valid_pid_set = set(pid2pub_date)

# =========================
# worker: reference
# =========================
def get_reference_from_file(file_path):
    pid2up = {}
    pid2down = defaultdict(list)

    with gzip.open(file_path, 'rb') as f:
        for line in f:
            line = json.loads(line)
            if 'id' not in line:
                continue

            id_ = line['id'][21:]
            if id_ not in valid_pid_set:
                continue

            source_date = pid2pub_date[id_]
            refs = line.get('referenced_works')
            if not refs:
                continue

            valid_refs = [
                r[21:] for r in refs
                if r[21:] in valid_pid_set
                and pid2pub_date[r[21:]] <= source_date
            ]

            if valid_refs:
                valid_refs = list(set(valid_refs))
                pid2up[id_] = valid_refs
                for r in valid_refs:
                    pid2down[r].append(id_)

    return pid2up, pid2down


# =========================
# reference 主流程
# =========================
if not os.path.exists(os.path.join(save_dire, 'cited_pid2citing_pids_list.pkl')):

    pid2up_refer = {}
    pid2down_citation = defaultdict(list)

    with ProcessPoolExecutor(max_workers=process_count) as executor:
        with tqdm(total=len(all_files), desc='process reference (file-level)') as pbar:
            for up, down in executor.map(get_reference_from_file, all_files):
                pid2up_refer.update(up)
                for k, v in down.items():
                    pid2down_citation[k].extend(v)
                pbar.update(1)

    pid2down_citation = {
        pid: list(set(citers))
        for pid, citers in tqdm(pid2down_citation.items(), desc='dedup citation')
    }

    with open(os.path.join(save_dire, 'citing_pid2cited_pids_list.pkl'), 'wb') as f:
        pickle.dump(pid2up_refer, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_dire, 'cited_pid2citing_pids_list.pkl'), 'wb') as f:
        pickle.dump(pid2down_citation, f, pickle.HIGHEST_PROTOCOL)

    print("size of pid2up_refer:", len(pid2up_refer))
    print("size of pid2down_citation:", len(pid2down_citation))

print("finished")
