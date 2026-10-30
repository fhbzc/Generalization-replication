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

import re
ARXIV_PATTERN = re.compile(r"^https?://arxiv\.org/abs/(.+)$")
def extract_arxiv_id(x):
    match = ARXIV_PATTERN.match(x)
    return match.group(1) if match else None
# print(extract_arxiv_id('http://arxiv.org/abs/physics/0401019'))
# print(extract_arxiv_id('https://arxiv.org/abs/physics/0401019'))
# print(extract_arxiv_id('https://openalex.org/I205783295'))
# assert False

path_openalex = OPENALEX_DIRE
save_dire = META_DIRE
os.makedirs(save_dire, exist_ok=True)

process_count = 16

print("save_dire", save_dire)
print("process_count", process_count)

# ===============================
# Collect all valid gz files
# ===============================
path_work = os.path.join(path_openalex, 'openalex-snapshot/data/works/')
dirs = sorted(os.listdir(path_work))

gz_files = []
for d in dirs:
    if '_updated_date' not in d and 'updated_date' in d:
        dpath = os.path.join(path_work, d)
        for f in os.listdir(dpath):
            if f.endswith('.gz') and '_part' not in f:
                gz_files.append(os.path.join(dpath, f))

print("Total gz files:", len(gz_files))

# ===============================
# File-level worker: metadata
# ===============================
def get_valid_pid_from_file(file_path):
    pid2arxiv_id_list = defaultdict(list)

    with gzip.open(file_path, 'rb') as f:
        for line in f:
            line = json.loads(line)
            if not line:
                continue

            if (
                line.get('id')
            ):
                pid = line['id'][21:]
                if line and line['publication_year'] and line['authorships'] and line['type'] and (line['type']=='article' or line['type'] == 'preprint'):
                    for location in line.get('locations', []):

                        if location.get('landing_page_url', None) is not None:

                            landing_page_url = location['landing_page_url']

                            extracted_item = extract_arxiv_id(landing_page_url)
                            if extracted_item is not None:
                                pid2arxiv_id_list[pid].append(extracted_item)

                                
    return pid2arxiv_id_list

# ===============================
# Build metadata (parallel)
# ===============================
# for file in gz_files:
#     get_valid_pid_from_file(file)

# assert False

if not os.path.exists(os.path.join(save_dire, 'pid2arxiv_id_list.pkl')):
# if True:

    pid2primary_location_M = {}

    if process_count > 1:
        with ProcessPoolExecutor(max_workers=process_count) as executor:
            for pid2primary_location in tqdm(
                executor.map(get_valid_pid_from_file, gz_files),
                total=len(gz_files),
                desc='processing metadata'
            ):
                pid2primary_location_M.update(pid2primary_location)
    else:
        for file in tqdm(gz_files):
            pid2primary_location = get_valid_pid_from_file(file)
            pid2primary_location_M.update(pid2primary_location)

    pid2primary_location_M = {pid: list(set(aid_list)) for pid, aid_list in tqdm(pid2primary_location_M.items())}
    with open(os.path.join(save_dire, 'pid2arxiv_id_list.pkl'), 'wb') as f:
        pickle.dump(pid2primary_location_M, f, pickle.HIGHEST_PROTOCOL)
    print("size of pid2arxiv_id_list", len(pid2primary_location_M))

print("finished")