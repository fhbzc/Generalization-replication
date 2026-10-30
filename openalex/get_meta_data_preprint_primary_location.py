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
os.makedirs(save_dire, exist_ok=True)

process_count = 8

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
    pid2primary_location = {}
    id2displayname = {}

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
                    if (
                        line.get('primary_location')
                        and line['primary_location'].get('source')
                        and line['primary_location']['source'].get('id')
                    ):
                        source = line['primary_location']['source']
                        source_id = source['id'][21:]
                        pid2primary_location[pid] = source_id
                        # source id -> human-readable venue name
                        id2displayname[source_id] = source.get('display_name')
    return pid2primary_location, id2displayname

# ===============================
# Build metadata (parallel)
# ===============================

if not os.path.exists(os.path.join(save_dire, 'pid2primay_location.pkl')):

    pid2primary_location_M = {}
    id2displayname_M = {}

    if process_count > 1:
        with ProcessPoolExecutor(max_workers=process_count) as executor:
            for pid2primary_location, id2displayname in tqdm(
                executor.map(get_valid_pid_from_file, gz_files),
                total=len(gz_files),
                desc='processing metadata'
            ):
                pid2primary_location_M.update(pid2primary_location)
                id2displayname_M.update(id2displayname)
    else:
        for file in tqdm(gz_files):
            pid2primary_location, id2displayname = get_valid_pid_from_file(file)
            pid2primary_location_M.update(pid2primary_location)
            id2displayname_M.update(id2displayname)


    with open(os.path.join(save_dire, 'source_id2display_name.pkl'), 'wb') as f:
        pickle.dump(id2displayname_M, f, pickle.HIGHEST_PROTOCOL)
    print("size of id2displayname_M", len(id2displayname_M))

    with open(os.path.join(save_dire, 'pid2primay_location.pkl'), 'wb') as f:
        pickle.dump(pid2primary_location_M, f, pickle.HIGHEST_PROTOCOL)
    print("size of pid2primary_location_M", len(pid2primary_location_M))

print("finished")