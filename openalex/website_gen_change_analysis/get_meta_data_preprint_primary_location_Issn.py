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
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)             # utils.py is in the parent directory
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
    pid2primary_location_issn = {}   # pid -> str (issn_l of primary_location source)
    pid2location_issn = {}           # pid -> list of unique issns across all locations

    with gzip.open(file_path, 'rb') as f:
        for line in f:
            line = json.loads(line)
            if not line:
                continue

            if not line.get('id'):
                continue

            pid = line['id'][21:]

            if not (line.get('publication_year') and line.get('authorships') and line.get('type')
                    and line['type'] in ('article', 'preprint')):
                continue

            # --- pid2primary_location_issn ---
            primary_loc = line.get('primary_location')
            if (primary_loc
                    and primary_loc.get('source')
                    and primary_loc['source'].get('issn_l')):
                pid2primary_location_issn[pid] = primary_loc['source']['issn_l']

            # --- pid2location_issn ---
            issns = set()
            for loc in line.get('locations', []):
                src = loc.get('source')
                if src and src.get('issn'):
                    for issn in src['issn']:
                        if issn:
                            issns.add(issn)
            if issns:
                pid2location_issn[pid] = list(issns)

    return pid2primary_location_issn, pid2location_issn


# ===============================
# Build metadata (parallel)
# ===============================

out_primary = os.path.join(save_dire, 'pid2primary_location_issn.pkl')
out_location = os.path.join(save_dire, 'pid2location_issn_list.pkl')

if not os.path.exists(out_primary) or not os.path.exists(out_location):

    pid2primary_location_issn_M = {}
    pid2location_issn_M = {}

    if process_count > 1:
        with ProcessPoolExecutor(max_workers=process_count) as executor:
            for pid2primary, pid2loc in tqdm(
                executor.map(get_valid_pid_from_file, gz_files),
                total=len(gz_files),
                desc='processing metadata'
            ):
                pid2primary_location_issn_M.update(pid2primary)
                pid2location_issn_M.update(pid2loc)
    else:
        for file in tqdm(gz_files):
            pid2primary, pid2loc = get_valid_pid_from_file(file)
            pid2primary_location_issn_M.update(pid2primary)
            pid2location_issn_M.update(pid2loc)

    with open(out_primary, 'wb') as f:
        pickle.dump(pid2primary_location_issn_M, f, pickle.HIGHEST_PROTOCOL)
    print("size of pid2primary_location_issn_M", len(pid2primary_location_issn_M))

    with open(out_location, 'wb') as f:
        pickle.dump(pid2location_issn_M, f, pickle.HIGHEST_PROTOCOL)
    print("size of pid2location_issn_M", len(pid2location_issn_M))

print("finished")