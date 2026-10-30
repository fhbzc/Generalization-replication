import json
import os
import gzip
import pickle
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)             # utils.py is in the parent directory
from utils import OPENALEX_DIRE, META_DIRE

path_openalex = OPENALEX_DIRE
save_dire = META_DIRE
os.makedirs(save_dire, exist_ok=True)

process_count = 8
out_path = os.path.join(save_dire, 'pid2pmid.pkl')

print("save_dire:", save_dire)
print("process_count:", process_count)
print("out_path:", out_path)

# ── Collect gz files ──────────────────────────────────────────────────────────
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
# ── Worker: extract pid -> pmid from one gz file ──────────────────────────────
def extract_pid2pmid(file_path):
    result = {}
    error_count = 0
    with gzip.open(file_path, 'rb') as f:
        for line in f:
            line = json.loads(line)
            if not line or not line.get('id'):
                continue
            ids = line.get('ids')
            if not ids or 'pmid' not in ids:
                continue
            pmid_str = ids['pmid'].rsplit('/', 1)[-1]
            if not pmid_str.isdigit():
                error_count += 1
                continue
            pid = line['id'][21:]   # strip 'https://openalex.org/'
            result[pid] = pmid_str
    return result, error_count

# ── Build pid2pmid (parallel) ─────────────────────────────────────────────────
pid2pmid = {}
total_error_count = 0

if process_count > 1:
    with ProcessPoolExecutor(max_workers=process_count) as executor:
        for partial, error_count in tqdm(
            executor.map(extract_pid2pmid, gz_files),
            total=len(gz_files),
            desc='extracting pid->pmid'
        ):
            pid2pmid.update(partial)
            total_error_count += error_count
else:
    for file in tqdm(gz_files, desc='extracting pid->pmid'):
        partial, error_count = extract_pid2pmid(file)
        pid2pmid.update(partial)
        total_error_count += error_count

print("pid2pmid size:", len(pid2pmid))
print("error_count (pmid not digit):", total_error_count)

with open(out_path, 'wb') as f:
    pickle.dump(pid2pmid, f, pickle.HIGHEST_PROTOCOL)

print("Saved to:", out_path)
print("finished")
