"""
Build the citation network exactly like get_meta_data_preprint.py, but WITHOUT the
temporal filter on references: a reference edge is kept as long as BOTH endpoints are
valid pids, regardless of publication year/date (the original drops edges whose cited
work is not older than the citing work, i.e. pid2pub_date[ref] < source_work_date).

It reuses the metadata produced by get_meta_data_preprint.py (pid2pub_year.pkl defines
the valid pid set), and only rebuilds the reference network, writing to its own
'_no_year_filter' output files so the originals are left untouched.

Input  : OPENALEX_DIRE/openalex-snapshot/data/works/**/*.gz   (OpenAlex works)
         META_DIRE/pid2pub_year.pkl                            (get_meta_data_preprint.py)
Output : META_DIRE/citing_pid2cited_pids_list_no_year_filter.pkl
         META_DIRE/cited_pid2citing_pids_list_no_year_filter.pkl
"""

import json
import os
import gzip
import pickle
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)               # utils.py is in this same directory
from utils import OPENALEX_DIRE, META_DIRE  # noqa: E402

path_openalex = OPENALEX_DIRE
save_dire = META_DIRE
os.makedirs(save_dire, exist_ok=True)
process_count = 1

print("save_dire", save_dire)
print("process_count", process_count)


def get_reference(file_path):
    pid2up_refer_single = {}
    pid2down_refer_single = defaultdict(list)
    # file-level worker: process a single .gz file (parallelism is over files)
    with gzip.open(file_path, 'rb') as f:
        for line in f:
            line = json.loads(line)

            id_ = line['id'][21:]

            if id_ in valid_pid_set:
                if line['referenced_works']:
                    # NO pub_year/pub_date filter: keep every reference whose pid is
                    # valid, regardless of temporal ordering.
                    refs = set([r[21:] for r in line['referenced_works']
                                if r[21:] in valid_pid_set])
                    refs = list(refs)
                    pid2up_refer_single[id_] = refs
                    for ref in refs:
                        pid2down_refer_single[ref].append(id_)
    return pid2up_refer_single, pid2down_refer_single


path_work = os.path.join(path_openalex, 'openalex-snapshot/data/works/')
dirs = sorted(os.listdir(path_work))

# ===============================
# Collect all valid gz files (file-level parallelism, like get_meta_data_preprint.py)
# ===============================
gz_files = []
for d in dirs:
    if '_updated_date' not in d and 'updated_date' in d:
        dpath = os.path.join(path_work, d)
        for fname in sorted(os.listdir(dpath)):
            if fname.endswith('.gz') and '_part' not in fname:
                gz_files.append(os.path.join(dpath, fname))
print("Total gz files:", len(gz_files))

# ===============================
# valid pid set from the metadata already produced by get_meta_data_preprint.py
# (no temporal filter here, so we only need the pid keys, not the dates)
# ===============================
pub_year_path = os.path.join(save_dire, 'pid2pub_year.pkl')
assert os.path.exists(pub_year_path), (
    f"{pub_year_path} not found; run get_meta_data_preprint.py first to build the metadata.")
with open(pub_year_path, 'rb') as f:
    pid2pub_year = pickle.load(f)
valid_pid_set = set(pid2pub_year)
print("valid_pid_set size:", len(valid_pid_set))

# ===============================
# build the (temporally-unfiltered) reference network
# ===============================
citing_out = os.path.join(save_dire, 'citing_pid2cited_pids_list_no_year_filter.pkl')
cited_out = os.path.join(save_dire, 'cited_pid2citing_pids_list_no_year_filter.pkl')

if os.path.exists(cited_out) is False:
    pid2up_refer = {}
    pid2down_citation = {}

    if process_count > 1:
        with ProcessPoolExecutor(max_workers=process_count) as executor:
            with tqdm(total=len(gz_files), desc='process reference') as pbar:
                for pid2up_refer_single, pid2down_refer_single in executor.map(get_reference, gz_files):
                    pbar.update(1)
                    pid2up_refer.update(pid2up_refer_single)
                    for pid in pid2down_refer_single:
                        if pid not in pid2down_citation:
                            pid2down_citation[pid] = []
                        pid2down_citation[pid].extend(pid2down_refer_single[pid])
    else:
        for file_path in tqdm(gz_files, desc='process reference'):
            pid2up_refer_single, pid2down_refer_single = get_reference(file_path)
            pid2up_refer.update(pid2up_refer_single)
            for pid in pid2down_refer_single:
                if pid not in pid2down_citation:
                    pid2down_citation[pid] = []
                pid2down_citation[pid].extend(pid2down_refer_single[pid])

    with open(citing_out, 'wb') as f:
        pickle.dump(pid2up_refer, f, pickle.HIGHEST_PROTOCOL)

    pid2down_citation = {pid: list(set(pid2down_citation[pid]))
                         for pid in tqdm(pid2down_citation, desc='process pid2down_citation')}
    with open(cited_out, 'wb') as f:
        pickle.dump(pid2down_citation, f, pickle.HIGHEST_PROTOCOL)

    print("size of pid2up_refer", len(pid2up_refer))
    print("size of pid2down_citation", len(pid2down_citation))

print("finished")
