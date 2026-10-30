"""
For every paper kept by get_meta_data.py (the keys of pid2pub_year.pkl), look up its
venue / source title from the WoS papers parquet shards and save pid -> source_title.
A paper whose source_title is missing/empty is stored as None.

WoS ids in the parquet carry the 'WOS:' prefix; pid2pub_year.pkl stores them with the
prefix stripped, so we strip it here too before matching. The parquet shards are
scanned in parallel (same set get_meta_data.py iterates over).

Set use_with_review to match get_meta_data[_with_review].py: when True it reads the
'_with_review' valid-pid set (the doctypes == 'review'-inclusive papers) and tags its
own output with the same suffix.

Input  : WOS_DIRE/wos_papers.parquet/*.parquet      (EXTERNAL WoS dataset)
         META_DIRE/pid2pub_year{data_suffix}.pkl     (get_meta_data[_with_review].py)
Output : META_DIRE/pid2source_title{data_suffix}.pkl (pid -> source_title or None)
"""

import os
import sys
import pickle
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from tqdm import tqdm

# utils.py lives in the parent wos/ directory; make it importable regardless of CWD.
WOS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WOS_DIR)
from utils import META_DIRE, WOS_DIRE  # noqa: E402

# match get_meta_data[_with_review].py: when True read the '_with_review' inputs (the
# doctypes == 'review'-inclusive set) and tag outputs with the same suffix.
use_with_review = True
data_suffix = '_with_review' if use_with_review else ''
print("use_with_review", use_with_review)
print("data_suffix", repr(data_suffix))

max_workers = 6
print("max_workers", max_workers)

paper_dire_raw = os.path.join(WOS_DIRE, 'wos_papers.parquet')
print("paper_dire_raw", paper_dire_raw)
output_dire = META_DIRE
print("output_dire", output_dire)


def process_venue(file_name):
    path = os.path.join(paper_dire_raw, file_name)
    df_data = pd.read_parquet(path, columns=['UID', 'source_title'])

    pid2source_title = {}
    for uid, source_title in zip(df_data['UID'], df_data['source_title']):
        if uid is None or uid[:4] != 'WOS:':
            continue
        short_pid = uid[4:]
        if short_pid not in valid_pid_set:
            continue
        # empty / missing source_title -> None
        if source_title is None or (isinstance(source_title, str) and len(source_title) == 0):
            pid2source_title[short_pid] = None
        else:
            pid2source_title[short_pid] = source_title
    return pid2source_title


if __name__ == '__main__':

    # valid pids = the papers get_meta_data[_with_review].py kept
    with open(os.path.join(output_dire, f'pid2pub_year{data_suffix}.pkl'), 'rb') as f:
        pid2pub_year_M = pickle.load(f)
    valid_pid_set = set(pid2pub_year_M)
    print("valid_pid_set size:", len(valid_pid_set))

    save_path = os.path.join(output_dire, f'pid2source_title{data_suffix}.pkl')
    if os.path.exists(save_path):
        print("already exists, nothing to do:", save_path)
        sys.exit(0)

    file_name_list = [f for f in os.listdir(paper_dire_raw) if f.endswith('.parquet')]

    pid2source_title_M = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(file_name_list), desc='process source_title') as pbar:
            for pid2source_title in executor.map(process_venue, file_name_list):
                pid2source_title_M.update(pid2source_title)
                pbar.update(1)

    # any valid pid not seen in the parquet (shouldn't happen) -> None
    n_missing = 0
    for pid in valid_pid_set:
        if pid not in pid2source_title_M:
            pid2source_title_M[pid] = None
            n_missing += 1
    print("pids with no parquet row (set to None):", n_missing)

    n_none = sum(1 for v in pid2source_title_M.values() if v is None)
    print("pid2source_title size:", len(pid2source_title_M), "| None values:", n_none)

    with open(save_path, 'wb') as f:
        pickle.dump(pid2source_title_M, f, pickle.HIGHEST_PROTOCOL)
    print("saved to:", save_path)
    print("finished")
