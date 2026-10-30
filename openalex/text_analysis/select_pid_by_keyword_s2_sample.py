"""
For every keyword produced by text_analysis/select_pid_by_keyword.py, randomly sample
up to 100 papers whose TITLE contains that keyword and write them to one xlsx per keyword.

Each xlsx has a FROZEN header row and the columns:
    pid, title

Keyword -> pid lists are read from the per-keyword pickles that select_pid_by_keyword.py
writes under META_DIRE/keyword_pids/ (each file = [keyword, [pid, ...]]). Titles come from
the same title table that script used (pid2title_no_html_before_2025.pkl, falling back to
pid2title.pkl), so the titles match exactly the text that was keyword-matched.

Input :
  META_DIRE/keyword_pids/<safe_keyword>.pkl        (one per keyword; from select_pid_by_keyword.py)
  META_DIRE/pid2title_no_html_before_2025.pkl      (preferred) or META_DIRE/pid2title.pkl
Output:
  META_DIRE/sample/<safe_keyword>_sample50.xlsx    (one per keyword; first row frozen)
"""

import os
import sys
import glob
import pickle
import random

import pandas as pd

# utils.py lives one level up (openalex/), not in this text_analysis/ directory; make it
# importable regardless of CWD. BASE_DIR (this script's own dir) is still used for ./temp below.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(BASE_DIR))
from utils import META_DIRE  # noqa: E402

meta_dire = META_DIRE
print("meta_dire", meta_dire)

SAMPLE_N = 50
RANDOM_SEED = 42          # fixed so the sampled set is reproducible across runs
random.seed(RANDOM_SEED)

# ==================================================
# 1. locate the per-keyword pid lists written by select_pid_by_keyword.py
# ==================================================
keyword_pids_dir = os.path.join(meta_dire, 'keyword_pids')
assert os.path.isdir(keyword_pids_dir), (
    f"{keyword_pids_dir} not found -- run text_analysis/select_pid_by_keyword.py first."
)
kw_files = sorted(glob.glob(os.path.join(keyword_pids_dir, '*.pkl')))
assert kw_files, f"no keyword pickles under {keyword_pids_dir}"
print(f"found {len(kw_files)} keyword files")

# ==================================================
# 2. load titles (same source / preference as select_pid_by_keyword.py)
# ==================================================
title_no_html_path = os.path.join(meta_dire, 'pid2title_no_html_before_2025.pkl')
title_raw_path = os.path.join(meta_dire, 'pid2title.pkl')
title_path = title_no_html_path if os.path.exists(title_no_html_path) else title_raw_path
assert os.path.exists(title_path), title_path
print("title_path", title_path)
with open(title_path, 'rb') as f:
    pid2title = pickle.load(f)
print("pid2title load finished | size:", len(pid2title))

# ==================================================
# 3. output directory (META_DIRE/sample, alongside the keyword_pids input)
# ==================================================
out_dir = os.path.join(meta_dire, 'sample')
os.makedirs(out_dir, exist_ok=True)
print("out_dir", out_dir)

# ==================================================
# 4. one xlsx per keyword: sample up to SAMPLE_N pids, dump [pid, title], freeze header row
# ==================================================
for path in kw_files:
    with open(path, 'rb') as f:
        kw, pids = pickle.load(f)

    n = min(SAMPLE_N, len(pids))
    sampled = random.sample(list(pids), n)  # without replacement; all pids <= SAMPLE_N -> all

    df = pd.DataFrame(
        [{'pid': pid, 'title': pid2title.get(pid)} for pid in sampled],
        columns=['pid', 'title'],
    )

    safe = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, f'{safe}_sample{SAMPLE_N}.xlsx')
    # freeze_panes=(1, 0): freeze the first (header) row so it stays visible while scrolling
    df.to_excel(out_path, index=False, freeze_panes=(1, 0))
    print(f"  '{kw}': sampled {n}/{len(pids)} -> {out_path}")

print("finished")
