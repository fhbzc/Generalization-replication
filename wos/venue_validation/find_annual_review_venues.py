"""
Find every WoS source_title whose (lower-cased) text contains BOTH the words
"annual" AND "review", from get_meta_data_get_venue.py's pid2source_title.pkl
(pid -> source_title, with None for missing).

The matched venue names are saved:
    - as a pickled list -> META_DIRE/annual_review_source_titles.pkl   (../wos/data)
    - as a CSV (to read) -> ./annual_review_source_titles.csv          (next to this script)
The CSV also records how many papers carry each matched source_title.

The match AND the grouping are on the lower-cased name (substring, so "reviews"
also matches "review"), so the same venue in different casing is merged; the
lower-cased source_title is what gets stored.

Set use_with_review to match get_meta_data_get_venue.py: when True it reads the
'_with_review' pid2source_title and tags its outputs with the same suffix.

Input  : META_DIRE/pid2source_title{data_suffix}.pkl   (get_meta_data_get_venue.py)
Output : META_DIRE/annual_review_source_titles{data_suffix}.pkl
         ./annual_review_source_titles{data_suffix}.csv
"""

import os
import sys
import csv
import pickle
from collections import Counter

# utils.py lives in the parent wos/ directory
WOS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WOS_DIR)
from utils import META_DIRE  # noqa: E402

# match get_meta_data_get_venue.py: when True read the '_with_review' input and tag
# outputs with the same suffix.
use_with_review = True
data_suffix = '_with_review' if use_with_review else ''
print("use_with_review", use_with_review)
print("data_suffix", repr(data_suffix))


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# --- pid -> source_title (get_meta_data_get_venue.py) ---
pid2source_title = load_pkl(os.path.join(META_DIRE, f'pid2source_title{data_suffix}.pkl'))
print("pid2source_title size:", len(pid2source_title))

# --- count papers per lower-cased source_title whose name has BOTH 'annual' & 'review' ---
title2count = Counter()
for source_title in pid2source_title.values():
    if not source_title:                 # None or empty
        continue
    low = source_title.lower()
    if 'annual' in low and 'review' in low:
        title2count[low] += 1            # group case-insensitively

# stable, readable order (by name)
matched = sorted(title2count.items(), key=lambda x: x[0])
print("matched source_titles (contain 'annual' and 'review'):", len(matched))
print("papers in matched venues:", sum(title2count.values()))

# --- save the list of venue names as a pickle in META_DIRE (../wos/data) ---
list_path = os.path.join(META_DIRE, f'annual_review_source_titles{data_suffix}.pkl')
with open(list_path, 'wb') as f:
    pickle.dump([name for name, _ in matched], f, pickle.HIGHEST_PROTOCOL)
print("wrote list:", list_path)

# --- save name + paper count as a CSV next to this script ---
out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f'annual_review_source_titles{data_suffix}.csv')
with open(out_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['source_title', 'paper_count'])
    for name, count in matched:
        writer.writerow([name, count])
print("wrote csv:", out_csv)
print("finished")
