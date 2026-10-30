"""
Find every venue whose (lower-cased) display name contains BOTH the words
"annual" AND "review", from source_id2display_name.pkl (source id -> venue
display name).

Set USE_REVIEW_META below to choose which meta directory to read from:
    * False -> META_DIRE                  (../openalex/data, from
               get_meta_data_preprint_primary_location.py; article/preprint only)
    * True  -> META_DIRE + '_with_review' (../openalex/data_with_review, from
               get_meta_data_preprint_review.py, whose KEEP_TYPES also includes
               'review'). Annual-review papers are type == 'review', so to validate
               annual-review venues you usually want this version.
The .pkl format (source id -> display name) is identical either way; only the
directory changes.

The matched (sid, name) pairs are saved (file names carry a '_with_review' suffix when
USE_REVIEW_META is True, so the two variants never overwrite each other):
    - as a pickled list  -> <META>/annual_review_venues[_with_review].pkl  (chosen meta dir)
    - as a CSV (to read)  -> ./annual_review_venues[_with_review].csv      (next to this script)

The match is on the lower-cased name (substring of the whole string, so words like
"reviews" also match "review"); the ORIGINAL display name is what gets stored.
"""

import os
import sys
import csv
import pickle

# utils.py lives in the parent openalex/ directory
OPENALEX_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, OPENALEX_DIR)
from utils import META_DIRE  # noqa: E402

# ---- option: read/write the review-inclusive meta (get_meta_data_preprint_review.py) ----
USE_REVIEW_META = True
META = META_DIRE + '_with_review' if USE_REVIEW_META else META_DIRE
# suffix so the two variants' output files never collide (the CSV in particular sits
# next to this script regardless of META, so it would otherwise overwrite)
NAME_SUFFIX = '_with_review' if USE_REVIEW_META else ''
print("USE_REVIEW_META:", USE_REVIEW_META)
print("META dir:", META)


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# --- source id -> venue display name ---
source_id2display_name = load_pkl(os.path.join(META, 'source_id2display_name.pkl'))
print("source_id2display_name size:", len(source_id2display_name))

# --- keep venues whose lower-cased name contains BOTH 'annual' and 'review' ---
matched = []  # list of (sid, name)
for sid, name in source_id2display_name.items():
    if not name:
        continue
    low = name.lower()
    if 'annual' in low and 'review' in low:
        matched.append((sid, name))

# stable, readable order (by name)
matched.sort(key=lambda x: x[1].lower())
print("matched venues (name contains 'annual' and 'review'):", len(matched))

# --- save the list as a pickle in the chosen meta dir ---
list_path = os.path.join(META, f'annual_review_venues{NAME_SUFFIX}.pkl')
with open(list_path, 'wb') as f:
    pickle.dump(matched, f, pickle.HIGHEST_PROTOCOL)
print("wrote list:", list_path)

# --- save the same as a CSV next to this script ---
out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f'annual_review_venues{NAME_SUFFIX}.csv')
with open(out_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['venue_id', 'display_name'])
    for sid, name in matched:
        writer.writerow([sid, name])
print("wrote csv:", out_csv)
print("finished")
