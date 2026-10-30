import os
import sys
import csv
import pickle
from collections import Counter

# utils.py lives in the parent openalex/ directory
OPENALEX_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, OPENALEX_DIR)
from utils import META_DIRE


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# --- result of get_meta_data_preprint_primary_location.py ---
primary_location_path = os.path.join(META_DIRE, 'pid2primay_location.pkl')
pid2primary_location = load_pkl(primary_location_path)

# the dict only stores pids whose primary_location -> source -> id is present,
# but count explicitly for clarity / robustness
n_total_pl = len(pid2primary_location)
n_valid_pl = sum(1 for v in pid2primary_location.values() if v is not None)

print("pid2primay_location.pkl")
print("  total pids in dict      :", n_total_pl)
print("  pids with valid primary location (non-None):", n_valid_pl)

# --- pid2pub_year (faster to load than pid2title; only used for the total pid count) ---
pub_year_path = os.path.join(META_DIRE, 'pid2pub_year.pkl')
pid2pub_year = load_pkl(pub_year_path)
print("pid2pub_year.pkl")
print("  total pids:", len(pid2pub_year))

# --- overlap ---
overlap = set(pid2primary_location) & set(pid2pub_year)
print("overlap (pids in both):", len(overlap))

# --- per-venue pid count -> csv ---
# pid2primary_location maps pid -> primary source (venue) id, so counting the values
# gives the number of pids per venue.
venue2count = Counter(v for v in pid2primary_location.values() if v is not None)

# source id -> human-readable venue name (from get_meta_data_preprint_primary_location.py)
source_id2display_name = load_pkl(os.path.join(META_DIRE, 'source_id2display_name.pkl'))

out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'venue_pid_count.csv')
with open(out_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['venue_id', 'display_name', 'pid_count'])
    for venue_id, count in venue2count.most_common():
        writer.writerow([venue_id, source_id2display_name.get(venue_id, ''), count])
print("wrote per-venue pid count:", out_csv)
print("  number of venues:", len(venue2count))
