"""
Augment the merged E/F/G table (get_efg_s2_merge.py output) with each paper's
single top (root-level) concept, then apply the reference / citation / max_year
filters.

Top concept comes from pid2root_concepts_level_{root_level}.pkl
(process_top_concepts.py), which maps pid -> list of root concepts (or None).
Papers with NO top concept (None / empty / pid missing from the map) or with
MORE THAN ONE top concept are DROPPED (the dropped counts are reported), so the
surviving 'top_concept' column is unambiguous: exactly one concept per paper.

Order (as requested): first attach top_concept and drop ambiguous/missing pids,
THEN apply the citation / reference / max_year thresholds.

Input  : META_DIRE/efg_computation/gen_V3_y{year_offset}.csv        (get_efg_s2_merge.py)
         META_DIRE/pid2root_concepts_level_{root_level}.pkl         (process_top_concepts.py)
Output : META_DIRE/efg_computation/
           gen_V3_y{year_offset}_with_top_concept_rt_{reference_threshold}_ct_{citation_threshold}_ybefore_{max_year}.csv
"""

import os
import sys
import pickle

import pandas as pd

# utils.py is in this same directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from utils import META_DIRE  # noqa: E402

meta_dire = META_DIRE
print("meta_dire", meta_dire)

load_dire = os.path.join(meta_dire, 'efg_computation')

year_offset = 5    # must match get_efg_s2_merge.py / get_efg.py
root_level = 0     # must match process_top_concepts.py
print("year_offset", year_offset)
print("root_level", root_level)

# filters applied AFTER the top-concept step
reference_threshold = 1
citation_threshold = 5
# pub_year cutoff leaves room for the citation window (matches the repo convention
# pub_year < 2025 - year_offset). For year_offset == -1 (all future citations,
# no fixed window) this is 2026, i.e. all data before 2025.
max_year = 2025 - year_offset
print("reference_threshold", reference_threshold)
print("citation_threshold", citation_threshold)
print("max_year", max_year)

citation_col = f'citation_count_y{year_offset}'

# ==================================================
# 1. load merged E/F/G table (get_efg_s2_merge.py output)
# ==================================================
obs_path = os.path.join(load_dire, f'gen_V3_y{year_offset}.csv')
assert os.path.exists(obs_path), obs_path
df = pd.read_csv(obs_path)
print("loaded gen_V3 shape:", df.shape)

# ==================================================
# 2. load pid -> root concepts and attach the single top concept
# ==================================================
concept_path = os.path.join(meta_dire, f'pid2root_concepts_level_{root_level}.pkl')
assert os.path.exists(concept_path), concept_path
with open(concept_path, 'rb') as f:
    pid2root_concepts = pickle.load(f)
print("pid2root_concepts size:", len(pid2root_concepts))

# classify every pid in the table: keep only those with EXACTLY one top concept
top_concepts = []
n_no_concept = 0   # None / empty / pid missing from the map
n_multiple = 0     # more than one top concept
for pid in df['pid'].values:
    v = pid2root_concepts.get(pid)
    if isinstance(v, list) and len(v) == 1:
        top_concepts.append(v[0])
    else:
        if isinstance(v, list) and len(v) > 1:
            n_multiple += 1
        else:                       # None, [], or pid not in the map
            n_no_concept += 1
        top_concepts.append(None)

df['top_concept'] = top_concepts

# ==================================================
# 3. drop pids with no / ambiguous top concept (report how many)
# ==================================================
n_before = len(df)
df = df[df['top_concept'].notna()].copy()
n_after = len(df)
n_deleted = n_before - n_after
assert n_deleted == n_no_concept + n_multiple
print(f"top-concept step: {n_before} -> {n_after} papers")
print(f"  deleted {n_deleted} pids "
      f"({n_no_concept} with no top concept, {n_multiple} with >1 top concept)")

# ==================================================
# 4. citation / reference / max_year filter (matches get_efg_2026_merge.py)
# ==================================================
n_before_filter = len(df)
df = df[
    (df['up_refer_count'] >= reference_threshold) &
    (df[citation_col] >= citation_threshold) &
    (df['pub_year'] < max_year)
].copy()
n_after_filter = len(df)
print(f"threshold filter: {n_before_filter} -> {n_after_filter} papers "
      f"(removed {n_before_filter - n_after_filter})")

# ==================================================
# 5. save
# ==================================================
save_path = os.path.join(
    load_dire,
    f'gen_V3_y{year_offset}_with_top_concept_'
    f'rt_{reference_threshold}_'
    f'ct_{citation_threshold}_'
    f'ybefore_{max_year}.csv'
)
df.to_csv(save_path, index=False)
print("output shape:", df.shape)
print("saved to:", save_path)
print("finished")
