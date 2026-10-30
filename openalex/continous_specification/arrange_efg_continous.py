"""
Concatenate the per-division continuous-spec E/F/G CSVs from get_efg_continous.py
and apply the reference / citation thresholds, producing a single FILTERED table that
visualize_efg_continous.py reads directly. Keeps rows with
up_refer_count >= reference_threshold and citation_count_y{year_offset} >=
citation_threshold. year_offset and division_count must match get_efg_continous.py.

Input  : META_DIRE/efg_computation/gen_continous_V3_y{year_offset}_d_{division}_max_{division_count}.csv
         (+ ..._finished_... markers, from get_efg_continous.py)
Output : META_DIRE/efg_computation/gen_continous_V3_y{year_offset}_rt_{reference_threshold}_ct_{citation_threshold}.csv
"""

import os
import sys

import pandas as pd
from tqdm import tqdm

# utils.py is in the parent directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
from utils import META_DIRE  # noqa: E402

meta_dire = META_DIRE
print("meta_dire", meta_dire)

division_count = 20  # must match get_efg_continous.py
print("division_count", division_count)

year_offset = 5  # must match get_efg_continous.py
print("year_offset", year_offset)

reference_threshold = 1
citation_threshold = 5
print("reference_threshold", reference_threshold)
print("citation_threshold", citation_threshold)

load_dire = os.path.join(meta_dire, 'efg_computation')
citation_col = f'citation_count_y{year_offset}'

# ==================================================
# 1. load & concat all divisions
# ==================================================
df_list = []
for division_index in tqdm(range(division_count), desc="loading divisions"):
    finish_dire = os.path.join(
        load_dire,
        f'gen_continous_V3_y{year_offset}_finished_d_{division_index}_max_{division_count}.pkl'
    )
    assert os.path.exists(finish_dire)

    csv_path = os.path.join(
        load_dire,
        f'gen_continous_V3_y{year_offset}_d_{division_index}_max_{division_count}.csv'
    )
    assert os.path.exists(csv_path)

    df_list.append(pd.read_csv(csv_path))

df_combined = pd.concat(df_list, ignore_index=True)
print("concat df shape:", df_combined.shape)

# ==================================================
# 2. apply the reference / citation thresholds & save
# ==================================================
df_combined = df_combined[df_combined['up_refer_count'] >= reference_threshold]
df_combined = df_combined[df_combined[citation_col] >= citation_threshold]
print("after threshold filter:", df_combined.shape)

save_path = os.path.join(
    load_dire,
    f'gen_continous_V3_y{year_offset}_rt_{reference_threshold}_ct_{citation_threshold}.csv'
)
df_combined.to_csv(save_path, index=False)
print("saved to:", save_path)
print("finished")
