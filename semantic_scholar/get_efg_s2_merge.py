"""
Semantic Scholar (S2AG) version of the OpenAlex get_efg_s2_merge.py.

Concatenate the per-division E/F/G CSVs into a single table. No filtering is done here:
this step only stitches the shards back together; downstream scripts apply their own
pub_year / reference / citation thresholds.

Set use_disp_f to match which upstream script produced the shards:
  False -> get_efg.py            (gen_V3_y{year_offset}...)
  True  -> get_efg_v2_disp_f.py  (gen_V3_disp_f_y{year_offset}..., with the three disp_f columns)

Set use_openalex_augmented to match get_efg.py / get_efg_v2_disp_f.py: when True it merges the
per-division output computed on the OpenAlex-augmented remove_circle network (from
augment_s2_citation_with_openalex.py); the choice is encoded in the input/output filenames via
network_suffix so the two variants never collide.

Input  : META_DIRE/efg_computation/gen_V3[_disp_f]_y{year_offset}[_openalex_augmented]_d_{division}_max_{division_count}.csv
         (+ ..._finished_d_{division}_max_{division_count}.pkl markers; from get_efg.py / get_efg_v2_disp_f.py)
Output : META_DIRE/efg_computation/gen_V3[_disp_f]_y{year_offset}[_openalex_augmented].csv
"""

import os
import sys

import pandas as pd
from tqdm import tqdm

# utils.py is in this same directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from utils import META_DIRE  # noqa: E402


meta_dire = META_DIRE
print("meta_dire", meta_dire)

division_count = 20
print("division_count", division_count)

load_dire = os.path.join(meta_dire, 'efg_computation')

year_offset = -1  # -1 for no offset; must match get_efg.py
print("year_offset", year_offset)

# which per-division output to merge:
#   False -> get_efg.py output               (gen_V3_y{year_offset}...)
#   True  -> get_efg_v2_disp_f.py output, which additionally carries the three
#            disruption_f_count / disruption_f (disp_f) columns (gen_V3_disp_f_y{year_offset}...)
use_disp_f = True
file_prefix = 'gen_V3_disp_f' if use_disp_f else 'gen_V3'
print("use_disp_f", use_disp_f)

# must match get_efg.py / get_efg_v2_disp_f.py: when True, merge the per-division output
# computed on the OpenAlex-augmented remove_circle network (augment_s2_citation_with_openalex.py).
use_openalex_augmented = False
network_suffix = '_openalex_augmented' if use_openalex_augmented else ''
print("use_openalex_augmented", use_openalex_augmented)
print("network_suffix", repr(network_suffix))

# base name shared by the per-division CSVs and finished-flags
# (matches get_efg.py / get_efg_v2_disp_f.py)
file_base = f'{file_prefix}_y{year_offset}{network_suffix}'

# ==================================================
# 1. load & concat all divisions
# ==================================================
df_list = []

for division_index in tqdm(range(division_count), desc="loading divisions"):
    finish_dire = os.path.join(
        load_dire,
        f'{file_base}_finished_d_{division_index}_max_{division_count}.pkl'
    )
    assert os.path.exists(finish_dire)

    csv_path = os.path.join(
        load_dire,
        f'{file_base}_d_{division_index}_max_{division_count}.csv'
    )
    assert os.path.exists(csv_path)

    df = pd.read_csv(csv_path)
    df_list.append(df)

df_res = pd.concat(df_list, ignore_index=True)
print("concat df shape:", df_res.shape)

# ==================================================
# 2. save
# ==================================================
# No filtering here: this step only concatenates the per-division output.
# Downstream scripts apply their own pub_year / reference / citation thresholds.
save_path = os.path.join(
    load_dire,
    f'{file_base}.csv'
)

df_res.to_csv(save_path, index=False)
print("saved to:", save_path)
print("finished")
