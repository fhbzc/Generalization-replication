"""
Concatenate the per-division E/F/G CSVs written by get_efg.py (WoS) into a single
table. Asserts every division's finished-flag and CSV exist, then concatenates them
with no filtering (downstream scripts apply their own pub_year / reference /
citation thresholds). `year_offset` and `division_count` must match get_efg.py.

Set use_with_review to match get_efg.py: when True it reads/writes the '_with_review'
variant (E/F/G computed over the doctypes=='review'-inclusive citation network).

Set use_disp_f to match the run being merged:
  * False -> get_efg.py output           (gen_V3_y{year_offset}...)
  * True  -> get_efg_v2_disp_f.py output, which additionally carries the three disp_f
             variant count / share columns (gen_V3_disp_f_y{year_offset}...)
The choice is encoded in the per-division input filenames and the merged output filename
via file_prefix, so the two runs never collide.

Input  : META_DIRE/efg_computation/{gen_V3|gen_V3_disp_f}_y{year_offset}[_with_review]_d_{division}_max_{division_count}.csv
         (+ ..._finished_... markers, from get_efg.py / get_efg_v2_disp_f.py)
Output : META_DIRE/efg_computation/{gen_V3|gen_V3_disp_f}_y{year_offset}[_with_review].csv
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

# must match get_efg.py: read/write the '_with_review' variant when True
use_with_review = True
data_suffix = '_with_review' if use_with_review else ''
print("use_with_review", use_with_review)

# Which run to merge:
#   False -> get_efg.py output          (gen_V3_y{year_offset}...)
#   True  -> get_efg_v2_disp_f.py output (gen_V3_disp_f_y{year_offset}...; adds the disp_f columns)
# The prefix is encoded in the per-division input filenames and the merged output filename,
# so the two runs never collide.
use_disp_f = False
file_prefix = 'gen_V3_disp_f' if use_disp_f else 'gen_V3'
print("use_disp_f", use_disp_f)
print("file_prefix", file_prefix)

# base name shared by the per-division CSVs and finished-flags (matches get_efg.py / get_efg_v2_disp_f.py)
file_base = f'{file_prefix}_y{year_offset}{data_suffix}'

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

    # pid must stay a string: WoS ids are digit-like and would otherwise be inferred
    # inconsistently across chunks (DtypeWarning) and lose leading zeros, breaking
    # downstream joins on pid.
    df = pd.read_csv(csv_path, dtype={'pid': str})
    df_list.append(df)

df_res = pd.concat(df_list, ignore_index=True)
print("concat df shape:", df_res.shape)

# ==================================================
# 2. save
# ==================================================
# No filtering here: this step only concatenates the per-division output.
# Downstream scripts apply their own pub_year / reference / citation thresholds.
save_path = os.path.join(load_dire, f'{file_base}.csv')

df_res.to_csv(save_path, index=False)
print("saved to:", save_path)
print("finished")
