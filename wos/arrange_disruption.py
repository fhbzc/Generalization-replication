"""
WoS version of the OpenAlex arrange_disruption.py: concatenate the per-division
disruption CSVs written by get_disruption.py into a single disp_full.csv.

Set use_with_review to match get_disruption.py: when True it reads/writes the
'_with_review' variant (disruption computed over the doctypes=='review'-inclusive
citation network).

Set k_require_after_focal to match get_disruption.py: the dropped-lower-bound K variant
tags its files with '_k_no_lower_bound' (k_suffix).

Input  : META_DIRE/disruption_division/disp_paper[_y{year_offset_main}][_with_review][_k_no_lower_bound]_d_{division}_max_{division_count}.csv
         (from get_disruption.py)
Output : META_DIRE/disruption/disp_full[_y{year_offset_main}][_with_review][_k_no_lower_bound].csv
         The citation window appears in the names only when it is NOT the default 5, so the
         5-year run keeps the names it already has on disk and any other window (e.g. -1, the
         no-offset variant) lands beside it instead of overwriting it. year_offset_main must
         match the get_disruption.py run being concatenated.
"""

import os
import sys

import pandas as pd
from tqdm import tqdm

# utils.py is in this same directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from utils import META_DIRE  # noqa: E402

main_dire = META_DIRE
print("main_dire", main_dire)
load_dire = os.path.join(main_dire, 'disruption_division')
save_dire = os.path.join(main_dire, 'disruption')
os.makedirs(save_dire, exist_ok=True)
division_count = 20
print("division_count", division_count)

# must match get_disruption.py: read/write the '_with_review' variant when True
use_with_review = True
data_suffix = '_with_review' if use_with_review else ''
print("use_with_review", use_with_review)

# must match get_disruption.py: the K-window variant to read/write. False (default K)
# -> ''; True for the dropped-lower-bound K -> '_k_no_lower_bound'.
k_require_after_focal = True
k_suffix = '' if k_require_after_focal else '_k_no_lower_bound'
print("k_require_after_focal", k_require_after_focal)
print("k_suffix", repr(k_suffix))

# must match get_disruption.py: the citation window. It is part of the filenames only when it
# is NOT the default 5, so the 5-year run keeps the names it already has on disk.
year_offset_main = -1
year_tag = '' if year_offset_main == 5 else '_y%s' % year_offset_main
print("year_offset_main", year_offset_main, "| year_tag", repr(year_tag))

list_of_df = []
for division_index in tqdm(range(division_count), desc='load data'):

    df = pd.read_csv(os.path.join(load_dire, 'disp_paper%s%s%s_d_%s_max_%s.csv' % (year_tag, data_suffix, k_suffix, division_index, division_count)))
    list_of_df.append(df)


df_combined = pd.concat(list_of_df, ignore_index=True)

save_path = os.path.join(save_dire, 'disp_full%s%s%s.csv' % (year_tag, data_suffix, k_suffix))
if os.path.exists(save_path) == False:
    df_combined.to_csv(save_path, index=False)
assert len(df_combined) == len(set(df_combined['pid']))
print("save full finished, size", len(df_combined))
print("saved to:", save_path)

print("finished")
