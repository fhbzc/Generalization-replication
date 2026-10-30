"""
MAG version of the WoS arrange_disruption.py: concatenate the per-division
disruption CSVs written by get_disruption.py into a single disp_full.csv.

There is NO '_with_review' variant here (the MAG network is journal-only, so
get_disruption.py has no data_suffix).

Set k_require_after_focal to match get_disruption.py: the dropped-lower-bound K variant
tags its files with '_k_no_lower_bound' (k_suffix).

Input  : META_DIRE/disruption_division/disp_paper{year_tag}[_k_no_lower_bound]_d_{division}_max_{division_count}.csv
         (year_tag = '_y{year_offset_main}', empty at the default 5)
         (from get_disruption.py)
Output : META_DIRE/disruption/disp_full[_k_no_lower_bound].csv
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

# must match get_disruption.py: the K-window variant to read/write. False (default K)
# -> ''; True for the dropped-lower-bound K -> '_k_no_lower_bound'.
k_require_after_focal = True
k_suffix = '' if k_require_after_focal else '_k_no_lower_bound'

# Must match get_disruption.py's year_offset_main: it tags every per-division file it writes
# with year_tag, so this step has to look for the same tag (and write its disp_full under it)
# or it would merge a DIFFERENT citation window's output. BACKWARD COMPATIBILITY:
# year_offset_main = 5 keeps the untagged names, so the already-saved files resolve as before.
year_offset_main = -1
year_tag = '' if year_offset_main == 5 else f'_y{year_offset_main}'
print("year_offset_main", year_offset_main, "| year_tag", repr(year_tag))
print("k_require_after_focal", k_require_after_focal)
print("k_suffix", repr(k_suffix))

list_of_df = []
for division_index in tqdm(range(division_count), desc='load data'):

    csv_path = os.path.join(load_dire, 'disp_paper%s%s_d_%s_max_%s.csv'
                            % (year_tag, k_suffix, division_index, division_count))
    assert os.path.exists(csv_path), (
        "%s\nRun get_disruption.py with year_offset_main=%s for every division first."
        % (csv_path, year_offset_main))
    df = pd.read_csv(csv_path)
    list_of_df.append(df)


df_combined = pd.concat(list_of_df, ignore_index=True)

save_path = os.path.join(save_dire, 'disp_full%s%s.csv' % (year_tag, k_suffix))
if os.path.exists(save_path) == False:
    df_combined.to_csv(save_path, index=False)
assert len(df_combined) == len(set(df_combined['pid']))
print("save full finished, size", len(df_combined))
print("saved to:", save_path)

print("finished")
