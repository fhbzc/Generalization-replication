"""
Semantic Scholar (S2AG) version of the OpenAlex arrange_disruption.py.

Concatenate the per-division disruption CSVs written by get_disruption.py into a
single table. No filtering is done here: this step only stitches the shards back
together; downstream scripts apply their own pub_year / reference / citation
thresholds.

Set use_openalex_augmented / skip_incomplete_window to match get_disruption.py: they are
encoded in the input/output filenames via network_suffix / window_suffix so the variants
never collide.
  - use_openalex_augmented=True merges the per-division output computed on the
    OpenAlex-augmented remove_circle network (from augment_s2_citation_with_openalex.py).
  - skip_incomplete_window=True merges the run that dropped focal papers whose year_offset
    citation window is not fully observed.

Input  : META_DIRE/disruption_division/disp_paper[_y{year_offset}][_openalex_augmented][_complete_window]_d_{division}_max_{division_count}.csv
         (+ disp_finished[_openalex_augmented][_complete_window]_d_{division}_max_{division_count}.pkl markers; from get_disruption.py)
Output : META_DIRE/disruption/disp_full[_y{year_offset}][_openalex_augmented][_complete_window].csv
         The window appears in the names only when it is NOT the default 5, so the 5-year run
         keeps the names it already has on disk and any other window (e.g. -1, the no-offset
         variant) lands beside it instead of overwriting it.
         year_offset must match the get_disruption.py run being concatenated: it is part of
         both filenames, so a mismatch fails loudly instead of mixing two citation windows.
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

# must match get_disruption.py: when True, merge the per-division output computed on
# the OpenAlex-augmented remove_circle network (augment_s2_citation_with_openalex.py).
use_openalex_augmented = False
network_suffix = '_openalex_augmented' if use_openalex_augmented else ''
print("use_openalex_augmented", use_openalex_augmented)
print("network_suffix", repr(network_suffix))

# must match get_disruption.py: when True, merge the run that skipped focal papers whose
# year_offset citation window is not fully observed.
skip_incomplete_window = True
window_suffix = None      # set below, once year_offset is known

# must match get_disruption.py: the citation window. It is part of the filenames only when it
# is NOT the default 5, so the 5-year run keeps the names it already has on disk.
year_offset = -1
year_tag = '' if year_offset == 5 else '_y%s' % year_offset
print("year_offset", year_offset, "| year_tag", repr(year_tag))

# The flag only does anything when there IS a window to be incomplete: at
# year_offset == -1 the citation window is unbounded, get_disruption.py's skip test
# short-circuits, and the run is identical either way -- so the tag is dropped there,
# and one -1 run is not written under two names that hold the same data.
window_suffix = ('_complete_window'
                  if (skip_incomplete_window and year_offset != -1) else '')
print("skip_incomplete_window", skip_incomplete_window)
print("window_suffix", repr(window_suffix))

load_dire = os.path.join(meta_dire, 'disruption_division')
save_dire = os.path.join(meta_dire, 'disruption')
os.makedirs(save_dire, exist_ok=True)

# ==================================================
# 1. load & concat all divisions
# ==================================================
list_of_df = []
for division_index in tqdm(range(division_count), desc='load data'):
    finish_dire = os.path.join(
        load_dire, f'disp_finished{year_tag}{network_suffix}{window_suffix}_d_{division_index}_max_{division_count}.pkl')
    assert os.path.exists(finish_dire), finish_dire

    csv_path = os.path.join(
        load_dire, f'disp_paper{year_tag}{network_suffix}{window_suffix}_d_{division_index}_max_{division_count}.csv')
    assert os.path.exists(csv_path), csv_path

    df = pd.read_csv(csv_path)
    list_of_df.append(df)

df_combined = pd.concat(list_of_df, ignore_index=True)

# ==================================================
# 2. save
# ==================================================
save_path = os.path.join(save_dire, f'disp_full{year_tag}{network_suffix}{window_suffix}.csv')
if os.path.exists(save_path) == False:
    df_combined.to_csv(save_path, index=False)
assert len(df_combined) == len(set(df_combined['pid']))
print("save full finished, size", len(df_combined))
print("saved to:", save_path)

print("finished")
