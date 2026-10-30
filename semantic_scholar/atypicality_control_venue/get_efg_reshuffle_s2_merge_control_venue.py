"""
Semantic Scholar (S2AG) version of the OpenAlex
get_efg_reshuffle_s2_merge_control_venue.py.

Merge the per-division E/F/G output produced by get_efg_reshuffle_control_venue.py
(the venue-controlled reshuffled / null-model network variant) into a single
concatenated table.

It only concatenates the per-division shards. No pub_year / reference / citation
filtering is applied here -- downstream scripts apply their own thresholds. Multiple
reshuffle iterations are processed sequentially, one merged output per iteration; an
iteration whose merged CSV already exists is skipped (resumable, no re-merge). An
iteration is also skipped (not merged) unless every division has its finished marker, so
a partially-computed shuffle is never merged. It reads from / writes to the
efg_computation_control_venue/ tree.

Input  : META_DIRE/efg_computation_control_venue/
           gen_V3_y{year_offset}_iter{iteration}_before_{upper_year}_d_{division}_max_{division_count}.csv
           (+ ..._finished_... markers) from get_efg_reshuffle_control_venue.py
Output : META_DIRE/efg_computation_control_venue/
           gen_V3_y{year_offset}_iter{iteration}_before_{upper_year}.csv   (one per iteration)
"""

import os
import pickle
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import pandas as pd
import numpy as np
import sys

# utils.py is in the parent (semantic_scholar/) directory; make it importable regardless of CWD
# (matches get_efg_reshuffle_control_venue.py).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
from utils import META_DIRE  # noqa: E402


meta_dire = META_DIRE
print("meta_dire", meta_dire)

division_count = 20
print("division_count", division_count)

load_dire = os.path.join(meta_dire, 'efg_computation_control_venue')

year_offset = 5  # -1 for no offset; must match get_efg_reshuffle_control_venue.py
print("year_offset", year_offset)

# must match get_efg_reshuffle_control_venue.py: selects which shards to merge,
# processed sequentially (one output each). Each entry is either an int reshuffle
# iteration, or None for the OBSERVED venue-filtered run (target_iteration=None there).
iteration_list = [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# keep a deterministic order; None (observed) first, then the int iterations sorted.
# (plain sorted() would raise on a mix of None and ints).
iteration_list = ([it for it in iteration_list if it is None]
                  + sorted(it for it in iteration_list if it is not None))
upper_year = 2025  # must match the upper_year used in get_efg_reshuffle_control_venue.py
print("iteration_list", iteration_list)
print("upper_year", upper_year)


def make_file_suffix(target_iteration):
    """File suffix for a target, matching get_efg_reshuffle_control_venue.py exactly:
    None -> the observed venue-filtered run; an int -> that reshuffle iteration."""
    if target_iteration is None:
        return f'_obs_valid_venue_before_{upper_year}'
    return f'_iter{target_iteration}_before_{upper_year}'


def merge_iteration(target_iteration):
    """Concatenate the per-division shards of a single target and save. `target_iteration`
    is an int reshuffle iteration, or None for the observed venue-filtered run."""
    label = 'obs' if target_iteration is None else f'iter {target_iteration}'
    file_suffix = make_file_suffix(target_iteration)
    print("file_suffix", file_suffix)

    save_path = os.path.join(
        load_dire,
        f'gen_V3_y{year_offset}{file_suffix}.csv'
    )

    # ----- 0. skip if this target is already merged -----
    if os.path.exists(save_path):
        print(f"{label}: already merged, skip:", save_path)
        return

    # ----- 0b. require EVERY division's finished marker; otherwise skip this
    #            iteration (it isn't fully computed yet) instead of merging a
    #            partial result -----
    missing = [
        division_index
        for division_index in range(division_count)
        if not os.path.exists(os.path.join(
            load_dire,
            f'gen_V3_y{year_offset}{file_suffix}_finished_d_{division_index}_max_{division_count}.pkl'
        ))
    ]
    if missing:
        print(f"{label}: not all divisions finished, skip "
              f"(missing finished markers for divisions {missing})")
        return

    # ----- 1. load & concat all divisions -----
    df_list = []
    for division_index in tqdm(range(division_count),
                               desc=f"{label}: loading divisions"):
        csv_path = os.path.join(
            load_dire,
            f'gen_V3_y{year_offset}{file_suffix}_d_{division_index}_max_{division_count}.csv'
        )
        assert os.path.exists(csv_path)

        df = pd.read_csv(csv_path)
        df_list.append(df)

    df_res = pd.concat(df_list, ignore_index=True)
    print(f"{label}: concat df shape:", df_res.shape)

    # ----- 2. save -----
    # No filtering here: this step only concatenates the per-division output.
    # Downstream scripts apply their own pub_year / reference / citation thresholds.
    df_res.to_csv(save_path, index=False)
    print(f"{label}: saved to:", save_path)


# ==================================================
# process every iteration sequentially
# ==================================================
for target_iteration in iteration_list:
    print("=" * 50)
    print("processing target_iteration", target_iteration)
    merge_iteration(target_iteration)

print("finished")
