"""
Semantic Scholar (S2AG) version of the OpenAlex get_efg_zscore_control_venue.py.

Compute per-paper E/F/G z-scores against the VENUE-CONTROLLED reshuffled (null-model)
networks.

Both the observed and the reshuffled tables come from the venue-controlled pipeline
(get_efg_reshuffle_control_venue.py -> get_efg_reshuffle_s2_merge_control_venue.py),
so obs and the null model are computed on the SAME upper_year edge subset and are
directly comparable:
  - observed  : the target_iteration=None run (venue-filtered network, restricted to
                source/citing year < upper_year).
  - random    : each int reshuffle iteration in iteration_list.

The observed table is first filtered on up_refer_count / citation_count / pub_year
(reference_threshold / citation_threshold / pub_year_upper); z-scores are then
computed only for the surviving papers.

For every surviving id (the 'pid' column, an S2AG corpusid):
    z = (obs - mean_random) / std_random
where mean_random / std_random are taken across the reshuffled-network iterations
listed in `iteration_list` (each merged by get_efg_reshuffle_s2_merge_control_venue.py).

Two ways of handling ids whose random std is (near-)zero for some index are both
expressed in ONE output via a dummy column `has_full_variance`, since an id with
genuine variance in ALL of e/f/g gets the SAME z-score either way (the modes only
differ on the zero-std ids):
  - Mode 1 ("backfill", the whole table): any (near-)zero random std is replaced
    with the smallest non-zero std among the same-year papers for that index, so no
    paper is dropped. has_full_variance == 0 marks the ids that needed this backfill.
  - Mode 2 ("drop", the has_full_variance == 1 subset): keep ONLY ids whose e/f/g
    random distributions all had genuine (non-zero) variance, i.e. whose z-scores
    never used a backfilled std. Recover it by filtering has_full_variance == 1.

The script also saves, per id:
    - pub_year (carried over from gen_V3, so downstream needn't re-merge it)
    - obs e/f/g (normalized shares)        : obs_e, obs_f, obs_g
    - obs up_refer_count and citation_count
    - the mean e/f/g score across all random iterations : random_mean_e/f/g
    - the std  e/f/g score across all random iterations : random_std_e/f/g
    - has_full_variance : dummy (1/0) splitting the two groups described above

Input  : META_DIRE/efg_computation_control_venue/
           gen_V3_y{year_offset}_obs_valid_venue_before_{upper_year}.csv  (obs, None-run)
           gen_V3_y{year_offset}_iter{it}_before_{upper_year}.csv         (one per iteration)
           (both from get_efg_reshuffle_s2_merge_control_venue.py)
Output : META_DIRE/efg_computation_control_venue/
           efg_zscore_y{year_offset}_before_{upper_year}_{iter_tag}.csv
           where iter_tag (e.g. "iter0-1-2") reflects the exact iteration_list used.
"""

import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

# utils.py is in the parent (semantic_scholar/) directory; make it importable regardless of CWD
# (matches get_efg_reshuffle_control_venue.py).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
from utils import META_DIRE  # noqa: E402


meta_dire = META_DIRE
print("meta_dire", meta_dire)

load_dire = os.path.join(meta_dire, 'efg_computation_control_venue')

year_offset = 5    # must match get_efg_reshuffle_s2_merge_control_venue.py
upper_year = 2025  # must match the upper_year used in get_efg_reshuffle_s2_merge_control_venue.py
print("year_offset", year_offset)
print("upper_year", upper_year)

# list of venue-controlled reshuffle iterations (different random-reshuffle versions) to
# use as the null model. Fill in the iterations you have merged with
# get_efg_reshuffle_s2_merge_control_venue.py.
iteration_list = [0, 1, 2]
iteration_list = sorted(iteration_list)  # keep iterations in a deterministic order
# tag the output filename with the exact iterations used (not just the count): e.g.
# [0,1,2] -> "iter0-1-2". [0,2,5] and [0,1,2] are different null models but would
# otherwise both collide on "niter3".
iter_tag = 'iter' + '-'.join(str(it) for it in iteration_list)
print("iteration_list", iteration_list)
print("iter_tag", iter_tag)

# sample std (ddof=1); set ddof=0 for population std
std_ddof = 1

# A random std of exactly 0 (or a tiny float that is effectively 0) makes the
# z-score undefined and would drop the paper. Instead, any std below this
# threshold is replaced with the smallest NON-zero std among the same-year papers
# for the SAME index (e/f/g). This keeps every filtered paper in the output.
std_zero_threshold = 1e-10

# obs filter applied BEFORE computing z-scores (z-scores are only computed for the
# papers that survive this filter)
reference_threshold = 1
citation_threshold = 5
# dynamic pub_year cutoff so the year_offset-year citation window fits within the data
pub_year_upper = 2024 - year_offset + 1
print("reference_threshold", reference_threshold)
print("citation_threshold", citation_threshold)
print("pub_year_upper", pub_year_upper)

# column names in both the obs and the reshuffle merges
e_col = f'e_y{year_offset}'
f_col = f'f_y{year_offset}'
g_col = f'g_y{year_offset}'
citation_col = f'citation_count_y{year_offset}'

# ==================================================
# 1. load observed (venue-filtered None-run) merge, then filter on citation /
#    reference / pub_year. Everything downstream is computed only for the surviving papers.
# ==================================================
obs_path = os.path.join(
    load_dire,
    f'gen_V3_y{year_offset}_obs_valid_venue_before_{upper_year}.csv'
)
assert os.path.exists(obs_path), obs_path
df_obs = pd.read_csv(obs_path).set_index('pid')
print("obs shape (before filter):", df_obs.shape)

n_before = len(df_obs)
df_obs = df_obs[
    (df_obs['up_refer_count'] >= reference_threshold)
    & (df_obs[citation_col] >= citation_threshold)
    & (df_obs['pub_year'] < pub_year_upper)
]
print(f"obs filter: {n_before} -> {len(df_obs)} papers "
      f"(up_refer_count >= {reference_threshold}, "
      f"{citation_col} >= {citation_threshold}, pub_year < {pub_year_upper})")
valid_pids = df_obs.index

# ==================================================
# 2. load every reshuffle iteration; stack e/f/g across iterations (id x iteration),
#    restricted to the surviving papers
# ==================================================
e_frames = []
f_frames = []
g_frames = []

for it in tqdm(iteration_list, desc="loading reshuffle iterations"):
    rand_path = os.path.join(
        load_dire,
        f'gen_V3_y{year_offset}_iter{it}_before_{upper_year}.csv'
    )
    assert os.path.exists(rand_path), rand_path

    df_rand = pd.read_csv(rand_path).set_index('pid')
    col_name = f'iter_{it}'
    # reindex to valid_pids so the random matrices only hold the filtered papers
    e_frames.append(df_rand[e_col].reindex(valid_pids).rename(col_name))
    f_frames.append(df_rand[f_col].reindex(valid_pids).rename(col_name))
    g_frames.append(df_rand[g_col].reindex(valid_pids).rename(col_name))

e_mat = pd.concat(e_frames, axis=1)
f_mat = pd.concat(f_frames, axis=1)
g_mat = pd.concat(g_frames, axis=1)
print("random matrix shape (id x iteration):", e_mat.shape)

# ==================================================
# 3. assemble output: obs e/f/g, obs counts, random mean, z-scores
#    (aligned on obs ids; pandas aligns the random series by id)
# ==================================================
res = pd.DataFrame(index=df_obs.index)

# obs e/f/g and obs counts (+ pub_year, so downstream needn't re-merge from gen_V3)
res['pub_year'] = df_obs['pub_year']
res['obs_e'] = df_obs[e_col]
res['obs_f'] = df_obs[f_col]
res['obs_g'] = df_obs[g_col]
res['up_refer_count'] = df_obs['up_refer_count']
res[citation_col] = df_obs[citation_col]

# z-score per metric:
#   z = (obs - mean) / std
#   - any (near-)zero std is first replaced with the smallest non-zero std among
#     the same-year papers for this index, so no paper is dropped for std == 0
#   - if (obs - mean) == 0 the z-score is 0
metrics = [('e', e_col, e_mat), ('f', f_col, f_mat), ('g', g_col, g_mat)]

pub_year = df_obs['pub_year']  # aligned to df_obs.index

# accumulator for the group dummy: an id has "full variance" only if NONE of its
# e/f/g random distributions had a (near-)zero std. This flags the rest.
has_zero_std_any = pd.Series(False, index=df_obs.index)

for key, col, mat in metrics:
    mean = mat.mean(axis=1).reindex(df_obs.index)
    std = mat.std(axis=1, ddof=std_ddof).reindex(df_obs.index)
    numerator = df_obs[col] - mean

    # which ids have a (near-)zero random std for THIS index
    is_zero_std = std < std_zero_threshold          # True only for finite < thr
    has_zero_std_any = has_zero_std_any | is_zero_std

    # --- replace (near-)zero std with the smallest NON-zero std among same-year
    #     papers for THIS index, so these papers are kept (not removed) ---
    # keep only the genuine (non-zero) stds, then take the per-year minimum
    nonzero_std = std.where(~is_zero_std)           # near-zero -> NaN (excluded)
    per_year_min = nonzero_std.groupby(pub_year).transform('min')
    # fallback if an entire year has no non-zero std at all: use global min std
    global_min = nonzero_std.min()
    fill_values = per_year_min.fillna(global_min)
    n_filled = int((is_zero_std & fill_values.notna()).sum())
    std = std.mask(is_zero_std, fill_values)        # assign the per-year min

    res[f'random_mean_{key}'] = mean
    res[f'random_std_{key}'] = std

    z = numerator / std.replace(0, np.nan)
    z = z.mask(numerator == 0, 0.0)  # obs == mean -> 0 regardless of std
    res[f'{key}_zscore'] = z

    n_num_zero = int((numerator == 0).sum())
    n_std_zero = int(is_zero_std.sum())
    # "normal" z-scores: a genuine value, i.e. obs-mean != 0 AND std != 0
    n_normal = int(((numerator != 0) & (~is_zero_std)).sum())
    print(f"[{key}] ids with obs-mean == 0 : {n_num_zero}")
    print(f"[{key}] ids with std < {std_zero_threshold:g} : {n_std_zero} "
          f"(replaced with same-year min non-zero std: {n_filled})")
    print(f"[{key}] ids with normal z-score (obs-mean != 0 & std != 0) : {n_normal}")

# --- group dummy: 1 = genuine random variance in ALL of e/f/g (no backfill needed);
#     0 = at least one index had a (near-)zero std and was backfilled above.
#     Mode 1 ("backfill") = use the whole table; the backfilled z-scores apply to the
#     dummy==0 rows. Mode 2 ("drop") = subset to has_full_variance == 1, i.e. only the
#     ids whose e/f/g all had variance (their z-scores never used a backfilled std).
res['has_full_variance'] = (~has_zero_std_any).astype(int)
print("has_full_variance == 1 (variance in all e/f/g; no backfill):",
      int(res['has_full_variance'].sum()))
print("has_full_variance == 0 (>=1 index had zero std, backfilled):",
      int((res['has_full_variance'] == 0).sum()))

# ==================================================
# 4. drop ids with any undefined z-score. After the per-year std backfill above,
#    std == 0 no longer causes drops; the only remaining NaNs are ids missing
#    from the random iterations (mean/std undefined).
# ==================================================
n_before = len(res)
res = res.dropna(subset=['e_zscore', 'f_zscore', 'g_zscore'])
n_after = len(res)
print(f"deleted {n_before - n_after} ids with undefined z-score "
      f"(missing in random iterations)")

res = res.reset_index()  # id back as a column
print("output shape:", res.shape)

# ==================================================
# 5. save
# ==================================================
save_path = os.path.join(
    load_dire,
    f'efg_zscore_y{year_offset}_before_{upper_year}_{iter_tag}.csv'
)
res.to_csv(save_path, index=False)
print("saved to:", save_path)
print("finished")
