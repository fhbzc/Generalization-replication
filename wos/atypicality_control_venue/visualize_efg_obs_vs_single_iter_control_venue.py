"""
WoS version of the OpenAlex visualize_efg_obs_vs_single_iter_control_venue.py.

Visualize the observed E/F/G shares against one or more venue-controlled reshuffle
iterations (the null model), as trends over publication year.

This script reads the raw merged tables directly. The observed table is the PLAIN
E/F/G run from get_efg.py -> get_efg_s2_merge.py (the real network, no venue filter,
no upper_year edge restriction); the reshuffled tables come from the venue-controlled
pipeline (get_efg_reshuffle_control_venue.py -> get_efg_reshuffle_s2_merge_control_venue.py):

  - observed run                             : gen_V3_y{year_offset}.csv
                                               (get_efg.py -> get_efg_s2_merge.py)
  - each venue-controlled reshuffle iteration: gen_V3_y{year_offset}_iter{it}_before_{upper_year}.csv

It produces ONE figure with three sub-panels (F, E, G). Each panel has:
  - one Observed (obs) e/f/g share line  (the plain real network)
  - one Random e/f/g share line PER iteration in iteration_list (each its own line)

Every share is computed over the SAME set of papers, and that set is the INTERSECTION
of papers that pass the citation + reference thresholds in EVERY sample -- the observed
run AND each random iteration in iteration_list. A pid is kept only if it satisfies
up_refer_count >= reference_threshold and citation_count >= citation_threshold in the
observed table and in all iterations (and is present in all of them). The observed
pub_year window is applied once (pub_year is intrinsic to the paper, unchanged by
reshuffling). The surviving pids' e/f/g are then averaged per pub_year for obs and for
each iteration.

The per-pub_year aggregates (observed + each iteration's random means) are cached to CSV
under ./figures/atypicality_control_venue/cached_data/. On a redraw, if both caches exist
they are loaded directly and the big gen_V3 CSVs are not read -- only the figure is rebuilt.
Delete the cache CSVs (or set force_recompute=True) to recompute.

Input  : META_DIRE/efg_computation/
           gen_V3_y{year_offset}.csv                                              (observed, from get_efg_s2_merge.py)
         META_DIRE/efg_computation_control_venue/
           gen_V3_y{year_offset}_iter{it}_before_{upper_year}.csv                 (one per iteration,
           from get_efg_reshuffle_s2_merge_control_venue.py)
Cache  : ./figures/atypicality_control_venue/cached_data/
           obs_agg_{param_suffix}.csv
           rand_agg_{param_suffix}.csv
Output : ./figures/atypicality_control_venue/
           efg_obs_vs_{iter_tag}_ay{ay}_ct{ct}_rt{rt}_before_{upper_year}.svg
           where iter_tag (e.g. "iter0-1-2") reflects the exact iteration_list used.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless / cluster-safe
import matplotlib.pyplot as plt  # noqa: E402

# utils.py is in the parent (wos/) directory; make it importable regardless of CWD
# (matches get_efg_reshuffle_control_venue.py).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
from utils import META_DIRE  # noqa: E402

# repo plotting convention; fall back silently if the font is unavailable (e.g. Linux)
try:
    plt.rcParams['font.family'] = 'Gill Sans MT'
except Exception:
    pass
# Gill Sans MT lacks the Unicode minus glyph (U+2212); use the ASCII hyphen instead.
plt.rcParams['axes.unicode_minus'] = False

meta_dire = META_DIRE
print("meta_dire", meta_dire)

# match get_efg_reshuffle_s2_merge_control_venue.py: read the '_with_review' E/F/G tree
# (and tag the figure) when True so the two variants never collide.
use_with_review = False
data_suffix = '_with_review' if use_with_review else ''
print("use_with_review", use_with_review)
print("data_suffix", repr(data_suffix))

# Observed E/F/G is the PLAIN get_efg.py run (merged by get_efg_s2_merge.py), living in
# efg_computation/ with data_suffix on the FILENAME. The reshuffled iterations live in
# the venue-controlled tree (efg_computation_control_venue{data_suffix}/, suffix on the
# DIRECTORY). So obs and rand come from DIFFERENT directories.
obs_load_dire = os.path.join(meta_dire, 'efg_computation')
rand_load_dire = os.path.join(meta_dire, f'efg_computation_control_venue{data_suffix}')

# ---- must match the merges ----
year_offset = 5
upper_year = 2025  # must match get_efg_reshuffle_s2_merge_control_venue.py
# the venue-controlled reshuffle iteration(s) to plot against the observed data; each
# iteration is drawn as its own random line in every panel.
iteration_list = [0]
iteration_list = sorted(iteration_list)  # keep iterations in a deterministic order
# tag for the saved filename so it reflects WHICH iterations were drawn.
iter_tag = 'iter' + '-'.join(str(it) for it in iteration_list)
print("year_offset", year_offset)
print("upper_year", upper_year)
print("iteration_list", iteration_list)
print("iter_tag", iter_tag)

# ---- filtering (matches get_efg_zscore_control_venue.py) ----
reference_threshold = 1
citation_threshold = 5
print("reference_threshold", reference_threshold)
print("citation_threshold", citation_threshold)

e_col = f'e_y{year_offset}'
f_col = f'f_y{year_offset}'
g_col = f'g_y{year_offset}'
citation_col = f'citation_count_y{year_offset}'

figure_dire_save = os.path.join(BASE_DIR, 'figures', 'atypicality_control_venue')
os.makedirs(figure_dire_save, exist_ok=True)

# ==================================================
# 0. cache: the per-pub_year aggregates (obs_agg + per-iteration random aggregates) are
#    cached to CSV. On a redraw, if BOTH caches exist they are loaded directly and the big
#    gen_V3 CSVs are not touched -- only the figure is rebuilt. The cache key carries every
#    knob that affects the aggregates (iterations, offset, thresholds, upper_year, suffix),
#    so different settings keep their own cache. Set force_recompute=True (or delete the
#    cache CSVs) to recompute.
# ==================================================
force_recompute = False

cache_dire = os.path.join(figure_dire_save, 'cached_data')
os.makedirs(cache_dire, exist_ok=True)
# 'isect' marks the cross-sample intersection pid set (obs AND all iterations must pass
# the thresholds), so caches from the old "obs-only filter" logic are never reused.
param_suffix = (f'isect_{iter_tag}_ay{year_offset}_ct{citation_threshold}_'
                f'rt{reference_threshold}_before_{upper_year}{data_suffix}')
obs_cache_path = os.path.join(cache_dire, f'obs_agg_{param_suffix}.csv')
rand_cache_path = os.path.join(cache_dire, f'rand_agg_{param_suffix}.csv')

cache_ready = (not force_recompute
               and os.path.exists(obs_cache_path)
               and os.path.exists(rand_cache_path))

if cache_ready:
    obs_agg = pd.read_csv(obs_cache_path)
    rand_long = pd.read_csv(rand_cache_path)
    # reconstruct {iteration -> per-pub_year mean random shares} from the long-form cache
    iter2agg = {int(it): g.drop(columns='iteration').reset_index(drop=True)
                for it, g in rand_long.groupby('iteration')}
    print("loaded cache:", obs_cache_path, "and", rand_cache_path,
          "| iterations:", sorted(iter2agg))
else:
    print("cache not found (or force_recompute) -> computing from gen_V3 CSVs ...")

    # ==================================================
    # 1. load observed merge, apply the pub_year window + thresholds. This yields the
    #    OBSERVED-passing candidates; the final pid set is narrowed further below to the
    #    cross-sample intersection.
    # ==================================================
    # the observed plain merge (get_efg.py -> get_efg_s2_merge.py)
    obs_path = os.path.join(
        obs_load_dire,
        f'gen_V3_y{year_offset}{data_suffix}.csv'
    )
    assert os.path.exists(obs_path), obs_path
    # pid kept as str: WoS ids are digit-like and must not be coerced to int (leading zeros).
    df_obs = pd.read_csv(obs_path, dtype={'pid': str}).set_index('pid')
    print("obs shape (before filter):", df_obs.shape)

    # dynamic pub_year cutoff so the year_offset-year citation window fits within the data.
    # pub_year is intrinsic to the paper (unchanged by reshuffling), so it is applied once.
    pub_year_upper = 2024 - year_offset + 1
    n_before = len(df_obs)
    df_obs = df_obs[
        (df_obs['up_refer_count'] >= reference_threshold)
        & (df_obs[citation_col] >= citation_threshold)
        & (df_obs['pub_year'].notna())
        & (df_obs['pub_year'] < pub_year_upper)
    ].copy()
    df_obs['pub_year'] = df_obs['pub_year'].astype(int)
    print(f"obs filter: {n_before} -> {len(df_obs)} papers pass in the observed run "
          f"(up_refer_count >= {reference_threshold}, "
          f"{citation_col} >= {citation_threshold}, pub_year < {pub_year_upper})")
    obs_pass_pids = df_obs.index  # observed candidates (before cross-sample intersection)

    # per-sample count of pids satisfying the conditions; 'obs' first, then each iteration.
    sample_pass_counts = [('obs', len(obs_pass_pids))]

    # ==================================================
    # 2. each reshuffle iteration: independently count the pids that satisfy the SAME
    #    conditions (thresholds + pub_year window) in that iteration's own table. The
    #    e/f/g columns are stashed (restricted to obs candidates) so the CSV is read once.
    # ==================================================
    iter2sub = {}        # iteration -> per-pid e/f/g restricted to obs candidates
    pass_pid_sets = [set(obs_pass_pids)]  # threshold-passing pid sets across all samples
    for it in iteration_list:
        rand_path = os.path.join(
            rand_load_dire,
            f'gen_V3_y{year_offset}_iter{it}_before_{upper_year}.csv'
        )
        assert os.path.exists(rand_path), rand_path
        df_rand = pd.read_csv(rand_path, dtype={'pid': str}).set_index('pid')

        # this iteration's OWN satisfying set (same conditions as the observed filter)
        keep_mask = ((df_rand['up_refer_count'] >= reference_threshold)
                     & (df_rand[citation_col] >= citation_threshold)
                     & (df_rand['pub_year'].notna())
                     & (df_rand['pub_year'] < pub_year_upper))
        keep_pids = df_rand.index[keep_mask]
        pass_pid_sets.append(set(keep_pids))
        sample_pass_counts.append((f'iter{it}', len(keep_pids)))
        # stash only the e/f/g shares we need downstream (final set is a subset of obs cands)
        iter2sub[it] = df_rand.reindex(obs_pass_pids)[[e_col, f_col, g_col]]
        print(f"iter {it}: {len(keep_pids)} papers pass the thresholds in this iteration")

    # ==================================================
    # 3. final pid set = papers passing the thresholds in EVERY sample (obs + all iters)
    # ==================================================
    intersection = set.intersection(*pass_pid_sets)
    # preserve the observed order for determinism
    valid_pids = obs_pass_pids[obs_pass_pids.isin(intersection)]

    # ---- per-sample vs final summary ----
    print("-" * 60)
    print("pids satisfying the conditions per sample:")
    for name, cnt in sample_pass_counts:
        print(f"    {name:>8}: {cnt}")
    print(f"final (intersection of all {len(sample_pass_counts)} samples): {len(valid_pids)}")
    print("-" * 60)

    df_obs = df_obs.loc[valid_pids]

    # ==================================================
    # 4. observed share per pub_year (mean across the surviving papers) — drawn once
    # ==================================================
    obs_agg = df_obs.groupby('pub_year').agg(
        obs_e=(e_col, 'mean'),
        obs_f=(f_col, 'mean'),
        obs_g=(g_col, 'mean'),
        n=(e_col, 'count'),
    ).reset_index().sort_values('pub_year')

    # ==================================================
    # 5. each iteration's random share per pub_year, over the SAME final pid set
    # ==================================================
    iter2agg = {}  # target_iteration -> per-pub_year mean random shares
    for it in iteration_list:
        sub = iter2sub[it].loc[valid_pids]  # all present & passing by construction
        cur = pd.DataFrame(index=valid_pids)
        cur['pub_year'] = df_obs['pub_year']
        cur['rand_e'] = sub[e_col]
        cur['rand_f'] = sub[f_col]
        cur['rand_g'] = sub[g_col]

        iter2agg[it] = cur.groupby('pub_year').agg(
            rand_e=('rand_e', 'mean'),
            rand_f=('rand_f', 'mean'),
            rand_g=('rand_g', 'mean'),
        ).reset_index().sort_values('pub_year')

    # ---- write the cache: obs_agg + long-form per-iteration random aggregates ----
    obs_agg.to_csv(obs_cache_path, index=False)
    rand_long = pd.concat(
        [agg.assign(iteration=it) for it, agg in iter2agg.items()],
        ignore_index=True,
    )
    rand_long.to_csv(rand_cache_path, index=False)
    print("wrote cache:", obs_cache_path, "and", rand_cache_path)

# x-axis starts at 1945; decade ticks from there up to the last available year
x_start = 1945
last_year = int(obs_agg['pub_year'].max())
xticks = list(range(x_start, last_year + 1, 10))

# ==================================================
# 4. FIGURE — obs vs every random iteration, three sub-panels (F, E, G)
# ==================================================
# order F, E, G to match the repo's Foundation / Extension / Generalization order
panels = [
    ('Foundation', 'obs_f', 'rand_f'),
    ('Extension', 'obs_e', 'rand_e'),
    ('Generalization', 'obs_g', 'rand_g'),
]
obs_color = 'black'
# distinct color per iteration
cmap = plt.get_cmap('viridis')
if len(iteration_list) > 1:
    iter_colors = {it: cmap(i / (len(iteration_list) - 1))
                   for i, it in enumerate(iteration_list)}
else:
    iter_colors = {iteration_list[0]: 'tab:red'}

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
for i, (title, obs_key, rand_key) in enumerate(panels):
    ax = axes[i]
    # observed line (drawn once, on top in black)
    ax.plot(obs_agg['pub_year'], obs_agg[obs_key],
            color=obs_color, linewidth=2.5, label='Observed', zorder=len(iteration_list) + 1)
    # one random line per iteration; with a single iteration the legend just says 'Random'
    for it in iteration_list:
        agg_it = iter2agg[it]
        rand_label = 'Random' if len(iteration_list) == 1 else f'Random (iter {it})'
        ax.plot(agg_it['pub_year'], agg_it[rand_key],
                color=iter_colors[it], linestyle='--', linewidth=1.2, alpha=0.8,
                label=rand_label)
    ax.set_title(title, fontsize=18)
    ax.set_xlabel('Publication Year', fontsize=18)
    if i == 0:
        ax.set_ylabel('Average index', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(x_start, last_year)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks])
plt.tight_layout()
out = os.path.join(
    figure_dire_save,
    f'efg_obs_vs_{iter_tag}_ay{year_offset}_ct{citation_threshold}_rt{reference_threshold}_before_{upper_year}{data_suffix}.svg'
)
plt.savefig(out, format='svg', transparent=True)
plt.close()
print("[figure saved]", out)

print("finished")
