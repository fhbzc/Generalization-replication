"""
Semantic Scholar (S2AG) version of the OpenAlex visualize_efg_zscore_control_venue.py.

Visualize the VENUE-CONTROLLED E/F/G atypicality results produced by
get_efg_zscore_control_venue.py.

Two figures, both as trends over publication year (mean across papers per year):

  1. efg_zscore_trend  : the change of the e/f/g z-scores over pub_year.
                         One panel, three lines (E, F, G z-score).

  2. efg_obs_vs_random : three sub-panels (F, E, G). Each panel has two lines:
                         - mean random e/f/g  (random_mean_*, the null model)
                         - mean observed e/f/g (obs_*, the real network)

The z-score file carries pub_year and the `has_full_variance` dummy (from
get_efg_zscore_control_venue.py), so no merge is needed here.

variance_mode selects which group of papers to plot, matching the two modes encoded
by get_efg_zscore_control_venue.py's has_full_variance dummy:
  - 'backfill' : use ALL papers (zero-std indices were backfilled with the same-year
                 min non-zero std). The whole table.
  - 'drop'     : keep ONLY papers whose e/f/g all had genuine random variance, i.e.
                 the has_full_variance == 1 subset (no backfilled std used).
The mode is encoded in the cache/figure filenames so the two variants never collide.

The per-year aggregate is cached to CSV (efg_zscore_plotcache_*) keyed by the full
specification (year_offset / thresholds / iter_tag / variance_mode). If that cache
already exists, the script reloads it and redraws directly, skipping the raw file.

Input  : META_DIRE/efg_computation_control_venue/
           efg_zscore_y{year_offset}_before_{upper_year}_{iter_tag}.csv  (get_efg_zscore_control_venue.py)
         (skipped when the plot cache below already exists)
Output : ./figures/atypicality_control_venue/
           efg_zscore_plotcache_ay{ay}_ct{ct}_rt{rt}_{iter_tag}_{variance_mode}.csv  (yearly aggregate, reused on re-run)
           efg_zscore_trend_ay{ay}_ct{ct}_rt{rt}_{iter_tag}_{variance_mode}.svg
           efg_obs_vs_random_ay{ay}_ct{ct}_rt{rt}_{iter_tag}_{variance_mode}.svg
         where iter_tag (e.g. "iter0-1-2") reflects the exact iteration_list used.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless / cluster-safe
import matplotlib.pyplot as plt  # noqa: E402

# utils.py is in the parent (semantic_scholar/) directory; make it importable regardless of CWD
# (matches get_efg_zscore_control_venue.py).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
from utils import META_DIRE  # noqa: E402

# repo plotting convention; fall back silently if the font is unavailable (e.g. Linux)
try:
    plt.rcParams['font.family'] = 'Gill Sans MT'
except Exception:
    pass
# Gill Sans MT lacks the Unicode minus glyph (U+2212), so negative tick labels
# (z-scores go below 0) render blank. Use the ASCII hyphen as the minus sign.
plt.rcParams['axes.unicode_minus'] = False

meta_dire = META_DIRE
print("meta_dire", meta_dire)

load_dire = os.path.join(meta_dire, 'efg_computation_control_venue')

# ---- must match get_efg_zscore_control_venue.py ----
year_offset = 5
upper_year = 2025
# same as get_efg_zscore_control_venue.py: the null-model iterations used.
iteration_list = [0, 1, 2]
iteration_list = sorted(iteration_list)
n_iter = len(iteration_list)
# tag for saved filenames so they reflect WHICH iterations were used, not just the
# count (e.g. [0,2,5] and [0,1,2] both give niter3 but are different null models).
iter_tag = 'iter' + '-'.join(str(it) for it in iteration_list)
print("year_offset", year_offset)
print("upper_year", upper_year)
print("iteration_list", iteration_list)
print("n_iter", n_iter)
print("iter_tag", iter_tag)

# ---- variance mode: which group of papers to plot (see module docstring) ----
#   'backfill' -> all papers (zero-std indices backfilled)
#   'drop'     -> only has_full_variance == 1 papers (genuine variance in all e/f/g)
variance_mode = 'drop'
assert variance_mode in ('backfill', 'drop'), variance_mode
print("variance_mode", variance_mode)

# ---- filtering (matches visualize_efg_trend.py main trend) ----
reference_threshold = 1
citation_threshold = 5
print("reference_threshold", reference_threshold)
print("citation_threshold", citation_threshold)

citation_col = f'citation_count_y{year_offset}'

figure_dire_save = os.path.join(BASE_DIR, 'figures', 'atypicality_control_venue')
os.makedirs(figure_dire_save, exist_ok=True)

# the yearly-aggregate plotting cache: same specification (year_offset / thresholds
# / iterations / variance_mode) -> same file, so re-runs draw straight from it.
cache_path = os.path.join(
    figure_dire_save,
    f'efg_zscore_plotcache_ay{year_offset}_ct{citation_threshold}_rt{reference_threshold}_{iter_tag}_{variance_mode}.csv'
)

if os.path.exists(cache_path):
    # ==================================================
    # 1'. fast path: reuse the cached yearly aggregate, skip the raw files
    # ==================================================
    agg = pd.read_csv(cache_path)
    print("[cache loaded]", cache_path, "| shape:", agg.shape)
else:
    # ==================================================
    # 1. load z-score output (pub_year + has_full_variance are carried in it)
    # ==================================================
    zscore_path = os.path.join(
        load_dire,
        f'efg_zscore_y{year_offset}_before_{upper_year}_{iter_tag}.csv'
    )
    assert os.path.exists(zscore_path), zscore_path
    df = pd.read_csv(zscore_path)
    print("zscore shape:", df.shape)

    # ==================================================
    # 2. variance-mode subset, then filter (same logic as the main e/f/g trend)
    # ==================================================
    if variance_mode == 'drop':
        assert 'has_full_variance' in df.columns, (
            "z-score file has no 'has_full_variance' column; re-run "
            "get_efg_zscore_control_venue.py to add the group dummy.")
        n_pre = len(df)
        df = df[df['has_full_variance'] == 1]
        print(f"variance_mode=drop: {n_pre} -> {len(df)} papers "
              f"(kept has_full_variance == 1)")

    n_before = len(df)
    df = df[df['pub_year'].notna()]
    # dynamic pub_year cutoff so the citation window fits within the data
    df = df[df['pub_year'] < (2024 - year_offset + 1)]
    df = df[df['up_refer_count'] >= reference_threshold]
    df = df[df[citation_col] >= citation_threshold]
    df = df.copy()
    df['pub_year'] = df['pub_year'].astype(int)
    print(f"filtered {n_before} -> {len(df)} papers")

    # ==================================================
    # 3. aggregate per pub_year (mean across papers)
    # ==================================================
    agg = df.groupby('pub_year').agg(
        e_zscore=('e_zscore', 'mean'),
        f_zscore=('f_zscore', 'mean'),
        g_zscore=('g_zscore', 'mean'),
        obs_e=('obs_e', 'mean'),
        obs_f=('obs_f', 'mean'),
        obs_g=('obs_g', 'mean'),
        random_mean_e=('random_mean_e', 'mean'),
        random_mean_f=('random_mean_f', 'mean'),
        random_mean_g=('random_mean_g', 'mean'),
        n=('pid', 'count'),
    ).reset_index().sort_values('pub_year')
    print("years:", agg['pub_year'].min(), "->", agg['pub_year'].max())

    # save the plotting cache so the next run with the same spec skips the raw files
    agg.to_csv(cache_path, index=False)
    print("[cache saved]", cache_path)

# x-axis starts at 1945; decade ticks from there up to the last available year
x_start = 1945
last_year = int(agg['pub_year'].max())
xticks = list(range(x_start, last_year + 1, 10))

# ==================================================
# 4. FIGURE 1 — e/f/g z-score trend (one panel, three lines)
# ==================================================
# order F, E, G to match the repo's Foundation / Extension / Generalization order
zscore_lines = [
    ('Foundation (F)', 'f_zscore', 'tab:blue'),
    ('Extension (E)', 'e_zscore', 'tab:green'),
    ('Generalization (G)', 'g_zscore', 'tab:red'),
]

fig, ax = plt.subplots(figsize=(7, 5))
for label, col, color in zscore_lines:
    ax.plot(agg['pub_year'], agg[col], label=label, color=color, linewidth=2)
ax.axhline(0, color='gray', linewidth=1, linestyle=':')
ax.set_title(f'E / F / G z-score over time ({variance_mode})', fontsize=18)
ax.set_xlabel('Publication Year', fontsize=14)
ax.set_ylabel('Mean z-score', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(x_start, last_year)
ax.set_xticks(xticks)
ax.set_xticklabels([str(x) for x in xticks])
ax.legend(loc='best', fontsize=12)
plt.tight_layout()
out1 = os.path.join(
    figure_dire_save,
    f'efg_zscore_trend_ay{year_offset}_ct{citation_threshold}_rt{reference_threshold}_{iter_tag}_{variance_mode}.svg'
)
plt.savefig(out1, format='svg', transparent=True)
plt.close()
print("[figure saved]", out1)

# ==================================================
# 5. FIGURE 2 — obs vs random, three sub-panels (F, E, G)
#    dual y-axes: Observed on the left axis, Random (null) on the right axis,
#    so each gets its own range (obs and null shares differ in magnitude).
# ==================================================
panels = [
    ('Foundation (F)', 'obs_f', 'random_mean_f'),
    ('Extension (E)', 'obs_e', 'random_mean_e'),
    ('Generalization (G)', 'obs_g', 'random_mean_g'),
]
obs_color = 'black'
rand_color = 'tab:red'

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
for i, (title, obs_col, rand_col) in enumerate(panels):
    ax = axes[i]                       # left axis: observed
    l_obs, = ax.plot(agg['pub_year'], agg[obs_col],
                     color=obs_color, linewidth=2, label='Observed')
    ax.set_title(title, fontsize=18)
    ax.set_xlabel('Publication Year', fontsize=14)
    if i == 0:
        ax.set_ylabel('Observed average share', fontsize=14, color=obs_color)
    ax.tick_params(axis='y', labelcolor=obs_color, labelsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.spines['top'].set_visible(False)

    ax_r = ax.twinx()                  # right axis: random (null), own range
    l_rand, = ax_r.plot(agg['pub_year'], agg[rand_col],
                        color=rand_color, linestyle='--', linewidth=1.8,
                        label='Random (null)')
    if i == len(panels) - 1:
        ax_r.set_ylabel('Random (null) average share', fontsize=14, color=rand_color)
    ax_r.tick_params(axis='y', labelcolor=rand_color, labelsize=12)
    ax_r.spines['top'].set_visible(False)

    ax.set_xlim(x_start, last_year)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks])
    ax.legend(handles=[l_obs, l_rand], loc='best', fontsize=12)
plt.tight_layout()
out2 = os.path.join(
    figure_dire_save,
    f'efg_obs_vs_random_ay{year_offset}_ct{citation_threshold}_rt{reference_threshold}_{iter_tag}_{variance_mode}.svg'
)
plt.savefig(out2, format='svg', transparent=True)
plt.close()
print("[figure saved]", out2)

print("finished")
