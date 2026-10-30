"""
Semantic Scholar (S2AG) version of the OpenAlex
visualize_efg_obs_vs_single_iter_control_venue.py.

Visualize the observed E/F/G shares against one or more venue-controlled reshuffle
iterations (the null model), as trends over publication year.

This script reads the raw merged tables directly. BOTH the observed and the reshuffled
tables come from the venue-controlled pipeline (get_efg_reshuffle_control_venue.py ->
get_efg_reshuffle_s2_merge_control_venue.py):

  - observed (venue-filtered) run            : gen_V3_y{year_offset}_obs_valid_venue_before_{upper_year}.csv
                                               (get_efg_reshuffle_control_venue.py with target_iteration=None)
  - each venue-controlled reshuffle iteration: gen_V3_y{year_offset}_iter{it}_before_{upper_year}.csv

It produces ONE figure with three sub-panels (F, E, G). Each panel has:
  - one Observed (obs) e/f/g share line  (the venue-filtered real network)
  - one Random e/f/g share line PER iteration in iteration_list (each its own line)

Both observed and reshuffled are computed on the SAME upper_year edge subset (the
observed None-run is restricted to source year < upper_year), so the comparison is
like-for-like. Every share is also computed over the SAME set of papers: obs filter
-> valid ids -> each iteration's random values are taken for exactly those ids, then
averaged per pub_year. (An id missing from a given iteration is dropped for that
iteration only.)

Input  : META_DIRE/efg_computation_control_venue/
           gen_V3_y{year_offset}_obs_valid_venue_before_{upper_year}.csv   (observed)
           gen_V3_y{year_offset}_iter{it}_before_{upper_year}.csv          (one per iteration)
           (both from get_efg_reshuffle_s2_merge_control_venue.py)
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

# utils.py is in the parent (semantic_scholar/) directory; make it importable regardless of CWD
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

# Both observed and reshuffled E/F/G live in the venue-controlled tree: the observed
# run is get_efg_reshuffle_control_venue.py with target_iteration=None (the venue-
# filtered network, restricted to the same upper_year edge subset as the reshuffles),
# merged by get_efg_reshuffle_s2_merge_control_venue.py. So obs and rand are the SAME
# directory and differ only by file suffix.
obs_load_dire = os.path.join(meta_dire, 'efg_computation_control_venue')
rand_load_dire = os.path.join(meta_dire, 'efg_computation_control_venue')

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
# 1. load observed merge, then filter on reference / citation / pub_year
# ==================================================
# the observed venue-filtered merge (get_efg_reshuffle_control_venue.py target_iteration=None)
obs_path = os.path.join(
    obs_load_dire,
    f'gen_V3_y{year_offset}_obs_valid_venue_before_{upper_year}.csv'
)
assert os.path.exists(obs_path), obs_path
df_obs = pd.read_csv(obs_path).set_index('pid')
print("obs shape (before filter):", df_obs.shape)

# dynamic pub_year cutoff so the year_offset-year citation window fits within the data
pub_year_upper = 2024 - year_offset + 1
n_before = len(df_obs)
df_obs = df_obs[
    (df_obs['up_refer_count'] >= reference_threshold)
    & (df_obs[citation_col] >= citation_threshold)
    & (df_obs['pub_year'].notna())
    & (df_obs['pub_year'] < pub_year_upper)
].copy()
df_obs['pub_year'] = df_obs['pub_year'].astype(int)
print(f"obs filter: {n_before} -> {len(df_obs)} papers "
      f"(up_refer_count >= {reference_threshold}, "
      f"{citation_col} >= {citation_threshold}, pub_year < {pub_year_upper})")
valid_pids = df_obs.index

# ==================================================
# 2. observed share per pub_year (mean across the surviving papers) — drawn once
# ==================================================
obs_agg = df_obs.groupby('pub_year').agg(
    obs_e=(e_col, 'mean'),
    obs_f=(f_col, 'mean'),
    obs_g=(g_col, 'mean'),
    n=(e_col, 'count'),
).reset_index().sort_values('pub_year')

# ==================================================
# 3. each reshuffle iteration: take its e/f/g for the SAME ids, then average per year
# ==================================================
iter2agg = {}  # target_iteration -> per-pub_year mean random shares
for it in iteration_list:
    rand_path = os.path.join(
        rand_load_dire,
        f'gen_V3_y{year_offset}_iter{it}_before_{upper_year}.csv'
    )
    assert os.path.exists(rand_path), rand_path
    df_rand = pd.read_csv(rand_path).set_index('pid')

    cur = pd.DataFrame(index=valid_pids)
    cur['pub_year'] = df_obs['pub_year']
    cur['rand_e'] = df_rand[e_col].reindex(valid_pids)
    cur['rand_f'] = df_rand[f_col].reindex(valid_pids)
    cur['rand_g'] = df_rand[g_col].reindex(valid_pids)

    n_before = len(cur)
    cur = cur.dropna(subset=['rand_e', 'rand_f', 'rand_g'])
    print(f"iter {it}: dropped {n_before - len(cur)} ids missing from this iteration; "
          f"{len(cur)} papers remain")

    iter2agg[it] = cur.groupby('pub_year').agg(
        rand_e=('rand_e', 'mean'),
        rand_f=('rand_f', 'mean'),
        rand_g=('rand_g', 'mean'),
    ).reset_index().sort_values('pub_year')

# x-axis starts at 1945; decade ticks from there up to the last available year
x_start = 1945
last_year = int(obs_agg['pub_year'].max())
xticks = list(range(x_start, last_year + 1, 10))

# ==================================================
# 4. FIGURE — obs vs every random iteration, three sub-panels (F, E, G)
# ==================================================
# order F, E, G to match the repo's Foundation / Extension / Generalization order
panels = [
    ('Foundation (F)', 'obs_f', 'rand_f'),
    ('Extension (E)', 'obs_e', 'rand_e'),
    ('Generalization (G)', 'obs_g', 'rand_g'),
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
    # one random line per iteration
    for it in iteration_list:
        agg_it = iter2agg[it]
        ax.plot(agg_it['pub_year'], agg_it[rand_key],
                color=iter_colors[it], linestyle='--', linewidth=1.2, alpha=0.8,
                label=f'Random (iter {it})')
    ax.set_title(title, fontsize=18)
    ax.set_xlabel('Publication Year', fontsize=14)
    if i == 0:
        ax.set_ylabel('Average share', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(x_start, last_year)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks])
# single shared legend (avoid repeating the long iteration list in every panel)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', fontsize=10, framealpha=0.9)
plt.tight_layout(rect=[0, 0, 0.9, 1])  # leave room on the right for the legend
out = os.path.join(
    figure_dire_save,
    f'efg_obs_vs_{iter_tag}_ay{year_offset}_ct{citation_threshold}_rt{reference_threshold}_before_{upper_year}.svg'
)
plt.savefig(out, format='svg', transparent=True)
plt.close()
print("[figure saved]", out)

print("finished")
