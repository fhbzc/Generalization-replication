"""
Visualize the observed E/F/G shares against one or more topic-controlled reshuffle null
models, as trends over publication year.

This is the topic-hierarchy twin of
atypicality_control_venue/visualize_efg_obs_vs_single_iter_control_venue.py. BOTH the
observed and the reshuffled tables come from the topic-controlled pipeline
(get_efg_reshuffle_control_topic.py -> get_efg_reshuffle_s2_merge_control_topic.py).

Unlike the venue twin (which overlays multiple reshuffle *iterations* of one network), each
observed network here is reshuffled only ONCE per control level -> the single iteration is 0.
The thing that actually varies between reshuffles is therefore the CONTROL LEVEL (which topic
level -- topic/subfield/field/domain -- was held fixed while citations were reshuffled). So
this script overlays one Random line PER CONTROL LEVEL, all at the same iteration:

  - observed (subfield-filtered) run     : gen_V3_y{year_offset}_obs_valid_primary_{FILTER_LEVEL}_before_{upper_year}.csv
                                           (get_efg_reshuffle_control_topic.py with target_iteration=None)
  - each control-level reshuffle (iter0) : gen_V3_y{year_offset}_control_{level}_iter{iteration}_before_{upper_year}.csv
                                           (one per level in control_level_list)

It produces ONE figure with three sub-panels (F, E, G). Each panel has:
  - one Observed (obs) e/f/g share line  (the subfield-filtered real network)
  - one Random e/f/g share line PER control level in control_level_list (each its own line)

Both observed and reshuffled are computed on the SAME upper_year edge subset (the observed
None-run is restricted to source year < upper_year), so the comparison is like-for-like.
Every sample's share is computed over the SAME set of papers -- the OVERLAP set: pids that
pass the citation + reference thresholds in the observed run AND in EVERY control-level
reshuffle simultaneously. Each sample is first filtered to its own passing pids, then all
samples are intersected down to this shared overlap set, and the obs/random e/f/g are
averaged per pub_year over exactly those pids. This guarantees obs and every control level
are built on an identical set of papers.

Input  : META_DIRE_2026/efg_computation_control_2026/
           gen_V3_y{year_offset}_obs_valid_primary_{FILTER_LEVEL}_before_{upper_year}.csv   (observed, shared)
           gen_V3_y{year_offset}_control_{level}_iter{iteration}_before_{upper_year}.csv     (one per control level)
           (both from get_efg_reshuffle_s2_merge_control_topic.py)
Cache  : ./figures/atypicality_control_topic/multi_level/cached_data/
           obs_agg_{param_suffix}.csv
           rand_agg_{param_suffix}.csv
Output : ./figures/atypicality_control_topic/multi_level/
           efg_obs_vs_{level_tag}_ay{ay}_ct{ct}_rt{rt}_before_{upper_year}.svg
           where level_tag (e.g. "levels_topic-subfield-field-domain") reflects the exact
           control_level_list used.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# utils.py is in openalex/, two levels up from this file
# (openalex/2026_version/atypicality_control_topic/<this file>).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OPENALEX_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.insert(0, OPENALEX_DIR)
from utils import META_DIRE_2026  # noqa: E402

# canonical order of the topic levels (coarsest last); used to keep control_level_list and
# the drawn/legend order deterministic regardless of how the list below is written.
LEVEL_ORDER = ['topic', 'subfield', 'field', 'domain']

# ---- which topic level(s) to control for -----------------------------------------------
# one Random line is drawn per entry; each must match a reshuffle produced by
# get_efg_reshuffle_control_topic.py (i.e. an existing _control_{level}_iter{iteration}_ file).
control_level_list = ['topic', 'subfield', 'field']
for _lv in control_level_list:
    assert _lv in LEVEL_ORDER, _lv
# de-duplicate while forcing canonical order for determinism (cache key + legend order).
control_level_list = [lv for lv in LEVEL_ORDER if lv in set(control_level_list)]
# tag for the saved filename / cache so it reflects WHICH control levels were drawn.
level_tag = 'levels_' + '-'.join(control_level_list)
print("control_level_list", control_level_list)
print("level_tag", level_tag)

# the Observed table comes from the single shared subfield-filtered network (labelled by
# FILTER_LEVEL, shared across all control levels); must match get_efg_reshuffle_control_topic.py.
FILTER_LEVEL = 'subfield'
print("FILTER_LEVEL", FILTER_LEVEL)

# repo plotting convention (see visualize_efg_zscore.py)
try:
    plt.rcParams['font.family'] = 'Gill Sans MT'
except Exception:
    pass
# Gill Sans MT lacks the Unicode minus glyph (U+2212); use the ASCII hyphen instead.
plt.rcParams['axes.unicode_minus'] = False

meta_dire = META_DIRE_2026
print("meta_dire", meta_dire)

# Both Observed and Random tables live in ONE fixed folder, efg_computation_control_2026
# (no level in the folder name). Observed is shared (labelled _obs_valid_primary_subfield);
# each Random table is level-specific (labelled _control_<level>_iter{n}).
obs_load_dire = os.path.join(meta_dire, 'efg_computation_control_2026')
rand_load_dire = os.path.join(meta_dire, 'efg_computation_control_2026')

# ---- must match the merges ----
year_offset = 5
upper_year = 2025  # must match get_efg_reshuffle_s2_merge_control_topic.py
# each observed network is reshuffled only ONCE per control level -> the single iteration is 0.
iteration = 0
print("year_offset", year_offset)
print("upper_year", upper_year)
print("iteration", iteration)

# ---- filtering (matches visualize_efg_zscore.py / get_efg_zscore.py) ----
reference_threshold = 1
citation_threshold = 5
print("reference_threshold", reference_threshold)
print("citation_threshold", citation_threshold)

e_col = f'e_y{year_offset}'
f_col = f'f_y{year_offset}'
g_col = f'g_y{year_offset}'
citation_col = f'citation_count_y{year_offset}'

# figures for the multi-control-level overlay go in their own folder so they never overwrite
# any single-level output.
figure_dire_save = os.path.join(BASE_DIR, 'figures', 'atypicality_control_topic', 'multi_level')
os.makedirs(figure_dire_save, exist_ok=True)

# ==================================================
# 0. cache: the per-pub_year aggregates (obs_agg + per-control-level random aggregates) are
#    cached to CSV. On a redraw, if BOTH caches exist they are loaded directly and the big
#    gen_V3 CSVs are not touched -- only the figure is rebuilt. The cache key carries every
#    knob that affects the aggregates (control levels, iteration, offset, thresholds,
#    upper_year), so different settings keep their own cache. Set force_recompute=True (or
#    delete the cache CSVs) to recompute.
# ==================================================
force_recompute = False

cache_dire = os.path.join(figure_dire_save, 'cached_data')
os.makedirs(cache_dire, exist_ok=True)
# 'isect' marks that all samples are intersected down to the OVERLAP set (pids valid in obs
# AND every control-level reshuffle) before aggregating, so caches from the old 'indep'
# independent-filtering logic are never reused.
param_suffix = (f'isect_{level_tag}_it{iteration}_ay{year_offset}_ct{citation_threshold}_'
                f'rt{reference_threshold}_before_{upper_year}')
obs_cache_path = os.path.join(cache_dire, f'obs_agg_{param_suffix}.csv')
rand_cache_path = os.path.join(cache_dire, f'rand_agg_{param_suffix}.csv')
# per-sample paper counts (obs + each control level's satisfying count + final intersection),
# cached so they can be reported even on a cache hit.
counts_cache_path = os.path.join(cache_dire, f'counts_{param_suffix}.csv')


def print_paper_counts(counts_df):
    """Print each network's own passing count plus the final 'overlap' (rows: 'obs',
    'ctrl_<level>', 'overlap'). Printed in BOTH the compute path and the cache-hit path so the
    number of valid pids after filtering is always reported. The figure is drawn over the
    'overlap' set -- pids valid in the observed run AND every randomized control level."""
    print("-" * 60)
    print("valid pids after filtering (per sample; figure uses the 'overlap' set):")
    for _, row in counts_df.iterrows():
        print(f"    {row['sample']:>14}: {int(row['count'])} valid pids")
    print("-" * 60)


cache_ready = (not force_recompute
               and os.path.exists(obs_cache_path)
               and os.path.exists(rand_cache_path)
               and os.path.exists(counts_cache_path))

if cache_ready:
    obs_agg = pd.read_csv(obs_cache_path)
    rand_long = pd.read_csv(rand_cache_path)
    # reconstruct {control_level -> per-pub_year mean random shares} from the long-form cache
    level2agg = {str(lv): g.drop(columns='control_level').reset_index(drop=True)
                 for lv, g in rand_long.groupby('control_level')}
    print("loaded cache:", obs_cache_path, "and", rand_cache_path,
          "| control levels:", sorted(level2agg))
    # report each network's paper count from the cached counts table
    print_paper_counts(pd.read_csv(counts_cache_path))
else:
    print("cache not found (or force_recompute) -> computing from gen_V3 CSVs ...")

    # ==================================================
    # 1. load observed merge, apply the pub_year window + thresholds. This yields the
    #    observed sample's OWN passing pids; each control level is filtered the same way
    #    below, then ALL samples are intersected down to the shared overlap set (section 3).
    # ==================================================
    # the observed subfield-filtered merge (get_efg_reshuffle_control_topic.py target_iteration=None)
    obs_path = os.path.join(
        obs_load_dire,
        f'gen_V3_y{year_offset}_obs_valid_primary_{FILTER_LEVEL}_before_{upper_year}.csv'
    )
    assert os.path.exists(obs_path), obs_path
    df_obs = pd.read_csv(obs_path).set_index('pid')
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

    # per-sample count of pids satisfying the conditions; 'obs' first, then each control level,
    # and finally the 'overlap' intersection appended in section 3.
    sample_pass_counts = [('obs', len(df_obs))]

    # ==================================================
    # 2. each control level's reshuffle (iter{iteration}): filter to that level's OWN
    #    threshold-passing pids (same conditions as the observed filter) and KEEP the filtered
    #    frame so all samples can be intersected to the shared overlap set in section 3.
    # ==================================================
    level2df = {}  # control_level -> its own threshold-passing frame (pre-overlap)
    for lv in control_level_list:
        rand_path = os.path.join(
            rand_load_dire,
            f'gen_V3_y{year_offset}_control_{lv}_iter{iteration}_before_{upper_year}.csv'
        )
        assert os.path.exists(rand_path), rand_path
        df_rand = pd.read_csv(rand_path).set_index('pid')

        # this control level's OWN satisfying set (same conditions as the observed filter)
        keep_mask = ((df_rand['up_refer_count'] >= reference_threshold)
                     & (df_rand[citation_col] >= citation_threshold)
                     & (df_rand['pub_year'].notna())
                     & (df_rand['pub_year'] < pub_year_upper))
        df_rand = df_rand[keep_mask].copy()
        df_rand['pub_year'] = df_rand['pub_year'].astype(int)
        sample_pass_counts.append((f'ctrl_{lv}', len(df_rand)))
        print(f"control level {lv}: {len(df_rand)} papers pass the thresholds in its reshuffle")
        level2df[lv] = df_rand

    # ==================================================
    # 3. OVERLAP: keep only pids valid in the observed run AND in EVERY control-level
    #    reshuffle. All samples (obs + each control level) are then restricted to this shared
    #    pid set, so the observed and every randomized line are computed over identical papers.
    # ==================================================
    overlap_pids = set(df_obs.index)
    for lv in control_level_list:
        overlap_pids &= set(level2df[lv].index)
    overlap_pids = pd.Index(sorted(overlap_pids))
    sample_pass_counts.append(('overlap', len(overlap_pids)))
    print(f"overlap: {len(overlap_pids)} pids valid in obs AND all "
          f"{len(control_level_list)} control levels ({'-'.join(control_level_list)})")
    df_obs = df_obs.loc[overlap_pids]

    # per-sample paper counts (also cached for cache-hit reporting)
    counts_df = pd.DataFrame(sample_pass_counts, columns=['sample', 'count'])
    print_paper_counts(counts_df)

    # ==================================================
    # 4. observed share per pub_year (mean across the shared overlap papers)
    # ==================================================
    obs_agg = df_obs.groupby('pub_year').agg(
        obs_e=(e_col, 'mean'),
        obs_f=(f_col, 'mean'),
        obs_g=(g_col, 'mean'),
        n=(e_col, 'count'),
    ).reset_index().sort_values('pub_year')

    # ==================================================
    # 5. each control level's share per pub_year over the SAME overlap papers, so obs and
    #    every random line are averaged across an identical set of pids.
    # ==================================================
    level2agg = {}  # control_level -> per-pub_year mean random shares
    for lv in control_level_list:
        df_rand = level2df[lv].loc[overlap_pids]
        level2agg[lv] = df_rand.groupby('pub_year').agg(
            rand_e=(e_col, 'mean'),
            rand_f=(f_col, 'mean'),
            rand_g=(g_col, 'mean'),
        ).reset_index().sort_values('pub_year')

    # ---- write the cache: obs_agg + long-form per-control-level random aggregates ----
    obs_agg.to_csv(obs_cache_path, index=False)
    rand_long = pd.concat(
        [agg.assign(control_level=lv) for lv, agg in level2agg.items()],
        ignore_index=True,
    )
    rand_long.to_csv(rand_cache_path, index=False)
    counts_df.to_csv(counts_cache_path, index=False)
    print("wrote cache:", obs_cache_path, ",", rand_cache_path, "and", counts_cache_path)

# x-axis starts at 1945; decade ticks from there up to the last available year
x_start = 1945
last_year = int(obs_agg['pub_year'].max())
xticks = list(range(x_start, last_year + 1, 10))

# ==================================================
# 4. FIGURE — obs vs every control-level random, three sub-panels (F, E, G)
# ==================================================
# order F, E, G to match the repo's Foundation / Extension / Generalization order.
# Each panel's OBSERVED line takes its own colour (Foundation=red, Extension=orange,
# Generalization=blue); the last element is the inset bottom anchor in main-axis DATA
# y-units (see the inset block below).
#
# Inset bottom-edge anchors per (year_offset, citation_threshold, reference_threshold): the
# axis range and the curves shift with the thresholds, so each parameter set needs its own
# tuning. All insets are flush-right (inset_x0 below); the anchor only sets the vertical spot.
INSET_Y0_MAP = {
    # y1 / ct2 / rt1 (axes ~ Foundation[-0.003,0.064], Extension[-0.03,0.80], Gen[0.17,1.03]):
    #   * Foundation: obs 0.021-0.028 & random <=0.008 on the right -> the strip above obs is
    #     empty but short; the legend sits at the very top and the inset tucks just below it
    #     (0.031), still above the obs curve.
    #   * Extension: obs 0.61-0.75 (top), random <=0.25 (bottom) -> empty middle band; 0.30
    #     centers the inset.
    #   * Generalization: obs <=0.36 (bottom), random >=0.75 (top) -> empty middle band; 0.40.
    (1, 2, 1): [0.031, 0.30, 0.40],
}
inset_y0_key = (year_offset, citation_threshold, reference_threshold)
inset_y0_list = INSET_Y0_MAP.get(inset_y0_key, [0.045, 0.30, 0.40])
print("inset_y0_key", inset_y0_key, "-> inset_y0_list", inset_y0_list)

panels = [
    ('Foundation', 'obs_f', 'rand_f', 'red', inset_y0_list[0]),
    ('Extension', 'obs_e', 'rand_e', 'orange', inset_y0_list[1]),
    ('Generalization', 'obs_g', 'rand_g', 'blue', inset_y0_list[2]),
]

# Where/how to draw the per-figure legend (only on the leftmost / Foundation panel):
# dict(loc, bbox, fontsize). For y1 the inset sits on Foundation's flush-right strip, so the
# legend goes at the very top-right (above the inset) at a compact font; otherwise upper-right.
DEFAULT_LEGEND = dict(loc='upper right', bbox=None, fontsize=11)
LEGEND_PLACEMENT = {
    (1, 2, 1): dict(loc='upper left', bbox=(0.66, 1.0), fontsize=9),
}
legend_spec = LEGEND_PLACEMENT.get(inset_y0_key, DEFAULT_LEGEND)
print("legend_spec", legend_spec)
# stable GRAY shade per control level (fixed mapping so a level keeps its shade across runs
# regardless of which subset is drawn); coarser levels get lighter grays. Matplotlib reads
# the string floats as gray levels ('0'=black .. '1'=white). Fall back to a mid-gray ramp.
level_palette = {
    'topic': '0.15',
    'subfield': '0.40',
    'field': '0.60',
    'domain': '0.75',
}
level_colors = {}
for i, lv in enumerate(control_level_list):
    if lv in level_palette:
        level_colors[lv] = level_palette[lv]
    else:
        denom = max(len(control_level_list) - 1, 1)
        level_colors[lv] = str(round(0.30 + 0.45 * (i / denom), 2))  # 0.30..0.75 gray ramp

obs_lw = 2.5   # observed -> bold
rand_lw = 1.2  # random   -> thin

# --- inset: rebase every series to its value at inset_start_year (so obs & each control
# level start from the SAME y-coordinate = 0 there), then show the change in RAW units
# (value - value_at_start), not a ratio. Every inset sits on the RIGHT of its panel and has
# the SAME size; only its vertical position differs, anchored (bottom edge) at the per-panel
# data value in `panels`. ---
inset_start_year = 1990
inset_x0, inset_w, inset_h = 0.66, 0.34, 0.30  # uniform geometry (axes fractions); flush right
# (x0 + w = 1.0: the inset's y-ticks are on its LEFT, so no tick room is needed on the right)


def relative_change(df, val_col, start_year):
    """Return (years, value - value_at_start) for years >= start_year, shifted so the
    series equals 0 at (the row nearest to) start_year. None if unavailable."""
    d = df[['pub_year', val_col]].dropna().sort_values('pub_year')
    d = d[d['pub_year'] >= start_year]
    if d.empty:
        return None, None
    base_idx = (d['pub_year'] - start_year).abs().idxmin()
    base = d.loc[base_idx, val_col]
    return d['pub_year'].values, (d[val_col] - base).values


fig, axes = plt.subplots(1, 3, figsize=(24, 6), sharex=True)
for i, (title, obs_key, rand_key, obs_color, inset_y0) in enumerate(panels):
    ax = axes[i]
    # observed line (drawn once, on top, coloured per index)
    ax.plot(obs_agg['pub_year'], obs_agg[obs_key],
            color=obs_color, linewidth=obs_lw, label='Observed', zorder=len(control_level_list) + 1)
    # one random line per control level; the legend labels each line by its control level
    # (e.g. "Randomized (by subfield)"), so different control levels are visibly different lines.
    for lv in control_level_list:
        agg_lv = level2agg[lv]
        rand_label = f'Randomized (by {lv})'
        ax.plot(agg_lv['pub_year'], agg_lv[rand_key],
                color=level_colors[lv], linestyle='--', linewidth=rand_lw, alpha=0.85,
                label=rand_label)
    ax.set_title(title, fontsize=24)
    if i == 0:
        ax.legend(loc=legend_spec['loc'], bbox_to_anchor=legend_spec['bbox'],
                  fontsize=legend_spec['fontsize'], frameon=False)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(x_start, last_year)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks])

    # --- inset: change since inset_start_year (obs + every control level, rebased to 0) ---
    # right side of the panel; uniform size, bottom anchored at inset_y0 (main-axis DATA units)
    ymin, ymax = ax.get_ylim()
    yf0 = (inset_y0 - ymin) / (ymax - ymin)
    axins = ax.inset_axes([inset_x0, yf0, inset_w, inset_h])
    # observed (coloured per index)
    xo, yo = relative_change(obs_agg, obs_key, inset_start_year)
    if xo is not None:
        axins.plot(xo, yo, color=obs_color, linewidth=obs_lw * 0.7)
    # one random line per control level
    for lv in control_level_list:
        xr, yr = relative_change(level2agg[lv], rand_key, inset_start_year)
        if xr is not None:
            axins.plot(xr, yr, color=level_colors[lv], linestyle='--', linewidth=rand_lw)
    axins.axhline(0.0, color='lightgray', linewidth=0.8, zorder=0)
    axins.set_xlim(inset_start_year, last_year)
    # one x-tick every 10 years from inset_start_year (matches the main-axis 10-year cadence)
    inset_xticks = list(range(inset_start_year, last_year + 1, 10))
    axins.set_xticks(inset_xticks)
    axins.set_xticklabels([str(x) for x in inset_xticks])
    axins.tick_params(axis='both', labelsize=9)
    axins.spines['top'].set_visible(False)
    axins.spines['right'].set_visible(False)
    # label only on the leftmost (Foundation) panel, as text in the inset's lower-left corner
    if i == 0:
        axins.text(0.03, 0.04, f'Change since {inset_start_year}',
                   transform=axins.transAxes, fontsize=10, ha='left', va='bottom')

# single shared x/y labels (figure-level, like the combined figure) so the panels — and
# therefore the insets — are the same size as that figure's.
fig.supxlabel('Publication Year', fontsize=24)
fig.supylabel('Average index', fontsize=24, x=0.01)
fig.tight_layout()
out = os.path.join(
    figure_dire_save,
    f'efg_obs_vs_{level_tag}_ay{year_offset}_ct{citation_threshold}_rt{reference_threshold}_before_{upper_year}.svg'
)
plt.savefig(out, format='svg', transparent=True)
plt.close()
print("[figure saved]", out)

print("finished")
