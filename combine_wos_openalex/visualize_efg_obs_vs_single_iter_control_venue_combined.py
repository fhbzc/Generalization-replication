"""
Combined OpenAlex + WoS version of visualize_efg_obs_vs_single_iter_control_venue.py.

This stitches the two per-dataset figures into ONE figure with a 1 x 3 grid of panels
(one per index); both datasets and both conditions are overlaid in each panel:

      Foundation        Extension        Generalization
  +----------------+----------------+----------------+
  |                |                |                |
  +----------------+----------------+----------------+

Within every panel the three visual channels encode:
  colour    -> index      : Foundation=red, Extension=orange, Generalization=blue
  linestyle -> dataset     : OpenAlex=solid, WoS=dashed
  linewidth -> condition   : Observed=bold, Randomized=thin
A single style legend (upper-right) is drawn on the first panel only. Each panel shows the
observed E/F/G share against one venue-controlled reshuffle iteration (the null model), as
a trend over publication year -- exactly the same content as the two source scripts:

  - openalex/atypicality_control_venue/visualize_efg_obs_vs_single_iter_control_venue.py
  - wos/atypicality_control_venue/visualize_efg_obs_vs_single_iter_control_venue.py

The per-dataset computation (load merged tables, apply the pub_year window + citation /
reference thresholds, intersect the threshold-passing pids across the observed run and
every iteration, then average e/f/g per pub_year) is identical to those scripts and is
factored into compute_dataset() below.

This combined script does NOT read the standalone scripts' caches. On a first run it
computes both datasets from the raw gen_V3 CSVs (on the server) and writes its OWN cache
under ./figures/atypicality_control_venue/cached_data/ (one obs_agg / rand_agg pair per
dataset, namespaced by dataset name). On a redraw those local caches are loaded directly
and the big gen_V3 CSVs are not touched. Delete the cache CSVs (or set force_recompute=
True) to recompute from the raw tables.

The combined figure uses a SINGLE shared x-label ("Publication Year") and a SINGLE shared
y-label ("Average index") via fig.supxlabel / fig.supylabel; the per-panel axis labels are
removed. Datasets and conditions are distinguished by line style / weight (see the legend).

At the paper's settings (year_offset=1, upper_year=2025, iteration_list=[0],
reference_threshold=1, citation_threshold=2, wos_data_suffix='') this figure IS 'Fig. 3b'
(is_fig3b): the figure and the two per-dataset aggregate caches are then named after the
panel instead of carrying the parameter stub. Every other setting keeps the descriptive
names. A cache written under the pre-Fig. 3b names is still picked up and re-saved under the
new ones, so the rename never forces a recompute from the gen_V3 tables.

Cache  : ./figures/atypicality_control_venue/cached_data/
           at the deliverable setting (is_fig3b), per dataset in {openalex, wos}:
             Fig. 3b_obs_agg_{dataset}.csv, Fig. 3b_rand_agg_{dataset}.csv
           otherwise (param_suffix = isect_{iter_tag}_ay{ay}_ct{ct}_rt{rt}_before_{upper_year},
           plus the WoS data_suffix on the WoS side):
             obs_agg_{dataset}_{param_suffix}.csv, rand_agg_{dataset}_{param_suffix}.csv
Output : ./figures/atypicality_control_venue/
           Fig. 3b.svg at the deliverable setting, else
           efg_obs_vs_{iter_tag}_ay{ay}_ct{ct}_rt{rt}_before_{upper_year}_combined.svg
"""

import os
import sys
import importlib.util

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless / cluster-safe
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# locations: this script lives in .../replication_cleaned_NHB_rebuild/combine_wos_openalex/
# the two datasets live in sibling dirs .../openalex/ and .../wos/ (each has its own utils.py)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REBUILD_DIR = os.path.dirname(BASE_DIR)
OPENALEX_DIR = os.path.join(REBUILD_DIR, 'openalex')
WOS_DIR = os.path.join(REBUILD_DIR, 'wos')


def load_meta_dire(parent_dir):
    """Import META_DIRE from the utils.py living in a dataset's parent directory."""
    utils_path = os.path.join(parent_dir, 'utils.py')
    spec = importlib.util.spec_from_file_location(
        f'utils_{os.path.basename(parent_dir)}', utils_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.META_DIRE


META_DIRE_OA = load_meta_dire(OPENALEX_DIR)
META_DIRE_WOS = load_meta_dire(WOS_DIR)
print("META_DIRE (openalex)", META_DIRE_OA)
print("META_DIRE (wos)     ", META_DIRE_WOS)

# repo plotting convention; fall back silently if the font is unavailable (e.g. Linux)
try:
    plt.rcParams['font.family'] = 'Gill Sans MT'
except Exception:
    pass
# Gill Sans MT lacks the Unicode minus glyph (U+2212); use the ASCII hyphen instead.
plt.rcParams['axes.unicode_minus'] = False

# ---- shared parameters (must match the merges / both source scripts) ----
year_offset = 1
upper_year = 2025
iteration_list = [0]
iteration_list = sorted(iteration_list)
iter_tag = 'iter' + '-'.join(str(it) for it in iteration_list)

# ---- filtering ----
reference_threshold = 1
citation_threshold = 2

force_recompute = False

# WoS side: data_suffix='' matches the standalone script's default (use_with_review=False).
# Defined here (rather than next to the WoS compute call) so the deliverable gate below can
# see it -- it changes which WoS tables are read, hence what the panels show.
wos_data_suffix = ''

print("year_offset", year_offset)
print("upper_year", upper_year)
print("iteration_list", iteration_list)
print("reference_threshold", reference_threshold)
print("citation_threshold", citation_threshold)
print("wos_data_suffix", repr(wos_data_suffix))

# ---- deliverable naming ----------------------------------------------------------------
# This combined 1x3 obs-vs-randomized figure at the paper's settings IS Fig. 3b, so the
# figure AND the per-dataset aggregate caches behind it are named after the panel; every
# other setting keeps the descriptive parameter names. The gate lists every parameter that
# changes what the panels show, so no other run can land on this name.
FIG_NAME = 'Fig. 3b'
is_fig3b = (year_offset == 1 and upper_year == 2025 and iteration_list == [0]
            and reference_threshold == 1 and citation_threshold == 2
            and wos_data_suffix == '')
print("is_fig3b", is_fig3b, ("-> %s" % FIG_NAME) if is_fig3b else "")

e_col = f'e_y{year_offset}'
f_col = f'f_y{year_offset}'
g_col = f'g_y{year_offset}'
citation_col = f'citation_count_y{year_offset}'


def compute_dataset(name, obs_load_dire, rand_load_dire, obs_path, pid_as_str,
                    cache_dire, param_suffix):
    """Replicate the per-dataset obs vs reshuffle computation (with the same caching as the
    standalone scripts) and return (obs_agg, iter2agg)."""
    print("=" * 70)
    print(f"[{name}] computing obs vs reshuffle aggregates")
    print(f"[{name}] obs_path     :", obs_path)
    print(f"[{name}] rand_load_dire:", rand_load_dire)

    os.makedirs(cache_dire, exist_ok=True)
    # namespace the cache files by dataset name: openalex and wos share the same
    # param_suffix, so without the name they would collide in this one local cache dir.
    # At the deliverable setting the caches are named after the panel instead (still one
    # pair per dataset, since the figure overlays both).
    if is_fig3b:
        obs_cache_path = os.path.join(cache_dire, f'{FIG_NAME}_obs_agg_{name}.csv')
        rand_cache_path = os.path.join(cache_dire, f'{FIG_NAME}_rand_agg_{name}.csv')
    else:
        obs_cache_path = os.path.join(cache_dire, f'obs_agg_{name}_{param_suffix}.csv')
        rand_cache_path = os.path.join(cache_dire, f'rand_agg_{name}_{param_suffix}.csv')
    # a run from before the Fig. 3b naming wrote the SAME aggregates under the descriptive
    # names; reuse them rather than re-reading the multi-GB gen_V3 tables (only the file name
    # changed), and re-save under the deliverable names so later runs load them directly.
    legacy_obs_cache_path = os.path.join(cache_dire, f'obs_agg_{name}_{param_suffix}.csv')
    legacy_rand_cache_path = os.path.join(cache_dire, f'rand_agg_{name}_{param_suffix}.csv')

    load_obs, load_rand = obs_cache_path, rand_cache_path
    if not (os.path.exists(load_obs) and os.path.exists(load_rand)):
        load_obs, load_rand = legacy_obs_cache_path, legacy_rand_cache_path
    cache_ready = (not force_recompute
                   and os.path.exists(load_obs)
                   and os.path.exists(load_rand))

    if cache_ready:
        obs_agg = pd.read_csv(load_obs)
        rand_long = pd.read_csv(load_rand)
        iter2agg = {int(it): grp.drop(columns='iteration').reset_index(drop=True)
                    for it, grp in rand_long.groupby('iteration')}
        print(f"[{name}] loaded cache:", load_obs, "and", load_rand,
              "| iterations:", sorted(iter2agg))
        if load_obs != obs_cache_path:
            obs_agg.to_csv(obs_cache_path, index=False)
            rand_long.to_csv(rand_cache_path, index=False)
            print(f"[{name}] cache re-saved under the deliverable names:",
                  obs_cache_path, "and", rand_cache_path)
        return obs_agg, iter2agg

    print(f"[{name}] cache not found (or force_recompute) -> computing from gen_V3 CSVs ...")

    read_kwargs = {'dtype': {'pid': str}} if pid_as_str else {}

    # ---- 1. observed merge: pub_year window + thresholds ----
    assert os.path.exists(obs_path), obs_path
    df_obs = pd.read_csv(obs_path, **read_kwargs).set_index('pid')
    print(f"[{name}] obs shape (before filter):", df_obs.shape)

    pub_year_upper = 2024 - year_offset + 1
    n_before = len(df_obs)
    df_obs = df_obs[
        (df_obs['up_refer_count'] >= reference_threshold)
        & (df_obs[citation_col] >= citation_threshold)
        & (df_obs['pub_year'].notna())
        & (df_obs['pub_year'] < pub_year_upper)
    ].copy()
    df_obs['pub_year'] = df_obs['pub_year'].astype(int)
    print(f"[{name}] obs filter: {n_before} -> {len(df_obs)} papers pass in the observed run")
    obs_pass_pids = df_obs.index

    sample_pass_counts = [('obs', len(obs_pass_pids))]

    # ---- 2. each reshuffle iteration: its own threshold-passing set + stashed e/f/g ----
    iter2sub = {}
    pass_pid_sets = [set(obs_pass_pids)]
    for it in iteration_list:
        rand_path = os.path.join(
            rand_load_dire,
            f'gen_V3_y{year_offset}_iter{it}_before_{upper_year}.csv'
        )
        assert os.path.exists(rand_path), rand_path
        df_rand = pd.read_csv(rand_path, **read_kwargs).set_index('pid')

        keep_mask = ((df_rand['up_refer_count'] >= reference_threshold)
                     & (df_rand[citation_col] >= citation_threshold)
                     & (df_rand['pub_year'].notna())
                     & (df_rand['pub_year'] < pub_year_upper))
        keep_pids = df_rand.index[keep_mask]
        pass_pid_sets.append(set(keep_pids))
        sample_pass_counts.append((f'iter{it}', len(keep_pids)))
        iter2sub[it] = df_rand.reindex(obs_pass_pids)[[e_col, f_col, g_col]]
        print(f"[{name}] iter {it}: {len(keep_pids)} papers pass the thresholds")

    # ---- 3. final pid set = intersection across obs + all iterations ----
    intersection = set.intersection(*pass_pid_sets)
    valid_pids = obs_pass_pids[obs_pass_pids.isin(intersection)]

    print("-" * 60)
    print(f"[{name}] pids satisfying the conditions per sample:")
    for sname, cnt in sample_pass_counts:
        print(f"    {sname:>8}: {cnt}")
    print(f"[{name}] final (intersection of all {len(sample_pass_counts)} samples): {len(valid_pids)}")
    print("-" * 60)

    df_obs = df_obs.loc[valid_pids]

    # ---- 4. observed share per pub_year ----
    obs_agg = df_obs.groupby('pub_year').agg(
        obs_e=(e_col, 'mean'),
        obs_f=(f_col, 'mean'),
        obs_g=(g_col, 'mean'),
        n=(e_col, 'count'),
    ).reset_index().sort_values('pub_year')

    # ---- 5. each iteration's random share per pub_year over the SAME final pid set ----
    iter2agg = {}
    for it in iteration_list:
        sub = iter2sub[it].loc[valid_pids]
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

    # ---- cache ----
    obs_agg.to_csv(obs_cache_path, index=False)
    rand_long = pd.concat(
        [agg.assign(iteration=it) for it, agg in iter2agg.items()],
        ignore_index=True,
    )
    rand_long.to_csv(rand_cache_path, index=False)
    print(f"[{name}] wrote cache:", obs_cache_path, "and", rand_cache_path)
    return obs_agg, iter2agg


# ---------------------------------------------------------------------------
# this combined script's OWN output + cache locations (independent of the standalone
# scripts). Both datasets cache into the SAME local cached_data dir; filenames are
# namespaced by dataset name in compute_dataset().
# ---------------------------------------------------------------------------
figure_dire_save = os.path.join(BASE_DIR, 'figures', 'atypicality_control_venue')
os.makedirs(figure_dire_save, exist_ok=True)
local_cache_dire = os.path.join(figure_dire_save, 'cached_data')

# ---------------------------------------------------------------------------
# compute both datasets from their merged tables on the server (own local cache)
# ---------------------------------------------------------------------------
# OpenAlex: obs and rand BOTH live in the venue-controlled tree (pid is int -> default read).
oa_param_suffix = (f'isect_{iter_tag}_ay{year_offset}_ct{citation_threshold}_'
                   f'rt{reference_threshold}_before_{upper_year}')
oa_obs_agg, oa_iter2agg = compute_dataset(
    name='openalex',
    obs_load_dire=os.path.join(META_DIRE_OA, 'efg_computation_control_venue'),
    rand_load_dire=os.path.join(META_DIRE_OA, 'efg_computation_control_venue'),
    obs_path=os.path.join(
        META_DIRE_OA, 'efg_computation_control_venue',
        f'gen_V3_y{year_offset}_obs_valid_primary_location_before_{upper_year}.csv'),
    pid_as_str=False,
    cache_dire=local_cache_dire,
    param_suffix=oa_param_suffix,
)

# WoS: obs is the PLAIN get_efg run (efg_computation/), rand is the venue-controlled tree.
# (wos_data_suffix is set in the parameter block at the top, where the Fig. 3b gate reads it.)
wos_param_suffix = (f'isect_{iter_tag}_ay{year_offset}_ct{citation_threshold}_'
                    f'rt{reference_threshold}_before_{upper_year}{wos_data_suffix}')
wos_obs_agg, wos_iter2agg = compute_dataset(
    name='wos',
    obs_load_dire=os.path.join(META_DIRE_WOS, 'efg_computation'),
    rand_load_dire=os.path.join(META_DIRE_WOS, f'efg_computation_control_venue{wos_data_suffix}'),
    obs_path=os.path.join(META_DIRE_WOS, 'efg_computation',
                          f'gen_V3_y{year_offset}{wos_data_suffix}.csv'),
    pid_as_str=True,
    cache_dire=local_cache_dire,
    param_suffix=wos_param_suffix,
)

# ---------------------------------------------------------------------------
# report how many papers from each dataset are finally used in the plot. obs_agg['n'] is the
# per-pub_year count over the FINAL (intersection) pid set, so its sum is the total number of
# papers actually plotted for that dataset (works on both a fresh compute and a cache hit).
# ---------------------------------------------------------------------------
oa_n_plot = int(oa_obs_agg['n'].sum())
wos_n_plot = int(wos_obs_agg['n'].sum())
print("=" * 70)
print("papers passing the filter (final intersection set) and drawn in the figure:")
print(f"    OpenAlex : {oa_n_plot}")
print(f"    WoS      : {wos_n_plot}")
# the two datasets are disjoint paper universes, so the combined figure covers their sum
print(f"    TOTAL    : {oa_n_plot + wos_n_plot}")
print("=" * 70)

# ---------------------------------------------------------------------------
# FIGURE — 1 row x 3 panels (Foundation, Extension, Generalization). Each panel
# overlays BOTH datasets and BOTH conditions, encoded as (cf. plot_longitudional_change.py):
#   colour     -> index      : Foundation=red, Extension=orange, Generalization=blue
#   linestyle  -> dataset     : OpenAlex=solid, WoS=dashed
#   linewidth  -> condition   : Observed=bold, Randomized=thin
# A single style legend (upper-right) is drawn on the first panel only.
# ---------------------------------------------------------------------------
from matplotlib.lines import Line2D  # noqa: E402

# Inset horizontal position (axes-fraction x of the inset's left edge) AND bottom-edge
# anchor (in each panel's main-axis DATA y-units), both per
# (year_offset, citation_threshold, reference_threshold). Every inset has the SAME size;
# only its position differs so it sits in that panel's empty band. The y-axis RANGE and the
# curves' layout both shift with the thresholds, so each parameter set needs its own tuning.
#
# INSET_X0_MAP: 0.07 = hug the LEFT y-axis; ~0.60 = sit on the RIGHT half of the panel.
# INSET_Y0_MAP: bottom-edge anchor per index [Foundation, Extension, Generalization].
INSET_X0_MAP = {
    (5, 5, 1): 0.07,          # left (hug the left y-axis)
    (1, 2, 1): 1.0 - 0.34,    # flush against the RIGHT edge (inset_w=0.34; no right margin,
                              # the inset's ticks are on its left/bottom so nothing is clipped)
    (1, 2, 0): 1.0 - 0.34,    # flush right as well (see the INSET_Y0_MAP note below)
}
INSET_Y0_MAP = {
    # y5 / ct5 / rt1: original tuning, insets on the LEFT
    (5, 5, 1): [0.10, 0.20, 0.45],
    # y1 / ct2 / rt1: insets on the RIGHT (x0=0.60 -> inset x-window is years ~1992-2018).
    #   Axes are Foundation~[0,0.09], Extension~[0.18,0.95], Generalization~[0.13,0.91].
    #   Within that recent-year window the curves have decayed/settled, leaving clean bands:
    #   * Foundation: obs 0.021-0.031 (low), rand ~0.007-0.010 -> whole upper-right (0.031-
    #     0.090) is empty; the legend sits at the very top-right and the flush-right inset
    #     tucks just below it (0.039), both above the low obs curves.
    #   * Extension: obs 0.77-0.91 (top), rand 0.29-0.36 (bottom) -> empty middle band; 0.40
    #     sits the inset a bit lower in that band (still clear of the randomized curves).
    #   * Generalization: obs 0.16-0.31 (bottom), rand 0.73-0.80 (top) -> empty middle band;
    #     0.45 centers the inset there.
    (1, 2, 1): [0.039, 0.40, 0.45],
    # y1 / ct2 / rt0: insets on the RIGHT (x0=0.66 -> inset x-window is years ~1992-2018).
    #   Anchors measured off the rendered SVG; axes are Foundation~[0.002,0.095],
    #   Extension~[0.053,0.844], Generalization~[0.129,0.937]. In the right-hand window the
    #   free vertical bands (and the anchors that centre the 0.30-high inset in them, biased
    #   slightly UP so the inset's own x tick labels have room below) are:
    #   * Foundation: legend on top (down to ~0.068 once shrunk to fontsize 10), obs curves
    #     0.023-0.033 below -> band 0.033-0.068; 0.037 leaves ~10px for the tick labels.
    #   * Extension: obs 0.57-0.71 (top), rand 0.17-0.25 (bottom) -> band 0.248-0.569; 0.30.
    #   * Generalization: rand 0.74-0.82 (top), obs 0.26-0.41 (bottom) -> band 0.406-0.743; 0.46.
    (1, 2, 0): [0.037, 0.30, 0.46],
}
# Where/how to draw the single shared style legend: dict(ax, loc, bbox, fontsize). The legend
# always lives on the LEFT-most (Foundation) panel. For the y1 layout the obs curves have
# decayed to ~0.022-0.031 on the right, so the strip above them is empty -> the legend sits at
# the very top-right and the flush-right inset tucks just beneath it, at a slightly smaller
# font. y5 keeps the roomy upper-right.
DEFAULT_LEGEND = dict(ax=0, loc='upper right', bbox=None, fontsize=13)
LEGEND_PLACEMENT = {
    # y1 / ct2 / rt1: enlarged to 14 together with the other text (see FONT_MAP). Foundation's
    # free right-hand strip runs from the inset's top edge (0.039 + 0.30*ylim) up to 1.0, i.e.
    # ~27% of the panel; four entries at fontsize 14 need ~22% of it, so it still clears the
    # inset. Going much beyond 14 here starts to overlap the flush-right inset.
    (1, 2, 1): dict(ax=0, loc='upper left', bbox=(0.655, 1.0), fontsize=14),
    # y1 / ct2 / rt0: same story -- at the default fontsize 13 the legend eats the top ~34% of
    # Foundation's right-hand strip and the flush-right inset no longer fits beneath it, so
    # shrink to 10 and pin it to the very top of that strip.
    (1, 2, 0): dict(ax=0, loc='upper left', bbox=(0.655, 1.0), fontsize=10),
}
inset_y0_key = (year_offset, citation_threshold, reference_threshold)
if inset_y0_key in INSET_Y0_MAP:
    inset_y0_list = INSET_Y0_MAP[inset_y0_key]
else:
    inset_y0_list = [0.10, 0.20, 0.45]
    print(f"[warn] no inset anchors tuned for {inset_y0_key}; using default {inset_y0_list} "
          f"(add an INSET_Y0_MAP entry and eyeball the empty band per panel)")
inset_x0 = INSET_X0_MAP.get(inset_y0_key, 0.07)
print("inset_y0_key", inset_y0_key, "-> inset_x0", inset_x0, "inset_y0_list", inset_y0_list)

# ---- text sizes, per (year_offset, citation_threshold, reference_threshold) ----
# The default set is the original styling; y1/ct2/rt1 uses a scaled-up set (panel titles, tick
# labels and the two shared axis labels) so the figure stays legible at print size. supy_x is
# the figure-fraction x of the shared y-label (it has to move right as the label grows, else
# the rotated text runs off the left edge) and tight_rect reserves that left strip for it.
# inset_text is the "Change since {year}" caption inside the leftmost inset; it only nudges up
# a little (the inset itself keeps its size, so a big caption would collide with its curves).
DEFAULT_FONTS = dict(title=24, tick=16, axis_label=24, inset_text=10,
                     supy_x=0.01, tight_rect=None)
FONT_MAP = {
    (1, 2, 1): dict(title=30, tick=22, axis_label=30, inset_text=13, supy_x=0.013,
                    tight_rect=(0.018, 0.0, 1.0, 1.0)),
}
fonts = dict(DEFAULT_FONTS, **FONT_MAP.get(inset_y0_key, {}))
print("font sizes:", fonts)

# (title, obs column, rand column, colour, inset bottom anchor in DATA y-units of the main axis)
panels = [
    ('Foundation', 'obs_f', 'rand_f', 'red', inset_y0_list[0]),
    ('Extension', 'obs_e', 'rand_e', 'orange', inset_y0_list[1]),
    ('Generalization', 'obs_g', 'rand_g', 'blue', inset_y0_list[2]),
]
# uniform inset geometry (axes fractions); inset_x0 set per-config above (left vs right)
inset_w, inset_h = 0.34, 0.30
# (label, linestyle, obs_agg, iter2agg)
datasets = [
    ('OpenAlex', '-', oa_obs_agg, oa_iter2agg),
    ('WoS', '--', wos_obs_agg, wos_iter2agg),
]

it0 = iteration_list[0]
obs_lw = 2.8       # Observed  -> bold
rand_lw = 1.3      # Randomized -> thin
rand_color = 'gray'  # Randomized -> always gray (regardless of index)

x_start = 1945
last_year = max(int(oa_obs_agg['pub_year'].max()),
                int(wos_obs_agg['pub_year'].max()))
xticks = list(range(x_start, last_year + 1, 10))

# --- inset: rebase every series to its value at inset_start_year so observed and
# randomized start from the SAME y-coordinate (=0) there, then show the relative change
# in RAW units (value - value_at_start), not a ratio. ---
inset_start_year = 1990


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


fig, axes = plt.subplots(1, 3, figsize=(24, 6))
for c, (title, obs_key, rand_key, color, inset_y0) in enumerate(panels):
    ax = axes[c]
    for ds_label, ls, obs_agg, iter2agg in datasets:
        # observed (bold)
        ax.plot(obs_agg['pub_year'], obs_agg[obs_key],
                color=color, linestyle=ls, linewidth=obs_lw)
        # randomized (thin, always gray)
        agg_it = iter2agg[it0]
        ax.plot(agg_it['pub_year'], agg_it[rand_key],
                color=rand_color, linestyle=ls, linewidth=rand_lw, alpha=0.9)
    ax.set_title(title, fontsize=fonts['title'])
    ax.tick_params(axis='both', which='major', labelsize=fonts['tick'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(x_start, last_year)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks])

    # --- inset: relative change since inset_start_year (obs & rand rebased to 0 there) ---
    # hug the left y-axis; uniform size, bottom anchored at inset_y0 (main-axis DATA units)
    ymin, ymax = ax.get_ylim()
    yf0 = (inset_y0 - ymin) / (ymax - ymin)
    # An anchor tuned for a DIFFERENT parameter set can fall outside this panel's ylim (e.g.
    # the default 0.10 exceeds Foundation's ymax when ct=2), which would silently float the
    # inset above the axes and over the title. Clamp it inside and say so, loudly.
    if not (0.0 <= yf0 <= 1.0 - inset_h):
        clamped = min(max(yf0, 0.0), 1.0 - inset_h)
        print(f"[warn] {title}: inset anchor {inset_y0} -> axes fraction {yf0:.3f} is outside "
              f"this panel (ylim {ymin:.4f}..{ymax:.4f}); clamped to {clamped:.3f}. "
              f"Tune INSET_Y0_MAP[{inset_y0_key}] for this parameter set.")
        yf0 = clamped
    axins = ax.inset_axes([inset_x0, yf0, inset_w, inset_h])
    for ds_label, ls, obs_agg, iter2agg in datasets:
        xo, yo = relative_change(obs_agg, obs_key, inset_start_year)
        if xo is not None:
            axins.plot(xo, yo, color=color, linestyle=ls, linewidth=obs_lw * 0.7)
        xr, yr = relative_change(iter2agg[it0], rand_key, inset_start_year)
        if xr is not None:
            axins.plot(xr, yr, color=rand_color, linestyle=ls, linewidth=rand_lw)
    axins.axhline(0.0, color='lightgray', linewidth=0.8, zorder=0)
    axins.set_xlim(inset_start_year, last_year)
    # one x-tick every 10 years from inset_start_year (matches the main-axis 10-year cadence)
    inset_xticks = list(range(inset_start_year, last_year + 1, 10))
    axins.set_xticks(inset_xticks)
    axins.set_xticklabels([str(x) for x in inset_xticks])
    axins.tick_params(axis='both', labelsize=10)
    axins.spines['top'].set_visible(False)
    axins.spines['right'].set_visible(False)
    # label only on the leftmost (Foundation) panel, as text in the inset's lower-left corner
    if c == 0:
        axins.text(0.03, 0.04, f'Change since {inset_start_year}',
                   transform=axins.transAxes, fontsize=fonts['inset_text'],
                   ha='left', va='bottom')

# style legend on the first panel only (colour-neutral: it keys line style / weight,
# since colour already encodes the index via the panel title)
legend_handles = [
    Line2D([0], [0], color='black', linestyle='-', linewidth=obs_lw, label='Observed (OpenAlex)'),
    Line2D([0], [0], color=rand_color, linestyle='-', linewidth=rand_lw, label='Randomized (OpenAlex)'),
    Line2D([0], [0], color='black', linestyle='--', linewidth=obs_lw, label='Observed (WoS)'),
    Line2D([0], [0], color=rand_color, linestyle='--', linewidth=rand_lw, label='Randomized (WoS)'),
]
legend_spec = LEGEND_PLACEMENT.get(inset_y0_key, DEFAULT_LEGEND)
print("legend placement:", legend_spec)
axes[legend_spec['ax']].legend(
    handles=legend_handles, loc=legend_spec['loc'],
    bbox_to_anchor=legend_spec['bbox'], frameon=False, fontsize=legend_spec['fontsize'])

# a single shared x-label and a single shared y-label for the whole figure
supx_text = fig.supxlabel('Publication Year', fontsize=fonts['axis_label'])
supy_text = fig.supylabel('Average index', fontsize=fonts['axis_label'], x=fonts['supy_x'])

if fonts['tight_rect'] is None:
    fig.tight_layout()
else:
    fig.tight_layout(rect=fonts['tight_rect'])

# Centre both shared labels ON THE PANELS rather than on the figure: fig.supxlabel/supylabel
# default to x=0.5 / y=0.5 of the whole FIGURE, but the panel band is not centred there -- the
# left y-tick labels (and the strip reserved for the y-label) push the panels right, and the
# x-label strip pushes them up. Do this after tight_layout, when the axes are in final position.
_panel_boxes = [ax.get_position() for ax in axes]
_panel_xmid = (min(b.x0 for b in _panel_boxes) + max(b.x1 for b in _panel_boxes)) / 2.0
_panel_ymid = (min(b.y0 for b in _panel_boxes) + max(b.y1 for b in _panel_boxes)) / 2.0
supx_text.set_position((_panel_xmid, supx_text.get_position()[1]))
supx_text.set_horizontalalignment('center')
supy_text.set_position((fonts['supy_x'], _panel_ymid))
supy_text.set_verticalalignment('center')
print(f"shared labels centred on the panels: x-label at figure x={_panel_xmid:.3f}, "
      f"y-label at figure y={_panel_ymid:.3f} (x={fonts['supy_x']})")

out = os.path.join(
    figure_dire_save,
    f'{FIG_NAME}.svg' if is_fig3b else
    (f'efg_obs_vs_{iter_tag}_ay{year_offset}_ct{citation_threshold}_'
     f'rt{reference_threshold}_before_{upper_year}_combined.svg')
)
plt.savefig(out, format='svg', transparent=True)
plt.close()
print("[figure saved]", out)
print("finished")
