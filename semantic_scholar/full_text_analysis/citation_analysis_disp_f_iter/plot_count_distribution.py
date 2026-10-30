"""
Standalone add-on HEATMAPS (does NOT touch / share files with the main pipeline).

From build_count_records.py's records (each citation carries up_count, down_count and a
{mention_position: frequency} map), draw HEATMAPS (./figures/citation_count/heatmap/):
x-axis = up_count, y-axis = down_count; one heatmap per metric (Frequency + each section), each
cell coloured by the mean of that metric over the citations with that (up_count, down_count).
Cells with fewer than MIN_CELL citations are left blank.

"Frequency" and the per-section ratio are defined exactly as analyze_refer_mention_position.py:
per citation, total = sum of its mention counts over ALL regions; the section ratio is that
region's count / total. The mean is taken over the citations in a cell.

The per-cell sums/counts are cached to an .npz (keyed by REQUIRE_REFS + MAX_HEATMAP), so redraws
(e.g. tweaking MIN_CELL / colours) reuse it instead of re-scanning every record; delete the cache
to force a refresh. The cache ALSO stores a bootstrap 95% CI of each cell's mean for every metric
(lo__/hi__ arrays); these are NOT drawn on the figures -- they are only persisted for downstream
use. The '_ci' tag in the cache filename keeps a CI-less older cache from being reused.

Set REQUIRE_REFS = True to restrict to citations whose CITED paper also has >= 1 reference in
the S2AG network (same "_hasref" restriction as analyze_refer_mention_position.py), so these
plots are comparable to the main hasref figures.

Input  : S2ORC_META_DIRE/citation_analysis_disp_f_iter/refer_mention_count_records_V2.pkl
             (from build_count_records.py)
         META_DIRE/citing2cited_cid_list_remove_circle.pkl   (only when REQUIRE_REFS)
Cache  : ./figures/citation_count/heatmap/heatmap_cache_updown_ci_hasref<0|1>_max<MAX_HEATMAP>.npz
             (per-cell [up_count, down_count] sums + counts + bootstrap 95% CI [lo/hi] for every
              metric; independent of MIN_CELL)
             -- at the deliverable setting (REQUIRE_REFS=False, MAX_HEATMAP=10) this is instead
             ./figures/citation_count/heatmap/Extended Data Fig. 3.npz
Output : ./figures/citation_count/heatmap/hasref<0|1>/heatmap_frequency.svg/.png +
             heatmap_<section>.svg/.png (one per metric), plus the combined 2x3 panel
             heatmap_combined_2x3.svg/.png -- named 'Extended Data Fig. 3.svg/.png' at the
             deliverable setting -- and heatmap_cell_mean_ci_hasref<0|1>_max<MAX_HEATMAP>.xlsx
             (per-cell mean + 95% CI, not a figure)
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# utils.py lives two levels up (semantic_scholar/); make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEM_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.insert(0, SEM_DIR)
from utils import S2ORC_META_DIRE, META_DIRE  # noqa: E402

# ---- parameters ----
REQUIRE_REFS = False          # keep only citations whose CITED paper has >=1 reference (hasref)
print('REQUIRE_REFS:', REQUIRE_REFS)

# heatmaps (x = up_count, y = down_count):
MAX_HEATMAP = 10             # both heatmap axes span 0..MAX_HEATMAP
MIN_CELL = 20                # a (up_count, down_count) cell needs >= this many citations to be coloured

# bootstrap 95% CI of each cell's mean (cached only, NOT drawn). A CI is computed for every cell
# with >= MIN_CI_CELL citations; smaller cells get NaN CI bounds.
N_BOOTSTRAP = 1000           # bootstrap resamples per cell
CI_ALPHA = 0.05              # -> 2.5 / 97.5 percentile CI
MIN_CI_CELL = 2              # a cell needs >= this many citations for a CI (else NaN)
BOOTSTRAP_SEED = 0           # fixed seed -> reproducible CI

meta_dire = S2ORC_META_DIRE
records_path = os.path.join(meta_dire, 'citation_analysis_disp_f_iter', 'refer_mention_count_records_V2.pkl')

# heatmap metrics: Frequency + EVERY section (all 8 regions). '__freq__' -> mean mentions per
# citation; a section key -> mean fraction of a citation's mentions in that region.
ALL_SECTIONS = [
    ('Before intro', 'before_intro'), ('Introduction', 'intro'), ('Method', 'method'),
    ('Result', 'result'), ('Result & Discussion', 'res_dis'), ('Discussion', 'discussion'),
    ('Conclusion', 'conclusion'), ('After conclusion', 'after_conclusion'),
]
HEATMAP_METRICS = [('Frequency', '__freq__')] + ALL_SECTIONS

# combined 2x3 panel: row1 = Frequency / Introduction / Method, row2 = Result / Result &
# Discussion / Discussion. Reuses the per-metric mean matrices (no separate cache).
COMBINED_LAYOUT = [
    [('Frequency', '__freq__'), ('Introduction', 'intro'), ('Method', 'method')],
    [('Result', 'result'), ('Result & Discussion', 'res_dis'), ('Discussion', 'discussion')],
]

XLABEL = '# of references of the focal paper cited together'
YLABEL = '# of citations of the focal paper cited together'

fig_root = os.path.join(BASE_DIR, 'figures', 'citation_count')
heatmap_dir = os.path.join(fig_root, 'heatmap')
os.makedirs(heatmap_dir, exist_ok=True)
# the output FIGURES for the two REQUIRE_REFS settings go into their own subdirectory so a
# hasref run and a non-hasref run never overwrite each other's figures. (The cache below already
# carries the hasref tag in its filename, so it stays in heatmap_dir and does not collide.)
fig_out_dir = os.path.join(heatmap_dir, f'hasref{int(REQUIRE_REFS)}')
os.makedirs(fig_out_dir, exist_ok=True)
# The combined 2x3 panel at the paper's default settings IS the Extended Data deliverable, so it
# and its cache are named after the figure; every other setting keeps the descriptive names. The
# gate covers exactly what the descriptive names encode (hasref + max), so nothing can collide.
ED_FIG_NAME = 'Extended Data Fig. 3'
is_ed_fig = (not REQUIRE_REFS) and MAX_HEATMAP == 10
print('is_ed_fig:', is_ed_fig, ('-> %s' % ED_FIG_NAME) if is_ed_fig else '')

# '_updown' marks the cell indexing [up_count, down_count] (was [up_count, total]); the tag
# keeps this version from loading an older total-indexed cache. '_ci' marks the schema that also
# stores the per-cell bootstrap 95% CI (lo__/hi__), so a pre-CI cache is never reused.
heatmap_cache = os.path.join(
    heatmap_dir,
    f'{ED_FIG_NAME}.npz' if is_ed_fig else
    f'heatmap_cache_updown_ci_hasref{int(REQUIRE_REFS)}_max{MAX_HEATMAP}.npz')


def accumulate_heatmaps(recs, metrics, max_hm, collect_values=False):
    """Per (up_count, down_count) cell, sum and count each metric over the records. Returns
    (sums, cnts): dict metric_key -> (max_hm+1) x (max_hm+1) array indexed [up_count, down_count].
    When collect_values, also returns `vals`: dict metric_key -> R x R nested lists holding every
    contributing per-citation value (used for the per-cell bootstrap CI); otherwise vals is None."""
    R = max_hm + 1
    sums = {key: np.zeros((R, R)) for _, key in metrics}
    cnts = {key: np.zeros((R, R)) for _, key in metrics}
    vals = ({key: [[[] for _ in range(R)] for _ in range(R)] for _, key in metrics}
            if collect_values else None)
    for r in recs:
        up = r['up_count']
        down = r['down_count']
        if up > max_hm or down > max_hm:
            continue
        pos = r['pos']
        tot = sum(pos.values())
        for _, key in metrics:
            if key == '__freq__':
                sums[key][up, down] += tot
                cnts[key][up, down] += 1
                if collect_values:
                    vals[key][up][down].append(tot)
            elif tot > 0:
                v = pos.get(key, 0) / tot
                sums[key][up, down] += v
                cnts[key][up, down] += 1
                if collect_values:
                    vals[key][up][down].append(v)
    return sums, cnts, vals


def _boot_mean_ci(arr, n_boot, alpha, rng):
    """Bootstrap (percentile) CI of the mean of `arr`. Resamples in memory-bounded blocks so a
    huge cell can't allocate an n_boot x len(arr) matrix at once. Returns (lo, hi)."""
    n = len(arr)
    if n == 0:
        return np.nan, np.nan
    means = np.empty(n_boot)
    max_elems = 5_000_000                       # cap on drawn indices held at once
    block = max(1, min(n_boot, max_elems // max(1, n)))
    done = 0
    while done < n_boot:
        b = min(block, n_boot - done)
        idx = rng.integers(0, n, size=(b, n))
        means[done:done + b] = arr[idx].mean(axis=1)
        done += b
    lo = np.percentile(means, 100.0 * (alpha / 2.0))
    hi = np.percentile(means, 100.0 * (1.0 - alpha / 2.0))
    return lo, hi


def bootstrap_cell_ci(vals, metrics, max_hm, n_boot, alpha, min_ci_cell, seed):
    """Per-cell bootstrap CI of the mean for every metric. Returns (ci_lo, ci_hi): dict
    metric_key -> R x R arrays indexed [up_count, down_count]; cells with < min_ci_cell citations
    stay NaN. Only the 11x11 cells are looped, so this is cheap despite the resampling."""
    R = max_hm + 1
    ci_lo = {key: np.full((R, R), np.nan) for _, key in metrics}
    ci_hi = {key: np.full((R, R), np.nan) for _, key in metrics}
    rng = np.random.default_rng(seed)
    for _, key in metrics:
        for up in range(R):
            for down in range(R):
                cell = vals[key][up][down]
                if len(cell) < min_ci_cell:
                    continue
                arr = np.asarray(cell, dtype=float)
                ci_lo[key][up, down], ci_hi[key][up, down] = _boot_mean_ci(
                    arr, n_boot, alpha, rng)
    return ci_lo, ci_hi


def plot_heatmap(mean_mat, title, cbar_label, out_stem, max_hm):
    """Heatmap of mean_mat[up_count, down_count]: x = up_count, y = down_count. NaN cells (too
    few citations) are drawn blank."""
    fig, ax = plt.subplots(figsize=(7.2, 6))
    # mean_mat is indexed [up_count, down_count]; imshow draws [row=y, col=x], so transpose to
    # put up_count on the x-axis and down_count on the y-axis.
    masked = np.ma.masked_invalid(mean_mat.T)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad('#e6e6e6')
    im = ax.imshow(masked, origin='lower', aspect='auto', cmap=cmap,
                   extent=[-0.5, max_hm + 0.5, -0.5, max_hm + 0.5])
    ax.set_xlabel(XLABEL, fontsize=13)
    ax.set_ylabel(YLABEL, fontsize=13)
    ax.set_title(title, fontsize=14)
    step = max(1, max_hm // 10)
    ax.set_xticks(range(0, max_hm + 1, step))
    ax.set_yticks(range(0, max_hm + 1, step))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_stem + '.svg', format='svg', transparent=True)
    fig.savefig(out_stem + '.png', dpi=150)
    plt.close(fig)


# colours for the E/F/G annotation drawn on the Frequency panel
FOUNDATIONAL_COLOR = 'red'
EXTENSIONAL_COLOR = '#F2A900'   # orange-yellow
GENERALIZATIONAL_COLOR = 'blue'
DISRUPTIVE_COLOR = '#DCDCDC'   # off-white / light grey


def annotate_ef_regions(ax, max_hm):
    """On the Frequency panel: draw the red x=y divider, then three 45-degree arrows from
    staggered points near the diagonal centre: upper-left 'Foundational' (red) and lower-right
    'Extensional' (orange-yellow), both PERPENDICULAR to x=y (|dx| == |dy|), plus lower-left
    'Generalizational' (blue) PARALLEL to x=y (offset off the line) toward the origin (up==0 &
    down==0). The whole x==0 (up_count==0) column is outlined with a grey box and labelled
    'Disruptive' via a grey arrow near the top."""
    lo, hi = -0.5, max_hm + 0.5
    c = max_hm / 2.0            # diagonal centre (e.g. 5,5 for max_hm=10)
    t = 0.28 * max_hm          # per-axis arrow length; equal on x and y -> exactly 45 degrees
    s = 0.6                     # stagger the two bases apart along the diagonal
    # red x = y divider across the panel
    ax.plot([lo, hi], [lo, hi], color='red', linewidth=2.0, zorder=5)

    # Foundational: base staggered DOWN the diagonal; arrow 45deg to the UPPER-LEFT (-t, +t).
    # Drawn in FRONT (high zorder) of the boxes / other arrows.
    fb = (c - s, c - s)
    ax.annotate('', xy=(fb[0] - t, fb[1] + t), xytext=fb,
                arrowprops=dict(arrowstyle='-|>', color=FOUNDATIONAL_COLOR, lw=2.2), zorder=10)
    ax.text(fb[0] - t, fb[1] + t + 0.2, 'Foundational', color=FOUNDATIONAL_COLOR, fontsize=12,
            fontweight='bold', ha='center', va='bottom', zorder=10)

    # Extensional: base staggered UP the diagonal; arrow 45deg to the LOWER-RIGHT (+t, -t)
    eb = (c + s, c + s)
    ax.annotate('', xy=(eb[0] + t, eb[1] - t), xytext=eb,
                arrowprops=dict(arrowstyle='-|>', color=EXTENSIONAL_COLOR, lw=2.2), zorder=6)
    ax.text(eb[0] + t, eb[1] - t - 0.2, 'Extensional', color=EXTENSIONAL_COLOR, fontsize=12,
            fontweight='bold', ha='center', va='top', zorder=6)

    # Generalizational: arrow PARALLEL to the diagonal (direction (-t, -t)), pointing toward the
    # origin (up==0 & down==0), but OFFSET off the x=y line by a perpendicular step (d, -d) (to
    # the lower-right side) so it runs alongside the red divider instead of on top of it.
    d = 0.45                                           # perpendicular offset from the diagonal
    gb = (c + d, c - d)
    ax.annotate('', xy=(gb[0] - t, gb[1] - t), xytext=gb,
                arrowprops=dict(arrowstyle='-|>', color=GENERALIZATIONAL_COLOR, lw=2.2), zorder=10)
    ax.text(gb[0] - t, gb[1] - t - 0.2, 'Generalizational', color=GENERALIZATIONAL_COLOR,
            fontsize=12, fontweight='bold', ha='center', va='top', zorder=10)

    # Disruptive: the entire x==0 (up_count == 0) column, outlined with a grey box and pointed
    # at by a grey arrow near the TOP, with the label to its right.
    ax.add_patch(Rectangle((-0.5, lo), 1.0, hi - lo, fill=False, edgecolor=DISRUPTIVE_COLOR,
                           linewidth=2.0, zorder=6))
    yb = max_hm - 0.7
    ax.annotate('', xy=(0.5, yb), xytext=(2.6, yb),
                arrowprops=dict(arrowstyle='-|>', color=DISRUPTIVE_COLOR, lw=2.2), zorder=6)
    ax.text(2.8, yb, 'Disruptive', color=DISRUPTIVE_COLOR, fontsize=12, fontweight='bold',
            ha='left', va='center', zorder=6)

    # keep the panel tight to the heatmap extent (the diagonal line would otherwise autoscale it)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)


def plot_combined_heatmaps(mean_by_key, layout, out_stem, max_hm):
    """Draw `layout` (a list of rows of (label, metric_key)) as a grid of heatmaps in one
    figure. Each panel keeps its own colour scale + colourbar; the x/y labels are drawn
    once (centred) for the whole figure instead of on every panel. The first panel (i=j=0,
    the Frequency panel) additionally gets the x=y divider + Foundational/Extensional arrows."""
    n_rows = len(layout)
    n_cols = max(len(row) for row in layout)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.6 * n_rows))
    axes = np.atleast_2d(axes)
    step = max(1, max_hm // 10)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad('#e6e6e6')

    for i, row in enumerate(layout):
        for j in range(n_cols):
            ax = axes[i, j]
            if j >= len(row):
                ax.axis('off')
                continue
            label, key = row[j]
            # mean_by_key[key] is indexed [up_count, down_count]; transpose so up_count is on x.
            masked = np.ma.masked_invalid(mean_by_key[key].T)
            im = ax.imshow(masked, origin='lower', aspect='auto', cmap=cmap,
                           extent=[-0.5, max_hm + 0.5, -0.5, max_hm + 0.5])
            ax.set_title(label, fontsize=17)
            ax.set_xticks(range(0, max_hm + 1, step))
            ax.set_yticks(range(0, max_hm + 1, step))
            cbar_label = 'Mean mentions per citation' if key == '__freq__' else 'Mean ratio'
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(cbar_label, fontsize=11)
            # first panel (Frequency): x=y divider + Foundational/Extensional arrows
            if i == 0 and j == 0:
                annotate_ef_regions(ax, max_hm)

    # single centred x/y label for the whole figure; nudge the y label further left (x=0.005)
    fig.supxlabel(XLABEL, fontsize=17)
    fig.supylabel(YLABEL, fontsize=17, x=0.005)
    fig.tight_layout()
    fig.savefig(out_stem + '.svg', format='svg', transparent=True)
    fig.savefig(out_stem + '.png', dpi=150)
    plt.close(fig)


# ==================================================
# heatmaps: x = up_count, y = down_count; one figure per metric
# (Frequency + each section). Per-cell sums/counts are cached to an .npz so redraws don't rescan.
# The records (and the optional hasref filter) are only loaded on a cache miss.
# ==================================================
print('\nheatmaps (x = up_count, y = down_count):')
if os.path.exists(heatmap_cache):
    z = np.load(heatmap_cache)
    sums = {key: z[f'sum__{key}'] for _, key in HEATMAP_METRICS}
    cnts = {key: z[f'cnt__{key}'] for _, key in HEATMAP_METRICS}
    ci_lo = {key: z[f'lo__{key}'] for _, key in HEATMAP_METRICS}
    ci_hi = {key: z[f'hi__{key}'] for _, key in HEATMAP_METRICS}
    print('  loaded heatmap cache:', heatmap_cache)
else:
    # load records (+ optional hasref filter) only when the cache needs to be built
    with open(records_path, 'rb') as f:
        records = pickle.load(f)
    print('records:', len(records))

    if REQUIRE_REFS:
        with open(os.path.join(META_DIRE, 'citing2cited_cid_list_remove_circle.pkl'), 'rb') as f:
            pid2up_list = pickle.load(f)
        n_before = len(records)
        records = [r for r in records if pid2up_list.get(r.get('cid'))]
        print(f'hasref filter: kept {len(records)} / {n_before} '
              f'(cited paper must have >= 1 reference)')

    # collect per-cell values too, so we can bootstrap a 95% CI of each cell's mean (cached only)
    sums, cnts, vals = accumulate_heatmaps(records, HEATMAP_METRICS, MAX_HEATMAP,
                                           collect_values=True)
    print(f'  bootstrapping per-cell 95% CI (n_boot={N_BOOTSTRAP}, seed={BOOTSTRAP_SEED}, '
          f'min cell {MIN_CI_CELL}) ...')
    ci_lo, ci_hi = bootstrap_cell_ci(vals, HEATMAP_METRICS, MAX_HEATMAP,
                                     N_BOOTSTRAP, CI_ALPHA, MIN_CI_CELL, BOOTSTRAP_SEED)
    np.savez_compressed(
        heatmap_cache,
        **{f'sum__{key}': sums[key] for _, key in HEATMAP_METRICS},
        **{f'cnt__{key}': cnts[key] for _, key in HEATMAP_METRICS},
        **{f'lo__{key}': ci_lo[key] for _, key in HEATMAP_METRICS},
        **{f'hi__{key}': ci_hi[key] for _, key in HEATMAP_METRICS})
    print('  wrote heatmap cache (with per-cell bootstrap CI):', heatmap_cache)

# totals that went into the heatmaps (up_count, down_count <= MAX_HEATMAP; works whether the
# cache was loaded or just built). A record in build_count_records.py is ONE citation LINK (a
# citing-cited pair): its in-text mentions were merged into the {position: frequency} 'pos' map.
#   - '__freq__' cnts add 1 per record   -> number of citation LINKS
#   - '__freq__' sums add tot=sum(pos)    -> number of citation INSTANCES (raw in-text mentions)
total_links = int(cnts['__freq__'].sum())
total_instances = int(sums['__freq__'].sum())
print(f'  total citation links (citing-cited pairs, up_count, down_count <= {MAX_HEATMAP}): '
      f'{total_links:,}')
print(f'  total citation instances (in-text mentions, up_count, down_count <= {MAX_HEATMAP}): '
      f'{total_instances:,}')

# ==================================================
# export per-cell mean + bootstrap 95% CI to an xlsx (easy to read; NOT the figures). One long
# 'all_metrics' sheet plus one sheet per metric; every (up_count, down_count) cell is a row with
# its citation count n, unmasked mean, and CI [lo, hi] (NaN where n < MIN_CI_CELL).
# ==================================================
ci_rows = []
for label, key in HEATMAP_METRICS:
    with np.errstate(invalid='ignore', divide='ignore'):
        full_mean = sums[key] / cnts[key]           # unmasked (every cell, incl. sparse ones)
    for up in range(MAX_HEATMAP + 1):
        for down in range(MAX_HEATMAP + 1):
            n = int(cnts[key][up, down])
            ci_rows.append({
                'metric': label, 'metric_key': key,
                'up_count': up, 'down_count': down, 'n': n,
                'mean': full_mean[up, down] if n > 0 else np.nan,
                'ci95_lo': ci_lo[key][up, down], 'ci95_hi': ci_hi[key][up, down],
            })
ci_df = pd.DataFrame(ci_rows)
xlsx_path = os.path.join(
    fig_out_dir, f'heatmap_cell_mean_ci_hasref{int(REQUIRE_REFS)}_max{MAX_HEATMAP}.xlsx')
with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
    ci_df.to_excel(writer, sheet_name='all_metrics', index=False)
    for label, key in HEATMAP_METRICS:
        # excel sheet names are <= 31 chars and cannot contain []:*?/\
        sheet = label[:31].replace('&', 'and')
        ci_df[ci_df['metric_key'] == key].drop(columns='metric_key').to_excel(
            writer, sheet_name=sheet, index=False)
print('  wrote per-cell mean/CI xlsx:', xlsx_path)

mean_by_key = {}
for label, key in HEATMAP_METRICS:
    with np.errstate(invalid='ignore', divide='ignore'):
        mean_mat = sums[key] / cnts[key]
    mean_mat[cnts[key] < MIN_CELL] = np.nan     # too-few-citation cells left blank
    mean_by_key[key] = mean_mat
    cbar_label = 'Mean mentions per citation' if key == '__freq__' else 'Mean ratio'
    fname = 'heatmap_frequency' if key == '__freq__' else f'heatmap_{key}'
    plot_heatmap(mean_mat, label, cbar_label, os.path.join(fig_out_dir, fname), MAX_HEATMAP)
    print(f'  {label}: figure')

# combined 2x3 panel (reuses the mean matrices above; no separate cache) -- this is the
# Extended Data deliverable, so at the default settings it carries the figure's name
plot_combined_heatmaps(mean_by_key, COMBINED_LAYOUT,
                       os.path.join(fig_out_dir,
                                    ED_FIG_NAME if is_ed_fig else 'heatmap_combined_2x3'),
                       MAX_HEATMAP)
print('  combined 2x3: figure')

print('\nsaved figures under:', fig_out_dir)
print('finished')
