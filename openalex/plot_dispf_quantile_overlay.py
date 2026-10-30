"""
Overlaid quantile-distribution curves for the disruptive-foundation (disp_f) variants
against Foundation / Generalization, split by the sign of the disruption index disp_y{off}.

Based on compute_efgd_quantile.py's per-paper quantile table. 1 x 2 subplots overlaying the
KDE of the selected {QUANTILE_GROUP} quantile of three metrics:
  LEFT  = disp > 0 papers, RIGHT = disp <= 0 papers
  three lines each: Foundation (red), disp_f (light red; the DISP_F_VARIANT below),
  Generalization (blue)

disp_f>T = "disruptive foundation" at threshold T (up_count == 0 & down_count > T), the three
shares get_efg_v2_disp_f.py -> compute_efgd_quantile.py (use_disp_f=True) quantile.

Input  : META_DIRE/efg_computation/
           efgd_within_year_domain_quantile_y{off}{data_suffix}_rt_{rt}_ct_{ct}_ybefore_{max_year}{tag}.csv
Cache  : ./figures/dispf_quantile_overlay/cache/{stem}_cache.csv
           (the finite-disp per-paper subset with the disp index + plotted quantile columns;
            if present it is loaded directly and the big quantile table is not touched --
            delete to refresh)
Output : ./figures/dispf_quantile_overlay/
           dispf_quantile_overlay_y{off}{data_suffix}_rt_{rt}_ct_{ct}_ybefore_{max_year}{tag}_q_{QUANTILE_GROUP}.svg (+ .png)
"""

import os
import sys
import math

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# utils.py is in this same directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from utils import META_DIRE  # noqa: E402

meta_dire = META_DIRE
efg_dire = os.path.join(meta_dire, 'efg_computation')

# ---- must match compute_efgd_quantile.py so the input filename resolves ----
year_offset = 5
reference_threshold = 1
citation_threshold = 5
with_review = False
data_suffix = '_with_review' if with_review else ''
# which pub_year cutoff / variant of the quantile table to read (see compute_efgd_quantile.py):
#   max_year = 2025, input_name_tag = '_full_cover' -> ALL papers before 2025
#   max_year = 2025 - year_offset, input_name_tag = '' -> complete-window papers only
max_year = 2025 - year_offset
input_name_tag = ''

# which quantile grouping to plot (a column suffix produced by compute_efgd_quantile.py):
#   'year_domain' -> {col}_quantile          (rank within pub_year x top_concept)
#   'year'        -> {col}_quantile_by_year   (rank within pub_year only)
#   'global'      -> {col}_quantile_global    (rank across all papers, no grouping)
QUANTILE_GROUP = 'year_domain'
_QUANTILE_SUFFIX = {
    'year_domain': '_quantile',
    'year': '_quantile_by_year',
    'global': '_quantile_global',
}
assert QUANTILE_GROUP in _QUANTILE_SUFFIX, QUANTILE_GROUP
qs = _QUANTILE_SUFFIX[QUANTILE_GROUP]

# number of points the KDE curve is evaluated at across the [0, 1] quantile range
N_KDE_POINTS = 200

print("year_offset", year_offset, "| rt", reference_threshold, "| ct", citation_threshold,
      "| max_year", max_year, "| tag", repr(input_name_tag),
      "| QUANTILE_GROUP", QUANTILE_GROUP, "->", repr(qs))

# ---- columns (mirror compute_efgd_quantile.py) ----
disp_col = f'disp_y{year_offset}'                              # disruption index (raw; filter disp>0)
f_q = f'f_y{year_offset}{qs}'                                  # Foundation quantile
g_q = f'g_y{year_offset}{qs}'                                  # Generalization quantile
dispf0_q = f'disruption_f_y{year_offset}{qs}'                  # disp_f>0 quantile
dispf1_q = f'disruption_f_down_gt1_y{year_offset}{qs}'         # disp_f>1 quantile
dispf2_q = f'disruption_f_down_gt2_y{year_offset}{qs}'         # disp_f>2 quantile

# which disruptive-foundation variant to draw as the single "disp_f" line.
DISP_F_VARIANT = 'disp_f_0'   # 'disp_f_0' (>0), 'disp_f_1' (>1) or 'disp_f_2' (>2)
_DISPF_Q = {'disp_f_0': dispf0_q, 'disp_f_1': dispf1_q, 'disp_f_2': dispf2_q}
_DISPF_LABEL = {'disp_f_0': 'Disruptive foundation (>0)',
                'disp_f_1': 'Disruptive foundation (>1)',
                'disp_f_2': 'Disruptive foundation (>2)'}
assert DISP_F_VARIANT in _DISPF_Q, DISP_F_VARIANT
dispf_q = _DISPF_Q[DISP_F_VARIANT]

# each panel draws THREE lines: Foundation (red), disp_f (light red), Generalization (blue).
FOUNDATION_COLOR = '#cb181d'     # red
DISPF_COLOR      = '#fca082'     # light red
GENERALIZATION_COLOR = '#2171b5' # blue
lines = [
    (f_q,     FOUNDATION_COLOR,     'Foundation'),
    (dispf_q, DISPF_COLOR,          _DISPF_LABEL[DISP_F_VARIANT]),
    (g_q,     GENERALIZATION_COLOR, 'Generalization'),
]

# the metric quantile columns the figure needs (all disp_f variants cached so DISP_F_VARIANT
# can be changed without rebuilding the cache)
plot_cols = [f_q, g_q, dispf0_q, dispf1_q, dispf2_q]

# outputs + a cache of the exact per-paper data the figure is built from, keyed by all the
# parameters so different settings never clash.
out_dir = os.path.join('./figures', 'dispf_quantile_overlay')
cache_dir = os.path.join(out_dir, 'cache')
os.makedirs(out_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)
stem = (f'dispf_quantile_overlay_y{year_offset}{data_suffix}_'
        f'rt_{reference_threshold}_ct_{citation_threshold}_'
        f'ybefore_{max_year}{input_name_tag}_q_{QUANTILE_GROUP}')
cache_path = os.path.join(cache_dir, f'{stem}_cache.csv')

# ==================================================
# 1. obtain the disp>0 per-paper quantile subset: load the cache if present, else build it
#    from compute_efgd_quantile.py's table (filter disp>0, keep the plotted columns) and save.
# ==================================================
if os.path.exists(cache_path):
    df = pd.read_csv(cache_path)
    print("loaded cache:", cache_path, df.shape)
    missing = [c for c in plot_cols + [disp_col] if c not in df.columns]
    assert not missing, (f"cache {cache_path} is missing {missing}; delete it to rebuild.")
else:
    src_path = os.path.join(
        efg_dire,
        f'efgd_within_year_domain_quantile_y{year_offset}{data_suffix}_'
        f'rt_{reference_threshold}_ct_{citation_threshold}_ybefore_{max_year}{input_name_tag}.csv')
    assert os.path.exists(src_path), f"{src_path}\nRun compute_efgd_quantile.py first."
    df_full = pd.read_csv(src_path)
    print("loaded quantile table shape:", df_full.shape)

    # ---- guard: the disp_f quantile columns need compute_efgd_quantile.py's use_disp_f=True ----
    needed = [disp_col, f_q, g_q, dispf0_q, dispf1_q, dispf2_q]
    missing = [c for c in needed if c not in df_full.columns]
    assert not missing, (
        f"quantile table is missing columns {missing}.\n"
        f"Re-run compute_efgd_quantile.py with use_disp_f=True (and matching rt/ct/year).")

    # ---- keep papers with a finite disruption index (both disp>0 and disp<=0 are plotted) ----
    disp_vals = pd.to_numeric(df_full[disp_col], errors='coerce')
    n_before = len(df_full)
    df_full = df_full[disp_vals.notna()].copy()
    print(f"finite-disp filter: {n_before} -> {len(df_full)} papers "
          f"(removed {n_before - len(df_full)})")
    assert len(df_full) > 0, "no papers with a disruption index"

    # keep only what the figure needs (disp column is needed to split by sign) and cache it
    keep = (['pid'] if 'pid' in df_full.columns else []) + [disp_col] + plot_cols
    df = df_full[keep].copy()
    df.to_csv(cache_path, index=False)
    print("wrote cache:", cache_path, df.shape)

# ---- split into the two disruption groups plotted on the two figure rows ----
_disp_num = pd.to_numeric(df[disp_col], errors='coerce')
df_pos = df[_disp_num > 0]
df_neg = df[_disp_num <= 0]
print(f"disp > 0: {len(df_pos)} papers  |  disp <= 0: {len(df_neg)} papers")
assert len(df_pos) > 0 and len(df_neg) > 0, (
    "need BOTH disp>0 and disp<=0 papers. If this is a stale disp!=0 cache from an earlier "
    f"version (which excluded disp==0), delete it and rerun:\n  {cache_path}")


# ==================================================
# 2. plotting helpers
# ==================================================
def kde_curve(vals, xs):
    """Gaussian-KDE density of `vals` on grid `xs`; falls back to a normalised histogram
    interpolated onto `xs` when KDE is undefined (fewer than 2 points / zero variance)."""
    vals = np.asarray(vals, dtype=float)
    if len(vals) >= 2 and np.ptp(vals) > 0:
        try:
            return gaussian_kde(vals)(xs)
        except np.linalg.LinAlgError:
            pass
    counts, edges = np.histogram(vals, bins=min(50, max(1, len(vals))), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return np.interp(xs, centers, counts, left=0.0, right=0.0)


def draw_overlay(ax, sub, lines, title):
    """Overlay the KDE of each metric's quantile (over the `sub` papers) onto `ax`."""
    xs = np.linspace(0.0, 1.0, N_KDE_POINTS)
    for qcol, color, label in lines:
        vals = pd.to_numeric(sub[qcol], errors='coerce').dropna().values
        if len(vals) == 0:
            print(f"  [warn] no data for {label} ({qcol})")
            continue
        ax.plot(xs, kde_curve(vals, xs), color=color, linewidth=2.0,
                label=f'{label} (n={len(vals):,}, mean={np.mean(vals):.3f}, '
                      f'median={np.median(vals):.3f})')
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(f'{QUANTILE_GROUP} quantile', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_xlim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=9, frameon=False, loc='upper center')


# ==================================================
# 3. figure: 1 x 2 subplots -- LEFT = disp>0, RIGHT = disp<=0. Each overlays the three
#    metric quantile distributions: Foundation (red), disp_f (light red), Generalization (blue).
# ==================================================
fig, (ax_pos, ax_neg) = plt.subplots(1, 2, figsize=(13, 5))
draw_overlay(ax_pos, df_pos, lines, 'disp > 0')
draw_overlay(ax_neg, df_neg, lines, 'disp <= 0')
fig.suptitle('Foundation / disruptive-foundation / Generalization quantile distributions',
             fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.96])

# ==================================================
# 4. save  (out_dir / stem defined above, next to the cache)
# ==================================================
svg_path = os.path.join(out_dir, f'{stem}.svg')
png_path = os.path.join(out_dir, f'{stem}.png')
fig.savefig(svg_path, format='svg', transparent=True)
fig.savefig(png_path, dpi=150)
plt.close(fig)
print("saved:", svg_path)
print("saved:", png_path)
print("finished")
