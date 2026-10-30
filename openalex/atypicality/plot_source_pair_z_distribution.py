"""
Plot the distribution of per-source-pair z-scores from compute_atypicality_s2.py, for
several upper_year cutoffs overlaid on one figure.

For each upper_year it loads
    META_DIRE/source_min2source_max2z_no_more_than_{upper_year}.pkl
which is a nested dict  {s_min: {s_max: z}}  carrying ONE z-score per source pair
(including the self-pairs where s_min == s_max), and collects every pair's z into a
single distribution -- each pair counted ONCE (unweighted; this is the distribution of
pair z-scores, not the per-paper weighted bag used in compute_atypicality_s3.py).

Excluded sources are dropped EXACTLY as in compute_atypicality_s3.py: there, references
whose primary source is in REMOVE_SOURCE_IDS are skipped before pairs are formed, which
is equivalent to discarding any pair (s_min, s_max) where EITHER end is a removed source.
Set DO_REMOVE accordingly (the REMOVE_SOURCE_IDS set below is copied verbatim from s3).

To avoid re-reading the (large) z pkls on every redraw, the extracted z array per
upper_year is cached to ./figures/atypicality/source_pair_z_cache/*.npy; delete a cache
file (or set FORCE_REBUILD) to force a refresh.

Input  : META_DIRE/source_min2source_max2z_no_more_than_{upper_year}.pkl  (compute_atypicality_s2.py)
Output : ./figures/atypicality/source_pair_z_distribution_{VERSION_TAG}_self{0,1}.svg
         ./figures/atypicality/source_pair_z_distribution_{VERSION_TAG}_self{0,1}_summary.csv
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# utils.py is in the parent directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
from utils import META_DIRE  # noqa: E402

# repo plotting convention
try:
    plt.rcParams['font.family'] = 'Gill Sans MT'
except Exception:
    pass
plt.rcParams['axes.unicode_minus'] = False

meta_dire = META_DIRE
print("meta_dire", meta_dire)

# which upper_year z pkls to plot (must match compute_atypicality_s2.py outputs that exist)
upper_year_list = [1970, 1990, 2010]
print("upper_year_list", upper_year_list)

# ---------------------------------------------------------------------------
# Source exclusion, IDENTICAL to compute_atypicality_s3.py:
#   DO_REMOVE = True  -> drop any pair touching a REMOVE_SOURCE_IDS source ("removed")
#   DO_REMOVE = False -> keep every pair ("full")
# ---------------------------------------------------------------------------
DO_REMOVE = True
REMOVE_SOURCE_IDS = set({
    "S4306525036",  # PubMed
    "S4306400194",  # arXiv (Cornell University)
    "S4306402512",  # HAL (Le Centre pour la Communication Scientifique Directe)
    "S4306401280",  # DOAJ (DOAJ: Directory of Open Access Journals)
    "S4210172589",  # SSRN Electronic Journal
    "S4306401271",  # RePEc: Research Papers in Economics
    "S41354064",    # ChemInform
    "S4317411217",  # Deleted Journal
    "S4306400562",  # Zenodo (CERN European Organization for Nuclear Research)
    "S4306402450",  # Research Square (Research Square)
    "S4210187579",  # Chemischer Informationsdienst
    "S4306402567",  # bioRxiv (Cold Spring Harbor Laboratory)
})
VERSION_TAG = 'removed' if DO_REMOVE else 'full'
print("VERSION_TAG", VERSION_TAG, "| DO_REMOVE", DO_REMOVE,
      "| remove_source_count", len(REMOVE_SOURCE_IDS))

# include the self-pairs (s_min == s_max)? compute_atypicality_s3.py uses them, so keep
# them by default; set False to look at cross-source pairs only.
INCLUDE_SELF_PAIRS = True
SELF_TAG = 'self1' if INCLUDE_SELF_PAIRS else 'self0'
print("INCLUDE_SELF_PAIRS", INCLUDE_SELF_PAIRS)

FORCE_REBUILD = False  # set True to ignore the per-year .npy caches and re-read the pkls

# x-range for the histograms: clip to these percentiles (pooled over all years) so a few
# extreme z-scores do not flatten the visible bulk. Set to (0, 100) for the full range.
CLIP_PCT = (0.5, 99.5)
N_BINS = 80
print("CLIP_PCT", CLIP_PCT, "| N_BINS", N_BINS, "| FORCE_REBUILD", FORCE_REBUILD)

figure_dire = os.path.join(BASE_DIR, 'figures', 'atypicality')
os.makedirs(figure_dire, exist_ok=True)
cache_dir = os.path.join(figure_dire, 'source_pair_z_cache')
os.makedirs(cache_dir, exist_ok=True)


def extract_pair_z(upper_year):
    """Return a 1-D float array of every kept pair's z-score for one upper_year, reading
    the .npy cache if present. Excludes pairs touching REMOVE_SOURCE_IDS (when DO_REMOVE)
    and self-pairs (when not INCLUDE_SELF_PAIRS); drops non-finite z."""
    cache_path = os.path.join(
        cache_dir, f'zpairs_{VERSION_TAG}_{SELF_TAG}_{upper_year}.npy')
    if os.path.exists(cache_path) and not FORCE_REBUILD:
        z = np.load(cache_path)
        print(f"  [cache] upper_year {upper_year}: {len(z):,} pairs <- {cache_path}")
        return z

    pkl_path = os.path.join(
        meta_dire, 'source_min2source_max2z_no_more_than_%s.pkl' % upper_year)
    if not os.path.exists(pkl_path):
        print(f"  [skip] upper_year {upper_year}: pkl not found ({pkl_path})")
        return None

    with open(pkl_path, 'rb') as f:
        s_min2s_max2z = pickle.load(f)

    vals = []
    n_removed_pairs = 0
    n_self = 0
    for s_min, s_max2z in s_min2s_max2z.items():
        # whole s_min row drops if its source is excluded (mirrors s3 reference removal)
        if DO_REMOVE and s_min in REMOVE_SOURCE_IDS:
            n_removed_pairs += len(s_max2z)
            continue
        for s_max, z in s_max2z.items():
            if DO_REMOVE and s_max in REMOVE_SOURCE_IDS:
                n_removed_pairs += 1
                continue
            if s_min == s_max:
                n_self += 1
                if not INCLUDE_SELF_PAIRS:
                    continue
            vals.append(z)

    z = np.asarray(vals, dtype=float)
    z = z[np.isfinite(z)]
    np.save(cache_path, z)
    print(f"  upper_year {upper_year}: kept {len(z):,} pairs "
          f"(self-pairs {n_self:,}; removed-source pairs dropped {n_removed_pairs:,}) "
          f"-> {cache_path}")
    return z


# ==================================================
# 1. gather the z arrays for each upper_year (cached)
# ==================================================
print("==== extracting per-pair z ====")
z_by_year = {}
for upper_year in upper_year_list:
    z = extract_pair_z(upper_year)
    if z is not None and len(z) > 0:
        z_by_year[upper_year] = z

assert z_by_year, "no z data found for any upper_year (check compute_atypicality_s2.py outputs)"
years_drawn = sorted(z_by_year)
print("years with data:", years_drawn)

# ==================================================
# 2. summary table
# ==================================================
summary_rows = []
for y in years_drawn:
    z = z_by_year[y]
    pct = np.percentile(z, [5, 10, 25, 50, 75, 90, 95])
    summary_rows.append({
        'upper_year': y, 'n_pairs': int(len(z)),
        'mean': float(np.mean(z)), 'std': float(np.std(z)),
        'min': float(np.min(z)), 'max': float(np.max(z)),
        'p5': pct[0], 'p10': pct[1], 'p25': pct[2], 'median': pct[3],
        'p75': pct[4], 'p90': pct[5], 'p95': pct[6],
        'frac_z_gt_0': float(np.mean(z > 0)),
    })
summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(
    figure_dire, f'source_pair_z_distribution_{VERSION_TAG}_{SELF_TAG}_summary.csv')
summary_df.to_csv(summary_path, index=False)
print("[summary saved]", summary_path)
print(summary_df.to_string(index=False))

# ==================================================
# 3. plot: all cohorts overlaid on ONE axes, with a vertical median line per cohort
# ==================================================
# shared bin edges from the pooled clip percentiles so the years are comparable
pooled = np.concatenate([z_by_year[y] for y in years_drawn])
lo, hi = np.percentile(pooled, list(CLIP_PCT))
if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
    lo, hi = float(pooled.min()), float(pooled.max())
if lo == hi:
    hi = lo + 1.0
bins = np.linspace(lo, hi, N_BINS + 1)

cmap = plt.get_cmap('viridis')
colors = {y: cmap(i / max(1, len(years_drawn) - 1)) for i, y in enumerate(years_drawn)}

fig, ax = plt.subplots(figsize=(9, 5.5))
for y in years_drawn:
    z = z_by_year[y]
    med = float(np.median(z))
    color = colors[y]
    # distribution (step density) + a dashed vertical line at this cohort's median
    ax.hist(z, bins=bins, density=True, histtype='step', linewidth=1.8,
            color=color, label=f'{y} (n={len(z):,}, median={med:.2f})')
    ax.axvline(med, color=color, linestyle='--', linewidth=1.4, alpha=0.9)

ax.axvline(0, color='0.6', linestyle=':', linewidth=1.0)
ax.set_xlabel('Source-pair z-score', fontsize=14)
ax.set_ylabel('Density', fontsize=14)
ax.tick_params(labelsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(lo, hi)
ax.legend(title='upper_year (dashed line = median)', fontsize=10, title_fontsize=11,
          frameon=False)
fig.tight_layout()

out_svg = os.path.join(
    figure_dire, f'source_pair_z_distribution_{VERSION_TAG}_{SELF_TAG}.svg')
fig.savefig(out_svg, format='svg', transparent=True)
plt.close(fig)
print("[figure saved]", out_svg)
print("finished")
