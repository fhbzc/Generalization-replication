"""
Plot, from compute_efgd_quantile.py's output, the relationship between a paper's
CITATION (as its within-(year, domain) citation quantile) and each of its four
E / F / G / D indices (as RAW values), pooling ALL publication years together.

For each domain (and for ALL domains pooled) we draw ONE figure of 4 subplots
(2 rows x 2 columns), one per index E / F / G / D:

        |   E vs citation     |   F vs citation
  ------+--------------------+--------------------
        |   G vs citation     |   D vs citation

In every subplot the x-axis is the citation quantile (0-100) and the y-axis is the
raw index value. A red OLS line shows the linear relationship and the title reports
the Pearson correlation r with its 95% confidence interval (Fisher z-transform) and
the sample size. The x/y axes (spines) are anchored at the center (citation quantile
50, index median) so they split the plane into four quadrants.

Citation quantile comes from compute_efgd_quantile.py (rank(pct=True), in (0, 1])
and is rescaled here to a 0-100 percentile; the E/F/G/D values are the raw indices.

To avoid re-reading the (large) source CSV on every redraw, the slim columns we
actually plot are cached to a small CSV; delete that cache (or pass --refresh) to
rebuild it from the source.

Input  : META_DIRE/efg_computation/efgd_within_year_domain_quantile_y{off}_rt_{rt}_ct_{ct}_ybefore_{max_year}.csv
         CONCEPT_DIRE (Concepts_definition.xlsx)   (top_concept id -> display_name)
Output : META_DIRE/efgd_quantile_plots/_cache_citation_efg.csv          (slim redraw cache)
         ./figures/citation_efg_cor/citation_efg_<domain>.svg           (transparent SVG, per domain)
         ./figures/citation_efg_cor/citation_efg_ALL.svg                (all domains pooled)
"""

import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress

# utils.py is in this same directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from utils import META_DIRE, CONCEPT_DIRE  # noqa: E402

meta_dire = META_DIRE
print("meta_dire", meta_dire)

REFRESH = '--refresh' in sys.argv  # force rebuild the slim cache from the source CSV

# ---- must match compute_efgd_quantile.py so the input filename resolves ----
year_offset = 5
reference_threshold = 1
citation_threshold = 5
max_year = 2025 - year_offset

# raw E/F/G/D index columns + the citation-quantile column from compute_efgd_quantile.py
e_col = f'e_y{year_offset}'
f_col = f'f_y{year_offset}'
g_col = f'g_y{year_offset}'
d_col = f'disp_y{year_offset}'
cit_q_col = f'citation_count_y{year_offset}_quantile'

# (display label, raw source column, slim cache column)
INDICES = [
    ('E', e_col, 'e_raw'),
    ('F', f_col, 'f_raw'),
    ('G', g_col, 'g_raw'),
    ('D', d_col, 'd_raw'),
]

efg_dire = os.path.join(meta_dire, 'efg_computation')
out_dir = os.path.join(meta_dire, 'efgd_quantile_plots')
os.makedirs(out_dir, exist_ok=True)
cache_path = os.path.join(out_dir, '_cache_citation_efg.csv')

# figures go into ./figures/citation_efg_cor (next to this script, works from any CWD)
fig_dir = os.path.join(BASE_DIR, 'figures', 'citation_efg_cor')
os.makedirs(fig_dir, exist_ok=True)

CACHE_COLS = ['domain', 'cit_q'] + [slim for _, _, slim in INDICES]
MAX_SCATTER = 4000  # subsample points for the scatter only; the fit uses ALL points


# ==================================================
# 1. build (or load) the slim redraw cache: domain, cit_q (0-100), e/f/g/d raw
# ==================================================
def build_cache():
    src_path = os.path.join(
        efg_dire,
        f'efgd_within_year_domain_quantile_y{year_offset}_'
        f'rt_{reference_threshold}_ct_{citation_threshold}_ybefore_{max_year}.csv'
    )
    assert os.path.exists(src_path), (
        f"{src_path}\nRun compute_efgd_quantile.py first.")
    use_cols = ['top_concept', cit_q_col] + [src for _, src, _ in INDICES]
    df = pd.read_csv(src_path, usecols=use_cols)
    print("loaded source quantile table:", df.shape)

    # top_concept id -> domain display name (same keying as the extract script)
    url_prefix_len = len("https://openalex.org/")
    df_concept = pd.read_excel(CONCEPT_DIRE)
    cid2name = {str(oid)[url_prefix_len:].lower(): name
                for oid, name in zip(df_concept['openalex_id'], df_concept['display_name'])}

    slim = pd.DataFrame({'domain': df['top_concept'].map(
        lambda c: cid2name.get(str(c).lower(), str(c)))})
    slim['cit_q'] = df[cit_q_col] * 100.0   # rescale (0,1] -> (0,100]
    for _, src, slim_col in INDICES:
        slim[slim_col] = df[src]
    slim = slim.dropna(subset=CACHE_COLS[1:])  # all numeric cols required

    slim.to_csv(cache_path, index=False)
    print("wrote redraw cache:", cache_path, slim.shape)
    return slim


if REFRESH or not os.path.exists(cache_path):
    data = build_cache()
else:
    data = pd.read_csv(cache_path)
    if not set(CACHE_COLS).issubset(data.columns):
        print("cache missing columns", set(CACHE_COLS) - set(data.columns), "-> rebuilding")
        data = build_cache()
    else:
        print("loaded redraw cache:", cache_path, data.shape)


# ==================================================
# 2. one subplot: scatter + OLS line, Pearson r with 95% CI in the title
# ==================================================
def fit_and_plot(ax, x, y, xlabel, ylabel):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)

    # scatter (subsample for display only; the fit below uses every point)
    if n > MAX_SCATTER:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, MAX_SCATTER, replace=False)
        ax.scatter(x[idx], y[idx], s=3, alpha=0.15, color='steelblue', linewidths=0)
    else:
        ax.scatter(x, y, s=3, alpha=0.15, color='steelblue', linewidths=0)

    if n >= 4 and np.ptp(x) > 0 and np.ptp(y) > 0:
        res = linregress(x, y)
        r = res.rvalue
        # 95% CI for Pearson r via the Fisher z-transform
        z = np.arctanh(r)
        se = 1.0 / np.sqrt(n - 3)
        lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, res.intercept + res.slope * xs, color='crimson', lw=1.8, zorder=3)
        title = f"r={r:.3g} [{lo:.3g}, {hi:.3g}], n={n}"
    else:
        title = f"n={n} (too few)"

    # citation quantile axis is fixed 0-100; the raw index axis autoscale
    ax.set_xlim(0, 100)

    # anchor the x/y axes (spines) at the center: citation quantile 50, index median
    cy = np.median(y) if n > 0 else 0.0
    ax.spines['left'].set_position(('data', 50))
    ax.spines['bottom'].set_position(('data', cy))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_title(title, fontsize=8)
    # axis labels sit at the ends of the (centered) axes: x at far right, y at top
    ax.set_xlabel(xlabel, fontsize=9, loc='right')
    ax.set_ylabel(ylabel, fontsize=9, loc='top', rotation=0, labelpad=2)
    ax.tick_params(labelsize=7)


# ==================================================
# 3. one figure (4 subplots, one per index) for a given dataframe slice
# ==================================================
def make_figure(df_slice, title, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for (label, _, slim_col), ax in zip(INDICES, axes.flat):
        fit_and_plot(ax, df_slice['cit_q'], df_slice[slim_col],
                     'citation quantile', f'{label} index')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, format='svg', transparent=True)
    plt.close(fig)
    print("  wrote", out_path, f"({len(df_slice)} rows)")


def safe_name(name):
    return re.sub(r'[^\w\-]+', '_', str(name)).strip('_')[:80] or 'domain'


# ==================================================
# 4. per-domain figures (largest domains first) + an ALL-domains figure
# ==================================================
make_figure(data, "ALL domains (all years pooled)",
            os.path.join(fig_dir, 'citation_efg_ALL.svg'))

domain_order = data.groupby('domain').size().sort_values(ascending=False).index
for domain in domain_order:
    df_d = data[data['domain'] == domain]
    make_figure(df_d, f"{domain} (all years pooled)",
                os.path.join(fig_dir, f'citation_efg_{safe_name(domain)}.svg'))

print("figures dir:", fig_dir)
print("cache file:", cache_path)
print("finished")
