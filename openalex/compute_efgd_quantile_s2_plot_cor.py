"""
Plot, from compute_efgd_quantile.py's output, the within-(year, domain)
relationship between the surprise indices F / G and the disruption index D, for
three citation-quantile subsets, pooling ALL publication years together.

For each domain (and for ALL domains pooled) we draw TWO figures, each of 8
subplots laid out 4 rows x 2 columns:
    - one based on the raw F / G / disp values        (fgd_raw_<domain>.png)
    - one based on the within-(year, domain) quantiles (fgd_quantile_<domain>.png)

        |   F vs D            |   G vs D
  -----+---------------------+---------------------
  cit  |  scatter + fit      |  scatter + fit       <- citation_quantile < 50
  cit  |  scatter + fit      |  scatter + fit       <- citation_quantile > 95
  cit  |  scatter + fit      |  scatter + fit       <- citation_quantile > 99
  cit  |  scatter + fit      |  scatter + fit       <- citation_quantile > 99.9 (top 0.1%)

In every subplot D is the x-axis; the y-axis is F (left column) or G (right
column). A red OLS line shows the linear relationship and the title reports the
Pearson correlation r with its 95% confidence interval (Fisher z-transform) and
the sample size. The x/y axes (spines)
are anchored at the center so they emanate from that point and split the plane
into four quadrants: in quantile mode the center is the 50th percentile (50, 50);
in raw mode it is the data median.

The citation subsets are always defined by the citation QUANTILE (cut at
50 / 95 / 99 / 99.9), regardless of whether the axes show raw values or quantiles.
Quantiles come from compute_efgd_quantile.py (rank(pct=True), in (0, 1]) and are
rescaled here to a 0-100 percentile for readability.

To avoid re-reading the (large) source CSV on every redraw, the slim columns we
actually plot are cached to a small CSV; delete that cache (or pass --refresh) to
rebuild it from the source.

Input  : META_DIRE/efg_computation/efgd_within_year_domain_quantile_y{off}_rt_{rt}_ct_{ct}_ybefore_{max_year}.csv
         CONCEPT_DIRE (Concepts_definition.xlsx)   (top_concept id -> display_name)
Output : META_DIRE/efgd_quantile_plots/_cache_fgd_quantile.csv          (slim redraw cache)
         ./figures/efgd_quantile_cor/fgd_raw_<domain>.svg               (raw values, per domain)
         ./figures/efgd_quantile_cor/fgd_quantile_<domain>.svg          (quantiles, per domain)
         ./figures/efgd_quantile_cor/fgd_{raw,quantile}_ALL.svg         (transparent SVG; all domains pooled)
"""

import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress, t

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

# raw index columns from compute_efgd_quantile.py
f_col = f'f_y{year_offset}'
g_col = f'g_y{year_offset}'
d_col = f'disp_y{year_offset}'
# quantile columns from compute_efgd_quantile.py (rank(pct=True), in (0, 1])
f_q_col = f'f_y{year_offset}_quantile'
g_q_col = f'g_y{year_offset}_quantile'
d_q_col = f'disp_y{year_offset}_quantile'
cit_q_col = f'citation_count_y{year_offset}_quantile'

# slim cache columns that every figure needs present
CACHE_COLS = ['domain', 'cit_q', 'f_q', 'g_q', 'd_q', 'f_raw', 'g_raw', 'd_raw']

efg_dire = os.path.join(meta_dire, 'efg_computation')
out_dir = os.path.join(meta_dire, 'efgd_quantile_plots')
os.makedirs(out_dir, exist_ok=True)
cache_path = os.path.join(out_dir, '_cache_fgd_quantile.csv')

# figures go into ./figures/efgd_quantile_cor (next to this script, works from any CWD)
fig_dir = os.path.join(BASE_DIR, 'figures', 'efgd_quantile_cor')
os.makedirs(fig_dir, exist_ok=True)

# the citation-quantile subsets (on the 0-100 percentile scale)
SUBSETS = [
    ('citation quantile < 50', lambda q: q < 50),
    ('citation quantile > 95', lambda q: q > 95),
    ('citation quantile > 99', lambda q: q > 99),
    ('citation quantile > 99.9 (top 0.1%)', lambda q: q > 99.9),
]
MAX_SCATTER = 4000  # subsample points for the scatter only; the fit uses ALL points


# ==================================================
# 1. build (or load) the slim redraw cache: domain, cit_q, f_q, g_q, d_q  (0-100)
# ==================================================
def build_cache():
    src_path = os.path.join(
        efg_dire,
        f'efgd_within_year_domain_quantile_y{year_offset}_'
        f'rt_{reference_threshold}_ct_{citation_threshold}_ybefore_{max_year}.csv'
    )
    assert os.path.exists(src_path), (
        f"{src_path}\nRun compute_efgd_quantile.py first.")
    use_cols = ['top_concept', cit_q_col,
                f_q_col, g_q_col, d_q_col,    # quantiles
                f_col, g_col, d_col]          # raw values
    df = pd.read_csv(src_path, usecols=use_cols)
    print("loaded source quantile table:", df.shape)

    # top_concept id -> domain display name (same keying as the extract script)
    url_prefix_len = len("https://openalex.org/")
    df_concept = pd.read_excel(CONCEPT_DIRE)
    cid2name = {str(oid)[url_prefix_len:].lower(): name
                for oid, name in zip(df_concept['openalex_id'], df_concept['display_name'])}
    df['domain'] = df['top_concept'].map(
        lambda c: cid2name.get(str(c).lower(), str(c)))

    slim = pd.DataFrame({
        'domain': df['domain'],
        'cit_q':  df[cit_q_col] * 100.0,   # rescale (0,1] -> (0,100]
        'f_q':    df[f_q_col] * 100.0,
        'g_q':    df[g_q_col] * 100.0,
        'd_q':    df[d_q_col] * 100.0,
        'f_raw':  df[f_col],
        'g_raw':  df[g_col],
        'd_raw':  df[d_col],
    }).dropna(subset=CACHE_COLS[1:])  # all numeric cols required

    slim.to_csv(cache_path, index=False)
    print("wrote redraw cache:", cache_path, slim.shape)
    return slim


if REFRESH or not os.path.exists(cache_path):
    data = build_cache()
else:
    data = pd.read_csv(cache_path)
    # rebuild if an older cache is missing the newer (raw) columns
    if not set(CACHE_COLS).issubset(data.columns):
        print("cache missing columns", set(CACHE_COLS) - set(data.columns), "-> rebuilding")
        data = build_cache()
    else:
        print("loaded redraw cache:", cache_path, data.shape)


# ==================================================
# 2. one subplot: scatter + OLS line, slope with 95% CI in the title
# ==================================================
def fit_and_plot(ax, x, y, xlabel, ylabel, lim=None, center=None, note=None):
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
    if note:
        title = f"{note}\n{title}"

    if lim is not None:               # fixed range (quantile mode); else autoscale (raw)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)

    # put the x/y axes (spines) themselves at the center, so they emanate from
    # (cx, cy) and split the plane into 4 quadrants. In quantile mode that center
    # is the 50th percentile (50, 50); in raw mode it is the data median.
    if center is not None:
        cx, cy = center
    elif n > 0:
        cx, cy = np.median(x), np.median(y)
    else:
        cx, cy = 0.0, 0.0
    ax.spines['left'].set_position(('data', cx))
    ax.spines['bottom'].set_position(('data', cy))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_title(title, fontsize=8)
    # axis labels sit at the ends of the (centered) axes: x at far right, y at top
    ax.set_xlabel(xlabel, fontsize=9, loc='right')
    ax.set_ylabel(ylabel, fontsize=9, loc='top', rotation=0, labelpad=2)
    ax.tick_params(labelsize=7)


# ==================================================
# 3. one figure (6 subplots) for a given dataframe slice, in 'raw' or 'quantile' mode
# ==================================================
# per-mode: (D col, F col, G col, axis label suffix, fixed limits, spine center)
# quantile mode anchors the axes at the 50th percentile (50, 50); raw mode at the
# data median (center=None -> fit_and_plot uses np.median).
MODES = {
    'quantile': ('d_q',   'f_q',   'g_q',   'quantile', (0, 100), (50, 50)),
    'raw':      ('d_raw', 'f_raw', 'g_raw', 'index',    None,     None),
}


def make_figure(df_slice, title, out_path, mode):
    dcol, fcol, gcol, suffix, lim, center = MODES[mode]
    nrows = len(SUBSETS)
    fig, axes = plt.subplots(nrows, 2, figsize=(9, 4 * nrows))
    for r, (label, cond) in enumerate(SUBSETS):
        sub = df_slice[cond(df_slice['cit_q'])]
        # left column: F vs D ; right column: G vs D ; D always on x-axis
        fit_and_plot(axes[r, 0], sub[dcol], sub[fcol],
                     f'D {suffix}', f'F {suffix}', lim=lim, center=center, note=label)
        fit_and_plot(axes[r, 1], sub[dcol], sub[gcol],
                     f'D {suffix}', f'G {suffix}', lim=lim, center=center, note=label)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, format='svg', transparent=True)
    plt.close(fig)
    print("  wrote", out_path, f"({len(df_slice)} rows)")


def safe_name(name):
    return re.sub(r'[^\w\-]+', '_', str(name)).strip('_')[:80] or 'domain'


# ==================================================
# 4. per-domain figures (largest domains first) + an ALL-domains figure,
#    each in both raw-value and quantile mode
# ==================================================
def make_both(df_slice, title, tag):
    for mode in ('raw', 'quantile'):
        make_figure(df_slice, f"{title} [{mode}]",
                    os.path.join(fig_dir, f'fgd_{mode}_{tag}.svg'), mode)


make_both(data, "ALL domains (all years pooled)", 'ALL')

domain_order = data.groupby('domain').size().sort_values(ascending=False).index
for domain in domain_order:
    df_d = data[data['domain'] == domain]
    make_both(df_d, f"{domain} (all years pooled)", safe_name(domain))

print("figures dir:", fig_dir)
print("cache file:", cache_path)
print("finished")
