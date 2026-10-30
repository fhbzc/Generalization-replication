import os
import pickle
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import sys

# utils.py is in the parent directory; import META_DIRE so this reads from EXACTLY where
# compute_atypicality_s3.py writes (META_DIRE/atypicality/atypicality_all_after_{year}.csv).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
from utils import META_DIRE  # noqa: E402

meta_dire = META_DIRE
print("meta_dire", meta_dire)



figure_dire = './figures/atypicality'
os.makedirs(figure_dire, exist_ok = True)

# Per-figure plot cache: the minimal data needed to redraw (Z_COL + the hue column) is
# cached here; if the cache exists it is read directly and the big source CSVs are not
# touched. Delete a cache file to force a refresh.
cache_dir = os.path.join(figure_dire, 'plot_cache')
os.makedirs(cache_dir, exist_ok = True)

# ---------------------------------------------------------------------------
# Pick which version of compute_atypicality_s3.py output to visualize:
#   * DO_REMOVE = False -> "full" version,  atypicality_all_after_{year}.csv
#   * DO_REMOVE = True  -> "removed" version, atypicality_all_after_{year}_removed.csv
# ---------------------------------------------------------------------------
DO_REMOVE = True
VERSION_TAG = 'removed' if DO_REMOVE else 'full'
FILE_SUFFIX = '_removed' if DO_REMOVE else ''
print("VERSION_TAG", VERSION_TAG)

# Column holding the per-paper median z-score in compute_atypicality_s3.py output.
Z_COL = 'Z_median'

# ---------------------------------------------------------------------------
# Two visualization modes:
#   MODE = 'per_atypicality_year' (Ours): fix ONE atypicality-year file
#       (embedding_year_before -> atypicality_all_after_{year}.csv) and split its
#       papers into 10-year publication periods -> one ECDF line per decade, all
#       scored against that SAME single baseline.
#   MODE = 'contemporaneous_years' (Declining disruption): one ECDF line per year Y, where each line
#       is the papers published in Y scored against the year-Y baseline
#       (atypicality_all_after_{Y}.csv, computed from papers up to Y). So the 2020
#       line = 2020 papers vs the 2020 baseline, the 2000 line = 2000 papers vs the
#       2000 baseline, etc. -- DIFFERENT papers per line, each vs its OWN baseline.
# ---------------------------------------------------------------------------
MODE = 'per_atypicality_year'
# list of years driving BOTH modes:
#   * per_atypicality_year  -> one figure per year Y (that year's atypicality_all_after_{Y}.csv)
#   * contemporaneous_years -> one ECDF line per year Y (papers in [Y, Y+CONTEMP_WINDOW) vs the year-Y baseline)
YEARS = [1960, 1970, 1980, 1990, 2000, 2010]  # each needs a matching atypicality_all_after_{Y}.csv
CONTEMP_WINDOW = 10   # (contemporaneous_years only) cohort width per line: 1 = exactly year Y; 10 = decade [Y, Y+10)
print("MODE", MODE)
print("CONTEMP_WINDOW", CONTEMP_WINDOW)


def _format_neg_pow2(x, pos):
    if x == 0:
        return '0'
    elif x < 0:
        return f'$-2^{{{int(np.log2(-x))}}}$'
    else:
        return f'$2^{{{int(np.log2(x))}}}$'


def _draw_ecdf_on_ax(ax, df_all, hue_col, title=None, show_text=True,
                     show_legend=True, axis_fontsize=18,
                     show_xlabel=True, show_ylabel=True):
    """Draw a single ECDF (one line per value of hue_col) onto the given Axes,
    applying the shared styling. Used by both the single-figure renderer and the
    combined subplot grid."""
    x_ticks = [-2**i for i in range(9, 0, -1)] + [0] + [2**i for i in range(1, 15)]
    sns.ecdfplot(data=df_all, x=Z_COL, hue=hue_col, ax=ax)
    if show_text:
        ax.text(-20, 0.3, 'Novel\ncombinations', ha='center', va='center', fontsize=axis_fontsize)
        ax.text(1000, 0.3, 'Conventional\ncombinations', ha='center', va='center', fontsize=axis_fontsize)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(_format_neg_pow2))
    ax.set_xlim(-2**9, 2**15)
    ax.axvspan(-2**9, 0, facecolor='#fddede', alpha=0.5, zorder=0)   # light red
    ax.axvspan(0, 2**15, facecolor='#fffdd0', alpha=0.5, zorder=0)    # light yellow
    ax.axvline(0, color='black', linewidth=2, zorder=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xscale('symlog', base=2, linthresh=1)  # linthresh defines the linear region around 0
    ax.set_xlabel('Median Z-score' if show_xlabel else '', fontsize=axis_fontsize)
    ax.set_ylabel('Cumulative Probability' if show_ylabel else '', fontsize=axis_fontsize)
    if title is not None:
        ax.set_title(title, fontsize=axis_fontsize)
    # put the (seaborn hue) legend uniformly in the upper-left corner
    if ax.get_legend() is not None:
        if show_legend:
            sns.move_legend(ax, 'upper left')
        else:
            ax.get_legend().remove()


def _plot_ecdf(df_all, hue_col, save_dire, title='Cumulative Distribution of Atypicality'):
    """Shared ECDF rendering: one line per value of hue_col, styled identically for
    both modes."""
    plt.figure(figsize=(8, 5))
    _draw_ecdf_on_ax(plt.gca(), df_all, hue_col)
    plt.savefig(save_dire,  format='svg')
    plt.close()


def _read_year_csv(atypicality_year):
    """Read one atypicality-year file (full table) with the standard filters applied;
    None if the input is missing."""
    csv_path = os.path.join(meta_dire, 'atypicality',
                            'atypicality_all_after_%s%s.csv' % (atypicality_year, FILE_SUFFIX))
    if not os.path.exists(csv_path):
        print("skip (input missing):", csv_path)
        return None
    df = pd.read_csv(csv_path, usecols=['pid', Z_COL, 'up_refer_count', 'pub_year'])
    df = df[df[Z_COL] != -999]
    df = df[df['up_refer_count'] >= 2]
    return df


def build_periods_df(atypicality_year):
    """Original mode: read the single atypicality-year file and split its papers into
    10-year publication periods. Returns df with [Z_COL, 'period'] (None if no data)."""
    df_data = _read_year_csv(atypicality_year)
    if df_data is None:
        return None
    focal_period = atypicality_year
    list_of_df = []
    while focal_period < 2030:
        sub_df = df_data[df_data['pub_year'] >= focal_period]
        sub_df = sub_df[sub_df['pub_year'] < (focal_period + 10)].copy()
        sub_df['period'] = '%s-%s'%(focal_period, focal_period + 9)
        list_of_df.append(sub_df)
        focal_period += 10
    if not list_of_df:
        return None
    return pd.concat(list_of_df)


def build_contemporaneous_df(years, window):
    """New mode: one group per year Y = papers published in [Y, Y+window) scored
    against the year-Y baseline (atypicality_all_after_{Y}.csv). Returns df with
    [Z_COL, 'Publication year'] (None if no data). Different groups are different
    papers, each measured against its OWN contemporaneous baseline."""
    list_of_df = []
    for y in years:
        df = _read_year_csv(y)
        if df is None:
            continue
        df = df[df['pub_year'] >= y]
        df = df[df['pub_year'] < (y + window)].copy()
        if len(df) == 0:
            print("no valid papers for year", y)
            continue
        label = str(y) if window == 1 else '%s-%s' % (y, y + window - 1)
        df['Publication year'] = label
        list_of_df.append(df)
        print("year %s: %s papers (window %s)" % (y, len(df), window))
    if not list_of_df:
        return None
    return pd.concat(list_of_df)


def render(cache_path, out_svg, hue_col, build_fn, title='Cumulative Distribution of Atypicality'):
    """Draw from cache if present, otherwise build the (Z_COL, hue_col) frame, cache it,
    and draw. The cache holds only what the ECDF needs, so a redraw never touches the
    big source CSVs. Returns the drawn (Z_COL, hue_col) frame (None if there was no
    data) so callers can reuse it without re-reading the cache from disk."""
    if os.path.exists(cache_path):
        df_all = pd.read_csv(cache_path)
        print("[cache] loaded:", cache_path, df_all.shape)
    else:
        df_all = build_fn()
        if df_all is None or len(df_all) == 0:
            print("no data to plot ->", out_svg)
            return None
        df_all = df_all[[Z_COL, hue_col]].copy()
        df_all.to_csv(cache_path, index=False)
        print("[cache] saved:", cache_path, df_all.shape)
    _plot_ecdf(df_all, hue_col, out_svg, title=title)
    return df_all


def _per_year_cache_path(year):
    return os.path.join(
        cache_dir, 'cache_per_atypicality_year_%s%s.csv' % (year, FILE_SUFFIX))


def plot_combined_per_atypicality_year(loaded_dfs):
    """One subplot per atypicality-year, all in a single figure. `loaded_dfs` is the
    {year: (Z_COL, 'period') frame} already loaded by the per-year loop, so this reuses
    those frames directly and never re-reads a cache or touches a source CSV."""
    out_svg = os.path.join(
        figure_dire, 'full_combined_per_atypicality_year%s.svg' % FILE_SUFFIX)

    # Keep only years that actually have data, in YEARS order.
    year_dfs = [(y, df) for y, df in loaded_dfs.items() if df is not None]
    if not year_dfs:
        print("no data for any year -> combined figure skipped")
        return

    n = len(year_dfs)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5 * nrows),
                             squeeze=False)
    flat_axes = axes.flatten()
    # Suppress every per-axes x/y label; a single shared label per figure is added below.
    for ax, (y, df_y) in zip(flat_axes, year_dfs):
        _draw_ecdf_on_ax(ax, df_y, 'period', title='Venue-level year %s' % y,
                         show_xlabel=False, show_ylabel=False)
    # hide any unused subplot cells
    for ax in flat_axes[len(year_dfs):]:
        ax.set_visible(False)

    # Reserve margin on the left/bottom so the shared labels sit clear of the tick numbers.
    fig.tight_layout(rect=[0.03, 0.05, 1, 1])
    # One shared xlabel centered along the bottom, one shared ylabel centered on the left.
    fig.supxlabel('Median Z-score', fontsize=22, y=0.01)
    fig.supylabel('Cumulative Probability', fontsize=22, x=0.005)
    fig.savefig(out_svg, format='svg')
    plt.close(fig)
    print("saved combined:", out_svg)


if MODE == 'per_atypicality_year':
    # Keep each year's drawn frame in memory (in YEARS order) so the combined figure
    # below can reuse it without re-reading the per-year caches from disk.
    loaded_dfs = OrderedDict()
    for embedding_year_before in tqdm(YEARS, position = 0, desc = 'plot full graph'):
        out_svg = os.path.join(
            figure_dire, 'full_embed_before_%s%s.svg' % (embedding_year_before, FILE_SUFFIX))
        cache_path = _per_year_cache_path(embedding_year_before)
        loaded_dfs[embedding_year_before] = render(
            cache_path, out_svg, 'period',
            (lambda y=embedding_year_before: build_periods_df(y)))

    # Combined figure: one subplot per year, drawn from the frames already loaded above.
    plot_combined_per_atypicality_year(loaded_dfs)

elif MODE == 'contemporaneous_years':
    out_svg = os.path.join(
        figure_dire, 'contemporaneous_years_w%s%s.svg' % (CONTEMP_WINDOW, FILE_SUFFIX))
    cache_path = os.path.join(
        cache_dir,
        'cache_contemporaneous_years_w%s%s_%s.csv' % (
            CONTEMP_WINDOW, FILE_SUFFIX, '_'.join(str(y) for y in YEARS)))
    render(cache_path, out_svg, 'Publication year',
           (lambda: build_contemporaneous_df(YEARS, CONTEMP_WINDOW)),
           title='Atypicality by publication year (each year vs its own baseline)')
    print("saved:", out_svg)

else:
    raise ValueError("unknown MODE: %s" % MODE)

print("finished")