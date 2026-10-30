"""
Plot the yearly-average E/F/G trend under the CONTINUOUS specification for the WoS
data (Extended Data Fig. 6l). Reads the already-filtered gen_continous_V3 table from
arrange_efg_continous.py (which has applied the reference/citation thresholds) and
plots the yearly mean of Foundation (f_y), Extension (e_y), Generalization (g_y) over
publication year as three panels, each with an "All" line plus lines split by exact
citation_count (norm_targets). The tidy plot data is cached to CSV before drawing.
activity_year / reference_threshold / citation_threshold must match
arrange_efg_continous.py; the figure label is 'Extended Data Fig. 6l' when
(activity_year, citation_threshold, reference_threshold) == (5, 5, 1), else a
parameterized name.

Input  : META_DIRE/efg_computation/gen_continous_V3_y{activity_year}_rt_{reference_threshold}_ct_{citation_threshold}.csv (from arrange_efg_continous.py)
Output : ./figures/validation/generalization_continous/Extended Data Fig. 6l.svg + plot_cache/*.csv
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless / cluster-safe
import matplotlib.pyplot as plt  # noqa: E402

# utils.py is in the parent directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
from utils import META_DIRE  # noqa: E402

# try the reference font but fall back silently if unavailable (e.g. on Linux)
try:
    import matplotlib.font_manager as fm
    if any("Gill Sans MT" in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = "Gill Sans MT"
except Exception:
    pass

# ----------------- Inputs -----------------
meta_dire = META_DIRE
print("meta_dire", meta_dire)

activity_year = 5
print("activity_year", activity_year)

figure_dire_save = './figures/validation/generalization_continous'
os.makedirs(figure_dire_save, exist_ok=True)

reference_threshold = 1
citation_threshold = 5
print("citation_threshold", citation_threshold)
print("reference_threshold", reference_threshold)

# input is arrange_efg_continous.py's filtered output
load_dire = os.path.join(meta_dire, 'efg_computation')
citation_col = f'citation_count_y{activity_year}'
use_cols = ['pub_year', citation_col,
            f'f_y{activity_year}', f'e_y{activity_year}', f'g_y{activity_year}']
indices = {
    'Foundation': f'f_y{activity_year}',
    'Extension': f'e_y{activity_year}',
    'Generalization': f'g_y{activity_year}',
}

# which "by citation count" lines to include
norm_targets = [1, 2, 3, 5, 10, 20, 30]
norm_targets = [target for target in norm_targets if target >= citation_threshold]
print("norm_targets", norm_targets)

# Colors for drawing
COLORS = [
    'black', 'tab:blue', 'tab:green', 'tab:red', 'tab:orange',
    'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
]

# Fixed colour per citation-count value (built from the full, unfiltered target list)
# so a given citation count uses the SAME colour in every figure, regardless of
# citation_threshold. Keep this mapping identical across all visualize_efg_* scripts.
CITATION_COLOR = {c: col for c, col in zip([1, 2, 3, 5, 10, 20, 30], COLORS[1:])}


def _legend_title_for_index(index_name: str) -> str:
    return 'Citation Count'


def _norm_col_for_index(index_name: str) -> str:
    return f'citation_count_y{activity_year}'


def _load_continuous_output() -> pd.DataFrame:
    """Read the already-filtered gen_continous_V3 table from arrange_efg_continous.py
    (reference/citation thresholds already applied there)."""
    path = os.path.join(
        load_dire,
        f'gen_continous_V3_y{activity_year}_rt_{reference_threshold}_ct_{citation_threshold}.csv')
    return pd.read_csv(path, usecols=use_cols)


# ----------------- Build & Save Plot Cache -----------------
def build_and_save_plot_cache(cache_path: str) -> str:
    """
    Build a tidy CSV that contains everything needed for the plot.

    Output columns:
      activity_year, reference_threshold, citation_threshold, index_name,
      legend_title, norm_kind (all/by_norm), norm_col, norm_value, pub_year,
      mean_value, n
    """
    df = _load_continuous_output()

    # reference/citation thresholds are already applied by arrange_efg_continous.py;
    # here we only bound the publication year to the plot range
    df = df[df['pub_year'] < (2024 - activity_year + 1)]
    df = df[df['pub_year'] >= 1945]

    rows = []
    for index_name, index_col in indices.items():
        df_use = df

        # --- ALL line
        g_all = df_use.groupby('pub_year')[index_col].agg(['mean', 'count']).reset_index()
        for _, r in g_all.iterrows():
            rows.append({
                'activity_year': activity_year,
                'reference_threshold': reference_threshold,
                'citation_threshold': citation_threshold,
                'index_name': index_name,
                'legend_title': _legend_title_for_index(index_name),
                'norm_kind': 'all',
                'norm_col': '',
                'norm_value': np.nan,
                'pub_year': int(r['pub_year']),
                'mean_value': float(r['mean']),
                'n': int(r['count']),
            })

        # --- lines split by exact citation count
        norm_col = _norm_col_for_index(index_name)
        for c in norm_targets:
            sub = df_use[df_use[norm_col] == c]
            if len(sub) == 0:
                continue
            g = sub.groupby('pub_year')[index_col].agg(['mean', 'count']).reset_index()
            for _, r in g.iterrows():
                rows.append({
                    'activity_year': activity_year,
                    'reference_threshold': reference_threshold,
                    'citation_threshold': citation_threshold,
                    'index_name': index_name,
                    'legend_title': _legend_title_for_index(index_name),
                    'norm_kind': 'by_norm',
                    'norm_col': norm_col,
                    'norm_value': int(c),
                    'pub_year': int(r['pub_year']),
                    'mean_value': float(r['mean']),
                    'n': int(r['count']),
                })

    cache_df = pd.DataFrame(rows).sort_values(['index_name', 'norm_kind', 'norm_value', 'pub_year'])
    cache_df.to_csv(cache_path, index=False)
    print(f"[Cache saved] {cache_path} (rows={len(cache_df)})")
    return cache_path


# ----------------- Draw Only From Cache -----------------
def plot_from_cache_csv(cache_csv_path: str, out_svg_path: str):
    dfc = pd.read_csv(cache_csv_path)

    if dfc.empty:
        print(f"No data in {cache_csv_path}")
        return

    index_order_pref = ['Foundation', 'Extension', 'Generalization']
    present = [ix for ix in index_order_pref if ix in dfc['index_name'].unique().tolist()]
    if not present:
        print("No recognized indices to plot.")
        return

    fig_w = 6 * len(present)
    fig, axes = plt.subplots(1, len(present), figsize=(fig_w, 5), sharex=True)
    if len(present) == 1:
        axes = [axes]

    for i, index_name in enumerate(present):
        ax = axes[i]
        dfi = dfc[dfc['index_name'] == index_name].copy()
        legend_title = dfi['legend_title'].dropna().iloc[0] if 'legend_title' in dfi.columns else _legend_title_for_index(index_name)

        # --- ALL line
        df_all = dfi[dfi['norm_kind'] == 'all'].sort_values('pub_year')
        if not df_all.empty:
            ax.plot(df_all['pub_year'], df_all['mean_value'], label='All', color=COLORS[0], linewidth=2)

        # --- lines split by citation count
        for j, c in enumerate(norm_targets):
            df_c = dfi[(dfi['norm_kind'] == 'by_norm') & (dfi['norm_value'] == c)].sort_values('pub_year')
            if not df_c.empty:
                ax.plot(df_c['pub_year'], df_c['mean_value'], label=f"{int(c)}", color=CITATION_COLOR[c], linestyle='--', linewidth=1.8)

        # index name as a text label in the upper-left corner (no subplot title)
        ax.text(0.03, 0.97, index_name, transform=ax.transAxes,
                ha='left', va='top', fontsize=18)
        ax.set_ylabel('Average index', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # legend hidden

        # x ticks every 10 years
        xticks = list(range(1949, 2021, 10))
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) for x in xticks])

    # single shared x-label centred under the row (instead of one per subplot).
    # reserve a small bottom band and place the label in it, just below the ticks.
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16)
    fig.supxlabel('Publication Year', fontsize=20, y=0.06)
    plt.savefig(out_svg_path, format='svg', transparent=True)
    plt.close()
    print(f"[Figure saved] {out_svg_path}")


# ----------------- Main -----------------
# Build cache CSV (once)
cache_dir = os.path.join(figure_dire_save, 'plot_cache')
os.makedirs(cache_dir, exist_ok=True)

if activity_year == 5 and citation_threshold == 5 and reference_threshold == 1:
    out_svg = os.path.join(figure_dire_save, 'Extended Data Fig. 6l.svg')
    cache_name = 'Extended Data Fig. 6l.csv'
else:
    out_svg = os.path.join(
        figure_dire_save,
        f"contious_gen_ay{activity_year}_c_t_{citation_threshold}_r_t_{reference_threshold}_fromcache.svg"
    )
    cache_name = (
        f"plotcache_generalization_continous_ay{activity_year}_ct{citation_threshold}_rt{reference_threshold}.csv"
    )

cache_path = os.path.join(cache_dir, cache_name)
if not os.path.exists(cache_path):
    cache_path = build_and_save_plot_cache(cache_path=cache_path)

plot_from_cache_csv(cache_path, out_svg)

print("finished")
