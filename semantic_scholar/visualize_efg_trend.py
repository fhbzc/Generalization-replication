"""
Semantic Scholar (S2AG) version of the OpenAlex visualize_efg_trend.py.

Plots the average Foundation / Extension / Generalization shares over publication
year (an "All" line plus one dashed line per fixed citation count) from the merged
E/F/G table produced by get_efg_s2_merge.py.

The one network variant on the S2 side is the OpenAlex-augmented remove_circle
network (augment_s2_citation_with_openalex.py): set use_openalex_augmented to match
get_efg.py / get_efg_s2_merge.py. It is encoded via network_suffix
('_openalex_augmented'), appended to the input filename AND the output figure/cache
names so the augmented and non-augmented variants never collide. The named "Extended
Data Fig." slots are reserved for the OpenAlex main analysis, so this S2 robustness
version always writes a descriptive, dataset-tagged filename.

Input  : META_DIRE/efg_computation/gen_V3_y{activity_year}[_openalex_augmented].csv  (get_efg_s2_merge.py)
Output : ./figures/generalization/*.svg  (+ plot_cache/*.csv)
"""

import pickle
import pandas as pd
import os
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)               # utils.py is in this same directory
from utils import META_DIRE
# ----------------- Inputs -----------------
meta_dire = META_DIRE
plt.rcParams['font.family'] = 'Gill Sans MT'

activity_year = 5
print("activity_year", activity_year)

figure_dire_save = './figures/generalization'
os.makedirs(figure_dire_save, exist_ok=True)

reference_threshold = 1
citation_threshold = 5
print("citation_threshold", citation_threshold)
print("reference_threshold", reference_threshold)

# must match get_efg.py / get_efg_s2_merge.py: when True, read the table computed on
# the OpenAlex-augmented remove_circle network (augment_s2_citation_with_openalex.py).
# network_suffix is appended to the input filename AND the output figure/cache names.
use_openalex_augmented = True
network_suffix = '_openalex_augmented' if use_openalex_augmented else ''
print("use_openalex_augmented", use_openalex_augmented)
print("network_suffix", repr(network_suffix))

# input is get_efg_s2_merge.py's output; this must match that script's settings
gen_file_load_dire = os.path.join(
    meta_dire, 'efg_computation',
    f'gen_V3_y{activity_year}{network_suffix}.csv'
)
print("gen_file_load_dire", gen_file_load_dire)

# What lines to include (same as your plot)
norm_targets = [2, 3, 5, 10, 20, 30]
print("norm_targets", norm_targets)


norm_targets = [target for target in norm_targets if target >= citation_threshold]
# Colors for drawing
COLORS = [
    'black', 'tab:blue', 'tab:green', 'tab:red', 'tab:orange',
    'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
]
# ----------------- Helpers -----------------
def _indices_and_usecols(activity_year: int):
    """
    Return (use_col_list, indices_dict) for the normalized shares
    (f_y / e_y / g_y) from get_efg_s2_merge.py output.
    """
    use_col_list = [
        "pub_year", "up_refer_count", f'citation_count_y{activity_year}',
        f'f_y{activity_year}',
        f'e_y{activity_year}',
        f'g_y{activity_year}',
    ]
    indices = {
        'Foundation': f'f_y{activity_year}',
        'Extension': f'e_y{activity_year}',
        'Generalization': f'g_y{activity_year}',
    }
    return use_col_list, indices


def _legend_title_for_index(index_name: str) -> str:
    return 'Citation Count'



def _norm_col_for_index(index_name: str) -> str:


    return f'citation_count_y{activity_year}'

def _read_source_csv(use_col_list: list) -> pd.DataFrame:
    """
    Reads the get_efg_s2_merge.py output (full table, all papers).
    """
    df = pd.read_csv(gen_file_load_dire, usecols=use_col_list)

    return df


# ----------------- Build & Save Plot Cache -----------------
def build_and_save_plot_cache(cache_dir: str,
                              topic: str = 'full') -> str:
    """
    Build a tidy CSV that contains everything needed for the plot.

    Output columns:
      topic, activity_year, reference_threshold, citation_threshold,
      index_name, norm_kind (all/by_norm),
      norm_col, norm_value, pub_year, mean_value, n
    """

    use_col_list, indices = _indices_and_usecols(activity_year)
    df = _read_source_csv(use_col_list)

    # Filtering to match your plotting logic
    # dynamic pub_year cutoff so the citation window fits within the data
    df = df[df['pub_year'] < (2024 - activity_year + 1)]
    df = df[df['up_refer_count'] >= reference_threshold]
    df = df[df[f'citation_count_y{activity_year}'] >= citation_threshold]
    # For Generalization norm

    df = df.copy()
    df['citation_up_sum'] = df['up_refer_count'] + df[f'citation_count_y{activity_year}']

    rows = []

    for index_name, index_col in indices.items():
        # For Generalization_exc you sometimes filter out negatives; if present, do it.
        if index_name == 'Generalization_exc' and f'generalization_index_y{activity_year}_exc_gen_one' in df.columns:
            df_use = df[df[f'generalization_index_y{activity_year}_exc_gen_one'] >= 0].copy()
        else:
            df_use = df

        # --- ALL line
        g_all = df_use.groupby('pub_year')[index_col].agg(['mean', 'count']).reset_index()
        for _, r in g_all.iterrows():
            rows.append({
                'topic': topic,
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

        # --- Normed lines (5/10/20/30)
        norm_col = _norm_col_for_index(index_name)
        for c in norm_targets:
            sub = df_use[df_use[norm_col] == c]
            if len(sub) == 0:
                continue
            g = sub.groupby('pub_year')[index_col].agg(['mean', 'count']).reset_index()
            for _, r in g.iterrows():
                rows.append({
                    'topic': topic,
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
    # Make a deterministic, informative filename

    cache_df.to_csv(cache_dir, index=False)
    print(f"[Cache saved] {cache_dir} (rows={len(cache_df)})")
    return cache_dir


# ----------------- Draw Only From Cache -----------------
def plot_from_cache_csv(cache_csv_path: str, out_svg_path: str):
    dfc = pd.read_csv(cache_csv_path)

    if dfc.empty:
        print(f"No data in {cache_csv_path}")
        return

    # what panels do we have?
    index_order_pref = ['Foundation', 'Extension', 'Generalization']
    present = [ix for ix in index_order_pref if ix in dfc['index_name'].unique().tolist()]
    if not present:
        print("No recognized indices to plot.")
        return

    # Make figure
    fig_w = 6 * len(present)
    fig, axes = plt.subplots(1, len(present), figsize=(fig_w, 5), sharex=True)
    if len(present) == 1:
        axes = [axes]

    for i, index_name in enumerate(present):
        ax = axes[i]
        dfi = dfc[dfc['index_name'] == index_name].copy()
        # Legend title (stored per row; take the first)
        legend_title = dfi['legend_title'].dropna().iloc[0] if 'legend_title' in dfi.columns else _legend_title_for_index(index_name)

        # --- ALL line
        df_all = dfi[dfi['norm_kind'] == 'all'].sort_values('pub_year')
        if not df_all.empty:
            ax.plot(df_all['pub_year'], df_all['mean_value'], label='All', color=COLORS[0], linewidth=2)

        # --- Norm lines
        for j, c in enumerate(norm_targets):
            df_c = dfi[(dfi['norm_kind'] == 'by_norm') & (dfi['norm_value'] == c)].sort_values('pub_year')
            if not df_c.empty:
                ax.plot(df_c['pub_year'], df_c['mean_value'], label=f"{int(c)}", color=COLORS[j+1], linestyle='--', linewidth=1.8)

        # index name as a text label in the upper-left corner (no subplot title)
        ax.text(0.03, 0.97, index_name, transform=ax.transAxes,
                ha='left', va='top', fontsize=18)
        if i == 0:
            ax.set_ylabel('Average index', fontsize=20)
        else:
            ax.set_ylabel("")
        ax.tick_params(axis='both', which='major', labelsize=12)         # tick label sizes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # only draw the (citation-count) legend on the leftmost subplot
        if i == 0:
            ax.legend(title=legend_title, loc='upper right', fontsize=12, title_fontsize=14)

        xticks = list(range(1953, 2024, 10))  # 1953, 1963, ..., 2013, 2023
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


# ----------------- Main Loop -----------------


# 1) Build cache CSV (once)
cache_dir = os.path.join(figure_dire_save, 'plot_cache')
os.makedirs(cache_dir, exist_ok=True)

# You were plotting 'full' only; keep that:
topic = 'full'

# The named "Extended Data Fig." slots are reserved for the OpenAlex main analysis;
# this Semantic Scholar (S2AG) robustness version always uses a descriptive,
# dataset-tagged filename so it never claims a named figure slot.
out_svg = os.path.join(
    figure_dire_save,
    f"s2ag_full_combined_index_trend_V2_ay{activity_year}_c_t_{citation_threshold}_r_t_{reference_threshold}{network_suffix}_fromcache.svg"
)
cache_name = (
    f"plotcache_generalization_s2ag_{topic}_ay{activity_year}_ct{citation_threshold}_rt{reference_threshold}{network_suffix}.csv"
)

cache_path = os.path.join(cache_dir, cache_name)
if not os.path.exists(cache_path):
    cache_path = build_and_save_plot_cache(cache_dir=cache_path, topic='full')

plot_from_cache_csv(cache_path, out_svg)

print("finished")
