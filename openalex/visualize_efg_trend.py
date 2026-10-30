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

reference_threshold = 0
citation_threshold = 5
print("citation_threshold", citation_threshold)
print("reference_threshold", reference_threshold)

# Must match get_efg.py / get_efg_s2_merge.py: selects which network variant's merged
# E/F/G table to plot.
#   'normal'           -> gen_V3_y{activity_year}.csv
#   'no_self_citation' -> gen_V3_y{activity_year}_no_self_citation.csv
#   'no_year_filter'   -> gen_V3_y{activity_year}_no_year_filter.csv
# network_suffix is appended to the input filename AND the output figure/cache names
# so the variants never overwrite each other.
network_mode = 'normal'
NETWORK_SUFFIX_MAP = {
    'normal': '',
    'no_self_citation': '_no_self_citation',
    'no_year_filter': '_no_year_filter',
}
assert network_mode in NETWORK_SUFFIX_MAP, network_mode
network_suffix = NETWORK_SUFFIX_MAP[network_mode]
print("network_mode", network_mode)
print("network_suffix", repr(network_suffix))

# input is get_efg_s2_merge.py's output; these must match that script's settings
# (reshuffle variant lives in atypicality/get_efg_reshuffle.py)
gen_file_load_dire = os.path.join(
    meta_dire, 'efg_computation',
    f'gen_V3_y{activity_year}{network_suffix}.csv'
)
print("gen_file_load_dire", gen_file_load_dire)

# What lines to include (master list; filtered per figure by citation_threshold)
NORM_TARGETS_ALL = [1, 2, 3, 5, 10, 20, 30]
norm_targets = [target for target in NORM_TARGETS_ALL if target >= citation_threshold]
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
def plot_from_cache_csv(cache_csv_path: str, out_svg_path: str, show_legend: bool = False):
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
                ax.plot(df_c['pub_year'], df_c['mean_value'], label=f"{int(c)}", color=CITATION_COLOR[c], linestyle='--', linewidth=1.8)

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
        # only draw the (citation-count) legend on the leftmost subplot, and only for
        # the figure that opts in (Extended Data Fig. 6a)
        if i == 0 and show_legend:
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


# ----------------- Main -----------------
cache_dir = os.path.join(figure_dire_save, 'plot_cache')
os.makedirs(cache_dir, exist_ok=True)

topic = 'full'

# When True, generate every named Extended Data Fig. in a single run (NORMAL network).
# When False, produce one figure from the manually-set
# activity_year / reference_threshold / citation_threshold above.
generate_all_named = True

# Single source of truth: (activity_year, reference_threshold, citation_threshold) -> label.
# Both the .svg and its .csv cache derive from the SAME label, so they can't drift.
# ay=5 main trend: 6a/6b/6c/6d ; different-data specs (feg_dif_data): 6f/6g
fig_lookup = {
    (5, 1, 0): 'Extended Data Fig. 6a',
    (5, 1, 1): 'Extended Data Fig. 6b',
    (5, 1, 2): 'Extended Data Fig. 6c',
    (5, 1, 5): 'Extended Data Fig. 6d',
    (1, 1, 2): 'Extended Data Fig. 6f',
    (10, 1, 5): 'Extended Data Fig. 6g',
}


def _output_names(ay, rt, ct):
    """(out_svg, cache_name) for one spec. The canonical Extended-Data-Fig labels are
    reserved for the NORMAL network only; any other network variant uses the generic
    descriptive filename (which carries network_suffix) so it never claims a slot."""
    label = fig_lookup.get((ay, rt, ct))
    if label is not None and network_suffix == '':
        return os.path.join(figure_dire_save, f'{label}.svg'), f'{label}.csv'
    out_svg = os.path.join(
        figure_dire_save,
        f"full_combined_index_trend_V2_ay{ay}_c_t_{ct}_r_t_{rt}{network_suffix}_fromcache.svg")
    cache_name = f"plotcache_generalization_{topic}_ay{ay}_ct{ct}_rt{rt}{network_suffix}.csv"
    return out_svg, cache_name


def generate_figure(ay, rt, ct):
    """Build the cache (if missing) and draw the figure for one spec, updating the
    module-level parameters the build/plot helpers read from."""
    global activity_year, reference_threshold, citation_threshold, norm_targets, gen_file_load_dire
    activity_year = ay
    reference_threshold = rt
    citation_threshold = ct
    norm_targets = [t for t in NORM_TARGETS_ALL if t >= ct]
    gen_file_load_dire = os.path.join(
        meta_dire, 'efg_computation', f'gen_V3_y{ay}{network_suffix}.csv')

    out_svg, cache_name = _output_names(ay, rt, ct)
    cache_path = os.path.join(cache_dir, cache_name)
    if not os.path.exists(cache_path):
        build_and_save_plot_cache(cache_dir=cache_path, topic=topic)
    # only Extended Data Fig. 6a carries the citation-count legend
    show_legend = fig_lookup.get((ay, rt, ct)) == 'Extended Data Fig. 6a'
    plot_from_cache_csv(cache_path, out_svg, show_legend=show_legend)


if generate_all_named and network_suffix == '':
    for (ay, rt, ct), label in fig_lookup.items():
        print(f"=== generating {label}  (ay={ay}, rt={rt}, ct={ct}) ===")
        try:
            generate_figure(ay, rt, ct)
        except FileNotFoundError as e:
            print(f"[skipped {label}] input not found: {e}")
else:
    generate_figure(activity_year, reference_threshold, citation_threshold)

print("finished")
