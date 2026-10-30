"""
Plot the yearly-average E/F/G trend for the WoS data (Extended Data Fig. 6h-n).
Reads the merged per-paper E/F/G table from get_efg_s2_merge.py and plots the
yearly mean of Foundation (f_y), Extension (e_y), Generalization (g_y) over
publication year, as three panels, with an "All" line plus lines split by exact
citation_count (norm_targets). The tidy plot data is cached to CSV before drawing.

The index columns are the citation-normalised shares f_y/e_y/g_y. activity_year /
reference_threshold / citation_threshold must match get_efg.py; the figure label is
chosen by (activity_year, reference_threshold, citation_threshold): the ay=5
main trend -> 3h/3i/3j/3k, and the feg_dif_data ("different data") specs with a different
activity_year -> 3m (ay=1, ct=2), 3n (ay=10, ct=5).

With generate_all_named=True the script produces every named figure above in a single
run (looping over NAMED_FIGS); set it False to produce just the one figure selected by
the manually-set activity_year / reference_threshold / citation_threshold.

Set use_with_review to match get_efg.py / get_efg_s2_merge.py: when True it reads the
'_with_review' merged table (the doctypes == 'review'-inclusive network). The named
"Extended Data Fig." slots are reserved for the article-only network (use_with_review
False); the '_with_review' robustness run always writes a descriptive, suffix-tagged
filename so it never claims a named figure slot.

Input  : META_DIRE/efg_computation/gen_V3_y{activity_year}[_with_review].csv (from
         get_efg_s2_merge.py; run get_efg.py/get_efg_s2_merge.py with
         year_offset=activity_year and the same use_with_review)
Output : ./figures/generalization/Extended Data Fig. 6{h,i,j,k,m,n}.svg
         (or full_combined_index_trend..._with_review_fromcache.svg) + plot_cache/*.csv
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless / cluster-safe
import matplotlib.pyplot as plt  # noqa: E402

# utils.py is in this same directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
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

activity_year = 5
print("activity_year", activity_year)

figure_dire_save = './figures/generalization'
os.makedirs(figure_dire_save, exist_ok=True)

reference_threshold = 0
citation_threshold = 5
print("citation_threshold", citation_threshold)
print("reference_threshold", reference_threshold)

# must match get_efg.py / get_efg_s2_merge.py: read the '_with_review' variant when True
# (the doctypes == 'review'-inclusive network)
use_with_review = False
data_suffix = '_with_review' if use_with_review else ''
print("use_with_review", use_with_review)

# input is get_efg_s2_merge.py's output; activity_year and use_with_review must match that script
gen_file_load_dire = os.path.join(meta_dire, 'efg_computation', f'gen_V3_y{activity_year}{data_suffix}.csv')
print("gen_file_load_dire", gen_file_load_dire)

# which "by citation count" lines to include (master list; filtered per figure by citation_threshold)
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
    """Columns to load from get_efg_s2_merge.py's output and the Foundation /
    Extension / Generalization index columns (citation-normalised shares
    f_y / e_y / g_y)."""
    f_col = f'f_y{activity_year}'
    e_col = f'e_y{activity_year}'
    g_col = f'g_y{activity_year}'

    use_col_list = [
        "pub_year", "up_refer_count", f'citation_count_y{activity_year}',
        f_col, e_col, g_col,
    ]
    indices = {'Foundation': f_col, 'Extension': e_col, 'Generalization': g_col}
    return use_col_list, indices


def _legend_title_for_index(index_name: str) -> str:
    return 'Citation Count'


def _norm_col_for_index(index_name: str) -> str:
    return f'citation_count_y{activity_year}'


def _read_source_csv(use_col_list: list) -> pd.DataFrame:
    """Reads get_efg_s2_merge.py's merged output (full table, all papers)."""
    return pd.read_csv(gen_file_load_dire, usecols=use_col_list)


# ----------------- Build & Save Plot Cache -----------------
def build_and_save_plot_cache(cache_path: str, topic: str = 'full') -> str:
    """
    Build a tidy CSV that contains everything needed for the plot.

    Output columns:
      topic, activity_year, reference_threshold, citation_threshold,
      index_name, legend_title, norm_kind (all/by_norm), norm_col, norm_value,
      pub_year, mean_value, n
    """
    use_col_list, indices = _indices_and_usecols(activity_year)
    df = _read_source_csv(use_col_list)

    # filters: dynamic pub_year cutoff so the citation window fits within the data,
    # plus reference/citation thresholds
    df = df[df['pub_year'] < (2024 - activity_year + 1)]
    df = df[df['up_refer_count'] >= reference_threshold]
    df = df[df[f'citation_count_y{activity_year}'] >= citation_threshold]

    df = df.copy()
    df['citation_up_sum'] = df['up_refer_count'] + df[f'citation_count_y{activity_year}']

    rows = []
    for index_name, index_col in indices.items():
        # --- ALL line
        g_all = df.groupby('pub_year')[index_col].agg(['mean', 'count']).reset_index()
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

        # --- lines split by exact citation count
        norm_col = _norm_col_for_index(index_name)
        for c in norm_targets:
            sub = df[df[norm_col] == c]
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
    cache_df.to_csv(cache_path, index=False)
    print(f"[Cache saved] {cache_path} (rows={len(cache_df)})")
    return cache_path


# ----------------- Draw Only From Cache -----------------
def plot_from_cache_csv(cache_csv_path: str, out_svg_path: str, show_legend: bool = False):
    dfc = pd.read_csv(cache_csv_path)
    dfc = dfc[dfc['pub_year'] >= 1945]
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
        if i == 0:
            ax.set_ylabel('Average index', fontsize=20)
        else:
            ax.set_ylabel("")
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # only draw the (citation-count) legend on the leftmost subplot, and only for
        # a figure that opts in (none of the WoS named figures do)
        if i == 0 and show_legend:
            ax.legend(title=legend_title, loc='upper right', fontsize=12, title_fontsize=14)

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

# When True, generate every named Extended Data Fig. in a single run (article-only
# network). When False, produce one figure from the manually-set
# activity_year / reference_threshold / citation_threshold above.
generate_all_named = True

# Single source of truth: (activity_year, reference_threshold, citation_threshold) -> label.
# Both the .svg and its .csv cache are derived from the SAME label, so they can't drift.
# ay=5 main trend -> 6h/6i/6j/6k; the feg_dif_data ("different data") specs use a different
# activity_year -> 6m (ay=1, ct=2), 6n (ay=10, ct=5).
NAMED_FIGS = {
    (5, 1, 0): 'Extended Data Fig. 6h',
    (5, 1, 1): 'Extended Data Fig. 6i',
    (5, 1, 2): 'Extended Data Fig. 6j',
    (5, 1, 5): 'Extended Data Fig. 6k',
    (1, 1, 2): 'Extended Data Fig. 6m',
    (10, 1, 5): 'Extended Data Fig. 6n',
}


def _output_names(ay, rt, ct):
    """(out_svg, cache_name) for one spec. Named slots are reserved for the
    article-only network; the '_with_review' variant always uses the descriptive
    suffix-tagged name so it never claims a named figure slot."""
    label = NAMED_FIGS.get((ay, rt, ct))
    if label is not None and not use_with_review:
        return os.path.join(figure_dire_save, f'{label}.svg'), f'{label}.csv'
    out_svg = os.path.join(
        figure_dire_save,
        f"full_combined_index_trend_V2_ay{ay}_c_t_{ct}_r_t_{rt}{data_suffix}_fromcache.svg")
    cache_name = f"plotcache_generalization_{topic}_ay{ay}_ct{ct}_rt{rt}{data_suffix}.csv"
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
        meta_dire, 'efg_computation', f'gen_V3_y{ay}{data_suffix}.csv')

    out_svg, cache_name = _output_names(ay, rt, ct)
    cache_path = os.path.join(cache_dir, cache_name)
    if not os.path.exists(cache_path):
        build_and_save_plot_cache(cache_path=cache_path, topic=topic)
    plot_from_cache_csv(cache_path, out_svg)


if generate_all_named and not use_with_review:
    for (ay, rt, ct), label in NAMED_FIGS.items():
        print(f"=== generating {label}  (ay={ay}, rt={rt}, ct={ct}) ===")
        try:
            generate_figure(ay, rt, ct)
        except FileNotFoundError as e:
            print(f"[skipped {label}] input not found: {e}")
else:
    generate_figure(activity_year, reference_threshold, citation_threshold)

print("finished")
