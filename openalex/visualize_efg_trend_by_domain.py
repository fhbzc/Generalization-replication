"""
Per-domain E/F/G trend grid (Extended Data Fig. 7a). For each domain, reads the
per-domain E/F/G table written by arrange_efg_by_domain.py and plots the yearly
mean of Foundation (f_y), Extension (e_y), Generalization (g_y) over publication
year, as a 4-column grid of dual-axis panels (Foundation on the left axis,
Extension/Generalization on the right). Per-domain yearly means are cached to small
CSVs so re-runs only redraw.

The plotted indices are the citation-normalised shares f_y/e_y/g_y. activity_year /
reference_threshold / citation_threshold must match get_efg.py.

Input  : META_DIRE/efg_computation/efg_by_domain/<domain>/efg_<domain>_y{activity_year}.csv
         (from arrange_efg_by_domain.py)
Output : ./figures/generalization/by_domain/Extended Data Fig. 7a.svg (+ .csv) at the
         deliverable setting (the 19 domains below, activity_year=5, rt=1, ct=5); otherwise
         efg_trend_by_domain_n{n}_ay{ay}_rt{rt}_ct{ct}.svg/.csv
         + ./figures/generalization/by_domain/cached_data/... (per-domain caches, shared)
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless / cluster-safe
import matplotlib.pyplot as plt  # noqa: E402
from tqdm import tqdm  # noqa: E402

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
figure_dire_save = './figures/generalization/by_domain'
os.makedirs(figure_dire_save, exist_ok=True)
file_save_dire = os.path.join(figure_dire_save, 'cached_data')
os.makedirs(file_save_dire, exist_ok=True)

# per-domain E/F/G tables from arrange_efg_by_domain.py
load_dire = os.path.join(meta_dire, 'efg_computation', 'efg_by_domain')

topic_list = ['Materials science', 'Environmental science', 'Physics', 'Chemistry',
              'Biology', 'Computer science', 'Mathematics', 'Political science',
              'Geography', 'Economics', 'Medicine', 'Psychology', 'Philosophy',
              'Geology', 'Engineering', 'Art', 'Business', 'Sociology', 'History']
topic_list = sorted(topic_list)

activity_year = 5

reference_threshold = 1
citation_threshold = 5

print("activity_year", activity_year)
print("citation_threshold", citation_threshold)
print("reference_threshold", reference_threshold)


# ------------- Column pickers -------------
def _indices_and_usecols(activity_year: int):
    """Columns of the per-domain efg CSV (arrange_efg_by_domain.py) to load, and the
    Foundation / Extension / Generalization index columns: the citation-normalised
    shares f_y / e_y / g_y.
    """
    f_col = f'f_y{activity_year}'
    e_col = f'e_y{activity_year}'
    g_col = f'g_y{activity_year}'

    use_col_list = [
        "pub_year", "up_refer_count", f'citation_count_y{activity_year}',
        f_col, e_col, g_col,
    ]
    indices = {'Foundation': f_col, 'Extension': e_col, 'Generalization': g_col}
    return use_col_list, indices


def _read_topic_csv(load_dire: str, topic: str, use_cols: list) -> pd.DataFrame:
    path = os.path.join(load_dire, topic, f'efg_{topic}_y{activity_year}.csv')
    return pd.read_csv(path, usecols=use_cols)


# ----------------- Caching helpers -----------------
CACHE_DIR_NAME = 'generalization_means_cache'


def _cache_key():
    # include all knobs that affect results so the cache auto-invalidates. The leading
    # 'gt-ratio' is a fixed literal, kept only so the already-computed per-domain caches
    # (written when the index columns were still selectable) still resolve.
    return (f"gt-ratio_ay-{activity_year}"
            f"_rt-{reference_threshold}_ct-{citation_threshold}")


def _cache_dir(base_dir: str) -> str:
    d = os.path.join(base_dir, CACHE_DIR_NAME)
    os.makedirs(d, exist_ok=True)
    return d


def _topic_cache_path(base_dir: str, topic: str) -> str:
    return os.path.join(_cache_dir(base_dir), f"{topic}_{_cache_key()}.csv")


def compute_yearly_means_for_topic_cached(cache_base: str, load_dire: str, topic: str,
                                          indices: dict, use_cols: list) -> pd.DataFrame:
    """
    Returns LONG-form df with columns ['pub_year','index_name','mean_value'].
    Uses a compact cached CSV if present; otherwise builds and writes it once.
    """
    cache_path = _topic_cache_path(cache_base, topic)
    if os.path.exists(cache_path):
        wide = pd.read_csv(cache_path)
        if 'pub_year' in wide:
            wide['pub_year'] = wide['pub_year'].astype(int)
        return wide.melt(id_vars=['pub_year'], var_name='index_name', value_name='mean_value')

    # cache miss: compute from the per-domain efg CSV
    df = _read_topic_csv(load_dire, topic, use_cols)

    # filters (match the trend script: dynamic pub_year cutoff so the citation
    # window fits within the data, plus reference/citation thresholds)
    df = df[df['pub_year'] < (2024 - activity_year + 1)]
    df = df[df['up_refer_count'] >= reference_threshold]
    df = df[df[f'citation_count_y{activity_year}'] >= citation_threshold]

    # group means for the three series
    g = df.groupby('pub_year')[[indices['Foundation'],
                                indices['Extension'],
                                indices['Generalization']]].mean().reset_index()

    g = g.rename(columns={
        indices['Foundation']: 'Foundation',
        indices['Extension']: 'Extension',
        indices['Generalization']: 'Generalization',
    }).sort_values('pub_year')

    g.to_csv(cache_path, index=False)

    return g.melt(id_vars=['pub_year'], var_name='index_name', value_name='mean_value')


# ------------- Plotting grid -------------
def plot_topics_grid(load_dire: str, topics: list, out_path: str):
    """
    4-column grid; each subplot has two axes:
      - left axis : Foundation
      - right axis: Extension, Generalization
    Every subplot shows the 1950-2020 (step 10) year ticks; the axis titles
    ("Publication Year", "Foundation Index", "Extension / Generalization Index")
    are drawn once, centred on the figure edges.
    """
    use_cols, indices = _indices_and_usecols(activity_year)

    n_topics = len(topics)
    ncols = 4
    nrows = int(np.ceil(n_topics / ncols))

    # panel size (tweakable)
    PANEL_W, PANEL_H = 5.2, 3.8
    fig_w, fig_h = PANEL_W * ncols, PANEL_H * nrows

    # do NOT share x: we set the ticks ourselves and want them on every subplot
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=False)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    xticks = np.arange(1950, 2021, 10)  # 1950..2020

    idx_left = ['Foundation']
    idx_right = ['Extension', 'Generalization']
    color_map = {'Foundation': 'red', 'Extension': 'orange', 'Generalization': 'blue'}

    # in-panel line labels (drawn on the FIRST panel only, in place of a legend)
    INLABEL_FS = 15          # font size of the Foundation/Extension/Generalization tags
    INLABEL_PAD = 0.012      # breathing room added around the text box, in axes fractions
    INLABEL_MIN_CLEAR = 0.02  # a placement must keep at least this gap to every curve

    def label_lines_in_panel(axL, axR, dft):
        """Tag each of the three curves with its name, in the curve's own colour.

        Each tag is placed at the spot CLOSEST TO ITS OWN CURVE among the spots whose text box
        stays clear of ALL three curves (and of the tags already placed), so the label reads as
        belonging to its line without any line running through the letters. Everything is done
        in AXES FRACTIONS -- the left and right axes have different scales, so only fractions
        reflect what the eye sees -- and the text box is MEASURED with the real renderer, so the
        result accounts for the actual glyph extent rather than an estimate.

        Must be called after the final subplots_adjust: the box size in axes fractions depends
        on the panel's pixel size.
        """
        series = {}
        for name, ax in [('Foundation', axL), ('Extension', axR), ('Generalization', axR)]:
            sub = dft[dft['index_name'] == name].sort_values('pub_year')
            if sub.empty:
                continue
            y0, y1 = ax.get_ylim()
            x0, x1 = ax.get_xlim()
            if y1 == y0 or x1 == x0:
                continue
            series[name] = {
                'ax': ax,
                'xf': (sub['pub_year'].to_numpy(dtype=float) - x0) / (x1 - x0),
                'yf': (sub['mean_value'].to_numpy(dtype=float) - y0) / (y1 - y0),
            }
        if not series:
            return

        renderer = fig.canvas.get_renderer()
        # candidate anchor points; the text is centred on each in turn
        gx = np.arange(0.06, 0.945, 0.02)
        gy = np.arange(0.06, 0.945, 0.02)
        placed = []          # boxes of the tags already positioned, as obstacles

        def box_to_points_dist(box, px, py):
            """Distance from an axes-fraction box to each point (0 inside the box)."""
            bx0, bx1, by0, by1 = box
            dx = np.maximum(np.maximum(bx0 - px, px - bx1), 0.0)
            dy = np.maximum(np.maximum(by0 - py, py - by1), 0.0)
            return np.hypot(dx, dy)

        for name in ('Foundation', 'Extension', 'Generalization'):
            if name not in series:
                continue
            ax = series[name]['ax']
            # measure the text once at a throwaway position to learn its size in axes fractions
            probe = ax.text(0.5, 0.5, name, transform=ax.transAxes, fontsize=INLABEL_FS,
                            ha='center', va='center')
            bb = probe.get_window_extent(renderer=renderer)
            (px0, py0), (px1, py1) = ax.transAxes.inverted().transform(
                [[bb.x0, bb.y0], [bb.x1, bb.y1]])
            probe.remove()
            hw = abs(px1 - px0) / 2 + INLABEL_PAD      # half width  + padding
            hh = abs(py1 - py0) / 2 + INLABEL_PAD      # half height + padding

            best = None      # (feasible, -dist_to_own, clearance, cx, cy)
            for cx in gx:
                if cx - hw < 0.005 or cx + hw > 0.995:
                    continue
                for cy in gy:
                    if cy - hh < 0.005 or cy + hh > 0.995:
                        continue
                    box = (cx - hw, cx + hw, cy - hh, cy + hh)
                    clearance = min(
                        float(box_to_points_dist(box, s['xf'], s['yf']).min())
                        for s in series.values())
                    for pbox in placed:      # keep tags off each other too
                        clearance = min(clearance, float(box_to_points_dist(
                            box, np.array([pbox[0], pbox[1], pbox[0], pbox[1]]),
                            np.array([pbox[2], pbox[2], pbox[3], pbox[3]])).min()))
                    own = float(box_to_points_dist(
                        box, series[name]['xf'], series[name]['yf']).min())
                    key = (clearance >= INLABEL_MIN_CLEAR, -own, clearance)
                    if best is None or key > best[0]:
                        best = (key, cx, cy, box, clearance)
            if best is None:
                continue
            (feasible, _, _), cx, cy, box, clearance = best
            # if the panel is too crowded for a genuinely clear spot, keep the text readable by
            # sitting it on a translucent white patch instead of letting a curve cross it
            bbox_kw = (None if feasible else
                       dict(facecolor='white', edgecolor='none', alpha=0.75,
                            boxstyle='square,pad=0.15'))
            ax.text(cx, cy, name, transform=ax.transAxes, color=color_map[name],
                    fontsize=INLABEL_FS, ha='center', va='center', bbox=bbox_kw)
            placed.append(box)

    first_panel = None
    k = 0
    for topic in tqdm(topics, desc='plot topics'):
        r, c = divmod(k, ncols)
        axL = axes[r, c]
        axR = axL.twinx()

        # ---- use cached means ----
        dft = compute_yearly_means_for_topic_cached(file_save_dire, load_dire, topic, indices, use_cols)

        # left axis: Foundation
        for name in idx_left:
            sub = dft[dft['index_name'] == name].sort_values('pub_year')
            if sub.empty:
                continue
            axL.plot(sub['pub_year'], sub['mean_value'],
                     label=name, color=color_map[name], linewidth=2)

        # right axis: Extension & Generalization
        for name in idx_right:
            sub = dft[dft['index_name'] == name].sort_values('pub_year')
            if sub.empty:
                continue
            axR.plot(sub['pub_year'], sub['mean_value'],
                     label=name, color=color_map[name], linewidth=2)

        # title & styling
        axL.set_title(topic.capitalize(), fontsize=18, pad=10)
        axL.tick_params(axis='both', labelsize=14)
        axR.tick_params(axis='y', labelsize=14)

        for side in ['top', 'right']:
            axL.spines[side].set_visible(False)
        axR.spines['top'].set_visible(False)
        axR.spines['left'].set_visible(False)

        # uniform x range & ticks; show ticks on every subplot
        axL.set_xlim(1950, 2020)
        axL.set_xticks(xticks)
        axL.tick_params(axis='x', labelbottom=True)

        # axis titles are drawn once for the whole figure (see below)
        if c == ncols - 1:
            axR.tick_params(axis='y', labelright=True)

        # the TOP-LEFT panel carries the in-panel curve names (there is no legend); it is
        # labelled after the layout is final, since the text size in axes fractions depends
        # on the panel's pixel size
        if r == 0 and c == 0:
            first_panel = (axL, axR, dft)

        k += 1

    # turn off any leftover empty axes
    for k in range(n_topics, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c].set_axis_off()

    # outer margins / spacing; reserve room on the right for the right-axis label
    fig.subplots_adjust(
        left=0.075, right=0.925, top=0.95, bottom=0.085,
        wspace=0.35, hspace=0.40
    )

    # single, centred axis titles for the whole figure (tweakable)
    AXLAB_FS = 24
    fig.supxlabel('Publication Year', fontsize=AXLAB_FS, y=0.040)
    fig.supylabel('Foundation Index', fontsize=AXLAB_FS, x=0.030)
    fig.text(0.960, 0.5, 'Extension / Generalization Index',
             fontsize=AXLAB_FS, rotation=270,
             ha='center', va='center')

    # name the three curves inside the top-left panel, now that the panel geometry is final
    if first_panel is not None:
        fig.canvas.draw()          # realise a renderer so the text can be measured
        label_lines_in_panel(*first_panel)

    plt.savefig(out_path, format='svg', transparent=True, bbox_inches='tight')
    plt.close()
    print(f"[Figure saved] {out_path}")


# ----------------- Main -----------------
topics = topic_list
print(f"Topics ({len(topics)}):", topics)

# the grid over THESE 19 domains at the paper's default settings IS the Extended Data
# deliverable; the topic list is spelled out (already sorted, like topic_list) so dropping or
# adding a domain visibly falls out of the deliverable naming instead of silently overwriting it
ED_FIG_NAME = 'Extended Data Fig. 7a'
ED_FIG_TOPICS = ['Art', 'Biology', 'Business', 'Chemistry', 'Computer science', 'Economics',
                 'Engineering', 'Environmental science', 'Geography', 'Geology', 'History',
                 'Materials science', 'Mathematics', 'Medicine', 'Philosophy', 'Physics',
                 'Political science', 'Psychology', 'Sociology']
is_ed_fig = (topic_list == ED_FIG_TOPICS and activity_year == 5
             and reference_threshold == 1 and citation_threshold == 5)
print("is_ed_fig", is_ed_fig, ("-> %s" % ED_FIG_NAME) if is_ed_fig else "")

if is_ed_fig:
    out_svg = os.path.join(figure_dire_save, f'{ED_FIG_NAME}.svg')
    out_csv = os.path.join(figure_dire_save, f'{ED_FIG_NAME}.csv')
else:
    stub = (f'efg_trend_by_domain_n{len(topic_list)}_ay{activity_year}'
            f'_rt{reference_threshold}_ct{citation_threshold}')
    out_svg = os.path.join(figure_dire_save, f'{stub}.svg')
    out_csv = os.path.join(figure_dire_save, f'{stub}.csv')

plot_topics_grid(load_dire, topics, out_svg)

# Save combined CSV (all domains merged) with the same base name as the figure
combined_rows = []
for topic in topics:
    cache_path = _topic_cache_path(file_save_dire, topic)
    if os.path.exists(cache_path):
        df_t = pd.read_csv(cache_path)
        df_t['topic'] = topic
        combined_rows.append(df_t)
if combined_rows:
    pd.concat(combined_rows, ignore_index=True).to_csv(out_csv, index=False)
    print(f"[Combined CSV saved] {out_csv}")

print("finished")
