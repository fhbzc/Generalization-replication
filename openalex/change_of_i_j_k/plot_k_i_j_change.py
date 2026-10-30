import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)             # utils.py is in the parent directory
from utils import META_DIRE

# ----------------------------
# Paths & thresholds
# ----------------------------
meta_dire = META_DIRE
load_dire = os.path.join(meta_dire, 'disruption')
fig_dir = os.path.join('./figures/i_j_k_change')
data_dir = os.path.join(fig_dir, 'data')
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

reference_threshold = 1
citation_threshold = 5

# Threshold-dependent filenames
suffix = f"ref{reference_threshold}_cit{citation_threshold}"

# Every per-paper metric averaged over the papers published in each year -- one line each.
#   key    the per-paper column computed below
#   label  how the line is named (legend, or the y-axis when it is alone on a figure)
#   color  its line colour
#   stem   its own yearly-average CSV
METRICS = [
    ('metric_cite_over_K',       '(I+J)/K',     'black',    'yearly_avg_citation_over_K'),
    ('metric_i_over_ij',         'I/(I+J)',     'tab:blue', 'yearly_avg_i_over_i_plus_J'),
    ('metric_i_minus_J_over_ij', '(I-J)/(I+J)', 'tab:red',
     'yearly_avg_i_minus_J_over_i_plus_J'),
]
METRIC_LABEL = {k: lbl for k, lbl, _c, _s in METRICS}
METRIC_COLOR = {k: c for k, _l, c, _s in METRICS}
csv_paths = {k: os.path.join(data_dir, f'{stem}_{suffix}.csv')
             for k, _l, _c, stem in METRICS}

# Which metrics share a panel.
#   keys     the metrics drawn on it, in legend order
#   ylabel   the y-axis text
#   stem     the figure name (the thresholds are appended to it)
#   hlines   dashed reference lines. The ratio panel marks 0 -- where (I-J)/(I+J) changes
#            sign -- and nothing else
#
# NOTE i/(i+j) and (i-j)/(i+j) are AFFINE images of each other: i + j = citation_count, so
# (i-j)/(i+j) = 2*[i/(i+j)] - 1. The two lines therefore have identical shape and differ only
# by a scale and a shift -- the second adds no information the first does not already carry.
# They share a panel so that relationship is visible rather than split across two files.
FIGURES = [
    (['metric_cite_over_K'], '(I+J)/K',
     'yearly_avg_citation_over_K', ()),
    (['metric_i_over_ij', 'metric_i_minus_J_over_ij'], 'Index',
     'yearly_avg_i_ratios', (0.0,)),
]

# ----------------------------
# Helper for plotting (matplotlib only)
# ----------------------------
def plot_lines(
    series, ylabel, fname_prefix,
    hlines=(), x='pub_year', y='avg',
    xlabel="Publication Year",
    x_labelsize=14, y_labelsize=14,
    x_ticksize=11,  y_ticksize=11,
    legend_size=18
):
    """Draw one or more yearly-average series on a single panel, saved under its OWN name.

    `series` is a list of (dataframe, label, colour). A legend is drawn only when there is more
    than one, so a lone curve looks exactly as it did before. `hlines` are dashed reference
    lines. Every figure is named after its own metrics plus the thresholds.

    fname_prefix used to be ignored -- every call wrote the same file -- which was harmless
    while there was one curve and silently destructive as soon as there were several.
    """
    fig, ax = plt.subplots(figsize=(10, 4.8))

    for df_i, label, color in series:
        ax.plot(df_i[x], df_i[y], color=color, linewidth=2, label=label)

    for h in hlines:
        ax.axhline(h, color='0.5', linestyle='--', linewidth=1)

    ax.set_xlabel(xlabel, fontsize=x_labelsize)
    ax.set_ylabel(ylabel, fontsize=y_labelsize)

    ax.tick_params(axis='x', labelsize=x_ticksize)
    ax.tick_params(axis='y', labelsize=y_ticksize)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.grid(False)
    # a single curve is named by the y-axis, so a legend would only repeat it
    if len(series) > 1:
        ax.legend(fontsize=legend_size, frameon=False)

    plt.tight_layout()
    # each figure is named after its own metrics, so they can never overwrite one another
    path = os.path.join(
        fig_dir, f'{fname_prefix}_rt_{reference_threshold}_ct_{citation_threshold}.svg')

    plt.savefig(path, transparent=True, format = 'svg')
    print("Saved:", path)
    plt.close()
# ----------------------------
# Main logic
# ----------------------------
# every curve must be cached for the shortcut to be taken: a run that added a metric would
# otherwise be short-circuited by the older, incomplete cache and never compute the new one
if all(os.path.exists(p) for p in csv_paths.values()):
    print("Found precomputed yearly CSVs, redrawing directly...")
    yearly = {k: pd.read_csv(csv_paths[k]) for k in csv_paths}
else:
    print("Precomputed CSVs not found, computing from raw disp_full.csv...")
    usecols = [
        'pub_year', 'up_refer_count', 'K_count_y5', 'citation_count_y5',
        'J_count_y5', 'I_iso_count_y5', 'I_start_count_y5'
    ]
    df = pd.read_csv(os.path.join(load_dire, 'disp_full.csv'), usecols=usecols)

    # Filter
    df = df[(df['pub_year'] >= 1945) & (df['pub_year'] < 2020)]
    df = df[df['up_refer_count'] >= reference_threshold]
    df = df[df['citation_count_y5'] >= citation_threshold]

    # Define i_y5 and check identity
    df['i_y5'] = df['I_iso_count_y5'] + df['I_start_count_y5']
    assert (df['i_y5'] + df['J_count_y5'] == df['citation_count_y5']).all()

    # Metrics
    df['metric_cite_over_K'] = np.where(
        df['K_count_y5'] > 0,
        df['citation_count_y5'] / df['K_count_y5'],
        np.nan
    )
    # i + j == citation_count (asserted just above), so citation_count IS the (i+j)
    # denominator -- no need to re-add the two counts, and it cannot disagree with them.
    ij = df['citation_count_y5']
    df['metric_i_over_ij'] = np.where(ij > 0, df['i_y5'] / ij, np.nan)
    df['metric_i_minus_J_over_ij'] = np.where(
        ij > 0, (df['i_y5'] - df['J_count_y5']) / ij, np.nan)

    # Yearly averages -- one frame per metric: the year, the mean, and the paper count
    yearly = {}
    for key, _label, _color, _stem in METRICS:
        y = df.groupby('pub_year')[key].agg(['mean', 'count']).reset_index()
        y.rename(columns={'mean': 'avg', 'count': 'n'}, inplace=True)
        yearly[key] = y
        y.to_csv(csv_paths[key], index=False)
        print(f"Saved yearly averages [{key}]:", csv_paths[key])

# ----------------------------
# Plot both figures
# ----------------------------
for keys, ylabel, stem, hlines in FIGURES:
    plot_lines(
        [(yearly[k], METRIC_LABEL[k], METRIC_COLOR[k]) for k in keys],
        ylabel, stem,
        hlines=hlines,
        x_labelsize=18, y_labelsize=18,
        x_ticksize=14,  y_ticksize=14
    )


print("finished")
