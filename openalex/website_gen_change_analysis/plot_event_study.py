import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from tqdm import tqdm
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)             # utils.py is in the parent directory
from utils import META_DIRE

DATA_DIRE = os.path.join(META_DIRE, 'full_text_online_analysis', 'R_generated')
SAVE_DIRE = os.path.join('./figures', 'full_text_online_analysis')
os.makedirs(SAVE_DIRE, exist_ok=True)

plt.rcParams['font.family'] = 'Gill Sans MT'
plt.rcParams['axes.unicode_minus'] = False

# ── Dataset registry: (dataset_key, file_mid, outcome_label) ─────────────────
ALL_DATASETS = [
    ("avg_g",          "",                      "G"),
    ("title_distance", "paper_title_distance_", "title distance"),
    ("ref_distance",   "paper_ref_distance_",   "reference distance"),
    ("cross_domain",   "paper_cross_domain_",   "cross-domain ratio"),
]
_DATASET_MAP = {key: (mid, label) for key, mid, label in ALL_DATASETS}

# ── Config list — each entry is one panel ────────────────────────────────────
# Configs with identical plot_datasets (and other params) but different k
# are automatically grouped into one figure as subplots.
# plot_datasets : list of dataset keys to include (from ALL_DATASETS)
# k, min_papers_per_year, reference_threshold, citation_threshold, same_domain_control
CONFIGS = [
    dict(plot_datasets=['avg_g'],          k=3, min_papers_per_year=2, reference_threshold=1, citation_threshold=1, same_domain_control=False),
    dict(plot_datasets=['avg_g'],          k=5, min_papers_per_year=2, reference_threshold=1, citation_threshold=1, same_domain_control=False),
    dict(plot_datasets=['avg_g'],          k=7, min_papers_per_year=2, reference_threshold=1, citation_threshold=1, same_domain_control=False),

    dict(plot_datasets=['title_distance'], k=5, min_papers_per_year=2, reference_threshold=1, citation_threshold=1, same_domain_control=False),
    dict(plot_datasets=['ref_distance'],   k=5, min_papers_per_year=2, reference_threshold=1, citation_threshold=1, same_domain_control=False),
    dict(plot_datasets=['cross_domain'],   k=5, min_papers_per_year=2, reference_threshold=1, citation_threshold=1, same_domain_control=False),

    # Combined 3-panel figure: cross_domain | ref_distance | title_distance
    dict(plot_datasets=['cross_domain', 'ref_distance', 'title_distance'],
         k=5, min_papers_per_year=2, reference_threshold=1, citation_threshold=1, same_domain_control=False),
]

# ── Visual ────────────────────────────────────────────────────────────────────
tick_size       = 16
label_size      = 20
panel_label_size = 20


def build_csv_path(file_mid, cfg, domain_tag):
    stem = (f"event_study_coefs_{file_mid}{domain_tag}"
            f"_k{cfg['k']}"
            f"_mpy{cfg['min_papers_per_year']}"
            f"_rt{cfg['reference_threshold']}"
            f"_ct{cfg['citation_threshold']}.csv")
    return os.path.join(DATA_DIRE, stem)


def draw_panel(ax, df, outcome_label, show_xlabel, show_ylabel, panel_label='', k=None):
    df = df.rename(columns={'conf.low': 'conf_low', 'conf.high': 'conf_high'})
    ax.axvspan(-0.7, -0.3, color='gray', alpha=0.15, zorder=0)
    ax.axhline(y=0, linestyle=':', color='gray', linewidth=1.8, zorder=1)
    sig_above = df['conf_low'] > 0
    sig_below = df['conf_high'] < 0
    for mask, color in [(sig_above, 'red'), (sig_below, 'blue'), (~sig_above & ~sig_below, 'black')]:
        sub = df[mask]
        if sub.empty:
            continue
        sub_yerr = [
            (sub['estimate'] - sub['conf_low']).values,
            (sub['conf_high'] - sub['estimate']).values,
        ]
        ax.errorbar(sub['ri'], sub['estimate'],
                    yerr=sub_yerr,
                    fmt='o', color=color, markersize=4,
                    capsize=3, capthick=1.2, elinewidth=1.2, zorder=3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.tick_params(axis='x', labelsize=tick_size)
    ax.tick_params(axis='y', labelsize=tick_size)
    if show_xlabel:
        ax.set_xlabel('Year index', fontsize=label_size)
    if show_ylabel:
        ax.set_ylabel(f'scaled({outcome_label})', fontsize=label_size)
    if panel_label:
        ax.text(0.02, 0.98, panel_label, transform=ax.transAxes,
                fontsize=panel_label_size, va='top', ha='left')
    if k is not None:
        x_min = df['ri'].min() - 0.5
        ax.set_xlim(x_min, k - 1 + 0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ── Group configs: same plot_datasets + same other params → one figure ────────
# Grouping key: plot_datasets tuple (k is intentionally excluded)
from collections import OrderedDict as _OD

def _group_key(cfg):
    return (
        tuple(cfg['plot_datasets']),
        cfg['min_papers_per_year'],
        cfg['reference_threshold'],
        cfg['citation_threshold'],
        cfg['same_domain_control'],
    )

_groups = _OD()
for cfg in CONFIGS:
    key = _group_key(cfg)
    _groups.setdefault(key, []).append(cfg)

# ── Main loop ─────────────────────────────────────────────────────────────────
for group_key, group_cfgs in tqdm(_groups.items(), desc='figures'):
    multi_k = len(group_cfgs) > 1  # show k= label only when grouped
    all_dataset_keys = [dk for cfg in group_cfgs for dk in cfg['plot_datasets']]
    multi_dataset = len(set(all_dataset_keys)) > 1  # multiple distinct datasets in one figure

    # Collect panels for each config in this group
    panels = []
    for cfg in group_cfgs:
        domain_tag  = "samedomain" if cfg['same_domain_control'] else "anydomain"
        panel_label = f"year={cfg['k']}" if multi_k else ''
        for dataset_key in cfg['plot_datasets']:
            file_mid, outcome_label = _DATASET_MAP[dataset_key]
            csv_path = build_csv_path(file_mid, cfg, domain_tag)
            if not os.path.exists(csv_path):
                print(f"  Skipping [{dataset_key} k={cfg['k']}]: not found at {csv_path}")
                continue
            df = pd.read_csv(csv_path).sort_values('ri').reset_index(drop=True)
            panels.append((outcome_label, panel_label, df, cfg))

    if not panels:
        continue

    n   = len(panels)
    mid = n // 2
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for i, (ax, (outcome_label, panel_label, df, cfg)) in enumerate(zip(axes, panels)):
        draw_panel(ax, df, outcome_label,
                   show_xlabel=(i == mid),
                   show_ylabel=(i == 0 or multi_dataset),
                   panel_label=panel_label,
                   k=cfg['k'])

    plt.tight_layout()
    first_cfg  = group_cfgs[0]
    domain_tag = "samedomain" if first_cfg['same_domain_control'] else "anydomain"
    outcome_label_first = panels[0][0]
    plot_datasets_tuple = tuple(first_cfg['plot_datasets'])

    # Special naming
    is_fig4b  = (multi_k and plot_datasets_tuple == ('avg_g',)
                 and not first_cfg['same_domain_control'])
    is_figs19 = (plot_datasets_tuple == ('cross_domain', 'ref_distance', 'title_distance')
                 and not first_cfg['same_domain_control'])

    if is_fig4b:
        svg_file = os.path.join(SAVE_DIRE, 'Fig. 4b.svg')
    elif is_figs19:
        svg_file = os.path.join(SAVE_DIRE, 'Extended Data Fig. 9a.svg')
    elif multi_k:
        k_range = '_'.join(str(c['k']) for c in group_cfgs)
        svg_file = os.path.join(
            SAVE_DIRE,
            f'event_study_{domain_tag}_{outcome_label_first.replace(" ", "_")}_k{k_range}.svg'
        )
    elif multi_dataset:
        svg_file = os.path.join(
            SAVE_DIRE,
            f'event_study_{domain_tag}_k{first_cfg["k"]}_combined.svg'
        )
    else:
        svg_file = os.path.join(
            SAVE_DIRE,
            f'event_study_{domain_tag}_k{first_cfg["k"]}_{outcome_label_first.replace(" ", "_")}.svg'
        )
    plt.savefig(svg_file, format='svg', bbox_inches='tight', transparent=True)
    plt.close()
    print(f"Saved: {svg_file}")

    csv_file = svg_file.replace('.svg', '.csv')
    csv_rows = []
    for outcome_label, panel_label, df, cfg in panels:
        df_copy = df.copy()
        df_copy['outcome'] = outcome_label
        df_copy['panel_label'] = panel_label
        df_copy['k'] = cfg['k']
        csv_rows.append(df_copy)
    pd.concat(csv_rows, ignore_index=True).to_csv(csv_file, index=False)
    print(f"Saved: {csv_file}")

print("\nAll done.")
