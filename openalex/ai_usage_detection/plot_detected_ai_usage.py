import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)             # utils.py is in the parent directory
from utils import META_DIRE

plt.rcParams['font.family'] = 'Gill Sans MT'
plt.rcParams['axes.unicode_minus'] = False

COEF_DIRE = os.path.join(META_DIRE, 'llm_usage_dection', 'R_generated')
SAVE_DIRE = os.path.join('./figures', 'llm_usage_dection')
os.makedirs(SAVE_DIRE, exist_ok=True)

RANDOM_SEED = 42   # must match get_author_adopt_time_no_cid.py

# ── Config list — each entry is one figure ────────────────────────────────────
# outcome   : "g_ratio" | "cross_domain_ratio" | "refer_dist" | "title_dist"
# ai_threshold : float
# datasets  : list of (dataset_name, paper_count_threshold)
CONFIGS = [
    dict(outcome='g_ratio',           ai_threshold=0.1,  datasets=[('arxiv',3),('biorxiv',3),('ssrn',2)]),
    dict(outcome='g_ratio',           ai_threshold=0.2,  datasets=[('arxiv',3),('biorxiv',3),('ssrn',2)]),
    dict(outcome='g_ratio',           ai_threshold=0.05, datasets=[('arxiv',3),('biorxiv',3),('ssrn',2)]),
    dict(outcome='cross_domain_ratio', ai_threshold=0.1, datasets=[('arxiv',3),('biorxiv',3),('ssrn',2)]),
    dict(outcome='refer_dist',         ai_threshold=0.1, datasets=[('arxiv',3),('biorxiv',3),('ssrn',2)]),
    dict(outcome='title_dist',         ai_threshold=0.1, datasets=[('arxiv',3),('biorxiv',3),('ssrn',2)]),
    # ── Add show_xlabel=False to suppress the x-axis label for a specific figure ─
    # dict(outcome='g_ratio', ai_threshold=0.1, datasets=[('arxiv',3)], show_xlabel=False),
]

# ── Outcome → y-axis display label ───────────────────────────────────────────
OUTCOME_LABEL = {
    'g_ratio':           'G reference ratio',
    'cross_domain_ratio': 'cross domain ratio',
    'refer_dist':        'reference distance',
    'title_dist':        'title distance',
}

# ── Visual ────────────────────────────────────────────────────────────────────
tick_size        = 16
label_size       = 20
panel_label_size = 20


def draw_panel(ax, int_df, dataset_label, show_xlabel, show_ylabel, outcome):
    ax.axvspan(-0.7, -0.3, color='gray', alpha=0.15, zorder=0)
    ax.axhline(y=0, linestyle=':', color='gray', linewidth=1.8, zorder=1)
    sig_above = int_df['conf_low'] > 0
    sig_below = int_df['conf_high'] < 0
    for mask, color in [(sig_above, 'red'), (sig_below, 'blue'), (~sig_above & ~sig_below, 'black')]:
        sub = int_df[mask]
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
        ax.set_xlabel('Paper order', fontsize=label_size)
    if show_ylabel:
        ax.set_ylabel(f'scaled({OUTCOME_LABEL.get(outcome, outcome)})', fontsize=label_size)
    ax.text(0.02, 0.98, dataset_label, transform=ax.transAxes,
            fontsize=panel_label_size, va='top', ha='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ── Main loop ─────────────────────────────────────────────────────────────────
for cfg in tqdm(CONFIGS, desc='configs'):
    outcome         = cfg['outcome']
    ai_threshold    = cfg['ai_threshold']
    datasets        = cfg['datasets']
    cfg_show_xlabel = cfg.get('show_xlabel', True)
    ai_threshold_str = str(ai_threshold).replace('.', 'p')

    panels = []
    for dataset_name, pct in datasets:
        csv_stem = (f'{dataset_name}_event_study_pct{pct}'
                    f'_ait{ai_threshold_str}_seed{RANDOM_SEED}_{outcome}')
        coef_csv = os.path.join(COEF_DIRE, f'{csv_stem}.csv')
        if not os.path.exists(coef_csv):
            print(f"  Skipping [{csv_stem}]: not found")
            continue
        int_df = pd.read_csv(coef_csv).sort_values('ri').reset_index(drop=True)
        label  = 'SSRN' if dataset_name == 'ssrn' else dataset_name.capitalize()
        panels.append((label, int_df))

    if not panels:
        continue

    n = len(panels)
    mid = n // 2
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for i, (ax, (label, int_df)) in enumerate(zip(axes, panels)):
        draw_panel(ax, int_df, label,
                   show_xlabel=(cfg_show_xlabel and i == mid),
                   show_ylabel=(i == 0),
                   outcome=outcome)

    plt.tight_layout()
    if outcome == 'g_ratio' and ai_threshold == 0.1:
        svg_file = os.path.join(SAVE_DIRE, 'Fig. 4c.svg')
    elif outcome == 'g_ratio' and ai_threshold == 0.05:
        svg_file = os.path.join(SAVE_DIRE, 'Extended Data Fig. 9b.svg')
    elif outcome == 'g_ratio' and ai_threshold == 0.2:
        svg_file = os.path.join(SAVE_DIRE, 'Extended Data Fig. 9c.svg')
    elif outcome == 'cross_domain_ratio' and ai_threshold == 0.1:
        svg_file = os.path.join(SAVE_DIRE, 'Extended Data Fig. 9d.svg')
    elif outcome == 'refer_dist' and ai_threshold == 0.1:
        svg_file = os.path.join(SAVE_DIRE, 'Extended Data Fig. 9e.svg')
    elif outcome == 'title_dist' and ai_threshold == 0.1:
        svg_file = os.path.join(SAVE_DIRE, 'Extended Data Fig. 9f.svg')
    else:
        svg_file = os.path.join(SAVE_DIRE, f'event_study_ait{ai_threshold_str}_seed{RANDOM_SEED}_{outcome}.svg')
    plt.savefig(svg_file, format='svg', bbox_inches='tight', transparent=True)
    plt.close()
    print(f"Saved: {svg_file}")

    csv_file = svg_file.replace('.svg', '.csv')
    csv_rows = []
    for label, int_df in panels:
        df_copy = int_df.copy()
        df_copy['dataset'] = label
        csv_rows.append(df_copy)
    pd.concat(csv_rows, ignore_index=True).to_csv(csv_file, index=False)
    print(f"Saved: {csv_file}")

# ── Combined figure: cross_domain_ratio / refer_dist / title_dist ─────────────
COMBINED_OUTCOMES = ['cross_domain_ratio', 'refer_dist', 'title_dist']
COMBINED_DATASETS = [('arxiv', 3), ('biorxiv', 3), ('ssrn', 2)]
COMBINED_AI_THR   = 0.1
ai_thr_str        = str(COMBINED_AI_THR).replace('.', 'p')

n_rows = len(COMBINED_OUTCOMES)
n_cols = len(COMBINED_DATASETS)
mid_row = n_rows // 2   # 1
mid_col = n_cols // 2   # 1

fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(5 * n_cols, 4 * n_rows),
                         sharey=False)

for r, outcome in enumerate(COMBINED_OUTCOMES):
    for c, (dataset_name, pct) in enumerate(COMBINED_DATASETS):
        csv_stem = (f'{dataset_name}_event_study_pct{pct}'
                    f'_ait{ai_thr_str}_seed{RANDOM_SEED}_{outcome}')
        coef_csv = os.path.join(COEF_DIRE, f'{csv_stem}.csv')
        ax = axes[r][c]
        if not os.path.exists(coef_csv):
            ax.set_visible(False)
            print(f"  Skipping [{csv_stem}]: not found")
            continue
        int_df = pd.read_csv(coef_csv).sort_values('ri').reset_index(drop=True)
        dataset_label = 'SSRN' if dataset_name == 'ssrn' else dataset_name.capitalize()
        draw_panel(ax, int_df, dataset_label,
                   show_xlabel=(r == n_rows - 1 and c == mid_col),
                   show_ylabel=(c == 0),
                   outcome=outcome)

plt.tight_layout()
combined_svg = os.path.join(SAVE_DIRE, 'combined.svg')
plt.savefig(combined_svg, format='svg', bbox_inches='tight', transparent=True)
plt.close()
print(f"Saved: {combined_svg}")

combined_csv_rows = []
for outcome in COMBINED_OUTCOMES:
    for dataset_name, pct in COMBINED_DATASETS:
        csv_stem = (f'{dataset_name}_event_study_pct{pct}'
                    f'_ait{ai_thr_str}_seed{RANDOM_SEED}_{outcome}')
        coef_csv = os.path.join(COEF_DIRE, f'{csv_stem}.csv')
        if not os.path.exists(coef_csv):
            continue
        df_tmp = pd.read_csv(coef_csv).sort_values('ri').reset_index(drop=True)
        df_tmp['outcome'] = outcome
        df_tmp['dataset'] = 'SSRN' if dataset_name == 'ssrn' else dataset_name.capitalize()
        combined_csv_rows.append(df_tmp)
if combined_csv_rows:
    combined_csv = combined_svg.replace('.svg', '.csv')
    pd.concat(combined_csv_rows, ignore_index=True).to_csv(combined_csv, index=False)
    print(f"Saved: {combined_csv}")

print("\nAll done.")
