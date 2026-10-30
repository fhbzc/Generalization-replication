import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)             # utils.py is in the parent directory
from utils import META_DIRE

plt.rcParams['font.family'] = 'Gill Sans MT'

# =========================
# Config
# =========================
period_length = 5
print("period_length", period_length)

# When True, read the _cited_has_ref variants of the e/f/g CSVs (produced by the feg
# scripts when their require_cited_has_reference flag is on) and write the figure /
# csv under a matching suffix so the two modes never overwrite each other.
require_cited_has_reference = True
print("require_cited_has_reference", require_cited_has_reference)
mode_suffix = '_cited_has_ref' if require_cited_has_reference else ''

meta_dire = META_DIRE
print("meta_dire", meta_dire)

save_fig_dire = './figures/combine_metric_validation'
os.makedirs(save_fig_dire, exist_ok=True)

disp_dire = os.path.join(meta_dire, 'reference_disp_g_e_f', 'two_group')
inter_domain_dire = os.path.join(meta_dire, 'reference_interdis_g_e_f', 'two_group')

# =========================
# Helpers
# =========================
def process_value(df):
    df = df.copy()
    df['group'] = pd.to_numeric(df['group'], errors='coerce')
    df = df.dropna(subset=['group'])
    df['group'] = df['group'].astype(int)
    v1 = df.loc[df['group'] == 1, 'mean'].values[0]
    v0 = df.loc[df['group'] == 0, 'mean'].values[0]
    return (v1 - v0) / v0

def add_dotted_v_break(ax, bar, at_y, n_dots=7, visible_frac=0.70, amp=0.035,
                       dot_size=18, edge=1.2, color="white"):
    x0, w = bar.get_x(), bar.get_width()
    left  = x0 + (1 - visible_frac) * 0.5 * w
    right = x0 + (1 + visible_frac) * 0.5 * w
    xs = np.linspace(left, right, n_dots)
    mid = n_dots // 2
    ys = np.empty_like(xs)
    ys[:mid+1] = np.linspace(at_y + amp, at_y - amp, mid + 1)
    ys[mid:]   = np.linspace(at_y - amp, at_y + amp, n_dots - mid)
    if at_y < 0:
        ys = 2*at_y - ys
    ax.plot(xs, ys, linestyle="None", marker="o",
            mfc=color, mec=color, ms=dot_size, mew=edge,
            clip_on=False, zorder=6)

FEG_COLORS = ['red', 'orange', 'blue']

BAR_W = 0.06                 
INNER_GAP = BAR_W / 2        
GROUP_GAP = 0.36             
CAP = 0.5
MAX_GROUPS = 3               

VALUE_LABEL_PAD = 0.03
GROUP_LABEL_PAD = 0.08
# empty band above the group labels (where the panel title sits), as a FRACTION
# of the panel's own y-span. A fraction keeps the title equally close to the bars
# in short-bar panels (first row) and tall-bar panels (bottom row); a fixed value
# left the first row's titles floating far above their short bars.
TOP_MARGIN_FRAC = 0.07

def load_groups_and_names(test_group):
    if test_group == 'disp':
        groups = [
            [process_value(pd.read_csv(os.path.join(disp_dire, f'fou_disp_raw{mode_suffix}.csv'))),
             process_value(pd.read_csv(os.path.join(disp_dire, f'ext_disp_raw{mode_suffix}.csv'))),
             process_value(pd.read_csv(os.path.join(disp_dire, f'gen_disp_raw{mode_suffix}.csv')))],
            [process_value(pd.read_csv(os.path.join(disp_dire, f'fou_disp_i_j_k{mode_suffix}.csv'))),
             process_value(pd.read_csv(os.path.join(disp_dire, f'ext_disp_i_j_k{mode_suffix}.csv'))),
             process_value(pd.read_csv(os.path.join(disp_dire, f'gen_disp_i_j_k{mode_suffix}.csv')))],
            [process_value(pd.read_csv(os.path.join(disp_dire, f'fou_disp_i_j{mode_suffix}.csv'))),
             process_value(pd.read_csv(os.path.join(disp_dire, f'ext_disp_i_j{mode_suffix}.csv'))),
             process_value(pd.read_csv(os.path.join(disp_dire, f'gen_disp_i_j{mode_suffix}.csv')))],
        ]
        group_names = ['(i-j)/(i+j+k)', 'i/(i+j+k)', 'i/(i+j)']

    elif test_group == 'inter':
        groups = [
            [process_value(pd.read_csv(os.path.join(inter_domain_dire, f'fou_inter_all{mode_suffix}.csv'))),
             process_value(pd.read_csv(os.path.join(inter_domain_dire, f'ext_inter_all{mode_suffix}.csv'))),
             process_value(pd.read_csv(os.path.join(inter_domain_dire, f'gen_inter_all{mode_suffix}.csv')))],
            [process_value(pd.read_csv(os.path.join(inter_domain_dire, f'fou_inter_main{mode_suffix}.csv'))),
             process_value(pd.read_csv(os.path.join(inter_domain_dire, f'ext_inter_main{mode_suffix}.csv'))),
             process_value(pd.read_csv(os.path.join(inter_domain_dire, f'gen_inter_main{mode_suffix}.csv')))],
        ]
        group_names = ['Original Domain', 'Top Domain']

    elif test_group == 'semantic_distance':
        base = os.path.join(meta_dire, 'reference_text_distance_g_e_f', 'two_group')
        groups = [
            [process_value(pd.read_csv(os.path.join(base, f'fou_title_dist_p_l_{period_length}{mode_suffix}.csv'))),
             process_value(pd.read_csv(os.path.join(base, f'ext_title_dist_p_l_{period_length}{mode_suffix}.csv'))),
             process_value(pd.read_csv(os.path.join(base, f'gen_title_dist_p_l_{period_length}{mode_suffix}.csv')))],
            [process_value(pd.read_csv(os.path.join(base, f'fou_abs_dist_p_l_{period_length}{mode_suffix}.csv'))),
             process_value(pd.read_csv(os.path.join(base, f'ext_abs_dist_p_l_{period_length}{mode_suffix}.csv'))),
             process_value(pd.read_csv(os.path.join(base, f'gen_abs_dist_p_l_{period_length}{mode_suffix}.csv')))],
        ]
        group_names = ['Title Distance', 'Abstract Distance']

    elif test_group == 'reference_distance':
        base = os.path.join(meta_dire, 'reference_refer_distance_g_e_f', 'two_group')
        groups = [
            [process_value(pd.read_csv(os.path.join(base, f'fou_refer_dist_p_l_{period_length}{mode_suffix}.csv'))),
             process_value(pd.read_csv(os.path.join(base, f'ext_refer_dist_p_l_{period_length}{mode_suffix}.csv'))),
             process_value(pd.read_csv(os.path.join(base, f'gen_refer_dist_p_l_{period_length}{mode_suffix}.csv')))],
            [process_value(pd.read_csv(os.path.join(base, f'fou_refer_paper_dist_p_l_{period_length}{mode_suffix}.csv'))),
             process_value(pd.read_csv(os.path.join(base, f'ext_refer_paper_dist_p_l_{period_length}{mode_suffix}.csv'))),
             process_value(pd.read_csv(os.path.join(base, f'gen_refer_paper_dist_p_l_{period_length}{mode_suffix}.csv')))],
        ]
        group_names = ['Venue Distance', 'Paper Distance']

    else:
        raise ValueError(test_group)

    return groups, group_names

def draw_panel(ax, test_group, panel_title):
    groups, group_names = load_groups_and_names(test_group)

    # Top of THIS panel's content = its own largest bar (capped at CAP). Using a
    # per-panel top keeps the group labels and the panel title close to the bars,
    # instead of floating far above the short-bar panels (e.g. the first row),
    # which happened when a single global top was shared by every panel.
    panel_top = min(CAP, max(abs(v) for g in groups for v in g))

    x_centers_full = np.arange(MAX_GROUPS) * GROUP_GAP
    x_centers_use = x_centers_full[:len(groups)]
    offsets = np.array([-BAR_W - INNER_GAP/2, 0.0, BAR_W + INNER_GAP/2])

    for i in range(3):
        heights = [g[i] for g in groups]
        for xc, h in zip(x_centers_use, heights):
            draw_h = np.sign(h) * min(abs(h), CAP)
            bar = ax.bar(xc + offsets[i], draw_h, width=BAR_W, color=FEG_COLORS[i])[0]
            y = draw_h + VALUE_LABEL_PAD * np.sign(draw_h if draw_h != 0 else 1)
            va = 'bottom' if h >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width()/2, y, f'{abs(h):.2f}',
                    ha='center', va=va, fontsize=9)
            if abs(h) > CAP:
                at_y = (CAP - 0.10) * (1 if h > 0 else -1)
                add_dotted_v_break(ax, bar, at_y=at_y,
                                   n_dots=7, visible_frac=1.0, amp=0.035,
                                   dot_size=1, edge=1.2, color="white")

    for xc, name in zip(x_centers_use, group_names):
        ax.text(xc, panel_top + GROUP_LABEL_PAD, name, ha='center', va='bottom', fontsize=14)

    # explicit y range so the title-to-bar gap scales with the panel's span,
    # keeping the title equally close in short-bar (first row) and tall-bar panels.
    drawn = [np.sign(v) * min(abs(v), CAP) for g in groups for v in g]
    content_top = panel_top + GROUP_LABEL_PAD
    panel_bottom = min(drawn + [0.0]) - VALUE_LABEL_PAD
    span = content_top - panel_bottom
    ax.set_xlim(x_centers_use[0] - GROUP_GAP/2, x_centers_use[-1] + GROUP_GAP/2)
    ax.set_ylim(panel_bottom, content_top + TOP_MARGIN_FRAC * span)

    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    for side in ['left', 'right', 'top', 'bottom']:
        ax.spines[side].set_visible(False)

    ax.set_title(panel_title, fontsize=20, pad=8)

# =========================
# Compose 2x2 grid
# =========================
test_groups = ['disp', 'inter', 'semantic_distance', 'reference_distance']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

draw_panel(axes[0, 0], 'disp', 'Disruption')
draw_panel(axes[1, 0], 'inter', 'Cross-Domain Ratio')
draw_panel(axes[0, 1], 'semantic_distance', 'Semantic Distance')
draw_panel(axes[1, 1], 'reference_distance', 'Reference Distance')

plt.tight_layout()
if period_length == 5:
    out_path = os.path.join(save_fig_dire, f'Extended Data Fig. 9{mode_suffix}.svg')
    csv_path = os.path.join(save_fig_dire, f'Extended Data Fig. 9{mode_suffix}.csv')
else:
    out_path = os.path.join(save_fig_dire, f'combine_metric_grid_uniform_y_thicker_p_l_{period_length}{mode_suffix}.svg')
    csv_path = None
plt.savefig(out_path, format='svg', transparent=True, bbox_inches='tight')
print("saved:", out_path)

if csv_path is not None:
    color_label = ['Foundation', 'Extension', 'Generalization']
    panel_titles = {'disp': 'Disruption', 'inter': 'Cross-Domain Ratio',
                    'semantic_distance': 'Semantic Distance', 'reference_distance': 'Reference Distance'}
    rows = []
    for tg in test_groups:
        groups, group_names = load_groups_and_names(tg)
        for gname, vals in zip(group_names, groups):
            row = {'test_group': panel_titles[tg], 'metric_name': gname}
            for label, val in zip(color_label, vals):
                row[label] = val
            rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print("saved:", csv_path)
