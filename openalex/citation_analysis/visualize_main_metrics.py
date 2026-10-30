import os
import pickle
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)             # utils.py is in the parent directory
from utils import META_DIRE


plt.rcParams['font.family'] = 'Gill Sans MT'

period_length = 5
print("period_length", period_length)

# When True, read the _cited_has_ref variants of the e/f/g CSVs (produced by the feg
# scripts when their require_cited_has_reference flag is on) and write the figure /
# csv under a matching suffix so the two modes never overwrite each other.
require_cited_has_reference = False
print("require_cited_has_reference", require_cited_has_reference)
mode_suffix = '_cited_has_ref' if require_cited_has_reference else ''


meta_dire = META_DIRE
print("meta_dire",meta_dire)

figure_dire = './figures/combine_metric'
os.makedirs(figure_dire, exist_ok=True)

text_dist_dire = os.path.join(meta_dire, 'reference_text_distance_g_e_f/two_group')
disp_dire = os.path.join(meta_dire, 'reference_disp_g_e_f/two_group')
inter_domain_dire = os.path.join(meta_dire, 'reference_interdis_g_e_f/two_group')


def process_value(df):
    df = df.copy()
    df['group'] = pd.to_numeric(df['group'], errors='coerce')
    df = df.dropna(subset=['group'])
    df['group'] = df['group'].astype(int)
    v1 = df.loc[df['group'] == 1, 'mean'].values[0]
    v0 = df.loc[df['group'] == 0, 'mean'].values[0]
    v = (v1 - v0) / v0
    return v

gen_inter = pd.read_csv(os.path.join(inter_domain_dire, f'gen_inter_all{mode_suffix}.csv'))
gen_inter = process_value(gen_inter)

ext_inter = pd.read_csv(os.path.join(inter_domain_dire, f'ext_inter_all{mode_suffix}.csv'))
ext_inter = process_value(ext_inter)

fou_inter = pd.read_csv(os.path.join(inter_domain_dire, f'fou_inter_all{mode_suffix}.csv'))
fou_inter = process_value(fou_inter)



gen_title_dist = pd.read_csv(os.path.join(text_dist_dire, "gen_title_dist_p_l_%s%s.csv"%(period_length, mode_suffix)))
gen_title_dist = process_value(gen_title_dist)

ext_title_dist = pd.read_csv(os.path.join(text_dist_dire, "ext_title_dist_p_l_%s%s.csv"%(period_length, mode_suffix)))
ext_title_dist = process_value(ext_title_dist)

fou_title_dist = pd.read_csv(os.path.join(text_dist_dire, "fou_title_dist_p_l_%s%s.csv"%(period_length, mode_suffix)))
fou_title_dist = process_value(fou_title_dist)




gen_distance_refer = pd.read_csv(os.path.join(meta_dire, 'reference_refer_distance_g_e_f/two_group', f'gen_refer_dist_p_l_{period_length}{mode_suffix}.csv'))
gen_distance_refer = process_value(gen_distance_refer)


ext_distance_refer = pd.read_csv(os.path.join(meta_dire, 'reference_refer_distance_g_e_f/two_group', f'ext_refer_dist_p_l_{period_length}{mode_suffix}.csv'))
ext_distance_refer = process_value(ext_distance_refer)


fou_distance_refer = pd.read_csv(os.path.join(meta_dire, 'reference_refer_distance_g_e_f/two_group', f'fou_refer_dist_p_l_{period_length}{mode_suffix}.csv'))
fou_distance_refer = process_value(fou_distance_refer)




groups = [[fou_inter, ext_inter, gen_inter], [fou_title_dist, ext_title_dist, gen_title_dist], [fou_distance_refer, ext_distance_refer, gen_distance_refer]]
group_names = ['Cross-Domain', 'Semantic Distance', 'Reference Distance']   # 每组名字






colors = ['red', 'orange', 'blue']                  
color_label = ['Foundation', 'Extension', 'Generalization']

w = 0.2
inner_gap = 0.03
group_gap = 1.0      # smaller -> groups sit closer together

x_centers = np.arange(len(groups)) * group_gap

offsets = np.array([-w - inner_gap/2, 0.0, w + inner_gap/2])

fig, ax = plt.subplots(figsize=(7, 5))

cluster_tops = []



cap = 0.5  

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

for i in range(3):  # F/E/G
    heights = [g[i] for g in groups]
    bars = []
    for j, (xc, h) in enumerate(zip(x_centers, heights)):
        if abs(h) > cap:
            draw_h = cap * np.sign(h)
        else:
            draw_h = h
        lbl = color_label[i] if j == 0 else "_nolegend_"
        bar = ax.bar(xc + offsets[i], draw_h, width=w, color=colors[i], label=lbl)[0]
        bars.append(bar)

        y = draw_h + 0.05*np.sign(draw_h if draw_h!=0 else 1)
        va = 'bottom' if h >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2, y, f'{abs(h):.2f}',
                ha='center', va=va, fontsize=10)

        if abs(h) > cap:
            at_y = (cap-0.1)
            if h < 0:
                at_y = -at_y
            add_dotted_v_break(ax, bar, at_y=at_y,
                               n_dots=7, visible_frac=1, amp=0.035,
                               dot_size=1, edge=1.2, color="white")

all_tops = [abs(val) for g in groups for val in g]
global_top = max(all_tops)
global_top = min(global_top, cap)
pad = 0.2   
for xc, name in zip(x_centers, group_names):
    ax.text(xc, global_top + pad, name, ha='center', va='bottom', fontsize=12)

ax.set_ylim(top=global_top + pad + 0.5)

title_y = global_top + pad           
legend_x = x_centers[1]              
legend_y = title_y - 0.04            

leg = ax.legend(
    loc='upper center',
    bbox_to_anchor=(legend_x, legend_y),
    bbox_transform=ax.transData,
    ncol=1,
    frameon=False,
    handlelength=1.1, handletextpad=0.4, labelspacing=0.3
)

ax.yaxis.set_visible(False)
ax.xaxis.set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)  

# period_length == 5 with require_cited_has_reference == False IS Fig. 2e, so the figure and
# the data table behind it are named after the panel. Both parameters are in the gate: they
# each change what the bars show (which upstream two-group CSVs are read), so no other run
# can land on this name -- the require_cited_has_reference variant falls back to the
# descriptive name instead of claiming Fig. 2e with different numbers.
FIG_NAME = 'Fig. 2e'
is_fig2e = (period_length == 5) and (not require_cited_has_reference)
print("is_fig2e", is_fig2e, ("-> %s" % FIG_NAME) if is_fig2e else "")

stem = FIG_NAME if is_fig2e else f'combine_metric_p_l_{period_length}{mode_suffix}'
svg_path = os.path.join(figure_dire, f'{stem}.svg')
# the data table is written for the 5-year specification (both modes); other period_lengths
# stay figure-only, as before.
csv_path = os.path.join(figure_dire, f'{stem}.csv') if period_length == 5 else None

plt.savefig(svg_path, format='svg', transparent=True, bbox_inches='tight')

if csv_path is not None:
    rows = []
    for metric_name, vals in zip(group_names, groups):
        for label, val in zip(color_label, vals):
            rows.append({'metric': metric_name, 'group': label, 'value': val})
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[Saved] {csv_path}")

print("finished")

