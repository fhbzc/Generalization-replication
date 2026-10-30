"""Distribution of e_y5 / f_y5 / g_y5 for the OpenAlex vs S2AG public-release CSVs (Zenodo).

usage: python plot_efg_distribution_public_release.py <dir with openalex_efg_y5.csv and s2ag_efg_y5.csv>
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

D = sys.argv[1]
COLS = ['e_y5', 'f_y5', 'g_y5']
SOURCES = [('OpenAlex', 'openalex_efg_y5.csv', '#2a78d6'),   # categorical slot 1
           ('S2AG', 's2ag_efg_y5.csv', '#eb6834')]           # categorical slot 2
TEXT1, TEXT2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
EDGES = np.linspace(0, 1, 51)

hists, stats = {}, []
for name, fn, _ in SOURCES:
    df = pd.read_csv(os.path.join(D, fn), usecols=COLS, dtype={c: 'float32' for c in COLS})
    for c in COLS:
        v = df[c].to_numpy()
        cnt, _ = np.histogram(v, bins=EDGES)
        hists[name, c] = 100 * cnt / len(v)
        stats.append((name, c, len(v), v.mean(), np.median(v), (v == 0).mean(), (v == 1).mean()))
    print(f'{name}: {len(df):,} papers', flush=True)

plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': TEXT2,
                     'xtick.color': TEXT2, 'ytick.color': TEXT2})
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True, facecolor=SURFACE)
for ax, c in zip(axes, COLS):
    ax.set_facecolor(SURFACE)
    for name, _, color in SOURCES:
        ax.stairs(hists[name, c], EDGES, color=color, linewidth=2, label=name)
    ax.set_title(f'{c[0].upper()} index ({c})', color=TEXT1, loc='left', fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_xlabel(c)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
axes[0].set_ylabel('% of papers (bin width 0.02)')
n = {name: s[2] for s in stats for name in [s[0]]}
fig.legend(*axes[0].get_legend_handles_labels(), loc='upper right', ncol=2, frameon=False,
           labelcolor=TEXT1, bbox_to_anchor=(0.99, 1.0))
fig.suptitle('Distribution of E / F / G indices (5-year window)', x=0.01, ha='left', color=TEXT1, fontsize=13)
fig.text(0.01, 0.905, f"citation_count_y5 ≥ 5, reference_count ≥ 1, 1945 ≤ pub_year < 2020 · "
         f"OpenAlex n = {n['OpenAlex']:,}, S2AG n = {n['S2AG']:,}", ha='left', color=TEXT2, fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.9))
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(D, f'efg_distribution.{ext}'), dpi=200, facecolor=SURFACE)

st = pd.DataFrame(stats, columns=['source', 'index', 'n', 'mean', 'median', 'share_eq_0', 'share_eq_1'])
print(st.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
