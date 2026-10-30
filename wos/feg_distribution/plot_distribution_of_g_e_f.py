"""
Fig S8.b: distribution (histograms) of the Foundation (f_y), Extension (e_y), and
Generalization (g_y) indices across WoS papers, after filtering by pub_year in
[min_year, max_year), up_refer_count >= reference_threshold, and citation_count_y{off}
>= citation_threshold. Reads the merged E/F/G table from get_efg_s2_merge.py; the
filtered three-column table is cached to CSV before drawing. year_offset must match
get_efg.py / get_efg_s2_merge.py.

Input  : META_DIRE/efg_computation/gen_V3_y{year_offset}.csv (from get_efg_s2_merge.py)
Output : ./figures/descriptive_stats/Fig S8.b.svg + Fig S8.b.csv
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless / cluster-safe
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

# utils.py is in the parent directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
from utils import META_DIRE  # noqa: E402

# try the reference font but fall back silently if unavailable (e.g. on Linux)
try:
    import matplotlib.font_manager as fm
    if any("Gill Sans MT" in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = "Gill Sans MT"
except Exception:
    pass

meta_dire = META_DIRE
print("meta_dire", meta_dire)

figure_dire = os.path.join('./figures/descriptive_stats')
os.makedirs(figure_dire, exist_ok=True)

min_year = 1945
max_year = 2020

reference_threshold = 1
citation_threshold = 5

# input is get_efg_s2_merge.py's output; year_offset must match that script
year_offset = 5
gen_file_load_dire = os.path.join(meta_dire, 'efg_computation', f'gen_V3_y{year_offset}.csv')

# get_efg output column names and their plot labels
citation_col = f'citation_count_y{year_offset}'
load_cols = ['pub_year', 'up_refer_count', citation_col,
             f'f_y{year_offset}', f'e_y{year_offset}', f'g_y{year_offset}']
rename_map = {f'f_y{year_offset}': 'foundation',
              f'e_y{year_offset}': 'extension',
              f'g_y{year_offset}': 'generalization'}

print("min_year", min_year)
print("max_year", max_year)
print("reference_threshold", reference_threshold)
print("citation_threshold", citation_threshold)

if reference_threshold == 1 and citation_threshold == 5 and min_year == 1945 and max_year == 2020:
    truncated_df_gen_dire = os.path.join(figure_dire, 'Fig S8.b.csv')
else:
    truncated_df_gen_dire = os.path.join(
        figure_dire,
        f'f_e_g_dist_min_{min_year}_max_{max_year}_r_f_{reference_threshold}_c_f_{citation_threshold}.csv')

if os.path.exists(truncated_df_gen_dire) == False:
    df_generalization = pd.read_csv(gen_file_load_dire, usecols=load_cols)
    df_generalization.rename(columns=rename_map, inplace=True)

    df_generalization = df_generalization[df_generalization['pub_year'] < max_year]
    df_generalization = df_generalization[df_generalization['pub_year'] >= min_year]
    df_generalization = df_generalization[df_generalization['up_refer_count'] >= reference_threshold]
    df_generalization = df_generalization[df_generalization[citation_col] >= citation_threshold]

    df_generalization = df_generalization[['foundation', 'extension', 'generalization']].copy()
    df_generalization.to_csv(truncated_df_gen_dire, index=False)
    print(f"Saved distribution data to {truncated_df_gen_dire}")
else:
    df_generalization = pd.read_csv(truncated_df_gen_dire)

# ---- descriptive stats ----
print(len(df_generalization))
print(len(df_generalization[df_generalization['foundation'] == 0]), 'foundation = 0')
print(len(df_generalization[df_generalization['foundation'] == 1]), 'foundation = 1')
foundation_list = list(df_generalization['foundation'])
print(np.mean(foundation_list), 'mean foundation')
print(np.median(foundation_list), 'median foundation')

print(len(df_generalization[df_generalization['extension'] == 0]), 'extension = 0')
print(len(df_generalization[df_generalization['extension'] == 1]), 'extension = 1')
ext_list = list(df_generalization['extension'])
print(np.mean(ext_list), 'mean ext')
print(np.median(ext_list), 'median ext')

print(len(df_generalization[df_generalization['generalization'] == 0]), 'generalization = 0')
print(len(df_generalization[df_generalization['generalization'] == 1]), 'generalization = 1')
gen_list = list(df_generalization['generalization'])
print(np.mean(gen_list), 'mean gen')
print(np.median(gen_list), 'median gen')

# =========================
# Plot distributions with independent y-axis and no top/right spines
# =========================
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

# Common font sizes
label_fontsize = 14
tick_fontsize = 12
title_fontsize = 16

# Foundation
sns.histplot(df_generalization['foundation'], bins=50, ax=axes[0], color="red")
axes[0].set_title("Foundation", fontsize=title_fontsize)
axes[0].set_xlabel("Index Value", fontsize=label_fontsize)
axes[0].set_ylabel("Paper Count", fontsize=label_fontsize)
axes[0].tick_params(axis="both", labelsize=tick_fontsize)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Extension
sns.histplot(df_generalization['extension'], bins=50, ax=axes[1], color="orange")
axes[1].set_title("Extension", fontsize=title_fontsize)
axes[1].set_xlabel("Index Value", fontsize=label_fontsize)
axes[1].set_ylabel("Paper Count", fontsize=label_fontsize)
axes[1].tick_params(axis="both", labelsize=tick_fontsize)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# Generalization
sns.histplot(df_generalization['generalization'], bins=50, ax=axes[2], color="blue")
axes[2].set_title("Generalization", fontsize=title_fontsize)
axes[2].set_xlabel("Index Value", fontsize=label_fontsize)
axes[2].set_ylabel("Paper Count", fontsize=label_fontsize)
axes[2].tick_params(axis="both", labelsize=tick_fontsize)
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)

plt.tight_layout()
if reference_threshold == 1 and citation_threshold == 5 and min_year == 1945 and max_year == 2020:
    svg_path = os.path.join(figure_dire, 'Fig S8.b.svg')
else:
    svg_path = os.path.join(
        figure_dire,
        f'f_e_g_dist_min_{min_year}_max_{max_year}_r_f_{reference_threshold}_c_f_{citation_threshold}.svg')
plt.savefig(svg_path, format='svg', transparent=True)

print("finished")
