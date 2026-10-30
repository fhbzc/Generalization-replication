import pickle
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)             # utils.py is in the parent directory
from utils import META_DIRE

meta_dire = META_DIRE
print("meta_dire",meta_dire)

figure_dire = os.path.join('./figures/descriptive_stats')
os.makedirs(figure_dire, exist_ok = True)

min_year = 1945
max_year = 2020

reference_threshold = 1
citation_threshold = 5

# input is get_efg_s2_merge.py's output; these must match that script's settings
year_offset = 5
# normal mode only (reshuffle variant lives in atypicality/get_efg_reshuffle.py)
gen_file_load_dire = os.path.join(
    meta_dire, 'efg_computation',
    f'gen_V3_y{year_offset}.csv'
)

# get_efg output column names and their plot labels
citation_col = f'citation_count_y{year_offset}'
load_cols = ['pub_year', 'up_refer_count', citation_col,
             f'f_y{year_offset}', f'e_y{year_offset}', f'g_y{year_offset}']
rename_map = {f'f_y{year_offset}': 'foundation',
              f'e_y{year_offset}': 'extension',
              f'g_y{year_offset}': 'generalization'}

print("min_year",min_year)
print("max_year",max_year)

print("reference_threshold",reference_threshold)
print("citation_threshold",citation_threshold)

truncated_df_gen_dire = os.path.join(figure_dire, 'Fig S8.a.csv')
if os.path.exists(truncated_df_gen_dire) == False:
    df_generalization = pd.read_csv(gen_file_load_dire, usecols = load_cols)

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

# =========================
# Plot distributions with independent y-axis and no top/right spines
# =========================
plt.rcParams['font.family'] = 'Gill Sans MT'

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


plt.savefig(os.path.join(figure_dire, "Fig S8.a.svg"), 
            dpi=300, format='svg', transparent=True)

print("finished")