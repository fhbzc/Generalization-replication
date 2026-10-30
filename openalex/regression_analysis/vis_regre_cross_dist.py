import os
import pickle
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)             # utils.py is in the parent directory
from utils import META_DIRE
# =========================
# Config
# =========================
plt.rcParams['font.family'] = 'Gill Sans MT'
import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False

meta_dire = META_DIRE
print("meta_dire", meta_dire)

by_domain_type = 'all'  # [all, top]
save_dire_main = os.path.join(meta_dire, 'cited_reference_text_distance_citing_t')
save_dire_main_by_domain = os.path.join(save_dire_main, 'by_domain')

max_year = 2020
min_year = 1951
period_length = 5
refer_threshold = 1
citation_threshold = 5

# output dirs
plot_dir = './figures/regression_cross_paper_dist_by_domain'
os.makedirs(plot_dir, exist_ok=True)
cache_dir = plot_dir
os.makedirs(cache_dir, exist_ok=True)

domains = ['Art', 'Biology', 'Business', 'Chemistry', 'Computer science', 'Economics',
           'Engineering', 'Environmental science', 'Geography', 'Geology', 'History',
           'Materials science', 'Mathematics', 'Medicine', 'Philosophy', 'Physics',
           'Political science', 'Psychology', 'Sociology']
domains = sorted(domains)
# expected columns
usecols = [
    'pub_year', 'citation_count', 'up_refer_count', 'pillar_ratio', 'extension_ratio',
    'generalization_ratio', 'refer_dist_mean', 'title_dist_mean'
]

# =========================
# Helpers
# =========================
def safe_zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors='coerce')
    m = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    z = (s - m) / sd
    z[~np.isfinite(z)] = 0.0
    return z

def add_logs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, newcol in [
        ('citation_count', 'log_citation_count'),
        ('up_refer_count', 'log_up_refer_count'),
    ]:
        x = pd.to_numeric(df[col], errors='coerce')
        df[newcol] = np.log1p(x.clip(lower=0))
    return df

def scale_key_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['scaled_title_dist_mean'] = safe_zscore(df['title_dist_mean'])
    df['scaled_refer_dist_mean'] = safe_zscore(df['refer_dist_mean'])
    df['scaled_pillar_ratio'] = safe_zscore(df['pillar_ratio'])
    df['scaled_extension_ratio'] = safe_zscore(df['extension_ratio'])
    df['scaled_generalization_ratio'] = safe_zscore(df['generalization_ratio'])
    return df

def fit_one(df: pd.DataFrame, dep_scaled_col: str):
    needed = [
        dep_scaled_col,
        'log_citation_count', 'log_up_refer_count', 'scaled_title_dist_mean', 'scaled_refer_dist_mean',
        'pub_year'
    ]
    d = df[needed].dropna()
    if len(d) < 20:
        return None
    
    formula = (
        f"{dep_scaled_col} ~ log_citation_count + log_up_refer_count + "
        f"scaled_title_dist_mean + scaled_refer_dist_mean + C(pub_year)"
    )
    try:
        model = smf.ols(formula=formula, data=d).fit()
        out = {}
        for var in ['scaled_title_dist_mean', 'scaled_refer_dist_mean']:
            if var in model.params.index:
                coef = model.params[var]
                ci_low, ci_high = model.conf_int(alpha=0.05).loc[var].values
                out[var] = (coef, ci_low, ci_high)
            else:
                out[var] = (np.nan, np.nan, np.nan)
        return out
    except Exception as e:
        print("Regression failed:", e)
        return None

# =========================
# Compute (or load) results
# =========================
cache_csv = os.path.join(
    cache_dir,
    f"domain_fe_reg_results_{by_domain_type}_miny_{min_year}_maxy_{max_year}_pl_{period_length}_rt_{refer_threshold}_ct_{citation_threshold}.csv"
)

if (not os.path.exists(cache_csv)):
    records = []
    for domain in tqdm(domains, desc="Domains"):
        fname = f"{by_domain_type}_{domain}_merged_dist_min_y_{min_year}_max_y_{max_year}_p_l_{period_length}_r_t_{refer_threshold}_c_t_{citation_threshold}_title.csv"
        fpath = os.path.join(save_dire_main_by_domain, fname)
        if not os.path.exists(fpath):
            print(f"[WARN] Missing file for {domain}: {fpath}")
            continue

        df = pd.read_csv(fpath, usecols=usecols)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=['pub_year'])
        df['pub_year'] = pd.to_numeric(df['pub_year'], errors='coerce')
        df = add_logs(df)
        df = scale_key_cols(df)

        for dep_name, dep_col in [
            ('pillar', 'scaled_pillar_ratio'),
            ('extension', 'scaled_extension_ratio'),
            ('generalization', 'scaled_generalization_ratio'),
        ]:
            res = fit_one(df, dep_col)
            if res is None:
                continue
            (t_coef, t_lo, t_hi) = res['scaled_title_dist_mean']
            (r_coef, r_lo, r_hi) = res['scaled_refer_dist_mean']

            records.append({
                'domain': domain,
                'model': dep_name,
                'coef_title': t_coef,
                'ci_low_title': t_lo,
                'ci_high_title': t_hi,
                'coef_refer': r_coef,
                'ci_low_refer': r_lo,
                'ci_high_refer': r_hi,
                'n_obs': df.shape[0]
            })

    res_df = pd.DataFrame.from_records(records)
    res_df.to_csv(cache_csv, index=False)
    print(f"[Saved] {cache_csv}")
else:
    print(f"[Load cached] {cache_csv}")
    res_df = pd.read_csv(cache_csv)

# 保证 domain 顺序
res_df['domain'] = pd.Categorical(res_df['domain'], categories=domains, ordered=True)
res_df = res_df.sort_values(['domain', 'model']).reset_index(drop=True)

# =========================
# Plotting
# =========================
color_map = {'pillar': '#d62728', 'extension': '#ff7f0e', 'generalization': '#1f77b4'}
marker_map = {'pillar': 'D', 'extension': 'D', 'generalization': 'D'}
offsets = {'pillar': -0.25, 'extension': 0.0, 'generalization': 0.25}
def plot_effect(res_df: pd.DataFrame, which: str, out_prefix: str, title: str):
    GAP = 1.6
    DOT_SIZE = 2
    EDGE_PAD = 0.18 

    fig, ax = plt.subplots(figsize=(18, 6))
    x_domains = [d for d in domains if d in set(res_df['domain'].dropna().unique())]
    x = np.arange(len(x_domains)) * GAP

    for model in ['pillar', 'extension', 'generalization']:
        true_model = model if model != 'pillar' else 'foundation'
        sub = res_df[(res_df['model'] == model) & (res_df['domain'].isin(x_domains))]
        sub = sub.set_index('domain').reindex(x_domains)

        y = sub[f'coef_{which}'].values
        lo = sub[f'ci_low_{which}'].values
        hi = sub[f'ci_high_{which}'].values
        yerr = np.vstack([y - lo, hi - y])

        ax.errorbar(
            x + offsets[model], y, yerr=yerr,
            fmt=marker_map[model], markersize=DOT_SIZE,
            elinewidth=1.2, capsize=4, lw=1.2,
            label=true_model.capitalize(), color=color_map[model]
        )

    for xi in x:
        left_edge  = xi + offsets['pillar'] - EDGE_PAD           
        right_edge = xi + offsets['generalization'] + EDGE_PAD    
        ax.axvline(left_edge,  color='#dddddd', lw=0.8, alpha=0.9, zorder=0)
        ax.axvline(right_edge, color='#dddddd', lw=0.8, alpha=0.9, zorder=0)

    ax.axhline(0.0, color='#888888', lw=1.0, linestyle='--', zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(x_domains, rotation=35, ha='right', fontsize=14)

    if len(x) > 0:
        x_min = x[0]  + offsets['pillar'] - EDGE_PAD - 0.2
        x_max = x[-1] + offsets['generalization'] + EDGE_PAD + 0.2
        ax.set_xlim(x_min, x_max)

    ax.set_ylabel('Estimated coefficient (95% CI)', fontsize=18)
    ax.legend(frameon=True, ncol=3, loc='upper left', fontsize = 18)

    fig.tight_layout()
    svg_path = os.path.join(plot_dir, f"{out_prefix}.svg")
    fig.savefig(svg_path, bbox_inches='tight', transparent=True, format='svg')
    plt.close(fig)
    print(f"[Saved] {svg_path}")

# =========================
# Draw
# =========================
if res_df.empty:
    print("[WARN] No results to plot.")
else:

    if by_domain_type == 'all' and min_year == 1951 and max_year == 2020 and period_length == 5 and refer_threshold == 1 and citation_threshold == 5:
        plot_effect(
            res_df,
            which='title',
            out_prefix= 'Extended Data Fig. 10c',
            title='Effect of semantic distance in title (scaled) on scaled X-ratio, with year fixed effects'
        )
        plot_effect(
            res_df,
            which='refer',
            out_prefix= 'Extended Data Fig. 10d',
            title='Effect of reference distance (scaled) on scaled X-ratio, with year fixed effects'
        )
        res_df.to_csv(os.path.join(plot_dir, 'Extended Data Fig. 10cd.csv'), index=False)
        print(f"[Saved] {os.path.join(plot_dir, 'Extended Data Fig. 10cd.csv')}")
    else:
        plot_effect(
            res_df,
            which='title',
            out_prefix=f"title_cross_paper_{by_domain_type}_miny_{min_year}_maxy_{max_year}_pl_{period_length}_rt_{refer_threshold}_ct_{citation_threshold}",
            title='Effect of semantic distance in title (scaled) on scaled X-ratio, with year fixed effects'
        )
        plot_effect(
            res_df,
            which='refer',
            out_prefix=f"refer_cross_paper_{by_domain_type}_miny_{min_year}_maxy_{max_year}_pl_{period_length}_rt_{refer_threshold}_ct_{citation_threshold}",
            title='Effect of reference distance (scaled) on scaled X-ratio, with year fixed effects'
        )

print("Done.")
