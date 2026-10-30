import os
import math
import pickle
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import sys
# utils.py lives two levels up (openalex/), not in this word_apperance/ directory;
# add it to the path so the import resolves regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import META_DIRE

plt.rcParams['font.family'] = 'Gill Sans MT'

# ----------------- Config -----------------
meta_dire = META_DIRE
print("meta_dire", meta_dire)

load_dire = os.path.join(meta_dire, 'efg_computation')
figure_dire = os.path.join('./figures/word_apperance_density/same_group')
save_file_dire = os.path.join(meta_dire, 'word_apperance_density/same_group')
os.makedirs(save_file_dire, exist_ok=True)
os.makedirs(figure_dire, exist_ok=True)

# ----- params -----
# must match get_efg.py / get_efg_s2_merge.py so the merged file name resolves
year_offset = 5
# normal mode only (reshuffle variant lives in atypicality/get_efg_reshuffle.py)
citation_threshold = 5
reference_threshold = 1
# F = foundation (pillar), E = extension, G = generalization (normalized shares)
foundation_col = f'f_y{year_offset}'
extension_col = f'e_y{year_offset}'
generalization_col = f'g_y{year_offset}'
citation_col = f'citation_count_y{year_offset}'
within_year_group_flag = False
year_threshold = 2020
N = 10                       
N_BOOT = 1000                
CI_LEVEL = 95                
RNG_SEED = 123               

word_list = [
    'device', 'tool', 'software',
    'review', 'guideline', 'tutorial',
    'discovery', 'dataset', 'observation',
    'theory', 'metric', 'hypothesis',
    'new', 'novel', 'innovative',
]
print("word_list",word_list)
dims = [
    ('pillar_ratio', 'Foundation', 'red'),
    ('extension_ratio', 'Extension', 'orange'),
    ('generalization_ratio', 'Generalization', 'blue'),
]

# At the deliverable setting (within_year_group_flag=False, reference_threshold=1,
# citation_threshold=5) this grid IS an Extended Data panel, so the figure AND the cached
# data table behind it both take the panel name. One constant, used in both places, so a
# renumbering can never leave the svg and the csv disagreeing.
ED_FIG_NAME = 'Extended Data Fig. 2'
is_ed_fig = (within_year_group_flag is False and reference_threshold == 1
             and citation_threshold == 5)
print("is_ed_fig", is_ed_fig, ("-> %s" % ED_FIG_NAME) if is_ed_fig else "")

ROW_SPACING = 0.6
BOTTOM_MARGIN = 0.13
LEGEND_Y = 0.05
TITLE_PAD = 18
TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 16

# ---------- Helpers ----------
def bootstrap_ci_from_counts(total, match, n_boot=1000, level=95, seed=None):

    if total <= 0:
        return np.nan, np.nan, np.nan
    p = match / total
    rng = np.random.default_rng(seed)
    # parametric bootstrap: Binomial(total, p)
    samples = rng.binomial(total, p, size=n_boot) / total
    low = np.percentile(samples, (100 - level) / 2) * 100.0
    high = np.percentile(samples, 100 - (100 - level) / 2) * 100.0
    mean = p * 100.0
    return float(mean), float(low), float(high)

def draw_panel(ax, df_word_one, dims, N):
    for name, label, color in dims:
        sub = df_word_one[df_word_one['metric'] == name].set_index('bin')
        x = list(range(N))
        y     = [sub.at[i, 'percent'] if i in sub.index else np.nan for i in x]
        ylow  = [sub.at[i, 'ci_low']  if i in sub.index else np.nan for i in x]
        yhigh = [sub.at[i, 'ci_high'] if i in sub.index else np.nan for i in x]

        ax.plot(x, y, linewidth=2, label=label, color=color)
        ax.fill_between(x, ylow, yhigh, color=color, alpha=0.20, linewidth=0)

    ax.set_xticks(range(N))
    ax.tick_params(axis='x', labelsize=TICK_FONTSIZE)
    ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)

    ax.set_title(df_word_one['word'].iloc[0].title(), fontsize=18, pad=TITLE_PAD)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def draw_grid(df_all, dims, N, figure_dire, within_year_group_flag, reference_threshold, citation_threshold):
    words = list(dict.fromkeys(df_all['word']))  
    ncols = 3
    nrows = math.ceil(len(words) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 4.2, nrows * 3.6),
        sharex=False, sharey=False
    )
    axes = np.array(axes).reshape(nrows, ncols)
    ax_flat = axes.ravel()

    for i, w in enumerate(words):
        df_w = df_all[df_all['word'] == w]
        draw_panel(ax_flat[i], df_w, dims, N)

    for j in range(len(words), nrows * ncols):
        fig.delaxes(ax_flat[j])

    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            if c == 0:
                ax.set_ylabel("% of papers containing the word", fontsize=16)
            if r == nrows - 1:
                ax.set_xlabel("Index Decile", fontsize=16)

    fig.tight_layout()
    fig.subplots_adjust(bottom=BOTTOM_MARGIN, top=0.94, hspace=ROW_SPACING)

    handles, labels_ = ax_flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels_,
        loc="lower center",
        bbox_to_anchor=(0.5, LEGEND_Y),
        ncol=3,
        frameon=False,
        fontsize=LEGEND_FONTSIZE
    )
    if is_ed_fig:
        out_name = f'{ED_FIG_NAME}.svg'
    else:
        out_name = (
            f"word_prevalence_grid_r_t_{reference_threshold}_c_t_{citation_threshold}"
            f"{'_within_year' if within_year_group_flag else ''}_bootstrap.svg"
        )
    out_path = os.path.join(figure_dire, out_name)
    fig.savefig(out_path, format='svg', transparent=True)
    plt.close(fig)

# ---------- Result CSV path ----------
if is_ed_fig:
    csv_all_path = os.path.join(save_file_dire, f'{ED_FIG_NAME}.csv')
else:
    csv_all_path = os.path.join(
        save_file_dire,
        f"word_prevalence_bins_r_t_{reference_threshold}_c_t_{citation_threshold}"
        f"{'_within_year' if within_year_group_flag else ''}_bootstrap.csv"
    )

# ---------- Compute or Load ----------
if os.path.exists(csv_all_path):
    print(f"[load] {csv_all_path}")
    df_all = pd.read_csv(csv_all_path)
else:
    print("[compute] building from raw tables...")

    csv_path = os.path.join(
        load_dire,
        f'gen_V3_y{year_offset}.csv'
    )
    usecols = [
        'pid', citation_col, 'up_refer_count', 'pub_year',
        foundation_col, extension_col, generalization_col
    ]
    df_gen = pd.read_csv(csv_path, usecols=usecols)
    df_gen = df_gen[
        (df_gen['up_refer_count'] >= reference_threshold) &
        (df_gen[citation_col] >= citation_threshold) &
        (df_gen['pub_year'] < year_threshold)
    ].copy()
    df_gen.rename(columns={
        foundation_col: 'pillar_ratio',
        extension_col: 'extension_ratio',
        generalization_col: 'generalization_ratio',
    }, inplace=True)

    for name in ['pillar_ratio', 'extension_ratio', 'generalization_ratio']:
        epsilon = 1e-9
        rng = np.random.default_rng(seed=42)
        df_gen[name] = df_gen[name] + rng.uniform(low=-epsilon, high=epsilon, size=len(df_gen))
        bin_col = f'{name}_bin'

        if within_year_group_flag:
            df_gen[bin_col] = np.nan
            for year, g in tqdm(df_gen.groupby('pub_year'), desc=f'qcut by year for {name}', leave=False):
                bins = pd.qcut(g[name], q=N, labels=False, duplicates='drop')
                df_gen.loc[g.index, bin_col] = bins
        else:
            df_gen[bin_col] = pd.qcut(df_gen[name], q=N, labels=False, duplicates='drop')

        df_gen[bin_col] = df_gen[bin_col].astype('Int64')

    method_string = 'lemma'  # [lemma, raw]
    token_pkl = os.path.join(meta_dire, f'pid2eng_token_list_{method_string}_before_{year_threshold}.pkl')
    with open(token_pkl, 'rb') as f:
        pid2token_list = pickle.load(f)
    print("load pid2token_list finished")

    word_set = set(word_list)
    word2contain_pid = defaultdict(list)
    for pid, token_list in tqdm(pid2token_list.items(), desc='generate tokens', leave=False):
        for w in (set(token_list) & word_set):
            word2contain_pid[w].append(pid)

    out_rows = []
    rng = np.random.default_rng(RNG_SEED)

    for target_word in tqdm(word_list, desc='compute per word'):
        contains_pid = set(word2contain_pid.get(target_word, []))
        contains = df_gen['pid'].isin(contains_pid).astype(int)

        df_tmp = df_gen[['pid'] + [f'{nm}_bin' for nm,_,_ in dims]].copy()
        df_tmp['contains_word'] = contains

        for name, _, _ in dims:
            bin_col = f'{name}_bin'
            g = (df_tmp
                 .dropna(subset=[bin_col])
                 .groupby(bin_col)['contains_word']
                 .agg(total='count', match='sum')
                 .reset_index())

            for _, row in g.iterrows():
                b = int(row[bin_col])
                total = int(row['total']); match = int(row['match'])
                mean, low, high = bootstrap_ci_from_counts(
                    total, match, n_boot=N_BOOT, level=CI_LEVEL, seed=rng.integers(0, 2**32-1)
                )
                out_rows.append({
                    'word': target_word,
                    'metric': name,
                    'bin': b,
                    'total': total,
                    'match': match,
                    'percent': mean,
                    'ci_low': low,
                    'ci_high': high
                })

    df_all = pd.DataFrame(out_rows)
    df_all.to_csv(csv_all_path, index=False)
    print(f"[saved] {csv_all_path}")

# ---------- Welch's t-test: bottom bin (0) vs top bin (N-1) ----------
def welch_ttest_from_counts(total1, match1, total2, match2):
    """Welch's t-test for two binomial proportions given aggregate counts."""
    if total1 < 2 or total2 < 2:
        return np.nan, np.nan, np.nan
    p1 = match1 / total1
    p2 = match2 / total2
    s1 = math.sqrt(p1 * (1 - p1) * total1 / (total1 - 1))
    s2 = math.sqrt(p2 * (1 - p2) * total2 / (total2 - 1))
    # ttest_ind_from_stats returns (statistic, pvalue); df via Welch-Satterthwaite
    result = stats.ttest_ind_from_stats(
        mean1=p1, std1=s1, nobs1=total1,
        mean2=p2, std2=s2, nobs2=total2,
        equal_var=False
    )
    # Compute Welch-Satterthwaite df explicitly
    v1 = s1 ** 2 / total1
    v2 = s2 ** 2 / total2
    df = (v1 + v2) ** 2 / (v1 ** 2 / (total1 - 1) + v2 ** 2 / (total2 - 1))
    return float(result.statistic), float(result.pvalue), float(df)

ttest_rows = []
bottom_bin = 0
top_bin = N - 1

for target_word in tqdm(word_list, desc='Welch t-test per word', position=0):
    df_w = df_all[df_all['word'] == target_word]
    for name, label, _ in tqdm(dims, desc=f'  metrics for "{target_word}"', leave=False, position=1):
        df_metric = df_w[df_w['metric'] == name].set_index('bin')
        if bottom_bin not in df_metric.index or top_bin not in df_metric.index:
            t_val, p_val, df_val = np.nan, np.nan, np.nan
        else:
            row_bot = df_metric.loc[bottom_bin]
            row_top = df_metric.loc[top_bin]
            t_val, p_val, df_val = welch_ttest_from_counts(
                int(row_bot['total']), int(row_bot['match']),
                int(row_top['total']), int(row_top['match'])
            )
        ttest_rows.append({
            'word': target_word,
            'metric': name,
            'metric_label': label,
            'bottom_bin': bottom_bin,
            'top_bin': top_bin,
            't_value': t_val,
            'p_value': p_val,
            'df': df_val,
        })

df_ttest = pd.DataFrame(ttest_rows)
ttest_csv_path = os.path.join(
    save_file_dire,
    f"word_prevalence_welch_ttest_bottom_vs_top"
    f"_r_t_{reference_threshold}_c_t_{citation_threshold}"
    f"{'_within_year' if within_year_group_flag else ''}.csv"
)
df_ttest.to_csv(ttest_csv_path, index=False)
print(f"[saved] Welch t-test results -> {ttest_csv_path}")
print(df_ttest.to_string(index=False))

# ---------- Plot ----------
draw_grid(
    df_all=df_all,
    dims=dims,
    N=N,
    figure_dire=figure_dire,
    within_year_group_flag=within_year_group_flag,
    reference_threshold=reference_threshold,
    citation_threshold=citation_threshold
)

print("finished")
