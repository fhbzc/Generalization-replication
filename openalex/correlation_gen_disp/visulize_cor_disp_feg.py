import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)             # utils.py is in the parent directory
from utils import META_DIRE
plt.rcParams['font.family'] = 'Gill Sans MT'

# ---- Inputs ----
meta_dire = META_DIRE
print("meta_dire", meta_dire)
save_dire = os.path.join(meta_dire, 'disruption')
activity_year = 5

reference_threshold = 1
citation_threshold = 5
bin_count = 10
print("activity_year", activity_year)
print("citation_threshold", citation_threshold)
print("reference_threshold", reference_threshold)


figure_directory = os.path.join('./figures/cor_disp')
os.makedirs(figure_directory, exist_ok=True)
file_directory = os.path.join(meta_dire, 'cor_disp')
os.makedirs(file_directory, exist_ok=True)

# generalization side comes from get_efg_s2_merge.py's output;
# these must match that script's settings
# normal mode only (reshuffle variant lives in atypicality/get_efg_reshuffle.py)
gen_file_load_dire = os.path.join(
    meta_dire, 'efg_computation',
    f'gen_V3_y{activity_year}.csv'
)
print("gen_file_load_dire", gen_file_load_dire)




# ---- Compute Pearson r on RAW df_merged ----
def pearson_r(df, x, y):
    s = df[[x, y]].dropna()
    if len(s) < 2:
        return np.nan
    return s.corr(method='pearson').iloc[0, 1]

label2color = {'generalization': 'blue', 'foundation': 'red', 'extension': 'orange'}
labels = ['foundation', 'extension', 'generalization']
label2r_file = os.path.join(file_directory, 'label2r_ratio.pkl')


# ---- Configs for saving ----
if reference_threshold == 1 and citation_threshold == 5:
    figure_filename = 'Fig 1.c.svg'
    csv_filename = 'Fig 1.c.csv'
else:
    figure_filename = f'cor_disp_gen_ratio_r_t_{reference_threshold}_c_t_{citation_threshold}.svg'
    csv_filename = f'disp_gen_ratio_ext_foun_r_t_{reference_threshold}_c_t_{citation_threshold}.csv'

csv_saved_dire = os.path.join(file_directory, csv_filename)

# ---- Bootstrap CI helper ----
def fast_bootstrap_mean_ci(data, num_samples=100, ci=95, random_state=None):
    data = np.asarray(data)
    n = len(data)
    if n == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed=random_state)
    idx = rng.integers(0, n, size=(num_samples, n))
    means = data[idx].mean(axis=1)
    lower = np.percentile(means, (100 - ci) / 2)
    upper = np.percentile(means, 100 - (100 - ci) / 2)
    mean = means.mean()
    return mean, lower, upper


if not os.path.exists(csv_saved_dire) or not os.path.exists(label2r_file):
    # ---- Load data (sampled for speed via nrows; remove nrows to use full data) ----


    data_gen = pd.read_csv(
        gen_file_load_dire,
        usecols=[
            "pid",
            f'f_y{activity_year}',
            f'e_y{activity_year}',
            f'g_y{activity_year}',
        ]
    )
    # ---- Filter ----

    data_disp = pd.read_csv(
        os.path.join(save_dire, 'disp_full.csv'),
        usecols=['pid', "pub_year", "up_refer_count", f"citation_count_y{activity_year}", f"disp_y{activity_year}"]
    )
    data_disp = data_disp[data_disp['pub_year'] < 2025 - activity_year]
    data_disp = data_disp[data_disp['up_refer_count'] >= reference_threshold]
    data_disp = data_disp[data_disp[f'citation_count_y{activity_year}'] >= citation_threshold]
    # ---- Merge & rename ----
    df_merged = pd.merge(data_gen, data_disp, on='pid', how='inner')
    df_merged.rename(columns={
        f'f_y{activity_year}': 'foundation',
        f'e_y{activity_year}': 'extension',
        f'g_y{activity_year}': 'generalization',
        f'disp_y{activity_year}': 'disp',
        f'citation_count_y{activity_year}': 'citations'
    }, inplace=True)
    # ---- Small noise to break ties (optional) ----
    rng = np.random.default_rng(seed=42)
    noise = rng.uniform(low=-1e-9, high=1e-9, size=len(df_merged))
    df_merged['disp'] = df_merged['disp'] + noise
    label2r = {
        'generalization': pearson_r(df_merged, 'disp', 'generalization'),
        'foundation': pearson_r(df_merged, 'disp', 'foundation'),
        'extension': pearson_r(df_merged, 'disp', 'extension'),
    }

    with open(label2r_file, 'wb') as f:
        pickle.dump(label2r, f, pickle.HIGHEST_PROTOCOL)

    # ---- Compute & Save CSV if needed ----

    # Bin by disruption into equal-count quantiles
    df_merged['disp_bin'] = pd.qcut(df_merged['disp'], q=bin_count, labels=False, duplicates='drop')

    type2bin2value_list = {}
    for label in tqdm(labels, desc='collect values per disp_bin'):
        type2bin2value_list[label] = (
            df_merged.groupby('disp_bin')[label].apply(list).to_dict()
        )

    plot_data = {'bin': list(range(bin_count))}
    for label in labels:
        bin2values = type2bin2value_list[label]
        count_list, mean_list, std_list = [], [], []
        min_list, max_list = [], []
        mean_boot_list, low_boot_list, up_boot_list = [], [], []
        se_list = []

        for b in range(bin_count):
            values = bin2values.get(b, [])
            n = len(values)
            if n > 0:
                arr = np.asarray(values)
                m = float(np.mean(arr))
                sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
                mn = float(np.min(arr))
                mx = float(np.max(arr))
                mb, lb, ub = fast_bootstrap_mean_ci(arr, num_samples=100, ci=95, random_state=123)
                se = sd / np.sqrt(n) if n > 1 else 0.0
            else:
                m = sd = mn = mx = mb = lb = ub = se = np.nan

            count_list.append(n)
            mean_list.append(m)
            std_list.append(sd)
            min_list.append(mn)
            max_list.append(mx)
            mean_boot_list.append(mb)
            low_boot_list.append(lb)
            up_boot_list.append(ub)
            se_list.append(se)

        plot_data[f'count_{label}'] = count_list
        plot_data[f'mean_{label}'] = mean_list
        plot_data[f'std_{label}'] = std_list
        plot_data[f'min_{label}'] = min_list
        plot_data[f'max_{label}'] = max_list
        plot_data[f'mean_boot_{label}'] = mean_boot_list
        plot_data[f'lower_boot_{label}'] = low_boot_list
        plot_data[f'upper_boot_{label}'] = up_boot_list
        plot_data[f'standard_error_{label}'] = se_list

    df_plot = pd.DataFrame(plot_data)
    df_plot.to_csv(csv_saved_dire, index=False)
else:
    df_plot = pd.read_csv(csv_saved_dire)


    with open(label2r_file, 'rb') as f:
        label2r = pickle.load(f)


# ---- Plot (two y-axes: left for foundation/extension, right for generalization) ----
fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()  # right y-axis for generalization

for label in labels:
    x = df_plot['bin'].tolist()
    y = df_plot[f'mean_boot_{label}'].tolist()
    ylow = df_plot[f'lower_boot_{label}'].tolist()
    yhigh = df_plot[f'upper_boot_{label}'].tolist()

    if label == 'foundation':
        ax = ax1
    else:
        ax = ax2

    ax.plot(x, y, label=label, color=label2color[label], linewidth = 1)
    ax.fill_between(x, ylow, yhigh, alpha=0.3, color=label2color[label])

    # ---- Inline colored "Name (r=..)" label next to each line (replaces the legend) ----
    # anchor bin + point offset chosen from the line geometry so each label sits in a
    # clean, unambiguous open stretch next to its own line:
    #   extension     -> in the gap just below its high, flat left plateau
    #   foundation    -> just above its isolated upper-left start (below the extension line)
    #   generalization-> bottom-left, lifted clearly ABOVE blue's plateau (not crossing it);
    #                    slightly smaller font so it fits between descending-red and rising-blue
    if len(x) > 0:
        if label == 'extension':
            w_idx, w_dx, w_dy, w_va, w_ha, w_fs = 3, 0, -7, 'top', 'center', 14
        elif label == 'foundation':
            w_idx, w_dx, w_dy, w_va, w_ha, w_fs = 0, 2, -6, 'bottom', 'left', 14
        elif label == 'generalization':
            w_idx, w_dx, w_dy, w_va, w_ha, w_fs = 0, 2, 30, 'bottom', 'left', 12
        else:
            assert False
        if not np.isnan(y[w_idx]):
            ax.annotate(
                f"{label.capitalize()} (r={label2r[label]:.2f})",
                xy=(x[w_idx], y[w_idx]),
                xytext=(w_dx, w_dy),
                textcoords='offset points',
                color=label2color[label],
                fontsize=w_fs,
                fontweight='bold',
                va=w_va,
                ha=w_ha,
            )

# Labels
ax1.set_xlabel('Disruption (Index Decile)', fontsize=18)
ax1.set_ylabel('Foundation Index', fontsize=18)
ax2.set_ylabel('Extension / Generalization Index', fontsize=18)
ax1.set_xticks(np.arange(bin_count))   # <--- force 0–9 ticks


ax1.tick_params(axis='x', labelsize=14)  # x 轴刻度
ax1.tick_params(axis='y', labelsize=14)  # y 轴刻度
ax2.tick_params(axis='y', labelsize=14)  # 第二个 y 轴刻度


# Optional: color-code the right y-axis label/ticks to match generalization color
# ---- ax1: 全部红色 ----
ax1.yaxis.label.set_color('red')  
ax1.tick_params(axis='y', colors='red')
ax1.spines['left'].set_color('red')  

# ---- ax2: 全部蓝色 ----
ax2.yaxis.label.set_color('blue')  
ax2.tick_params(axis='y', colors='blue')
ax2.spines['right'].set_color('blue')
# Title
# plt.title('Correlation between F/E/G and D')

# (legend removed; each line is labelled inline in its own colour above)

# Aesthetic tweaks
for ax in (ax1, ax2):
    ax.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)   # left axis right spine hidden; ax2 draws right spine
ax2.spines['left'].set_visible(False)   # left axis right spine hidden; ax2 draws right spine

ax1.spines['left'].set_color('red')
ax2.spines['right'].set_color('blue')

plt.tight_layout()
plt.savefig(os.path.join(figure_directory, figure_filename), format='svg', transparent=True)
plt.close()

print("Pearson r (disp, generalization):", f"{label2r['generalization']:.6f}")
print("Pearson r (disp, foundation):    ", f"{label2r['foundation']:.6f}")
print("Pearson r (disp, extension):     ", f"{label2r['extension']:.6f}")
print(f"Saved CSV to: {csv_saved_dire}")
print(f"Saved figure to: {os.path.join(figure_directory, figure_filename)}")
