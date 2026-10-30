import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)             # utils.py is in the parent directory
from utils import META_DIRE
# -----------------------------
# Paths & params
# -----------------------------
plt.rcParams['font.family'] = 'Gill Sans MT'

meta_dire = META_DIRE
print("meta_dire", meta_dire)

load_directory = os.path.join(meta_dire, 'merged_stats')
figure_directory = os.path.join('./figures/refer_title_distance_g_e_f_heatmap/conditioned')
os.makedirs(figure_directory, exist_ok=True)

period_length = 5
load_refer_threshold = 1
citation_threshold = 5
token_threshold = 2
if token_threshold < load_refer_threshold:
    token_threshold = load_refer_threshold
assert token_threshold >= load_refer_threshold

semantic_type = 'title'  # ['abs','title']
reference_type = 'venue' # [venue, paper]

print("load_refer_threshold", load_refer_threshold)
print("token_threshold", token_threshold)
print("semantic_type", semantic_type)
print("reference_type", reference_type)
print("period_length", period_length)
print("citation_threshold", citation_threshold)

if semantic_type == 'abs':
    load_file_name =  os.path.join(
        load_directory,
        'merged_dist_min_y_1986_max_y_2025_p_l_%s_r_t_%s_c_t_%s.csv'%(period_length, load_refer_threshold, citation_threshold)
    )
    min_year = 1986
    max_year = 2020
    text_name = 'abs'
    ylabel = 'Semantic Distance (Abstract) Quantile'
elif semantic_type == 'title':
    load_file_name = os.path.join(
        load_directory,
        'merged_dist_min_y_1951_max_y_2025_p_l_%s_r_t_%s_c_t_%s_no_abs.csv'
        % (period_length, load_refer_threshold, citation_threshold)
    )
    max_year = 2020
    min_year = 1951
    text_name = 'title'
    ylabel = 'Semantic Distance (Title) Quantile'
else:
    raise ValueError("semantic_type must be one of ['abs','title']")
print("min_year", min_year)
print("max_year", max_year)


if reference_type == 'venue':
    refer_name = 'refer'
    xlabel = 'Reference Distance (Venue) Quantile'
elif reference_type == 'paper':
    refer_name = 'refer_paper'
    xlabel = 'Reference Distance (Paper) Quantile'
else:
    assert False
# -----------------------------
# Load data
# -----------------------------
agg_type = 'median'
dist_type = 'mean'
print("agg_type", agg_type)
print("dist_type", dist_type)

if dist_type == 'mean':
    text_dist_name = f'{text_name}_dist_mean'
    refer_dist_name = f'{refer_name}_dist_mean'
else:
    raise ValueError("Only dist_type='mean' implemented here")

usecols = [
    'pub_year', refer_dist_name, text_dist_name, 'up_refer_count',
    'citation_count', 'pillar_ratio', 'extension_ratio', 'generalization_ratio',
    f'{text_name}_token_count', 'journal_refer_count'
]

# -----------------------------
# Heatmap / combine config
# -----------------------------
bin_method = "qcut"    
n_bins = 30
min_count_per_cell = 100        
min_rows_per_stratum = n_bins   
print("min_count_per_cell", min_count_per_cell)

INVERT_ALPHA = False
ALPHA_MIN, ALPHA_MAX = 0.05, 1.00
BG_RGBA = np.array([0.95, 0.95, 0.95, 0.25])

COLOR_MAP = {
    "pillar": mcolors.to_rgba("#E41A1C"),     # Foundation
    "extension": mcolors.to_rgba("#FFA500"),  # Extension
    "generalization": mcolors.to_rgba("#3772E7"),  # Generalization
}

def make_bins(s, n=10, method="qcut"):
    s = s.copy()
    if s.nunique(dropna=True) <= 1:
        return pd.Series(np.zeros(len(s), dtype=int), index=s.index, dtype="Int64")
    if method == "qcut":
        try:
            b = pd.qcut(s, q=n, labels=False, duplicates='drop')
        except ValueError:
            b = pd.cut(s, bins=n, labels=False, include_lowest=True)
        if b.notna().any():
            uniq = np.sort(b.dropna().unique())
            remap = {u: i for i, u in enumerate(uniq)}
            b = b.map(remap)
    else:
        b = pd.cut(s, bins=n, labels=False, include_lowest=True)
    return b.astype("Int64")

# up_refer_count & citation_count：<32 单值；>=32 2^k~2^{k+1}-1
def pow2_bucket_label_and_key(x, start=32):
    v = int(x)
    if v < start:
        return str(v), (0, v)
    k = int(np.floor(np.log2(v)))
    lo = max(start, 2 ** k)
    hi = (2 ** (k + 1)) - 1
    return f"[{lo}-{hi}]", (1, lo)

def sanitize_for_path(s: str):
    return (str(s)
            .replace(' ', '')
            .replace('[', '').replace(']', '')
            .replace('(', '').replace(')', '')
            .replace(',', '_')
            .replace('+', 'p')
            .replace('-', '_')
            .replace(':', '_')
            .replace('/', '_'))

def quantile_to_alpha(q, invert=False):
    if not np.isfinite(q):
        return None
    if invert:
        q = 1.0 - q
    return ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * float(q)

def draw_axis(ax, n_bins, ylabel, title_text=None):

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis='both', which='both', length=0)

    ax.set_xlim(0, n_bins)
    ax.set_ylim(0, n_bins)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    if title_text:
        ax.set_title(title_text, fontsize=14)

def add_cellwise_quantiles_global(df, metric_prefix: str):
    med_col = f"{metric_prefix}_agg"
    q_col   = f"{metric_prefix}_q_global"
    vals = df[med_col].to_numpy()
    srt = np.sort(vals)
    ranks = np.searchsorted(srt, vals, side="right")
    qs = ranks / srt.size
    df[q_col] = qs
    return df

def build_rgba_from_valid(valid_df, metric_key: str, n_bins: int):
    rgb = np.array(COLOR_MAP[metric_key][:3])
    img = np.ones((n_bins, n_bins, 4), dtype=float)
    img[:, :, :] = BG_RGBA
    q_col = f"{metric_key}_q_global"
    for _, row in valid_df.iterrows():
        x = int(row["refer_bin"])
        yb = int(row["title_bin"])
        q = row[q_col]
        a = quantile_to_alpha(q, INVERT_ALPHA)
        if a is None:
            continue
        img[n_bins - 1 - yb, x, :] = np.array([rgb[0], rgb[1], rgb[2], a])
    return img

# -----------------------------
# Output file paths (needed early for "redraw" path)
# -----------------------------
file_name =  f'heatmap_ALLDIV_pool_median_min_y_{min_year}_max_y_{max_year}_p_l_{period_length}_r_t_{load_refer_threshold}_c_t_{citation_threshold}_t_l_{token_threshold}_{text_name}_{refer_name}.svg'
# ONLY the title semantic distance x venue reference distance panel is a named deliverable
# (Fig. 2b); every other combination keeps the descriptive name above.
if period_length == 5 and load_refer_threshold == 1 and citation_threshold == 5 and token_threshold == 2:

    if semantic_type == 'title' and reference_type == 'venue':
        file_name = 'Fig. 2b.svg'



out_svg_all = os.path.join(figure_directory, file_name)
stats_csv = os.path.join(figure_directory, file_name.replace('.svg', '.csv'))

# -----------------------------
# If stats CSV exists → redraw directly; else compute and save
# -----------------------------
if os.path.exists(stats_csv):
    print(f"Found existing stats CSV: {stats_csv}\nRedrawing figure from CSV...")
    valid = pd.read_csv(stats_csv)

    required_cols = {"refer_bin","title_bin","n","pillar_agg","extension_agg","generalization_agg"}
    if not required_cols.issubset(set(valid.columns)):
        raise ValueError(f"CSV missing required columns. Found: {valid.columns}")

    valid = valid[valid["n"] >= min_count_per_cell].copy()

    for k in ["pillar","extension","generalization"]:
        q_col = f"{k}_q_global"
        if q_col not in valid.columns:
            valid = add_cellwise_quantiles_global(valid, k)

    rgba_p = build_rgba_from_valid(valid, "pillar", n_bins)
    rgba_e = build_rgba_from_valid(valid, "extension", n_bins)
    rgba_g = build_rgba_from_valid(valid, "generalization", n_bins)

else:
    print("No existing stats CSV. Computing pooled stats...")

    df_merged = pd.read_csv(load_file_name, usecols=usecols)
    df_merged = df_merged[df_merged['up_refer_count'] >= token_threshold]
    df_merged = df_merged[df_merged[f'{text_name}_token_count'] >= token_threshold]
    df_merged = df_merged[(df_merged['pub_year'] >= min_year) & (df_merged['pub_year'] < max_year)].copy()

    # Build division groups
    upref_labels_keys = df_merged['up_refer_count'].apply(lambda v: pow2_bucket_label_and_key(v, start=32))
    cite_labels_keys  = df_merged['citation_count'].apply(lambda v: pow2_bucket_label_and_key(v, start=32))

    df_merged['upref_group'] = upref_labels_keys.apply(lambda t: t[0])
    df_merged['upref_key0']  = upref_labels_keys.apply(lambda t: t[1][0])
    df_merged['upref_key1']  = upref_labels_keys.apply(lambda t: t[1][1])

    df_merged['cite_group']  = cite_labels_keys.apply(lambda t: t[0])
    df_merged['cite_key0']   = cite_labels_keys.apply(lambda t: t[1][0])
    df_merged['cite_key1']   = cite_labels_keys.apply(lambda t: t[1][1])

    year_vals = sorted(df_merged['pub_year'].dropna().unique().tolist())
    upref_groups = (
        df_merged[['upref_group','upref_key0','upref_key1']]
        .drop_duplicates().sort_values(['upref_key0','upref_key1'])
        ['upref_group'].tolist()
    )
    cite_groups = (
        df_merged[['cite_group','cite_key0','cite_key1']]
        .drop_duplicates().sort_values(['cite_key0','cite_key1'])
        ['cite_group'].tolist()
    )

    # Pool all divisions' samples (only needed columns)
    pool_chunks = []
    need_cols = ['refer_bin','title_bin','pillar_ratio','extension_ratio','generalization_ratio']

    for y in tqdm(year_vals):
        df_y = df_merged[df_merged['pub_year'] == y]
        if df_y.empty:
            continue
        for ugr in upref_groups:
            df_y_ur = df_y[df_y['upref_group'] == ugr]
            if df_y_ur.shape[0] < min_rows_per_stratum:
                continue
            for cgr in cite_groups:
                df_stratum = df_y_ur[df_y_ur['cite_group'] == cgr].copy()
                if df_stratum.shape[0] < min_rows_per_stratum:
                    continue

                df_stratum["refer_bin"] = make_bins(df_stratum[refer_dist_name], n_bins, bin_method)
                df_stratum["title_bin"] = make_bins(df_stratum[text_dist_name], n_bins, bin_method)
                df_stratum = df_stratum.dropna(subset=["refer_bin", "title_bin"]).copy()
                if df_stratum.empty:
                    continue

                pool_chunks.append(df_stratum[need_cols])

    if len(pool_chunks) == 0:
        raise SystemExit("No pooled samples after stratification and binning.")
    pooled = pd.concat(pool_chunks, ignore_index=True)

    # Aggregate per (refer_bin, title_bin) without weighting: sample-level medians
    agg_all = (
        pooled.groupby(["refer_bin","title_bin"])
              .agg(
                  n=("pillar_ratio","size"),
                  pillar_agg=("pillar_ratio","median"),
                  extension_agg=("extension_ratio","median"),
                  generalization_agg=("generalization_ratio","median")
              )
              .reset_index()
    )

    valid = agg_all[agg_all['n'] >= min_count_per_cell].copy()
    if valid.empty:
        raise SystemExit("No valid cells (n >= min_count_per_cell) after pooling.")

    # Add global quantiles for transparency mapping and save CSV for future redraws
    for k in ["pillar","extension","generalization"]:
        valid = add_cellwise_quantiles_global(valid, k)

    valid.to_csv(stats_csv, index=False)
    print("Saved pooled stats CSV:", stats_csv)

    # Build layers
    rgba_p = build_rgba_from_valid(valid, "pillar", n_bins)
    rgba_e = build_rgba_from_valid(valid, "extension", n_bins)
    rgba_g = build_rgba_from_valid(valid, "generalization", n_bins)

# -----------------------------
# Draw final 3-panel figure
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 8), constrained_layout=True)

axes[0].imshow(rgba_p, origin="upper", extent=[0, n_bins, 0, n_bins], interpolation="nearest", aspect="equal")
draw_axis(axes[0], n_bins, ylabel, title_text="Foundation")

axes[1].imshow(rgba_e, origin="upper", extent=[0, n_bins, 0, n_bins], interpolation="nearest", aspect="equal")
draw_axis(axes[1], n_bins, ylabel, title_text="Extension")

axes[2].imshow(rgba_g, origin="upper", extent=[0, n_bins, 0, n_bins], interpolation="nearest", aspect="equal")
draw_axis(axes[2], n_bins, ylabel, title_text="Generalization")

plt.savefig(out_svg_all, format="svg", transparent=True)
plt.close(fig)

print("Saved single figure:", out_svg_all)
print("finished")
