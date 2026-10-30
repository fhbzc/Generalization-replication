"""
Plot the longitudinal change of the E/F/G (Foundation / Extension / Generalization)
indices for OpenAlex vs WoS on a shared time axis.

Sources (both share the same e/f/g_count schema; both are unfiltered concatenations,
this script applies the rt / ct / pub_year thresholds):
  - OpenAlex : get_efg_2026_merge.py output
        META_DIRE_2026/gen_V3_keep_disp/gen_V3_y{activity_year}.csv
  - WoS      : wos/get_efg_s2_merge.py output
        WOS_META_DIRE/efg_computation/gen_V3_y{activity_year}.csv

Both merges expose, per paper, the already-normalized E/F/G index shares
(count / citation_count, set to 0 when citation_count == 0):
  pub_year, up_refer_count, citation_count_y{ay}, e_y{ay}, f_y{ay}, g_y{ay}
which map directly to:
  foundation = f_y, extension = e_y, generalization = g_y

NOTE: activity_year here must match the year_offset used when BOTH merges were run.

At the paper's settings (activity_year=1, reference_threshold=1, citation_threshold=2) this
figure IS 'Fig. 3a' (is_fig3a): the figure and the data table behind it are then named after
the panel; every other setting keeps the descriptive dynamic_feg_dual_axis_* name.

Cache  : ./figures/wos_openalex_joint_pattern/
           yearly_avg_wos_y{ay}_ref{rt}_cit{ct}.csv
           yearly_avg_openalex_y{ay}_ref{rt}_cit{ct}.csv
         (per-source yearly means; when BOTH exist they are loaded and the big gen_V3
          tables are not read. These keep their parameterized names in every run --
          delete them to recompute.)
Output : ./figures/wos_openalex_joint_pattern/
           Fig. 3a.svg + Fig. 3a.csv at the deliverable setting, else
           dynamic_feg_dual_axis_y{ay}_ref{rt}_cit{ct}.svg + .csv
         (the csv is always the svg name with the extension swapped: both sources'
          yearly means stacked, with a 'dataset' column of WoS / OpenAlex)
"""

import os
import importlib.util

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load THIS directory's utils.py explicitly by file path, so it can never be
# shadowed by another "utils" module on sys.path or in sys.modules. It derives
# OPENALEX_META / WOS_META from the OpenAlex and WoS projects' own utils.py, so
# they always match where the merge scripts wrote.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_utils_spec = importlib.util.spec_from_file_location(
    'combine_wos_openalex_utils', os.path.join(BASE_DIR, 'utils.py')
)
_utils = importlib.util.module_from_spec(_utils_spec)
_utils_spec.loader.exec_module(_utils)
OPENALEX_META = _utils.OPENALEX_META
WOS_META = _utils.WOS_META

# OpenAlex 2026 metadata root (get_efg_2026_merge.py writes under here);
# WoS metadata root (wos/get_efg_s2_merge.py writes under here).
openalex_meta = OPENALEX_META
wos_meta = WOS_META

# =========================
# Config
# =========================
marker_years = [1948, 1953, 1967, 1991, 2000, 2007, 2020, 2022]
activity_year = 1            # must match year_offset used in BOTH merge scripts
reference_threshold = 1      # must match get_efg_2026_merge.py (it is baked into the OA filename)
citation_threshold = 2       # must match get_efg_2026_merge.py (it is baked into the OA filename)
max_year = 2025
print("wos_meta", wos_meta)
print("openalex_meta", openalex_meta)
print("activity_year", activity_year)
print("reference_threshold", reference_threshold)
print("citation_threshold", citation_threshold)
print("max_year", max_year)
print("marker_years", marker_years)

# both sources share this schema (e_y/f_y/g_y are the already-normalized index shares)
use_col_list = [
    "pub_year", "up_refer_count", f'citation_count_y{activity_year}',
    f'e_y{activity_year}',
    f'f_y{activity_year}',
    f'g_y{activity_year}',
]

# =========================
# File paths
# =========================
fig_dir = os.path.join('./figures/wos_openalex_joint_pattern')
os.makedirs(fig_dir, exist_ok=True)

wos_csv_path = os.path.join(
    fig_dir,
    f'yearly_avg_wos_y{activity_year}_ref{reference_threshold}_cit{citation_threshold}.csv'
)
oa_csv_path = os.path.join(
    fig_dir,
    f'yearly_avg_openalex_y{activity_year}_ref{reference_threshold}_cit{citation_threshold}.csv'
)

# =========================
# Preprocess (shared: both sources have the same e/f/g_count schema)
# =========================
def preprocess(df, max_pub_year):
    df = df[df['pub_year'] >= 1945]
    # pub_year cutoff (source-specific: OpenAlex 2026 has one more year than WoS)
    df = df[df['pub_year'] < max_pub_year]
    df = df[df['up_refer_count'] >= reference_threshold]
    df = df[df[f'citation_count_y{activity_year}'] >= citation_threshold]
    df = df.rename(columns={
        f'f_y{activity_year}': 'foundation',
        f'e_y{activity_year}': 'extension',
        f'g_y{activity_year}': 'generalization',
    })
    return df


# =========================
# Load or compute yearly averages
# =========================
if os.path.exists(wos_csv_path) and os.path.exists(oa_csv_path):
    print("Loading saved yearly averages...")
    yearly_wos = pd.read_csv(wos_csv_path, index_col="pub_year")
    yearly_oa  = pd.read_csv(oa_csv_path, index_col="pub_year")
else:
    print("Computing yearly averages from raw data...")
    # WoS: wos/get_efg_s2_merge.py output (unfiltered concat)
    wos_path = os.path.join(
        wos_meta, 'efg_computation',
        f'gen_V3_y{activity_year}.csv'
    )
    # OpenAlex: get_efg_2026_merge.py output (unfiltered concat; thresholds applied below)
    openalex_path = os.path.join(
        openalex_meta, 'gen_V3_keep_disp',
        f'gen_V3_y{activity_year}.csv'
    )
    print("wos_path", wos_path)
    print("openalex_path", openalex_path)

    df_gen_wos_raw = pd.read_csv(wos_path, usecols=use_col_list)
    df_gen_openalex_raw = pd.read_csv(openalex_path, usecols=use_col_list)

    # same thresholds applied to both unfiltered concatenations, but a source-specific
    # pub_year cutoff: OpenAlex (2026 snapshot) has data one year later than WoS.
    #   activity_year=1 -> OpenAlex pub_year < 2025, WoS pub_year < 2024
    df_gen_wos = preprocess(df_gen_wos_raw, 2025 - activity_year)
    df_gen_openalex = preprocess(df_gen_openalex_raw, 2026 - activity_year)

    print('wos', len(df_gen_wos))
    print('openalex', len(df_gen_openalex))

    agg_cols = ['foundation', 'extension', 'generalization']
    yearly_wos = df_gen_wos.groupby('pub_year')[agg_cols].mean().sort_index()
    yearly_oa  = df_gen_openalex.groupby('pub_year')[agg_cols].mean().sort_index()

    yearly_wos.to_csv(wos_csv_path, index_label="pub_year")
    yearly_oa.to_csv(oa_csv_path, index_label="pub_year")
    print(f"Saved yearly averages:\n  {wos_csv_path}\n  {oa_csv_path}")
# =========================
# Plot (dual y-axes, refined)
# =========================
all_years = pd.Index(sorted(set(yearly_wos.index).union(set(yearly_oa.index))))
yearly_wos = yearly_wos.reindex(all_years)
yearly_oa  = yearly_oa.reindex(all_years)

plt.rcParams['font.family'] = 'Gill Sans MT'
fig, ax1 = plt.subplots(figsize=(16, 5.5))

# shared size for the x-label and BOTH y-labels (left Foundation axis / right E+G axis);
# a notch above the 14pt tick labels so the axis labels read clearly at print size
axis_label_fontsize = 19

c_foundation = 'red'
c_extension  = 'orange'
c_general    = 'blue'

# --- Left axis: Foundation ---
ax1.plot(all_years, yearly_oa['foundation'], color=c_foundation, linestyle='-',  linewidth=2.0, label='Foundation (OpenAlex)')
ax1.plot(all_years, yearly_wos['foundation'], color=c_foundation, linestyle='--', linewidth=2.0, label='Foundation (WoS)')
ax1.set_xlabel('Publication Year', fontsize=axis_label_fontsize)
ax1.set_ylabel('Foundation Index', fontsize=axis_label_fontsize, color=c_foundation)
ax1.tick_params(axis='y', labelcolor=c_foundation, labelsize=14)

# --- Right axis: Extension + Generalization ---
ax2 = ax1.twinx()
ax2.plot(all_years, yearly_oa['extension'],      color=c_extension, linestyle='-',  linewidth=2.0, label='Extension (OpenAlex)')
ax2.plot(all_years, yearly_wos['extension'],     color=c_extension, linestyle='--', linewidth=2.0, label='Extension (WoS)')
ax2.plot(all_years, yearly_oa['generalization'], color=c_general,   linestyle='-',  linewidth=2.0, label='Generalization (OpenAlex)')
ax2.plot(all_years, yearly_wos['generalization'],color=c_general,   linestyle='--', linewidth=2.0, label='Generalization (WoS)')
ax2.set_ylabel('Extension / Generalization Index', fontsize=axis_label_fontsize)
ax2.tick_params(axis='y', labelsize=14)

# --- X axis: 1945 → last_year, label last_year ---
last_year = int(all_years.max())
ax1.set_xlim(1945, last_year)  # exact range
xticks = list(range(1945, last_year + 1, 10))
if last_year not in xticks:
    xticks.append(last_year)
ax1.set_xticks(sorted(xticks))
ax1.tick_params(axis='x', labelsize=14)

for y in marker_years:
    ax1.axvline(
        x=y,
        color='gray',
        linestyle='--',
        linewidth=1.2,
        alpha=0.7,
        zorder=0
    )

# --- Remove top spine; keep right spine (for ax2) ---
sns.despine(ax=ax1, top=True, right=False)   # left axis: no top, keep right so twin spine aligns
ax2.spines['top'].set_visible(False)         # right axis: hide its top spine only

# --- Legend inside (upper-right), 3 rows × 2 cols: OA | WoS per row ---
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
handles = h1 + h2
labels  = l1 + l2

def idx(name): return labels.index(name)
order = [
    idx('Foundation (OpenAlex)'),     idx('Extension (OpenAlex)'),     idx('Generalization (OpenAlex)'),
    idx('Foundation (WoS)'),          idx('Extension (WoS)'),          idx('Generalization (WoS)')
]
ordered_handles = [handles[i] for i in order]
ordered_labels  = [labels[i]  for i in order]

ax2.legend(
    ordered_handles, ordered_labels,
    ncol=2,
    frameon=False,
    loc='upper right',
    bbox_to_anchor=(1, 0.9),
    fontsize=13
)

fig.tight_layout()

# This dual-axis OpenAlex-vs-WoS trend at the paper's settings IS Fig. 3a, so the figure and
# the data table behind it are named after the panel; every other setting keeps the
# descriptive name. The gate lists every parameter that changes what the lines show, so no
# other run can land on this name. (The per-source yearly_avg_* caches above stay
# parameterized -- they are keyed by the same thresholds and shared across runs.)
FIG_NAME = 'Fig. 3a'
is_fig3a = (activity_year == 1 and reference_threshold == 1 and citation_threshold == 2)
print("is_fig3a", is_fig3a, ("-> %s" % FIG_NAME) if is_fig3a else "")

if is_fig3a:
    out_path = os.path.join(fig_dir, f'{FIG_NAME}.svg')
else:
    out_path = os.path.join(
        fig_dir,
        f'dynamic_feg_dual_axis_y{activity_year}_ref{reference_threshold}_cit{citation_threshold}.svg'
    )
plt.savefig(out_path, format='svg', transparent=True, bbox_inches='tight')
plt.close()
print(f"Saved figure: {out_path}")

out_csv = out_path.replace('.svg', '.csv')
wos_combined = yearly_wos.copy()
wos_combined['dataset'] = 'WoS'
oa_combined = yearly_oa.copy()
oa_combined['dataset'] = 'OpenAlex'
combined_df = pd.concat([wos_combined, oa_combined])
combined_df.index.name = 'pub_year'
combined_df.to_csv(out_csv)
print(f"Saved CSV: {out_csv}")
