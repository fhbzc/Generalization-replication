import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)             # utils.py is in the parent directory
from utils import META_DIRE

plt.rcParams['font.family']       = 'Gill Sans MT'
plt.rcParams['axes.unicode_minus'] = False

SAVE_DIRE = os.path.join('./figures', 'author_analysis')
os.makedirs(SAVE_DIRE, exist_ok=True)

# ── Settings ──────────────────────────────────────────────────────────────────
reference_threshold    = 1
citation_threshold     = 5
max_year               = 2020
n_bootstrap            = 1000
tick_size              = 13
label_size             = 15
legend_size            = 11

# Countries to plot (ISO 3166-1 alpha-2 codes)
target_countries = ['US', 'CN', 'GB', 'DE', 'JP', 'IN']

# Full names used for the legend
country_full_name = {
    'US': 'United States',
    'CN': 'China',
    'GB': 'United Kingdom',
    'DE': 'Germany',
    'JP': 'Japan',
    'IN': 'India',
}

print("reference_threshold:", reference_threshold)
print("citation_threshold:", citation_threshold)
print("max_year:", max_year)
print("target_countries:", target_countries)

country_tag = '_'.join(sorted(target_countries))
if reference_threshold == 1 and citation_threshold == 5 and max_year == 2020:
    csv_path = os.path.join(SAVE_DIRE, 'Extended Data Fig. 10c.csv')
else:
    csv_path = os.path.join(SAVE_DIRE,
        f'country_feg_trend_{country_tag}_rt_{reference_threshold}_ct_{citation_threshold}_my_{max_year}.csv')

# ── Load or compute ───────────────────────────────────────────────────────────
if os.path.exists(csv_path):
    print(f"Loading cached data from {csv_path}")
    df_agg = pd.read_csv(csv_path)

else:
    # ── 1. Load generalization data ───────────────────────────────────────────
    print("Loading generalization data ...")
    df = pd.read_csv(
        os.path.join(META_DIRE, 'efg_computation', 'gen_V3_y5.csv'),
        usecols=['pid', 'pub_year', 'up_refer_count', 'citation_count_y5',
                 'f_y5', 'e_y5', 'g_y5']
    )

    df = df[df['pub_year']          < max_year]
    df = df[df['up_refer_count']    >= reference_threshold]
    df = df[df['citation_count_y5'] >= citation_threshold]

    df['f'] = df['f_y5']
    df['e'] = df['e_y5']
    df['g'] = df['g_y5']

    valid_pids = set(df['pid'])
    print(f"  papers after threshold filter: {len(df)}")

    # ── 2. Load pid2author_country_list ───────────────────────────────────────
    # Structure: {pid: [(author_id, [country_code, ...]), ...]}
    print("Loading pid2author_country_list ...")
    with open(os.path.join(META_DIRE, 'pid2author_country_list.pkl'), 'rb') as f:
        pid2author_country_list = pickle.load(f)

    # For each pid, collect the set of countries from all authors
    pid2countries = {}
    for pid, author_list in tqdm(pid2author_country_list.items(), desc='extracting countries'):
        if pid not in valid_pids:
            continue
        countries = set()
        for author_id, country_list in author_list:
            for c in country_list:
                countries.add(c)
        if countries:
            pid2countries[pid] = countries

    # ── 3. Build per-paper country membership rows ────────────────────────────
    print("Building country-paper mapping ...")
    rows = []
    for pid, countries in tqdm(pid2countries.items(), desc='mapping countries'):
        for country in target_countries:
            if country in countries:
                rows.append({'pid': pid, 'country': country})

    df_country = pd.DataFrame(rows)
    print(f"  (pid, country) pairs: {len(df_country)}")

    # ── 4. Merge with F/E/G data ──────────────────────────────────────────────
    df_merged = df_country.merge(df[['pid', 'pub_year', 'f', 'e', 'g']], on='pid', how='inner')
    print(f"  merged rows: {len(df_merged)}")

    # ── 5. Bootstrap 95% CI per (country, pub_year) ───────────────────────────
    rng = np.random.default_rng(42)
    result_rows = []
    groups = list(df_merged.groupby(['country', 'pub_year']))
    for (country, pub_year), grp in tqdm(groups, desc='bootstrap'):
        row = {'country': country, 'pub_year': pub_year, 'n': len(grp)}
        for col in ['f', 'e', 'g']:
            vals = grp[col].dropna().values
            if len(vals) == 0:
                row[f'{col}_mean'] = np.nan
                row[f'{col}_ci_low']  = np.nan
                row[f'{col}_ci_high'] = np.nan
                continue
            row[f'{col}_mean'] = vals.mean()
            boot_means = np.array([
                rng.choice(vals, size=len(vals), replace=True).mean()
                for _ in range(n_bootstrap)
            ])
            row[f'{col}_ci_low']  = np.percentile(boot_means, 2.5)
            row[f'{col}_ci_high'] = np.percentile(boot_means, 97.5)
        result_rows.append(row)

    df_agg = pd.DataFrame(result_rows)
    df_agg.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

print(df_agg.head())

# ── 6. Plot: 3 subplots (F, E, G), one line per country ──────────────────────
metrics = [
    ('f', 'Foundation'),
    ('e', 'Extension'),
    ('g', 'Generalization'),
]

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
country_color = {c: colors[i % len(colors)] for i, c in enumerate(target_countries)}

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

mid = len(metrics) // 2

for i, (ax, (col, name)) in enumerate(zip(axes, metrics)):
    for country in target_countries:
        sub = df_agg[df_agg['country'] == country].sort_values('pub_year')
        if sub.empty:
            continue
        color = country_color[country]
        ax.fill_between(sub['pub_year'], sub[f'{col}_ci_low'], sub[f'{col}_ci_high'],
                        alpha=0.15, color=color)
        ax.plot(sub['pub_year'], sub[f'{col}_mean'],
                linewidth=0.8, color=color,
                label=country_full_name.get(country, country))

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.tick_params(axis='x', labelsize=tick_size, rotation=45)
    ax.tick_params(axis='y', labelsize=tick_size)
    if i == mid:
        ax.set_xlabel('Publication Year', fontsize=label_size)
        ax.legend(frameon=False, fontsize=legend_size, loc='lower right', ncol=1)
    # a single shared y-label on the leftmost panel; each panel names its index in the
    # bottom-left corner instead (same placement/size as author_count_feg_agg.py's plot_panel)
    if i == 0:
        ax.set_ylabel('Average index', fontsize=label_size)
    ax.text(0.03, 0.03, name, transform=ax.transAxes,
            fontsize=label_size + 7, ha='left', va='bottom')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()

if reference_threshold == 1 and citation_threshold == 5 and max_year == 2020:
    svg_path = os.path.join(SAVE_DIRE, 'Extended Data Fig. 10c.svg')
else:
    svg_path = os.path.join(SAVE_DIRE,
        f'country_feg_trend_{country_tag}_rt_{reference_threshold}_ct_{citation_threshold}_my_{max_year}.svg')
plt.savefig(svg_path, format='svg', bbox_inches='tight', transparent=True)
plt.close()
print(f"Saved: {svg_path}")

print("Done.")
