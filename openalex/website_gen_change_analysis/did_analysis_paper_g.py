"""
did_analysis.py

Difference-in-Differences analysis of venue first-online-release effect on
paper-level F / E / G indices.

Treatment : venues whose earliest FSO online year is t  (and t >= MIN_FSO_YEAR).
Control   : venues whose earliest FSO online year is strictly later than t+k.

Output CSV columns
------------------
venue_issn, is_control, treatment_issn, treatment_year,
year_index, pub_year, paper_count,
avg_e, avg_f, avg_g, avg_citation_y5, avg_up_refer_count
"""

import os
import sys
import pickle
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)             # utils.py is in the parent directory
from utils import META_DIRE, FSO_DIRE
print("ridig control")
# ── Hyperparameters ──────────────────────────────────────────────────────────
k                      = 7       # exclusion-window half-width around treatment year
min_papers_per_year    = 2       # every single year in [t-k, t+k] must have >= this many papers
same_domain_control    = False    # True: control must be in same top_domain as treatment
random_seed            = 42
reference_threshold    = 1
citation_threshold     = 1
min_year               = 1950    # keep papers from this year onwards
max_year               = 2025 - k    # keep papers strictly before this year
MIN_FSO_YEAR           = 1990    # drop venues whose earliest online year < this

print("k:", k)
print("min_papers_per_year:", min_papers_per_year)
print("same_domain_control:", same_domain_control)
print("random_seed:", random_seed)
random.seed(random_seed)
print("reference_threshold:", reference_threshold)
print("citation_threshold:", citation_threshold)
print("min_year:", min_year)
print("max_year:", max_year)
print("MIN_FSO_YEAR:", MIN_FSO_YEAR)

save_dire = os.path.join(META_DIRE, 'full_text_online_analysis')
os.makedirs(save_dire, exist_ok=True)
print("save_dire:", save_dire)

# ── 1. Load FSO data → issn2earliest_year ────────────────────────────────────
print("\nLoading FSO data ...")
fso_path = os.path.join(FSO_DIRE, 'FSOe-2026.csv')
df_fso = pd.read_csv(fso_path, dtype=str, keep_default_na=False)

print("initial unique issn", len(set(df_fso['Issn'])))
# assert False
df_fso = df_fso[df_fso['StartYear'].str.strip() != '']
df_fso['StartYear'] = pd.to_numeric(df_fso['StartYear'], errors='coerce')
df_fso = df_fso.dropna(subset=['StartYear'])
df_fso['StartYear'] = df_fso['StartYear'].astype(int)
df_fso['Issn'] = df_fso['Issn'].str.strip()
df_fso = df_fso[df_fso['Issn'] != '']

# Delete records with StartYear <= MIN_FSO_YEAR
df_fso = df_fso[df_fso['StartYear'] > MIN_FSO_YEAR]

# Earliest start year per ISSN from the filtered records
issn2earliest = df_fso.groupby('Issn')['StartYear'].min()
print(f"  venues with earliest year > {MIN_FSO_YEAR}: {len(issn2earliest)}")

# ── 2. Load issn2top_domain ──────────────────────────────────────────────────
print("\nLoading issn2top_domain ...")
with open(os.path.join(META_DIRE, 'issn2top_domain.pkl'), 'rb') as f:
    issn2top_domain = pickle.load(f)
print(f"  issn2top_domain size: {len(issn2top_domain)}")

# ── 3. Load pid → primary-location ISSN ──────────────────────────────────────
print("\nLoading pid2primary_location_issn ...")
with open(os.path.join(META_DIRE, 'pid2primary_location_issn.pkl'), 'rb') as f:
    pid2primary_issn = pickle.load(f)
print(f"  pid2primary_issn size: {len(pid2primary_issn)}")

# Invert: issn_l → set of pids
issn2pids = defaultdict(set)
for pid, issn in tqdm(pid2primary_issn.items(), desc='building issn2pids'):
    issn2pids[issn].add(pid)
print(f"  unique ISSNs with pids: {len(issn2pids)}")

# ── 3. Load generalization data ───────────────────────────────────────────────
print("\nLoading generalization data ...")
df_gen = pd.read_csv(
    os.path.join(META_DIRE, 'efg_computation', 'gen_V3_y5.csv'),
    usecols=['pid', 'pub_year', 'up_refer_count', 'citation_count_y5',
             'f_y5', 'e_y5', 'g_y5']
)
print(f"  generalization rows: {len(df_gen)}")

df_gen = df_gen[df_gen['pub_year']          >= min_year]
df_gen = df_gen[df_gen['pub_year']          <  max_year]
df_gen = df_gen[df_gen['up_refer_count']    >= reference_threshold]
df_gen = df_gen[df_gen['citation_count_y5'] >= citation_threshold]
print(f"  after threshold filter: {len(df_gen)}")

df_gen['f'] = df_gen['f_y5']
df_gen['e'] = df_gen['e_y5']
df_gen['g'] = df_gen['g_y5']
df_gen['log_citation_count_y5'] = np.log1p(df_gen['citation_count_y5'])
df_gen['log_up_refer_count']    = np.log1p(df_gen['up_refer_count'])
df_gen = df_gen[['pid', 'pub_year', 'log_citation_count_y5', 'log_up_refer_count', 'f', 'e', 'g']]

pid2row  = df_gen.set_index('pid')
valid_pids = set(df_gen['pid'])
print(f"  valid pids in df_gen: {len(valid_pids)}")

# ── 4. Pre-compute per-issn per-year aggregate stats ─────────────────────────
print("\nPre-computing per-issn per-year stats ...")

issn2stats = {}   # issn -> DataFrame(pub_year, paper_count, avg_e, avg_f, avg_g, ...)

for issn, pids in tqdm(issn2pids.items(), desc='issn stats'):
    valid = [p for p in pids if p in valid_pids]
    if not valid:
        continue
    sub = pid2row.loc[valid].reset_index()
    rows = []
    for yr, grp in sub.groupby('pub_year'):
        rows.append({
            'pub_year':               int(yr),
            'paper_count':            len(grp),
            'avg_e':                  grp['e'].mean(),
            'avg_f':                  grp['f'].mean(),
            'avg_g':                  grp['g'].mean(),
            'avg_log_citation_y5':    grp['log_citation_count_y5'].mean(),
            'avg_log_up_refer_count': grp['log_up_refer_count'].mean(),
        })
    if rows:
        issn2stats[issn] = pd.DataFrame(rows)

print(f"  issns with stats: {len(issn2stats)}")

# Both treatment and control must be in FSO and have a known top domain
fso_issns_with_stats = set(issn2earliest.index) & set(issn2stats.keys()) & set(issn2top_domain.keys())
print(f"  FSO issns with stats and top domain: {len(fso_issns_with_stats)}")


def has_enough_papers_each_year(issn, t_year, k, min_per_year):
    """Return True if venue has >= min_per_year papers in EVERY year in [t-k, t+k]."""
    year_counts = issn2stats[issn].set_index('pub_year')['paper_count'].to_dict()
    return all(year_counts.get(yr, 0) >= min_per_year for yr in range(t_year - k, t_year + k + 1))


# Treatment-eligible: in FSO, treatment year < max_year - k, passes per-year paper-count check
treatment_issns = sorted([
    issn for issn in fso_issns_with_stats
    if int(issn2earliest[issn]) < max_year - k
    and has_enough_papers_each_year(issn, int(issn2earliest[issn]), k, min_papers_per_year)
])
print(f"  treatment-eligible issns: {len(treatment_issns)}")

# ── 5. Build DID panel ────────────────────────────────────────────────────────
print(f"\nBuilding DID panel (k={k}, min_papers_per_year={min_papers_per_year}) ...")

all_rows = []
no_control_count = 0

for t_issn in tqdm(treatment_issns, desc='treatment venues'):
    t_year     = int(issn2earliest[t_issn])
    window_lo  = t_year - k
    window_hi  = t_year + k

    # Treatment venue rows (only year_index in [-k, k])
    df_t = issn2stats[t_issn].copy()
    df_t['year_index'] = df_t['pub_year'] - t_year
    df_t = df_t[(df_t['year_index'] >= -k) & (df_t['year_index'] <= k)]
    t_domain = issn2top_domain[t_issn]

    if not df_t.empty:
        df_t.insert(0, 'venue_issn', t_issn)
        df_t['is_control']     = 0
        df_t['treatment_issn'] = t_issn
        df_t['treatment_year'] = t_year
        df_t['top_domain']     = t_domain
        all_rows.append(df_t)

    # Collect all eligible control candidates
    control_candidates = []
    for c_issn in treatment_issns:
        if c_issn == t_issn:
            continue
        if same_domain_control and issn2top_domain[c_issn] != t_domain:
            continue   # different domain — skip
        c_year = int(issn2earliest[c_issn])
        if c_year <= window_hi:
            continue   # earliest online not strictly after t+k — skip
        # Control must also have >= min_papers_per_year in every year of [t-k, t+k]
        if not has_enough_papers_each_year(c_issn, t_year, k, min_papers_per_year):
            continue
        control_candidates.append(c_issn)

    if not control_candidates:
        no_control_count += 1
        continue

    # Randomly pick one control venue
    c_issn = random.choice(sorted(control_candidates))
    df_c = issn2stats[c_issn].copy()
    df_c['year_index'] = df_c['pub_year'] - t_year
    df_c = df_c[(df_c['year_index'] >= -k) & (df_c['year_index'] <= k)]
    if not df_c.empty:
        df_c.insert(0, 'venue_issn', c_issn)
        df_c['is_control']     = 1
        df_c['treatment_issn'] = t_issn
        df_c['treatment_year'] = t_year
        df_c['top_domain']     = issn2top_domain[c_issn]
        all_rows.append(df_c)

print(f"  treatment venues with no eligible control: {no_control_count} / {len(treatment_issns)}")
print("Concatenating ...")
df_did = pd.concat(all_rows, ignore_index=True)
print(f"  total rows: {len(df_did)}")

col_order = [
    'venue_issn', 'top_domain', 'is_control', 'treatment_issn', 'treatment_year',
    'year_index', 'pub_year', 'paper_count',
    'avg_e', 'avg_f', 'avg_g', 'avg_log_citation_y5', 'avg_log_up_refer_count',
]
df_did = df_did[col_order]

domain_tag = "samedomain" if same_domain_control else "anydomain"
out_path = os.path.join(
    save_dire,
    f'did_panel_{domain_tag}_k{k}_mpy{min_papers_per_year}'
    f'_rt{reference_threshold}_ct{citation_threshold}_miny{min_year}_maxy{max_year}.csv'
)
df_did.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
print("Done.")
