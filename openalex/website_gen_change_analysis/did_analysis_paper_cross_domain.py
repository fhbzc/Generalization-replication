"""
did_analysis_paper_cross_domain.py

Difference-in-Differences analysis of venue first-online-release effect on
the paper-level cross-domain citation ratio.

For each paper p (with exactly one root-level concept), cross_domain_ratio_y5
is the fraction of incoming citations within 5 years after pub_year that come
from a citing paper also having exactly one root concept, where that root
concept differs from p's.

For each venue × pub_year we average cross_domain_ratio_y5 across all eligible
venue papers.  DID panel construction is identical to did_analysis.py.

Output CSV columns
------------------
venue_issn, top_domain, is_control, treatment_issn, treatment_year,
year_index, pub_year, paper_count, avg_cross_domain_ratio_y5
"""

import os
import pickle
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)             # utils.py is in the parent directory
from utils import META_DIRE, FSO_DIRE

print("rigid control")

# ── Hyperparameters ──────────────────────────────────────────────────────────
k                      = 3
min_papers_per_year    = 2
same_domain_control    = False
random_seed            = 42
reference_threshold    = 1
citation_threshold     = 1
min_year               = 1950
max_year               = 2025 - k
MIN_FSO_YEAR           = 1990
year_offset            = 5

print("k:", k)
print("min_papers_per_year:", min_papers_per_year)
print("same_domain_control:", same_domain_control)
print("random_seed:", random_seed)
print("reference_threshold:", reference_threshold)
print("citation_threshold:", citation_threshold)
print("year_offset:", year_offset)
print("min_year:", min_year)
print("max_year:", max_year)
print("MIN_FSO_YEAR:", MIN_FSO_YEAR)
random.seed(random_seed)

save_dire = os.path.join(META_DIRE, 'full_text_online_analysis')
os.makedirs(save_dire, exist_ok=True)
print("save_dire:", save_dire)

# ── 1. Load FSO data ──────────────────────────────────────────────────────────
print("\nLoading FSO data ...")
fso_path = os.path.join(FSO_DIRE, 'FSOe-2026.csv')
df_fso = pd.read_csv(fso_path, dtype=str, keep_default_na=False)
df_fso = df_fso[df_fso['StartYear'].str.strip() != '']
df_fso['StartYear'] = pd.to_numeric(df_fso['StartYear'], errors='coerce')
df_fso = df_fso.dropna(subset=['StartYear'])
df_fso['StartYear'] = df_fso['StartYear'].astype(int)
df_fso['Issn'] = df_fso['Issn'].str.strip()
df_fso = df_fso[df_fso['Issn'] != '']
df_fso = df_fso[df_fso['StartYear'] > MIN_FSO_YEAR]
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
issn2pids = defaultdict(set)
for pid, issn in tqdm(pid2primary_issn.items(), desc='building issn2pids'):
    issn2pids[issn].add(pid)
print(f"  unique ISSNs with pids: {len(issn2pids)}")

# ── 4. Load paper metadata and apply threshold filters ────────────────────────
print("\nLoading paper metadata ...")
df_meta = pd.read_csv(
    os.path.join(META_DIRE, 'efg_computation', 'gen_V3_y5.csv'),
    usecols=['pid', 'pub_year', 'up_refer_count', 'citation_count_y5']
)
df_meta = df_meta[(df_meta['pub_year'] >= min_year) & (df_meta['pub_year'] < max_year)]
df_meta = df_meta[
    (df_meta['up_refer_count']    >= reference_threshold) &
    (df_meta['citation_count_y5'] >= citation_threshold)
]
valid_pids  = set(df_meta['pid'])
pid2pubyear = df_meta.set_index('pid')['pub_year'].to_dict()
pid2log_citation = np.log1p(df_meta.set_index('pid')['citation_count_y5']).to_dict()
pid2log_refer    = np.log1p(df_meta.set_index('pid')['up_refer_count']).to_dict()
print(f"  valid pids: {len(valid_pids)}")

# ── 5. Load root concept dict; keep only papers with exactly one root concept ─
print("\nLoading pid2root_concepts_level_0 ...")
with open(os.path.join(META_DIRE, 'pid2root_concepts_level_0.pkl'), 'rb') as f:
    pid2root_concept_list_raw = pickle.load(f)

# pid2root_concept: pid -> single root concept string (only for papers with exactly 1)
pid2root_concept = {}
for pid, concepts in pid2root_concept_list_raw.items():
    if concepts is not None and len(concepts) == 1:
        pid2root_concept[pid] = next(iter(concepts))
del pid2root_concept_list_raw
print(f"  pids with exactly one root concept: {len(pid2root_concept)}")

# ── 6. Load citation dicts ────────────────────────────────────────────────────
print("\nLoading cited_pid2citing_pids_list ...")
with open(os.path.join(META_DIRE, 'cited_pid2citing_pids_list.pkl'), 'rb') as f:
    cited_pid2citing_pids = pickle.load(f)
print(f"  cited_pid2citing_pids size: {len(cited_pid2citing_pids)}")

# ── 7. Compute cross-domain ratios for ALL FSO candidate venues ───────────────
print("\nComputing cross-domain ratios for all FSO candidate venues ...")
fso_issns_candidate = set(issn2earliest.index) & set(issn2pids.keys()) & set(issn2top_domain.keys())

pid2issn_fso = {
    p: issn
    for issn in fso_issns_candidate
    if int(issn2earliest[issn]) < max_year - k
    for p in issn2pids[issn]
    if p in valid_pids
}

pid2cross_domain_ratio = {}

for pid in tqdm(pid2issn_fso, desc='per-paper cross-domain'):
    p_root = pid2root_concept.get(pid)
    if p_root is None:
        continue
    if pid not in cited_pid2citing_pids:
        continue

    p_year = pid2pubyear[pid]
    total  = 0
    cross  = 0

    for c in cited_pid2citing_pids[pid]:
        c_year = pid2pubyear.get(c)
        if c_year is None:
            continue
        if not (p_year < c_year <= p_year + year_offset):
            continue
        c_root = pid2root_concept.get(c)
        if c_root is None:
            continue
        total += 1
        if c_root != p_root:
            cross += 1

    if total > 0:
        pid2cross_domain_ratio[pid] = cross / total

print(f"  pids with cross_domain_ratio: {len(pid2cross_domain_ratio)}")

# Count valid-ratio pids per issn per year → issn2papercounts (for eligibility)
_issn2pc = defaultdict(lambda: defaultdict(int))
for p in pid2cross_domain_ratio:
    issn = pid2issn_fso[p]
    yr   = pid2pubyear[p]
    _issn2pc[issn][yr] += 1
issn2papercounts = {issn: dict(counts) for issn, counts in _issn2pc.items()}

def has_enough_papers_each_year(issn, t_year, k, min_per_year):
    counts = issn2papercounts.get(issn, {})
    return all(counts.get(yr, 0) >= min_per_year for yr in range(t_year - k, t_year + k + 1))

treatment_issns = sorted([
    issn for issn in fso_issns_candidate
    if issn in issn2papercounts
    and int(issn2earliest[issn]) < max_year - k
    and has_enough_papers_each_year(issn, int(issn2earliest[issn]), k, min_papers_per_year)
])
print(f"  treatment-eligible issns: {len(treatment_issns)}")

# ── 8. Select treatment/control pairs; derive relevant venue-years ────────────
print("\nSelecting treatment/control pairs ...")
panel_pairs      = []
no_control_count = 0

for t_issn in treatment_issns:
    t_year    = int(issn2earliest[t_issn])
    window_hi = t_year + k
    t_domain  = issn2top_domain[t_issn]

    control_candidates = [
        c_issn for c_issn in treatment_issns
        if c_issn != t_issn
        and (not same_domain_control or issn2top_domain[c_issn] == t_domain)
        and int(issn2earliest[c_issn]) > window_hi
        and has_enough_papers_each_year(c_issn, t_year, k, min_papers_per_year)
    ]
    if not control_candidates:
        no_control_count += 1
        continue
    c_issn = random.choice(sorted(control_candidates))
    panel_pairs.append((t_issn, t_year, c_issn))

print(f"  treatment venues with no eligible control: {no_control_count} / {len(treatment_issns)}")
print(f"  panel pairs: {len(panel_pairs)}")

# ── 9. (cross-domain ratios already computed in section 7) ───────────────────
print(f"  pids with cross_domain_ratio computed: {len(pid2cross_domain_ratio)}")

# ── 10. Aggregate per-issn per-year stats (all FSO candidate venues) ──────────
print("\nAggregating per-issn per-year stats ...")

issn2stats = {}

for issn in tqdm(set(pid2issn_fso.values()), desc='issn stats'):
    year2data = defaultdict(lambda: {'count': 0, 'sum': 0.0, 'n': 0,
                                     'log_cit_sum': 0.0, 'log_ref_sum': 0.0})
    for p in issn2pids[issn]:
        if p not in valid_pids:
            continue
        yr = pid2pubyear[p]
        year2data[yr]['count']       += 1
        year2data[yr]['log_cit_sum'] += pid2log_citation.get(p, 0.0)
        year2data[yr]['log_ref_sum'] += pid2log_refer.get(p, 0.0)
        if p in pid2cross_domain_ratio:
            year2data[yr]['sum'] += pid2cross_domain_ratio[p]
            year2data[yr]['n']   += 1

    rows = []
    for yr, d in sorted(year2data.items()):
        rows.append({
            'pub_year':                  int(yr),
            'paper_count':               d['count'],
            'avg_cross_domain_ratio_y5': d['sum'] / d['n'] if d['n'] > 0 else float('nan'),
            'avg_log_citation_y5':       d['log_cit_sum'] / d['count'] if d['count'] > 0 else float('nan'),
            'avg_log_up_refer_count':    d['log_ref_sum']  / d['count'] if d['count'] > 0 else float('nan'),
        })
    if rows:
        issn2stats[issn] = pd.DataFrame(rows)

print(f"  issns with stats: {len(issn2stats)}")

# ── 11. Build DID panel from pre-selected pairs ───────────────────────────────
print(f"\nBuilding DID panel (k={k}) ...")

all_rows = []

for t_issn, t_year, c_issn in tqdm(panel_pairs, desc='building panel'):
    t_domain = issn2top_domain[t_issn]

    if t_issn in issn2stats:
        df_t = issn2stats[t_issn].copy()
        df_t['year_index'] = df_t['pub_year'] - t_year
        df_t = df_t[(df_t['year_index'] >= -k) & (df_t['year_index'] <= k)]
        if not df_t.empty:
            df_t.insert(0, 'venue_issn', t_issn)
            df_t['is_control']     = 0
            df_t['treatment_issn'] = t_issn
            df_t['treatment_year'] = t_year
            df_t['top_domain']     = t_domain
            all_rows.append(df_t)

    if c_issn in issn2stats:
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

print("Concatenating ...")
df_did = pd.concat(all_rows, ignore_index=True)
print(f"  total rows: {len(df_did)}")

col_order = [
    'venue_issn', 'top_domain', 'is_control', 'treatment_issn', 'treatment_year',
    'year_index', 'pub_year', 'paper_count', 'avg_cross_domain_ratio_y5',
    'avg_log_citation_y5', 'avg_log_up_refer_count',
]
df_did = df_did[col_order]

domain_tag = "samedomain" if same_domain_control else "anydomain"
out_path = os.path.join(
    save_dire,
    f'did_panel_paper_cross_domain_{domain_tag}_k{k}_mpy{min_papers_per_year}'
    f'_rt{reference_threshold}_ct{citation_threshold}_miny{min_year}_maxy{max_year}.csv'
)
df_did.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
print("Done.")
