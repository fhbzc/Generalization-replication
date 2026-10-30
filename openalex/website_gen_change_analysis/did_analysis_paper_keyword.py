"""
did_analysis_paper_keyword.py

Difference-in-Differences analysis of venue first-online-release effect on the
PROBABILITY THAT A PAPER'S TITLE CONTAINS EACH OF A LIST OF KEYWORDS.

Step 1 (annotation).  For a user-specified `keyword_list` (see below), every
paper gets one BINARY variable per keyword: 1 if its title contains that
keyword, else 0.  The matching rule is exactly the one used by
text_analysis/select_pid_by_keyword.py:

  * PREFERRED — whole-token equality against the lemmatized title tokens in
    META_DIRE/pid2eng_token_list_lemma_before_2025.pkl: the keyword must equal
    one of the title's tokens as a string, so "review" hits the token "review"
    but NOT "reviews";
  * FALLBACK  — case-insensitive substring match on the raw title, used only if
    the token file is missing.

The keywords are annotated INDEPENDENTLY of one another: a title may contain any
number of them (zero, one, or several) and each gets its own 0/1.  A keyword that
occurs several times in the same title is still a single 1 — the outcome is the
probability that a title CONTAINS the keyword, not a count.

Per-keyword pid lists are cached under META_DIRE/keyword_pids/<safe_keyword>.pkl
in the SAME [keyword, [pids...]] format select_pid_by_keyword.py writes, so the
two scripts share one cache: whichever runs first pays the cost of scanning the
(large) token file, and a keyword already saved is never rescanned.  Inverting
those lists gives every paper's keyword annotation, so with all keywords cached
the token file is never opened.

The denominator is every paper of the venue-year that passes the usual
reference/citation filters: a pid in none of the keyword lists simply scores 0
for all of them.  Note this includes papers whose title the keyword scan could
not match on (e.g. non-English titles), which therefore enter as 0s.

Step 2 (DID panel).  For each venue x pub_year we average each binary variable
across all papers of that venue-year, giving the empirical
probability that a title of that venue-year contains the keyword.  Panel
construction (FSO treatment year, rigid control matching, +/-k window) is
identical to did_analysis_paper_title_distance.py.

Step 3 (estimation) is done in did_event_study.Rmd (dataset <- "keyword"),
which loops over every kw_*_rate column and reports one event study per keyword.

Output CSV columns
------------------
venue_issn, top_domain, is_control, treatment_issn, treatment_year,
year_index, pub_year, paper_count,
kw_<keyword>_rate ... (one per keyword),
avg_log_citation_y5, avg_log_up_refer_count
"""

import os
import re
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

# ── Keywords ─────────────────────────────────────────────────────────────────
# Fill these in; case does not matter (they are lower-cased before matching).
# Whole-token matching only makes sense for single-token keywords -- a
# multi-word phrase can match only under the substring fallback.
keyword_list = ['device', 'tool', 'software', 'review', 'guideline', 'tutorial',
                'new', 'novel', 'innovative', 'theory', 'metric', 'hypothesis',
                'discovery', 'dataset', 'observation', 'overview', 'method']

# normalize: lower-case + drop blanks/duplicates while preserving order
_seen = set()
keywords = []
for _kw in keyword_list:
    _kw = _kw.lower().strip()
    if _kw and _kw not in _seen:
        _seen.add(_kw)
        keywords.append(_kw)
print("keywords:", keywords)

# ── Hyperparameters ──────────────────────────────────────────────────────────
k                      = 7
min_papers_per_year    = 2
same_domain_control    = False
random_seed            = 42
reference_threshold    = 1
citation_threshold     = 1
min_year               = 1950
max_year               = 2025 - k
MIN_FSO_YEAR           = 1990

# must match parse_title_to_token.py (which wrote the before_2025 token file)
year_threshold         = 2025

print("k:", k)
print("min_papers_per_year:", min_papers_per_year)
print("same_domain_control:", same_domain_control)
print("random_seed:", random_seed)
print("reference_threshold:", reference_threshold)
print("citation_threshold:", citation_threshold)
print("min_year:", min_year)
print("max_year:", max_year)
print("MIN_FSO_YEAR:", MIN_FSO_YEAR)
random.seed(random_seed)

save_dire = os.path.join(META_DIRE, 'full_text_online_analysis')
os.makedirs(save_dire, exist_ok=True)
print("save_dire:", save_dire)

# ── 0. Annotate titles with the keywords (cached, shared with select_pid_by_keyword.py)
kw_dir = os.path.join(META_DIRE, 'keyword_pids')
os.makedirs(kw_dir, exist_ok=True)


def keyword_path(kw):
    # same mapping select_pid_by_keyword.py uses, so both scripts hit one cache
    safe = re.sub(r'[^0-9a-z]+', '_', kw).strip('_') or 'kw'
    return os.path.join(kw_dir, f'{safe}.pkl')


def keyword_column(kw):
    safe = re.sub(r'[^0-9a-z]+', '_', kw).strip('_') or 'kw'
    return f'kw_{safe}_rate'


token_path = os.path.join(META_DIRE, f'pid2eng_token_list_lemma_before_{year_threshold}.pkl')
use_tokens = os.path.exists(token_path)

todo = [kw for kw in keywords if not os.path.exists(keyword_path(kw))]
if todo:
    print(f"\nkeywords to annotate ({len(todo)}): {todo}")
else:
    print("\nall keyword pid lists already cached -- the title source is never opened")

keyword2pids = {kw: [] for kw in todo}

# The (large) token/title file is opened ONLY to annotate keywords that have no
# cached pid list yet.  With every keyword cached this branch is skipped entirely
# and the script reads nothing but the small per-keyword lists and the eng-flag dict.
if todo:
    if use_tokens:
        # ---- whole-token match: a keyword hits when it equals SOME token of the title ----
        # Independent per keyword: a title may hit any number of keywords (or none),
        # and repeated occurrences of the same keyword still count as a single 1.
        print("token_path (whole-token match):", token_path)
        with open(token_path, 'rb') as f:
            pid2token_list = pickle.load(f)
        print("pid2token_list load finished | size:", len(pid2token_list))

        for pid, token_list in tqdm(pid2token_list.items(), desc='scan tokens'):
            if not token_list:
                continue
            token_set = {t.lower() for t in token_list}
            for kw in todo:
                if kw in token_set:      # string equality: "review" hits "review", not "reviews"
                    keyword2pids[kw].append(pid)
        del pid2token_list
    else:
        # ---- fallback: case-insensitive SUBSTRING match on titles (HTML-stripped preferred) ----
        title_no_html_path = os.path.join(META_DIRE, 'pid2title_no_html_before_2025.pkl')
        title_raw_path     = os.path.join(META_DIRE, 'pid2title.pkl')
        title_path = title_no_html_path if os.path.exists(title_no_html_path) else title_raw_path
        print("token file not found -> substring title match. title_path:", title_path)
        assert os.path.exists(title_path), title_path

        with open(title_path, 'rb') as f:
            pid2title = pickle.load(f)
        print("pid2title load finished | size:", len(pid2title))

        for pid, title in tqdm(pid2title.items(), desc='scan titles'):
            if not title:
                continue
            title_l = title.lower()
            for kw in todo:
                if kw in title_l:
                    keyword2pids[kw].append(pid)
        del pid2title

    for kw in todo:
        with open(keyword_path(kw), 'wb') as f:
            pickle.dump([kw, keyword2pids[kw]], f, pickle.HIGHEST_PROTOCOL)
        print(f"  '{kw}': {len(keyword2pids[kw])} pids -> {keyword_path(kw)}")
    del keyword2pids

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

# ── 4. Load paper metadata ────────────────────────────────────────────────────
print("\nLoading paper metadata ...")
df_meta = pd.read_csv(
    os.path.join(META_DIRE, 'efg_computation', 'gen_V3_y5.csv'),
    usecols=['pid', 'pub_year', 'up_refer_count', 'citation_count_y5']
)
df_meta = df_meta[(df_meta['pub_year'] >= min_year) & (df_meta['pub_year'] < max_year)]

pid2pubyear_all = df_meta.set_index('pid')['pub_year'].to_dict()

df_valid = df_meta[
    (df_meta['up_refer_count']    >= reference_threshold) &
    (df_meta['citation_count_y5'] >= citation_threshold)
]
valid_pids  = set(df_valid['pid'])
pid2pubyear = {pid: pid2pubyear_all[pid] for pid in valid_pids}
pid2log_citation = np.log1p(df_valid.set_index('pid')['citation_count_y5']).to_dict()
pid2log_refer    = np.log1p(df_valid.set_index('pid')['up_refer_count']).to_dict()
print(f"  valid pids: {len(valid_pids)}")

# ── 5. Build pid2issn_fso; restrict the keyword annotation to those pids ──────
print("\nBuilding pid2issn_fso ...")
fso_issns_candidate = set(issn2earliest.index) & set(issn2pids.keys()) & set(issn2top_domain.keys())

pid2issn_fso = {
    p: issn
    for issn in fso_issns_candidate
    if int(issn2earliest[issn]) < max_year - k
    for p in issn2pids[issn]
    if p in valid_pids
}

print(f"  papers in FSO candidate venues: {len(pid2issn_fso)}")

# The denominator is simply every paper of the venue-year: a pid in none of the
# keyword lists is a 0, no title-side eligibility filter on top.
# per-keyword pid sets, restricted to those papers (keeps the big cached lists out of memory)
print("\nLoading per-keyword pid sets ...")
keyword2pid_set = {}
for kw in tqdm(keywords, desc='keyword pids'):
    with open(keyword_path(kw), 'rb') as f:
        saved_kw, pids = pickle.load(f)
    keyword2pid_set[kw] = {p for p in pids if p in pid2issn_fso}
    del pids
    print(f"  '{saved_kw}': {len(keyword2pid_set[kw])} papers "
          f"({100.0 * len(keyword2pid_set[kw]) / max(len(pid2issn_fso), 1):.3f}%)")

# ── 6. Venue eligibility, based on the paper count per year ──────────────────
_issn2pc = defaultdict(lambda: defaultdict(int))
for p in pid2issn_fso:
    _issn2pc[pid2issn_fso[p]][pid2pubyear[p]] += 1
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
print(f"\n  treatment-eligible issns: {len(treatment_issns)}")

# ── 7. Select treatment/control pairs ─────────────────────────────────────────
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

# ── 8. Aggregate per-issn per-year stats (all FSO candidate venues) ───────────
print("\nAggregating per-issn per-year stats ...")

kw_columns = [keyword_column(kw) for kw in keywords]
issn2stats = {}

for issn in tqdm(set(pid2issn_fso.values()), desc='issn stats'):
    year2data = defaultdict(lambda: {'count': 0,
                                     'log_cit_sum': 0.0, 'log_ref_sum': 0.0,
                                     'kw_hits': defaultdict(int)})
    for p in issn2pids[issn]:
        if p not in valid_pids:
            continue
        yr = pid2pubyear[p]
        d  = year2data[yr]
        d['count']       += 1
        d['log_cit_sum'] += pid2log_citation.get(p, 0.0)
        d['log_ref_sum'] += pid2log_refer.get(p, 0.0)
        for kw in keywords:
            if p in keyword2pid_set[kw]:
                d['kw_hits'][kw] += 1

    rows = []
    for yr, d in sorted(year2data.items()):
        row = {
            'pub_year':                  int(yr),
            'paper_count':               d['count'],
            'avg_log_citation_y5':       d['log_cit_sum'] / d['count'] if d['count'] > 0 else float('nan'),
            'avg_log_up_refer_count':    d['log_ref_sum'] / d['count'] if d['count'] > 0 else float('nan'),
        }
        for kw in keywords:
            # share of ALL papers of that venue-year whose title contains the keyword
            row[keyword_column(kw)] = (d['kw_hits'][kw] / d['count']) if d['count'] > 0 else float('nan')
        rows.append(row)
    if rows:
        issn2stats[issn] = pd.DataFrame(rows)

print(f"  issns with stats: {len(issn2stats)}")

# ── 9. Build DID panel from pre-selected pairs ────────────────────────────────
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
    'year_index', 'pub_year', 'paper_count',
] + kw_columns + [
    'avg_log_citation_y5', 'avg_log_up_refer_count',
]
df_did = df_did[col_order]

domain_tag = "samedomain" if same_domain_control else "anydomain"
out_path = os.path.join(
    save_dire,
    f'did_panel_paper_keyword_{domain_tag}_k{k}_mpy{min_papers_per_year}'
    f'_rt{reference_threshold}_ct{citation_threshold}_miny{min_year}_maxy{max_year}.csv'
)
df_did.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
print("keyword columns:", kw_columns)
print("Done.")
