"""
For every Nobel-prize-winning paper (as loaded by award_analysis/
nobel_logistic_regression_sep.py) that also appears in the per-paper quantile
table read by compute_efgd_quantile_s2_extract.py, write out a single CSV row
holding that paper's pub_year, its E/F/G/D values and (year, domain)-relative
quantiles, its top domain, and which Nobel domain file it came from.

Pipeline
--------
1. Load the SAME full-cover quantile table that compute_efgd_quantile_s2_extract.py
   reads (the '_full_cover' / all-papers-before-2025 variant from
   compute_efgd_quantile.py). This table already holds, for every paper that
   passed the reference / citation / max_year filters and has exactly one top
   domain, its (pub_year, top_domain)-relative quantile for each of
   E (e_y{off}), F (f_y{off}), G (g_y{off}) and D (disp_y{off}).

2. Reproduce nobel_logistic_regression_sep.py's Nobel-paper selection:
   for each {Medicine, Chemistry, Physics} publication-record CSV, take the
   'Is prize-winning paper' == 'YES' rows, collect their DOIs, and map each DOI
   to a pid via pid2doi.pkl (unique-DOI mapping only). Both sides of that join
   are lower-cased first (norm_doi), exactly as the regression does it, so this
   yields the same nobel_pids, tagged with their Nobel domain.

3. Keep the Nobel papers that appear in the quantile table and write one CSV
   with, per paper: pid, doi, title, nobel_domain (which prize file), domain
   (its OpenAlex top-concept display name), pub_year, group_size,
   up_refer_count, citation_count_y{off}, the four index values + quantiles, and
   an `in_extract_selection` flag (does it pass the extract script's
   pub_year >= MIN_YEAR and citation >= EXTRACT_CITATION_THRESHOLD cuts).

Input  : META_DIRE/efg_computation/efgd_within_year_domain_quantile_y{year_offset}_rt_{rt}_ct_{ct}_ybefore_2025_full_cover.csv
                                                                (from compute_efgd_quantile.py)
         META_DIRE/pid2doi.pkl                                  (from get_meta_data_preprint.py)
         META_DIRE/pid2title.pkl                                (from get_meta_data_preprint.py)
         CONCEPT_DIRE (Concepts_definition.xlsx)                (concept id -> display_name)
         NOBEL_DATA_DIRE/{Medicine,Chemistry,Physics} publication record.csv
Output : META_DIRE/efgd_quantile_top_extract/efgd_nobel_qct_{citation_threshold}_extract.csv
"""

import os
import sys
import pickle
from collections import defaultdict

import pandas as pd

# utils.py is in this same directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from utils import META_DIRE, CONCEPT_DIRE, NOBEL_DATA_DIRE  # noqa: E402

meta_dire = META_DIRE
print("meta_dire", meta_dire)

efg_dire = os.path.join(meta_dire, 'efg_computation')

# ---- must match compute_efgd_quantile.py so the input filename resolves ----
year_offset = 5
root_level = 0
reference_threshold = 1
citation_threshold = 100
# load the full-cover quantile table: ALL papers published before 2025 (the
# '_full_cover' variant from compute_efgd_quantile.py, ybefore_2025).
max_year = 2025
input_name_tag = '_full_cover'
print("year_offset", year_offset, "| reference_threshold", reference_threshold,
      "| citation_threshold", citation_threshold, "| max_year", max_year,
      "| input_name_tag", input_name_tag)

# ---- the extract script's display cuts (only used to set the informational flag) ----
MIN_YEAR = 1960
EXTRACT_CITATION_THRESHOLD = 100
print("MIN_YEAR", MIN_YEAR, "| EXTRACT_CITATION_THRESHOLD", EXTRACT_CITATION_THRESHOLD)

# ---- Nobel-prize publication-record files (must match nobel_logistic_regression_sep.py) ----
NOBEL_FILES = [
    {'file': 'Medicine publication record.csv',  'domain': 'Medicine'},
    {'file': 'Chemistry publication record.csv', 'domain': 'Chemistry'},
    {'file': 'Physics publication record.csv',   'domain': 'Physics'},
]


# ---- DOI normalisation ------------------------------------------------------------------
# DOIs are case-INSENSITIVE by spec, but the two sides of the join disagree in practice: the
# Nobel CSVs carry mixed-case DOIs (e.g. 10.1103/PhysRevLett.50.1153) while OpenAlex stores
# them lower-cased (get_meta_data_preprint_doi.py keeps the string as-is after stripping the
# 'https://doi.org/' prefix). A raw string comparison therefore drops those papers silently.
# Both sides now go through norm_doi() before they meet. Identical to the copy in
# award_analysis/nobel_logistic_regression_sep.py, whose selection this script reproduces.
# NOTE: lower-casing also MERGES pid2doi keys that differed only in case. A merged key holding
# several pids is then dropped by the one-to-one rule, so the net gain is smaller than the
# number of newly-matchable DOIs -- run award_analysis/temp.py to see both numbers.
def norm_doi(doi):
    """Canonical form for DOI comparison: stripped and lower-cased."""
    return str(doi).strip().lower()


# the four index columns + their quantile columns (must match compute_efgd_quantile.py)
index_names = ['e', 'f', 'g', 'disp']
index_cols = [f'e_y{year_offset}', f'f_y{year_offset}', f'g_y{year_offset}', f'disp_y{year_offset}']
quantile_cols = [f'{c}_quantile' for c in index_cols]
citation_col = f'citation_count_y{year_offset}'

# ==================================================
# 1. load the per-paper quantile table (compute_efgd_quantile.py output)
# ==================================================
src_path = os.path.join(
    efg_dire,
    f'efgd_within_year_domain_quantile_y{year_offset}_'
    f'rt_{reference_threshold}_ct_{citation_threshold}_ybefore_{max_year}{input_name_tag}.csv'
)
assert os.path.exists(src_path), (
    f"{src_path}\nRun compute_efgd_quantile.py first.")
df = pd.read_csv(src_path)
print("loaded quantile table shape:", df.shape)

# ==================================================
# 2. lookups: pid -> doi, pid -> title, root concept id -> display name
# ==================================================
print("\nLoading pid2doi ...")
with open(os.path.join(meta_dire, 'pid2doi.pkl'), 'rb') as f:
    pid2doi = pickle.load(f)
print(f"  pid2doi size: {len(pid2doi)}")

# keys are norm_doi()-canonicalised, so the Nobel-CSV side must be canonicalised too
doi2pid_list = defaultdict(list)
for pid, doi in pid2doi.items():
    doi2pid_list[norm_doi(doi)].append(pid)
print(f"  distinct lower-cased DOIs: {len(doi2pid_list)}")

with open(os.path.join(meta_dire, 'pid2title.pkl'), 'rb') as f:
    pid2title = pickle.load(f)
print("pid2title size:", len(pid2title))

URL_PREFIX_LEN = len("https://openalex.org/")
df_concept = pd.read_excel(CONCEPT_DIRE)
# key by lower-cased, prefix-stripped id so it matches the ids in top_concept
cid2name = {str(oid)[URL_PREFIX_LEN:].lower(): name
            for oid, name in zip(df_concept['openalex_id'], df_concept['display_name'])}

def concept_display(cid):
    return cid2name.get(str(cid).lower(), str(cid))

# ==================================================
# 3. reproduce nobel_logistic_regression_sep.py's Nobel-paper selection
#    (prize-winning DOIs -> unique pid), tagged with the Nobel domain
# ==================================================
pid2nobel_domain = {}   # pid -> Nobel domain (first file that claims it wins)
pid2nobel_doi    = {}    # pid -> the winning DOI it was matched through
for cfg in NOBEL_FILES:
    domain = cfg['domain']
    fpath = os.path.join(NOBEL_DATA_DIRE, cfg['file'])
    df_n = pd.read_csv(fpath, usecols=['Is prize-winning paper', 'DOI'], encoding='latin1')
    prize_rows = df_n[df_n['Is prize-winning paper'] == 'YES']
    # match on the canonical form, but remember the CSV's own spelling so the output 'doi'
    # column still shows the DOI as the prize record writes it
    doi_raw = {}
    for v in prize_rows['DOI'].dropna():
        doi_raw.setdefault(norm_doi(v), v)
    doi_set = set(doi_raw)
    print(f"\nDomain: {domain}  |  file: {cfg['file']}  |  Nobel DOIs: {len(doi_set)}")

    n_mapped = 0
    for doi in doi_set:
        if doi in doi2pid_list and len(doi2pid_list[doi]) == 1:
            pid = doi2pid_list[doi][0]
            n_mapped += 1
            # keep the first domain that claims a pid (stable across files)
            if pid not in pid2nobel_domain:
                pid2nobel_domain[pid] = domain
                pid2nobel_doi[pid] = doi_raw[doi]
    print(f"  Nobel PIDs (unique DOI mapping): {n_mapped}")

nobel_pids = set(pid2nobel_domain)
print(f"\nTotal unique Nobel PIDs across all files: {len(nobel_pids)}")

# ==================================================
# 4. keep the Nobel papers that appear in the quantile table
# ==================================================
df_nobel = df[df['pid'].isin(nobel_pids)].copy()
n_in_table = df_nobel['pid'].nunique()
print(f"Nobel PIDs present in the quantile table: {n_in_table} "
      f"(missing: {len(nobel_pids) - n_in_table})")

# annotate
df_nobel['doi']          = df_nobel['pid'].map(pid2nobel_doi)
df_nobel['title']        = df_nobel['pid'].map(lambda p: pid2title.get(p, ''))
df_nobel['nobel_domain'] = df_nobel['pid'].map(pid2nobel_domain)
df_nobel['domain']       = df_nobel['top_concept'].map(concept_display)
# does the paper survive the extract script's display cuts?
df_nobel['in_extract_selection'] = (
    (df_nobel['pub_year'] >= MIN_YEAR) &
    (df_nobel[citation_col] >= EXTRACT_CITATION_THRESHOLD)
)
print(f"  of those, in extract selection (pub_year >= {MIN_YEAR} & "
      f"{citation_col} >= {EXTRACT_CITATION_THRESHOLD}): "
      f"{int(df_nobel['in_extract_selection'].sum())}")

# ==================================================
# 5. write the output
#    - CSV : machine-readable (build_citation_depth_tree.py reads this via pd.read_csv)
#    - XLSX: same table, with the header row FROZEN (freeze_panes=(1, 0)) for viewing
# ==================================================
out_cols = (['pid', 'doi', 'title', 'nobel_domain', 'domain', 'top_concept',
             'pub_year', 'group_size', 'up_refer_count', citation_col]
            + index_cols + quantile_cols
            + ['in_extract_selection'])

# sort by Nobel domain then pub_year for a readable file
df_nobel = df_nobel.sort_values(['nobel_domain', 'pub_year', 'pid'])

out_dir = os.path.join(meta_dire, 'efgd_quantile_top_extract')
os.makedirs(out_dir, exist_ok=True)

csv_path = os.path.join(
    out_dir, f'efgd_nobel_qct_{citation_threshold}_extract.csv')
df_nobel[out_cols].to_csv(csv_path, index=False)
print(f"\nwrote {csv_path}  ({len(df_nobel)} rows)")

xlsx_path = os.path.join(
    out_dir, f'efgd_nobel_qct_{citation_threshold}_extract.xlsx')
# freeze_panes=(1, 0): freeze the first (header) row so it stays visible when scrolling
df_nobel[out_cols].to_excel(xlsx_path, index=False, freeze_panes=(1, 0),
                            engine='openpyxl')
print(f"wrote {xlsx_path}  (header row frozen)")

print("output dir:", out_dir)
print("finished")
