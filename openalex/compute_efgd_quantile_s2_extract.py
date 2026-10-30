"""
Extract, for each year (>= MIN_YEAR) and each top domain, ALL papers in the
(year, domain) cell and write them to per-year Excel workbooks (one sheet per
domain).

Reads the per-paper quantile table from compute_efgd_quantile.py, which already
holds, for every paper that passed the reference / citation / max_year filters and
has exactly one top domain, its (pub_year, top_domain)-relative quantile for each
of E (e_y{off}), F (f_y{off}), G (g_y{off}) and D (disp_y{off}).

Selection rule, applied within each (pub_year, top_domain) cell:
    keep EVERY paper in the cell (no per-quantile top-N restriction).
The only filters applied are pub_year >= MIN_YEAR and the highly-cited cutoff
(citation_count_y{off} >= EXTRACT_CITATION_THRESHOLD).

Each kept paper is annotated with:
    - its title           (pid2title.pkl)
    - its domain          (top concept's display_name from CONCEPT_DIRE)
    - top_e / top_f / top_g / top_disp flags (which criteria put it in)
    - the three disruptive-foundation (disp_f) shares disruption_f_y{off} /
      disruption_f_down_gt1_y{off} / disruption_f_down_gt2_y{off} and their
      (year, domain)-relative quantiles, when the input quantile table was built with
      use_disp_f=True (display-only; they do not affect selection or the sort order)
    - the disruptive-citation fraction i/(i+j) = ij_ratio_y{off} and its
      (year, domain)-relative quantile (display-only, same as disp_f)

Output: one .xlsx per year, one sheet per domain (sheet name = domain display name).
The filename carries both the quantile-computation citation cutoff (qct =
citation_threshold) and the display/extraction citation cutoff (dct =
EXTRACT_CITATION_THRESHOLD).

Input  : META_DIRE/efg_computation/efgd_within_year_domain_quantile_y{year_offset}_rt_{rt}_ct_{ct}_ybefore_2025_full_cover.csv
                                                                (the '_full_cover' / all-papers-before-2025
                                                                 variant from compute_efgd_quantile.py)
         META_DIRE/pid2title.pkl                                (from get_meta_data_preprint.py)
         CONCEPT_DIRE (Concepts_definition.xlsx)                (concept id -> display_name)
Output : META_DIRE/efgd_quantile_top_extract/efgd_topall_qct_{citation_threshold}_dct_{EXTRACT_CITATION_THRESHOLD}_extract_y{year}.xlsx   (its own folder)
"""

import os
import re
import sys
import pickle

import pandas as pd

# utils.py is in this same directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from utils import META_DIRE, CONCEPT_DIRE  # noqa: E402

meta_dire = META_DIRE
print("meta_dire", meta_dire)

efg_dire = os.path.join(meta_dire, 'efg_computation')

# ---- must match compute_efgd_quantile.py so the input filename resolves ----
year_offset = -1
root_level = 0
reference_threshold = 1
citation_threshold = 100
# load the full-cover quantile table: ALL papers published before 2025 (the
# '_full_cover' variant from compute_efgd_quantile.py, ybefore_2025), which also
# includes recent papers without a complete year_offset citation window.
max_year = 2025
input_name_tag = '_full_cover'
print("year_offset", year_offset, "| reference_threshold", reference_threshold,
      "| citation_threshold", citation_threshold, "| max_year", max_year,
      "| input_name_tag", input_name_tag)

# ---- extraction settings ----
MIN_YEAR = 1960   # only extract publication years >= this (the input is already < max_year)
# keep ALL papers in each (year, domain) cell (no per-quantile top-N restriction)
# additionally restrict to highly-cited papers (citation_count_y{off} >= this); the
# threshold is written into every output filename so different cutoffs don't collide.
EXTRACT_CITATION_THRESHOLD = 100
print("MIN_YEAR", MIN_YEAR, "| TOP_N all",
      "| EXTRACT_CITATION_THRESHOLD", EXTRACT_CITATION_THRESHOLD)

# the four index columns + their quantile columns (must match compute_efgd_quantile.py)
index_names = ['e', 'f', 'g', 'disp']
index_cols = [f'e_y{year_offset}', f'f_y{year_offset}', f'g_y{year_offset}', f'disp_y{year_offset}']
quantile_cols = [f'{c}_quantile' for c in index_cols]
citation_col = f'citation_count_y{year_offset}'

# the three "disruptive foundation" (disp_f) shares + their quantiles, present in the
# quantile table when compute_efgd_quantile.py was run with use_disp_f=True (down_count > 0/1/2).
# These are display-only extra columns; they do NOT enter the _sort_key or the top_* flags.
dispf_index_names = ['disp_f_0', 'disp_f_1', 'disp_f_2']
dispf_cols = [f'disruption_f_y{year_offset}',
              f'disruption_f_down_gt1_y{year_offset}',
              f'disruption_f_down_gt2_y{year_offset}']
dispf_quantile_cols = [f'{c}_quantile' for c in dispf_cols]

# the disruptive-citation fraction R = i/(i+j) (I_count / (I_count + J_count)) and its
# (year, domain) quantile, computed by compute_efgd_quantile.py. Like the disp_f columns
# these are display-only: they do NOT enter the _sort_key or the top_* flags.
ratio_index_names = ['ij_ratio']
ratio_cols = [f'ij_ratio_y{year_offset}']
ratio_quantile_cols = [f'{c}_quantile' for c in ratio_cols]

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

# disp_f columns are only present when compute_efgd_quantile.py ran with use_disp_f=True.
# If they are missing, fall back gracefully (drop them from the display) rather than crash.
missing_dispf = [c for c in dispf_cols + dispf_quantile_cols if c not in df.columns]
if missing_dispf:
    print(f"WARNING: disp_f columns absent from the quantile table ({missing_dispf}); "
          f"re-run compute_efgd_quantile.py with use_disp_f=True to include them. "
          f"Proceeding WITHOUT disp_f.")
    dispf_index_names = []
    dispf_cols = []
    dispf_quantile_cols = []
else:
    print(f"disp_f columns present: {dispf_cols} (+ *_quantile)")

# i/(i+j) is only present when the quantile table was written by a compute_efgd_quantile.py
# version that carries ij_ratio_y{off}; fall back gracefully if absent.
missing_ratio = [c for c in ratio_cols + ratio_quantile_cols if c not in df.columns]
if missing_ratio:
    print(f"WARNING: i/(i+j) columns absent from the quantile table ({missing_ratio}); "
          f"re-run compute_efgd_quantile.py to include them. Proceeding WITHOUT i/(i+j).")
    ratio_index_names = []
    ratio_cols = []
    ratio_quantile_cols = []
else:
    print(f"i/(i+j) columns present: {ratio_cols} (+ *_quantile)")

# keep only years we extract (input is already filtered to pub_year < max_year)
df = df[df['pub_year'] >= MIN_YEAR].copy()
print(f"after pub_year >= {MIN_YEAR}: {len(df)} papers")

# keep only highly-cited papers (citation_count_y{off} >= EXTRACT_CITATION_THRESHOLD)
n_before_cit = len(df)
df = df[df[citation_col] >= EXTRACT_CITATION_THRESHOLD].copy()
print(f"after {citation_col} >= {EXTRACT_CITATION_THRESHOLD}: {len(df)} papers "
      f"(removed {n_before_cit - len(df)})")

# ==================================================
# 2. lookups: pid -> title, root concept id -> display name
# ==================================================
with open(os.path.join(meta_dire, 'pid2title.pkl'), 'rb') as f:
    pid2title = pickle.load(f)
print("pid2title size:", len(pid2title))

URL_PREFIX_LEN = len("https://openalex.org/")
df_concept = pd.read_excel(CONCEPT_DIRE)
# key by lower-cased, prefix-stripped id so it matches the ids in pid2root_concepts
cid2name = {str(oid)[URL_PREFIX_LEN:].lower(): name
            for oid, name in zip(df_concept['openalex_id'], df_concept['display_name'])}

def concept_display(cid):
    return cid2name.get(str(cid).lower(), str(cid))

# ==================================================
# 3. within each (pub_year, top_concept) cell, keep ALL papers
# ==================================================
selected_frames = []
for (year, concept), df_g in df.groupby(['pub_year', 'top_concept']):
    # no per-quantile top-N restriction: keep every paper in the cell
    df_sel = df_g.copy()
    for name in index_names:
        df_sel[f'top_{name}'] = True
    selected_frames.append(df_sel)

df_sel_all = pd.concat(selected_frames, ignore_index=True)
print(f"selected {len(df_sel_all)} (paper, cell) rows across "
      f"{df_sel_all.groupby(['pub_year', 'top_concept']).ngroups} (year, domain) cells")

# ==================================================
# 4. annotate: title, domain display name, sort key (max of the 4 quantiles)
# ==================================================
df_sel_all['title'] = df_sel_all['pid'].map(lambda p: pid2title.get(p, ''))
df_sel_all['domain'] = df_sel_all['top_concept'].map(concept_display)
df_sel_all['_sort_key'] = df_sel_all[quantile_cols].max(axis=1)

out_cols = (['pid', 'title', 'domain', 'pub_year', 'group_size',
             'up_refer_count', citation_col]
            + index_cols + quantile_cols
            + dispf_cols + dispf_quantile_cols
            + ratio_cols + ratio_quantile_cols
            + [f'top_{name}' for name in index_names])

# ==================================================
# 5. write one xlsx per year, one sheet per domain
# ==================================================
# its own dedicated top-level folder (NOT nested under efg_computation)
out_dir = os.path.join(meta_dire, 'efgd_quantile_top_extract')
os.makedirs(out_dir, exist_ok=True)

def make_sheet_name(name, used):
    """Excel sheet names: <=31 chars, none of : \\ / ? * [ ]; ensure uniqueness."""
    s = re.sub(r'[:\\/?*\[\]]', '_', str(name)).strip() or 'sheet'
    s = s[:31]
    base, i = s, 1
    while s in used:
        suffix = f'_{i}'
        s = base[:31 - len(suffix)] + suffix
        i += 1
    used.add(s)
    return s

for year, df_year in df_sel_all.groupby('pub_year'):
    # filename carries BOTH citation thresholds:
    #   qct = the citation cutoff used when the quantiles were COMPUTED
    #         (citation_threshold; the population the percentile rank is taken over)
    #   dct = the citation cutoff used for DISPLAY / extraction here
    #         (EXTRACT_CITATION_THRESHOLD)
    out_path = os.path.join(
        out_dir,
        f'efgd_topall_qct_{citation_threshold}_dct_{EXTRACT_CITATION_THRESHOLD}_extract_y{int(year)}.xlsx')
    used = set()
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        # domains sorted by size (largest first) for a stable, readable order
        domain_order = (df_year.groupby('domain').size()
                        .sort_values(ascending=False).index)
        for domain in domain_order:
            df_dom = (df_year[df_year['domain'] == domain]
                      .sort_values('_sort_key', ascending=False))
            sheet = make_sheet_name(domain, used)
            # freeze_panes=(1, 0): freeze the first (header) row so it stays visible
            # when scrolling every sheet.
            df_dom[out_cols].to_excel(writer, sheet_name=sheet, index=False,
                                      freeze_panes=(1, 0))
    print(f"  wrote {out_path}  ({len(df_year)} rows, {len(used)} domain sheets)")

print("output dir:", out_dir)
print("finished")
