"""
Aggregate the per-YEAR Excel workbooks written by compute_efgd_quantile_s2_extract.py
into COMPACT, per-DOMAIN files.

compute_efgd_quantile_s2_extract.py writes one .xlsx per publication year, each with
one sheet per domain (the full column set: e/f/g/disp + quantiles, disp_f, i/(i+j),
top_* flags, ...). This script reads every one of those per-year files, re-groups all
their rows BY DOMAIN (across all years), keeps a compact column subset, and writes ONE
FILE PER DOMAIN into three sibling subdirectories:

    compact_by_domain_full/                <domain>.xlsx  ALL papers, sorted by citation
    compact_by_domain_top500_disruption/   <domain>.xlsx  top 500 by disp quantile
    compact_by_domain_top500_citation/     <domain>.xlsx  top 500 by citation count

The top-500 versions carry exactly the same columns; they differ only in the ranking
used to pick / order the 500 rows.

Compact column subset:
    publication_year, pid, title, citation_count,
    disp (+ quantile), f (+ quantile), g (+ quantile)

In every version the header row is frozen and cells are conditionally shaded:
    - disp > 0            -> disp_quantile cell shaded BLUE
    - f_quantile > 0.5    -> f_quantile   cell shaded RED
    - g_quantile > 0.5    -> g_quantile   cell shaded BLUE

Input  : META_DIRE/efgd_quantile_top_extract/efgd_topall_qct_{qct}_dct_{dct}_extract_y{year}.xlsx
Output : META_DIRE/efgd_quantile_top_extract/compact_by_domain_full/<domain>.xlsx
         META_DIRE/efgd_quantile_top_extract/compact_by_domain_top500_disruption/<domain>.xlsx
         META_DIRE/efgd_quantile_top_extract/compact_by_domain_top500_citation/<domain>.xlsx
"""

import os
import re
import sys

import pandas as pd
from openpyxl.styles import PatternFill

# utils.py is in this same directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from utils import META_DIRE  # noqa: E402

meta_dire = META_DIRE
print("meta_dire", meta_dire)

# ---- must match compute_efgd_quantile_s2_extract.py so the input filenames resolve ----
year_offset = -1
citation_threshold = 100          # qct: quantile-computation citation cutoff
EXTRACT_CITATION_THRESHOLD = 100  # dct: display/extraction citation cutoff
# the per-year files span these publication years (must cover the range
# compute_efgd_quantile_s2_extract.py wrote: MIN_YEAR .. max_year-1). Years with no
# file are simply skipped.
MIN_YEAR = 1960
MAX_YEAR = 2025
print("year_offset", year_offset, "| qct", citation_threshold,
      "| dct", EXTRACT_CITATION_THRESHOLD,
      "| years", MIN_YEAR, "..", MAX_YEAR - 1)

# source column names (as written by compute_efgd_quantile_s2_extract.py)
citation_col = f'citation_count_y{year_offset}'
# the three indices we keep in the compact view (drop e), each with its *_quantile
compact_indices = [
    ('disp', f'disp_y{year_offset}'),
    ('f',    f'f_y{year_offset}'),
    ('g',    f'g_y{year_offset}'),
]

in_dir = os.path.join(meta_dire, 'efgd_quantile_top_extract')

# ==================================================
# 1. collect every per-year workbook by building each expected filename explicitly
#    (year by year), NOT by scanning the directory.
# ==================================================
in_paths = []
missing_years = []
for year in range(MIN_YEAR, MAX_YEAR):
    fname = (f'efgd_topall_qct_{citation_threshold}_dct_{EXTRACT_CITATION_THRESHOLD}'
             f'_extract_y{year}.xlsx')
    path = os.path.join(in_dir, fname)
    if os.path.exists(path):
        in_paths.append(path)
    else:
        missing_years.append(year)
assert in_paths, (
    f"No per-year files found in {in_dir}\n"
    f"(looked for efgd_topall_qct_{citation_threshold}_dct_{EXTRACT_CITATION_THRESHOLD}"
    f"_extract_y{{{MIN_YEAR}..{MAX_YEAR - 1}}}.xlsx)\n"
    f"Run compute_efgd_quantile_s2_extract.py first.")
print(f"found {len(in_paths)} per-year workbooks "
      f"({len(missing_years)} of {MAX_YEAR - MIN_YEAR} years have no file)")

# ==================================================
# 2. read all sheets of all files into one long frame
# ==================================================
frames = []
for path in in_paths:
    # read EVERY sheet (each sheet is one domain); sheet_name=None -> dict of DataFrames.
    sheets = pd.read_excel(path, sheet_name=None)
    n_rows = 0
    for _sheet, df_sheet in sheets.items():
        if df_sheet.empty:
            continue
        frames.append(df_sheet)
        n_rows += len(df_sheet)
    print(f"  read {os.path.basename(path)}: {len(sheets)} sheets, {n_rows} rows")

df = pd.concat(frames, ignore_index=True)
print(f"total rows across all years: {len(df)}")

# 'domain' is a real column in every source sheet, so we group on it directly (no need
# to rely on the possibly-truncated sheet names).
assert 'domain' in df.columns, "'domain' column missing from source sheets"

# ==================================================
# 3. build the compact column subset
# ==================================================
# friendly-name -> source column, in the order they should appear.
out_map = [('publication_year', 'pub_year'),
           ('pid',              'pid'),
           ('title',            'title'),
           ('citation_count',   citation_col)]
for friendly, src in compact_indices:
    out_map.append((friendly, src))
    out_map.append((f'{friendly}_quantile', f'{src}_quantile'))

missing = [src for _f, src in out_map if src not in df.columns]
assert not missing, f"source columns missing from the per-year files: {missing}"

compact = df[[src for _f, src in out_map]].copy()
compact.columns = [f for f, _s in out_map]

# carry the domain along for grouping, then drop it from the written columns
compact = compact.assign(_domain=df['domain'].values)

# ==================================================
# 4. write one FILE PER DOMAIN into three subdirectories
# ==================================================
TOP_N = 10000

# three output flavours: (subdir, sort column, ascending, keep-top-N?, filename order tag)
# the top-N subdir names are built from TOP_N so they stay in sync if it changes.
# the order tag is appended to each per-domain filename so the ranking is visible from
# the filename itself (e.g. "<domain>_by_citation.xlsx" / "<domain>_by_disruption.xlsx").
VERSIONS = [
    ('compact_by_domain_full',                   'citation_count', False, None,  'by_citation'),   # all papers
    (f'compact_by_domain_top{TOP_N}_disruption', 'disp_quantile',  False, TOP_N, 'by_disruption'),  # top-N disruption
    (f'compact_by_domain_top{TOP_N}_citation',   'citation_count', False, TOP_N, 'by_citation'),   # top-N citation
]

# conditional-shading fills
BLUE = PatternFill(start_color='FFADD8E6', end_color='FFADD8E6', fill_type='solid')
RED = PatternFill(start_color='FFFFC7CE', end_color='FFFFC7CE', fill_type='solid')


def safe_filename(name):
    """Filesystem-safe file stem: strip chars Windows forbids in filenames."""
    s = re.sub(r'[\\/:*?"<>|]', '_', str(name)).strip().rstrip('.') or 'domain'
    return s[:150]


def sheet_name_of(name):
    """Excel sheet name: <=31 chars, none of : \\ / ? * [ ]."""
    return (re.sub(r'[:\\/?*\[\]]', '_', str(name)).strip() or 'sheet')[:31]


def write_domain_file(df_dom, out_path, sheet):
    """Write one domain's rows to its own workbook, freeze the header, and shade
    disp_quantile / f_quantile / g_quantile cells per the rules."""
    cols = list(df_dom.columns)
    dispq_i = cols.index('disp_quantile')  # shaded when the row's disp value > 0
    fq_i = cols.index('f_quantile')
    gq_i = cols.index('g_quantile')

    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        # freeze_panes=(1, 0): keep the header row visible while scrolling.
        df_dom.to_excel(writer, sheet_name=sheet, index=False, freeze_panes=(1, 0))
        ws = writer.sheets[sheet]
        # data rows start at Excel row 2 (row 1 is the header). openpyxl is 1-indexed.
        for offset, (_, row) in enumerate(df_dom.iterrows()):
            r = offset + 2
            if pd.notna(row['disp']) and row['disp'] > 0:
                ws.cell(row=r, column=dispq_i + 1).fill = BLUE
            if pd.notna(row['f_quantile']) and row['f_quantile'] > 0.5:
                ws.cell(row=r, column=fq_i + 1).fill = RED
            if pd.notna(row['g_quantile']) and row['g_quantile'] > 0.5:
                ws.cell(row=r, column=gq_i + 1).fill = BLUE


domains = sorted(compact['_domain'].dropna().unique())
print(f"{len(domains)} domains")

for subdir, sort_col, ascending, keep_n, order_tag in VERSIONS:
    out_dir = os.path.join(in_dir, subdir)
    os.makedirs(out_dir, exist_ok=True)
    used_names = set()
    n_files = 0
    for domain in domains:
        df_dom = (compact[compact['_domain'] == domain]
                  .drop(columns='_domain')
                  .sort_values(sort_col, ascending=ascending))
        if keep_n is not None:
            df_dom = df_dom.head(keep_n)
        if df_dom.empty:
            continue

        # one file per domain; the order tag makes the ranking visible in the filename.
        # ensure unique, filesystem-safe stems.
        stem = f'{safe_filename(domain)}_{order_tag}'
        cand, k = stem, 1
        while cand.lower() in used_names:
            cand = f'{stem}_{k}'
            k += 1
        used_names.add(cand.lower())

        out_path = os.path.join(out_dir, f'{cand}.xlsx')
        write_domain_file(df_dom, out_path, sheet_name_of(domain))
        n_files += 1
    print(f"  {subdir}: wrote {n_files} domain files -> {out_dir}")

print("finished")
