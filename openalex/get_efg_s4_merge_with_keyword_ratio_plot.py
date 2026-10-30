"""
Per-publication-year keyword prevalence from the get_efg_s4_merge_with_keyword.py
output: for every year, what fraction of papers contain each keyword in the title.

By default (ENGLISH_ONLY) the per-year population is restricted to papers whose
TITLE is English (flags from text_analysis/process_titles.py); keyword matching is
English-only, so non-English titles can never match and would only dilute the
denominator. Set ENGLISH_ONLY = False to count papers of every language.

For each pub_year and each chosen keyword (over the English-title papers only):
    ratio_<keyword> = (# papers that year with has_kw_<keyword> == True) / (# papers that year)
plus two summary columns:
    ratio_any_keyword = fraction of papers matching AT LEAST ONE chosen keyword
    ratio_no_keyword  = 1 - ratio_any_keyword  (the "remaining" share)

Each per-year ratio carries a 95% bootstrap CI (the ratio is a proportion, so the
bootstrap of the mean is sampled via the Binomial equivalence). The cached table
is tidy/long: one row per (keyword, pub_year) with columns
    keyword, pub_year, n_papers, k_count, ratio, ci_low, ci_high
where 'keyword' also includes two summary rows per year: 'any_keyword' and
'no_keyword'. The plot draws one subplot per chosen keyword (line + CI band).

Same conventions as get_efg_s4_merge_with_keyword_plot.py: the keywords to report
are specified here (keyword_list) as a SUBSET of the has_kw_* columns s4 produced;
each is resolved to its has_kw_<safe> column and asserted to exist. The table is
cached to a CSV keyed by a keyword-set tag (kw{N}_{hash}); if that cache exists the
script redraws straight from it and never opens the big file.

At the paper's settings (year_offset=5, reference_threshold=1, citation_threshold=5,
ENGLISH_ONLY=True and KEYWORD_ROWS = theory/hypothesis, observation/discovery,
review/guideline, tool/device) this figure IS 'Fig. 4a' (is_fig4a): the cache, figure and
keyword sidecar are then named after the panel instead of carrying the parameter stub.
Every other setting keeps the descriptive efg_keyword_ratio_{name_stub} names. The gate
matches the keyword ROWS, not the md5 kw_tag, so editing KEYWORD_ROWS drops visibly out of
the deliverable naming rather than silently overwriting Fig. 4a with a different keyword set.

Input  : META_DIRE/efg_computation/
           gen_V3_y{year_offset}_with_keyword_rt_{reference_threshold}_ct_{citation_threshold}_ybefore_{max_year}.csv
         META_DIRE/pid2eng_flag_before_{ENG_FLAG_YEAR}.pkl   (when ENGLISH_ONLY)
Output : ./figures/keyword/
         at the deliverable setting (is_fig4a) -> file_stem = 'Fig. 4a':
           Fig. 4a.csv / Fig. 4a.svg / Fig. 4a_keywords.txt
         otherwise file_stem = efg_keyword_ratio_{name_stub}
           (name_stub = ay{ay}_ct{ct}_rt{rt}_ybefore{max_year}_{kw_tag}, plus
            _engonly{ENG_FLAG_YEAR} when ENGLISH_ONLY):
           {file_stem}.csv         (tidy per-year ratio+CI table, reused on re-run)
           {file_stem}.svg         (one subplot per keyword, ratio + 95% CI)
           {file_stem}_keywords.txt
         A cache written under the pre-Fig. 4a descriptive name is still picked up and
         re-saved under the new name, so renaming never forces a rebuild.
"""

import os
import re
import sys
import pickle
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# utils.py is in this same directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from utils import META_DIRE  # noqa: E402

# repo plotting convention
try:
    plt.rcParams['font.family'] = 'Gill Sans MT'
except Exception:
    pass
plt.rcParams['axes.unicode_minus'] = False

meta_dire = META_DIRE
print("meta_dire", meta_dire)

load_dire = os.path.join(meta_dire, 'efg_computation')

# ---- must match get_efg_s4_merge_with_keyword.py so the input filename resolves ----
year_offset = 5
reference_threshold = 1
citation_threshold = 5
max_year = 2025 - year_offset   # same cutoff as s4 (pub_year < 2025 - year_offset)
print("year_offset", year_offset)
print("reference_threshold", reference_threshold)
print("citation_threshold", citation_threshold)
print("max_year", max_year)

FORCE_REBUILD = False             # set True to ignore an existing ratio cache
# Restrict the denominator (and numerator) to papers whose TITLE is ENGLISH.
# Keyword matching is English-only, so a non-English title can never match a
# keyword; including such papers only inflates the denominator and deflates every
# ratio. English flags come from text_analysis/process_titles.py, which writes
# pid2eng_flag_before_{ENG_FLAG_YEAR}.pkl ({pid: bool}, fastText lid.176). The
# english-only mode is baked into the cache/figure names so it never reuses or
# overwrites the all-language version.
ENGLISH_ONLY = True
ENG_FLAG_YEAR = 2025              # must match year_threshold in process_titles.py
# only draw a year whose subsample has at least this many papers (plot-time only;
# the cache still stores every year so the threshold can be changed without rebuild)
MIN_PAPERS_PER_YEAR = 10
# bootstrap settings for the 95% CI of each per-year ratio
N_BOOT = 1000
CI = 95
SEED = 42
print("FORCE_REBUILD", FORCE_REBUILD, "| MIN_PAPERS_PER_YEAR", MIN_PAPERS_PER_YEAR,
      "| N_BOOT", N_BOOT, "| CI", CI, "| SEED", SEED)

# ==================================================
# keywords to report (a subset of what get_efg_s4_merge_with_keyword.py produced)
# Subplot layout is 4 rows x 2 columns: each entry below is ONE ROW, given as
# (left-column keyword, right-column keyword). keyword_list is derived from this so the
# cache / has_kw_* column resolution stays in sync with exactly what is drawn.
# ==================================================
KEYWORD_ROWS = [
    ('theory', 'hypothesis'),
    ('observation', 'discovery'),
    ('review', 'guideline'),
    ('tool', 'device'),
]
keyword_list = [kw for row in KEYWORD_ROWS for kw in row]

# normalize (lower-case, dedup) then SORT so the hash / columns are deterministic
seen = set()
keywords = []
for kw in keyword_list:
    kw = kw.lower().strip()
    if kw and kw not in seen:
        seen.add(kw)
        keywords.append(kw)
keywords = sorted(keywords)
print("keywords:", keywords)

# same keyword -> column-suffix mapping as get_efg_s4_merge_with_keyword.py
def safe_name(kw):
    return re.sub(r'[^0-9a-z]+', '_', kw).strip('_') or 'kw'

figure_dire_save = os.path.join(BASE_DIR, 'figures', 'keyword')
os.makedirs(figure_dire_save, exist_ok=True)

# source = get_efg_s4_merge_with_keyword.py output
src_path = os.path.join(
    load_dire,
    f'gen_V3_y{year_offset}_with_keyword_'
    f'rt_{reference_threshold}_'
    f'ct_{citation_threshold}_'
    f'ybefore_{max_year}.csv'
)
assert os.path.exists(src_path), src_path

# resolve the chosen keywords to has_kw_* columns (verify they exist; cheap header read)
header = pd.read_csv(src_path, nrows=0)
kw_cols = []
for kw in keywords:
    col = f'has_kw_{safe_name(kw)}'
    assert col in header.columns, (
        f"keyword '{kw}' -> column '{col}' not in {src_path}; "
        f"regenerate get_efg_s4_merge_with_keyword.py with this keyword included."
    )
    kw_cols.append(col)
print("keyword columns:", kw_cols)

# keyword-set tag (matches get_efg_s4_merge_with_keyword_plot.py)
kw_blob = ','.join(sorted(keywords))
kw_tag = f'kw{len(keywords)}_{hashlib.md5(kw_blob.encode()).hexdigest()[:8]}'
print("keyword tag:", kw_tag)

name_stub = f'ay{year_offset}_ct{citation_threshold}_rt{reference_threshold}_ybefore{max_year}_{kw_tag}'
# english-only changes the denominator, so it must alter the cache/figure names
if ENGLISH_ONLY:
    name_stub += f'_engonly{ENG_FLAG_YEAR}'

# ---- deliverable naming ----------------------------------------------------------------
# This 4x2 keyword-ratio grid at the paper's settings IS Fig. 4a, so its figure, cached data
# table and keyword sidecar are named after the panel; every other setting keeps the
# descriptive parameter stub. The gate lists every parameter that changes what the panels
# show -- thresholds, the English-title denominator and the keyword rows themselves -- so no
# other run can land on this name. Like get_efg_s4_merge_with_keyword_plot.py's is_ed_fig, it
# matches the keyword ROWS rather than the md5 kw_tag, so editing KEYWORD_ROWS visibly drops
# out of the deliverable naming instead of silently overwriting Fig. 4a with other keywords.
FIG_NAME = 'Fig. 4a'
FIG_KEYWORD_ROWS = [
    ('theory', 'hypothesis'),
    ('observation', 'discovery'),
    ('review', 'guideline'),
    ('tool', 'device'),
]
is_fig4a = (year_offset == 5 and reference_threshold == 1 and citation_threshold == 5
            and ENGLISH_ONLY
            and [tuple(r) for r in KEYWORD_ROWS] == FIG_KEYWORD_ROWS)
print("is_fig4a", is_fig4a, ("-> %s" % FIG_NAME) if is_fig4a else "")

file_stem = FIG_NAME if is_fig4a else f'efg_keyword_ratio_{name_stub}'
cache_csv = os.path.join(figure_dire_save, f'{file_stem}.csv')
out_svg = os.path.join(figure_dire_save, f'{file_stem}.svg')
keywords_path = os.path.join(figure_dire_save, f'{file_stem}_keywords.txt')
# a run from before this naming wrote the SAME table under the descriptive name; reuse it
# rather than re-reading the multi-GB source (the contents are identical -- only the file
# name changed), and re-save it under the deliverable name so later runs load it directly.
legacy_cache_csv = os.path.join(figure_dire_save, f'efg_keyword_ratio_{name_stub}.csv')

with open(keywords_path, 'w', encoding='utf-8') as f:
    f.write('# keywords used for this efg keyword-ratio plot (one per line)\n')
    f.write(f'# tag: {kw_tag}\n')
    f.write('\n'.join(keywords) + '\n')
print("[keywords saved]", keywords_path)

# ==================================================
# bootstrap helper: 95% CI of a proportion k/n.
# The ratio is the mean of a 0/1 column, so a bootstrap resample of n papers has
# True-count ~ Binomial(n, k/n); dividing by n gives the resampled ratio. This is
# exactly the nonparametric bootstrap of the mean, but O(N_BOOT) instead of O(N_BOOT*n).
# ==================================================
def bootstrap_ci_proportion(k, n, n_boot, ci, rng):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    boot = rng.binomial(n, p, size=n_boot) / n
    lo = float(np.percentile(boot, (100 - ci) / 2))
    hi = float(np.percentile(boot, 100 - (100 - ci) / 2))
    return p, lo, hi

# ==================================================
# 1. build the per-year ratio table with bootstrap CI (or load it if cached).
# Tidy/long format: one row per (keyword, pub_year). 'keyword' also carries two
# summary pseudo-keywords: 'any_keyword' and 'no_keyword'.
# columns: keyword, pub_year, n_papers, k_count, ratio, ci_low, ci_high
# ==================================================
load_cache = None
if not FORCE_REBUILD:
    if os.path.exists(cache_csv):
        load_cache = cache_csv
    elif os.path.exists(legacy_cache_csv):     # pre-Fig. 4a name, same contents
        load_cache = legacy_cache_csv
if load_cache is not None:
    out = pd.read_csv(load_cache)
    print("[ratio cache loaded]", load_cache, "| shape:", out.shape)
    if load_cache != cache_csv:
        out.to_csv(cache_csv, index=False)
        print("[ratio cache re-saved under the deliverable name]", cache_csv)
else:
    use_cols = ['pub_year'] + kw_cols
    if ENGLISH_ONLY:
        use_cols = ['pid'] + use_cols   # need pid to look up the English flag
    df = pd.read_csv(src_path, usecols=use_cols)
    df = df[df['pub_year'].notna()].copy()
    df['pub_year'] = df['pub_year'].astype(int)
    for col in kw_cols:
        df[col] = df[col].astype(bool)

    if ENGLISH_ONLY:
        # keep only papers whose title fastText-detected as English; papers with no
        # flag entry (e.g. null title) are treated as non-English and dropped.
        eng_path = os.path.join(meta_dire, f'pid2eng_flag_before_{ENG_FLAG_YEAR}.pkl')
        assert os.path.exists(eng_path), (
            f"English-title flag not found: {eng_path}; run "
            f"text_analysis/process_titles.py (year_threshold={ENG_FLAG_YEAR}) first."
        )
        with open(eng_path, 'rb') as f:
            pid2eng = pickle.load(f)
        is_eng = df['pid'].map(pid2eng).fillna(False).astype(bool)
        n_before = len(df)
        df = df[is_eng].copy()
        print(f"English-title filter: {n_before} -> {len(df)} papers "
              f"(dropped {n_before - len(df)} non-English / unknown-title)")
    print("loaded source shape:", df.shape)

    g = df.groupby('pub_year')
    n_by_year = g.size()                       # papers per year (the denominator)
    k_by_year = g[kw_cols].sum().astype(int)   # True-count per keyword per year
    df['_any_kw'] = df[kw_cols].any(axis=1)
    any_by_year = df.groupby('pub_year')['_any_kw'].sum().astype(int)

    rng = np.random.default_rng(SEED)
    rows = []
    for year in n_by_year.index:
        n = int(n_by_year[year])
        # per chosen keyword
        for kw, col in zip(keywords, kw_cols):
            k = int(k_by_year.loc[year, col])
            ratio, lo, hi = bootstrap_ci_proportion(k, n, N_BOOT, CI, rng)
            rows.append({'keyword': kw, 'pub_year': int(year), 'n_papers': n,
                         'k_count': k, 'ratio': ratio, 'ci_low': lo, 'ci_high': hi})
        # summary: any keyword / no keyword
        ka = int(any_by_year[year])
        ratio, lo, hi = bootstrap_ci_proportion(ka, n, N_BOOT, CI, rng)
        rows.append({'keyword': 'any_keyword', 'pub_year': int(year), 'n_papers': n,
                     'k_count': ka, 'ratio': ratio, 'ci_low': lo, 'ci_high': hi})
        ratio, lo, hi = bootstrap_ci_proportion(n - ka, n, N_BOOT, CI, rng)
        rows.append({'keyword': 'no_keyword', 'pub_year': int(year), 'n_papers': n,
                     'k_count': n - ka, 'ratio': ratio, 'ci_low': lo, 'ci_high': hi})

    out = pd.DataFrame(rows).sort_values(['keyword', 'pub_year'])
    out.to_csv(cache_csv, index=False)
    print("[ratio cache saved]", cache_csv, "| shape:", out.shape)

# ==================================================
# 2. plot: one subplot per keyword (ratio over year, with 95% bootstrap CI band)
# ==================================================
# only draw years whose subsample has at least MIN_PAPERS_PER_YEAR papers
plot_df = out[out['n_papers'] >= MIN_PAPERS_PER_YEAR].sort_values('pub_year')

# decade-ish ticks; extend to max_year (e.g. 2020) so that endpoint shows even
# though the data stops at max_year - 1 (pub_year < max_year)
y_min = int(plot_df['pub_year'].min())
tick_start = y_min - (y_min % 10)
xticks = list(range(tick_start, max_year + 1, 20))

# subplot grid: n_rows x NCOLS, laid out ROW by ROW from KEYWORD_ROWS (each entry is one row
# given left-to-right), so the on-screen arrangement matches the spec exactly.
n_rows = len(KEYWORD_ROWS)
NCOLS = max(len(row) for row in KEYWORD_ROWS)
LINE_COLOR = 'tab:blue'   # all subplots share one color
TICK_FONTSIZE = 21        # x and y tick labels

# The ratios are stored as fractions (0-1) but the y-axis is labelled in PERCENT, so the
# tick labels are the fraction x100 (0.023 -> "2.3"). %g drops trailing zeros, which keeps
# the small-ratio panels readable (0.001 -> "0.1") without forcing a fixed decimal count.
def _pct_tick(v, _pos):
    return f'{v * 100:g}'

fig, axes = plt.subplots(
    n_rows, NCOLS,
    figsize=(4.2 * NCOLS, 3.4 * n_rows),
    sharex=True, squeeze=False,
)

used = set()
for row_idx, row_kws in enumerate(KEYWORD_ROWS):
    for col_idx, kw in enumerate(row_kws):
        ax = axes[row_idx][col_idx]
        used.add((row_idx, col_idx))
        d = plot_df[plot_df['keyword'] == kw.lower().strip()].sort_values('pub_year')
        ax.plot(d['pub_year'], d['ratio'], color=LINE_COLOR, linewidth=1.8)
        ax.fill_between(d['pub_year'], d['ci_low'], d['ci_high'],
                        color=LINE_COLOR, alpha=0.25, linewidth=0)
        # keyword name (capitalised) as a text label in the upper-left corner (no per-subplot title)
        ax.text(0.03, 0.97, kw.capitalize(), transform=ax.transAxes,
                ha='left', va='top', fontsize=24)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) for x in xticks], fontsize=TICK_FONTSIZE, rotation=45,
                           ha='right', rotation_mode='anchor')
        ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)
        ax.yaxis.set_major_formatter(FuncFormatter(_pct_tick))
        ax.set_xlim(y_min, max_year)   # extend so the (e.g. 2020) tick is visible
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

# hide any unused cells (e.g. a row with a single keyword)
for r in range(n_rows):
    for c in range(NCOLS):
        if (r, c) not in used:
            axes[r][c].axis('off')

# explicit margins (rather than tight_layout, which over-reserves space for the shared
# labels) so the single x/y labels can be tucked right up against the tick labels.
plt.subplots_adjust(left=0.14, right=0.975, top=0.96, bottom=0.115,
                    wspace=0.30, hspace=0.30)
# single shared axis labels: x centred under the columns, y centred on the left.
# supylabel is pinned essentially at the figure's left edge; the WIDER left margin
# (above) is what opens the gap to the now-larger y-tick labels. supxlabel nudged UP.
# centre the x-label over the AXES span (left..right), not the whole figure: the wide
# left margin would otherwise pull the default x=0.5 to the left of the two columns.
fig.supxlabel('Publication Year', fontsize=28, y=0.03, x=(0.14 + 0.975) / 2)
# centre the y-label over the AXES span (bottom..top), matching the x-label treatment.
# x=0.045: moved in from the old 0.006 (which sat right on the figure's left edge). The
# rotated 28pt label is ~0.023 fig-widths tall, so it spans ~0.034-0.056 -- still clear of
# the y-tick labels, which end around x=0.09 given the left margin of 0.14.
fig.supylabel('Ratio of papers (%)', fontsize=28, x=0.045, y=(0.115 + 0.96) / 2)
plt.savefig(out_svg, format='svg', transparent=True)
plt.close()
print("[figure saved]", out_svg)

print("finished")
