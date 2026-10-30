"""
Do Nobel papers disrupt more than the papers they sat next to?

This is the per-paper counterpart of nobel_disp_mean_quantile.py. That script locates ONE
number -- the Nobel papers' mean disruption -- on a corpus-wide axis. This one gives EVERY
paper its own quantile, computed inside the tightest comparison set the data supports, and then
compares the two distributions of quantiles.

THREE metrics go through the identical machinery:

    Disruption  D       = (I - J) / (C + K)    from disp_full, the column disp_y{off}
    Foundation  F/C     = f_count_y{off} / C   F from gen_V3, C from gen_V3
    Foundation  F/(C+K) = f_count_y{off}       F from gen_V3, C and K from disp_full
                          / (C + K)

C is citation_count_y{off}, the papers citing the focal one inside the window; K is the
disruption K. For disruption C = I + J, so (I-J)/(C+K) and (I-J)/(I+J+K) are one and the same
column and it is read straight from disp_full rather than recomputed.

The two Foundation ratios differ only in the denominator. F/C takes BOTH halves from gen_V3,
because f_count and that file's citation_count are counted over the same window by the same run
while disp_full's citation_count belongs to the disruption run. F/(C+K) instead uses the
DISRUPTION denominator -- disp_full's own C and K, whose sum is exactly the I+J+K that disp is
divided by -- so F sits on the identical footing as the disruption index. It is the same
quantity compute_efgd_quantile.py calls f_over_ijk.

THE COMPARISON SET.  A paper's quantile is its rank among the papers published in the SAME YEAR
and the SAME VENUE (WoS source_title). That cell is the unit of everything here:

  1. the population is the qualifying papers (rt / ct / pub_year < max_year, as everywhere
     else), with their venue attached from pid2source_title;
  2. only the cells that contain at least one Nobel paper are kept -- so every 'other' paper in
     this analysis shares a publication year AND a venue with some Nobel paper, and nothing
     else is compared against anything;
  3. a cell with fewer than MIN_CELL_SIZE papers is dropped whole. The default is 1, i.e. NO
     cut, so no Nobel paper is lost for having had few contemporaries in its journal -- at the
     price that a rank among three papers carries almost no information and a paper alone in
     its cell scores 1.0 by definition. The run prints, and the CSV records, how many Nobel
     papers are alone in their cell and how many sit in a cell of fewer than 5, so the reported
     mean can be read against that;
  4. inside each surviving cell every paper is ranked by EACH metric with
     rank(pct=True, method='average') -- the same convention compute_efgd_quantile.py uses --
     giving a number in (0, 1]. Ties share the average rank, which matters here: disp has a
     large atom at exactly 0. A paper with no value for one metric drops out of that metric's
     ranking only; the cells themselves are fixed by the disruption population, so both metrics
     are read inside exactly the same comparison sets.

WHAT THE FIGURES SHOW.  ONE PANEL PER METRIC -- disruption in red, both Foundation ratios in
blue (they are told apart by their marker labels and their filenames, not by colour). Each
draws that metric's Nobel quantile density and marks its own Nobel mean (dashed) and median
(dash-dotted) in the metric's colour. The two panels share the [0, 1] quantile axis, so they
can still be read against each other, and each against 0.5, side by side.

The other papers are not drawn (set DRAW_OTHERS to put them back): ranks inside a cell fill the
unit interval by construction, so their density is the uniform baseline whatever the data says
-- up to the atom at disp = 0 and the tie handling -- and the line says nothing the number 0.5
does not. That number IS the comparison: a Nobel mean above 0.5 is the whole finding, and every
statistic of the others (mean, median, the shares above 0.5 and 0.9, and a Mann-Whitney U test
of exactly this shift) is still computed on all of them, per metric, and written to the
printout and the summary CSV.

Nobel papers are identified exactly as in nobel_disp_mean_quantile.py: DOI first (canonicalised
by norm_doi, looked up in pid2doi.pkl), then TITLE (canonicalised by norm_title) for the rows no
DOI could place, both honouring require_unique_match. The title fallback is not optional in
practice -- the prize-winning papers have a median publication year around 1957 while WoS
carries a DOI essentially only from ~1998 on. See that script's header for the details.

Everything the figure and the tables need is cached to one pkl keyed by every setting below; a
re-run with an unchanged key redraws from it and never touches disp_full or the Nobel records.

Input  : META_DIRE/disruption/disp_full[_y{off}]{ds}{ks}.csv   (the _y tag only when off != 5)
             (arrange_disruption.py; supplies disp_y{off} and the filter columns pub_year /
              up_refer_count / citation_count_y{off})
         META_DIRE/efg_computation/gen_V3_y{off}{ds}.csv
                                              (get_efg_s2_merge.py; supplies f_count_y{off} and
                                               the citation_count_y{off} F/C divides by. F/(C+K)
                                               divides the same F count by disp_full's C + K)
         META_DIRE/pid2source_title{ds}.pkl   (wos/get_meta_data_get_venue.py, RUN IT FIRST
                                               with the same use_with_review)
         META_DIRE/pid2doi.pkl                (wos/get_meta_data_doi.py)
         META_DIRE/pid2title.pkl              (wos/get_meta_data.py; the title fallback's index)
         NOBEL_DATA_DIRE/<nobel csv>          (EXTERNAL Nobel publication records)
Cache  : ./figures/nobel_disp_compare_v2/
           nobel_disp_compare_v2_cache_{param_suffix}.pkl
Output : META_DIRE/nobel_disp_compare_v2/
           nobel_disp_compare_v2_{param_suffix}.csv
         one row per metric: both groups' n / mean / median / sd, the share of Nobel papers
         above 0.5 and 0.9, the Mann-Whitney result, the cell counts, and how many Nobel papers
         sit alone in their cell or in a cell of fewer than 5 papers.
           nobel_papers_{param_suffix}.csv
         one row per identified Nobel paper: pid, title, venue, pub_year, the raw counts, both
         raw metrics, BOTH within-cell quantiles, the cell's size, matched_by, and why it was
         dropped when it was (thresholds, no venue, cell too small).
           cells_{param_suffix}.csv
         one row per surviving (pub_year, venue) cell: its size, how many Nobel papers it
         holds, and the mean quantile of each group inside it.
         ./figures/nobel_disp_compare_v2/
           nobel_disp_compare_v2_{metric}_{param_suffix}.svg  (+ .png)
         one figure per metric, each named after the metric whose density it draws.
         {param_suffix} = y{off}{ds}{ks}_rt_{rt}_ct_{ct}_ybefore_{max_year}_
                          cell{MIN_CELL_SIZE}{match_tag}{title_tag}

Usage  : python nobel_disp_mean_quantile_compare_v2.py
"""

import os
import re
import sys
import pickle
import unicodedata
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)             # utils.py is in the parent directory
from utils import META_DIRE, NOBEL_DATA_DIRE  # noqa: E402

meta_dire = META_DIRE
nobel_prize_dire = NOBEL_DATA_DIRE          # META_DIRE/nobel_prize; see utils.py
disp_dire = os.path.join(meta_dire, 'disruption')
print("nobel_prize_dire", nobel_prize_dire)

# ---- must match the run that produced disp_full (same settings as nobel_disp_mean_quantile.py)
# There is deliberately no use_disp_f here: this script reads the raw disruption table, not any
# of compute_efgd_quantile.py's outputs.
year_offset = 5
reference_threshold = 1
citation_threshold = 1
# must match get_disruption.py: read the '_with_review' variant (the doctypes=='review'-
# inclusive network) when True. Part of the disp_full AND the pid2source_title filename.
use_with_review = False
data_suffix = '_with_review' if use_with_review else ''
# must match get_disruption.py / arrange_disruption.py: which K-window disruption variant to
# read. False -> the dropped-lower-bound K ('_k_no_lower_bound'); True -> the default K (no tag)
k_require_after_focal = True
k_suffix = '' if k_require_after_focal else '_k_no_lower_bound'
# pub_year cutoff leaves room for the citation window, as in compute_efgd_quantile.py
max_year = 2010
off = year_offset
citation_col = f'citation_count_y{off}'

print("year_offset", year_offset, "| rt", reference_threshold, "| ct", citation_threshold,
      "| max_year", max_year)
print("data_suffix", repr(data_suffix), "| k_suffix", repr(k_suffix))

# ---- the comparison cell ------------------------------------------------------------------
# The minimum number of papers a (pub_year, venue) cell must hold to be used at all. Cells
# below it are dropped WHOLE -- their other papers as well as their Nobel one.
#
# 1 means no cut, which is the default: every Nobel paper that has a venue is kept, and none is
# lost to having had few contemporaries in its journal.
#
# The cost of 1 is worth knowing, because it inflates the very number this script reports. A
# rank is only as meaningful as the set it is taken within: a cell of three papers yields ranks
# of 1/3, 2/3 and 1 whatever the disruption values are, and a Nobel paper ALONE in its cell
# scores exactly 1.0 by definition, contributing a perfect score that reflects no comparison at
# all. So the run prints how many Nobel papers sit in a cell of size 1 and of size < 5, and
# both counts go to the summary CSV -- read the mean against them. Raise MIN_CELL_SIZE to trade
# Nobel papers for cleaner ranks; it is in the output names, so two thresholds never overwrite
# each other.
MIN_CELL_SIZE = 1

# ---- Nobel-prize publication-record files (same list as award_quantile_analysis.py) ----
NOBEL_FILES = [
    {'file': 'Medicine publication record.csv',  'domain': 'Medicine'},
    {'file': 'Chemistry publication record.csv', 'domain': 'Chemistry'},
    {'file': 'Physics publication record.csv',   'domain': 'Physics'},
]

# The metrics to rank on, each with (short name, column, axis label, short label, colour).
# Both go through exactly the same machinery: the same (pub_year, venue) cells, the same
# rank(pct=True, method='average'), the same statistics.
#   disp     (I-J)/(C+K), read straight from disp_full -- see the header
#   f_over_c F/C, the Foundation COUNT over the citation count, both from gen_V3 (see
#            FOUNDATION below). Not F/(I+J+K): the denominator is C alone.
# The FIRST entry is the primary one -- its density is the curve the figure draws. Every
# other entry contributes its Nobel mean and median as marker lines on that same figure,
# which is legitimate because all of them live on the one [0, 1] quantile axis.
METRICS = [
    ('disp',      f'disp_y{off}',      'Disruption D = (I-J)/(C+K)', 'Disruption',      'tab:red'),
    ('f_over_c',  f'f_over_c_y{off}',  'Foundation F/C',             'Foundation F/C',  'tab:blue'),
    ('f_over_ck', f'f_over_ck_y{off}', 'Foundation F/(C+K)',         'Foundation F/(C+K)',
     'tab:blue'),
]

# ---- the two Foundation indices -----------------------------------------------------------
# Neither is in disp_full: the F count comes from get_efg_s2_merge.py's gen_V3 table. They
# differ only in the denominator.
#
#   F/C      the F count over gen_V3's OWN citation_count. Taking both halves from gen_V3 is
#            deliberate: they are counted over the same window by the same run, whereas
#            disp_full's citation_count belongs to the disruption run and the two differ
#            whenever the runs used different year_offsets.
#   F/(C+K)  the F count over the DISRUPTION denominator -- C + K with C = citation_count and K
#            = K_count, both from disp_full, so C + K is exactly the I + J + K that disp is
#            divided by. This is the same quantity compute_efgd_quantile.py calls f_over_ijk,
#            and it puts F on the identical footing as the disruption index: same denominator,
#            same window. Only the numerator crosses runs, which is unavoidable -- there is no
#            F count in disp_full -- and is exactly what the pipeline's own f_over_ijk does.
#
# A paper missing from gen_V3, or with a zero denominator, gets NaN and simply takes no part in
# THAT metric's ranking; it still takes part in the others.
efg_dire = os.path.join(meta_dire, 'efg_computation')
f_count_col = f'f_count_y{off}'

# ---- figure / KDE ----
N_KDE_POINTS = 200                 # grid points across [0, 1], as award_quantile_analysis.py
# gaussian_kde costs O(n_points x N_KDE_POINTS), and the 'others' group can be large, so the
# CURVE is built from at most this many values (drawn once, with KDE_SEED). Every NUMBER --
# both means, medians, shares and the Mann-Whitney test -- is always computed on ALL of them,
# so this affects only the drawn line. Set to None to use every value.
KDE_MAX_POINTS = 2_000_000
KDE_SEED = 20260911
# Each metric's curve and markers take that metric's OWN colour, set in METRICS above.
OTHER_COLOR = 'tab:gray'           # the 'others' baseline, only drawn when DRAW_OTHERS is on;
                                   # grey so it clashes with neither metric's colour
AXIS_LABEL_FONTSIZE = 16
MARKER_LABEL_FONTSIZE = 13         # the coloured text beside each marker line (there is no legend)
# The figure draws the NOBEL quantile density only, marked with its mean and its median. The
# other papers' curve is not drawn: it is the uniform baseline by construction (see the header),
# so the line carries no information the number 0.5 does not. Every OTHERS statistic -- mean,
# median, shares, the Mann-Whitney test -- is still computed on all of them and still goes to
# the printout and the CSV. Set this True to put the blue curve back; it is in the cache key, so
# switching rebuilds rather than redrawing a curve that was never computed.
DRAW_OTHERS = False
FORCE_RECOMPUTE = False            # ignore any existing cache and rebuild from the raw tables

# ---- how a prize record is matched to a WoS paper (identical to nobel_disp_mean_quantile.py)
USE_TITLE_FALLBACK = True
title_tag = '' if USE_TITLE_FALLBACK else '_doionly'
# a normalised title shorter than this is never used as a key: too generic to identify a paper
MIN_TITLE_CHARS = 10

# ---- id-matching strictness ----
#   require_unique_match = True  -> only a ONE-TO-ONE key counts; a DOI/title claimed by several
#                                   pids is DROPPED, so no paper is admitted on a guess.
#   require_unique_match = False -> EVERY pid claiming the key is taken.
require_unique_match = False
match_tag = '' if require_unique_match else '_multi_match'
print("MIN_CELL_SIZE", MIN_CELL_SIZE)
print("require_unique_match", require_unique_match, "| match_tag", repr(match_tag))
print("USE_TITLE_FALLBACK", USE_TITLE_FALLBACK, "| title_tag", repr(title_tag),
      "| MIN_TITLE_CHARS", MIN_TITLE_CHARS)


# ---- DOI normalisation (identical to nobel_disp_mean_quantile.py) --------------------------
_DOI_PREFIXES = ('https://doi.org/', 'http://doi.org/', 'https://dx.doi.org/',
                 'http://dx.doi.org/', 'doi:')


def norm_doi(doi):
    """Canonical form for DOI comparison: stripped, lower-cased, no resolver prefix."""
    s = str(doi).strip().lower()
    for prefix in _DOI_PREFIXES:
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s or None


# ---- title normalisation ----
_TITLE_NON_ALNUM = re.compile(r'[^a-z0-9]+')


def norm_title(title):
    """Canonical form for title comparison, or None when the title is too short to identify a
    paper (see MIN_TITLE_CHARS)."""
    s = unicodedata.normalize('NFKD', str(title))
    s = s.encode('ascii', 'ignore').decode('ascii').lower()
    s = _TITLE_NON_ALNUM.sub(' ', s).strip()
    return s if len(s) >= MIN_TITLE_CHARS else None


# ---- venue normalisation ----
def norm_venue(venue):
    """Canonical form for venue comparison, or None for a missing / empty source_title."""
    if venue is None or (isinstance(venue, float) and pd.isna(venue)):
        return None
    s = ' '.join(str(venue).split()).lower()
    return s or None


def resolve_ids(key, index):
    """The pids an external id resolves to, honouring require_unique_match."""
    pids = index.get(key) or []
    if require_unique_match and len(pids) != 1:
        return []
    return [str(p) for p in pids]


def kde_curve(vals, xs):
    """Gaussian-KDE density of `vals` on grid `xs`, with a histogram fallback.

    Returns (ys, how); gaussian_kde is undefined for fewer than two points or a zero-variance
    sample. NOTE the quantiles live on [0, 1] and a Gaussian kernel does not know that, so the
    curve leaks a little past both ends -- read the printed means, not the curve's tails."""
    vals = np.asarray(vals, dtype=float)
    if len(vals) >= 2 and np.ptp(vals) > 0:
        try:
            return gaussian_kde(vals)(xs), 'kde'
        except np.linalg.LinAlgError:
            pass
    counts, edges = np.histogram(vals, bins=min(50, max(1, len(vals))), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return np.interp(xs, centers, counts, left=0.0, right=0.0), 'histogram'


# ---- output + cache locations ----
param_suffix = (f'y{year_offset}{data_suffix}{k_suffix}_'
                f'rt_{reference_threshold}_ct_{citation_threshold}_'
                f'ybefore_{max_year}_cell{MIN_CELL_SIZE}{match_tag}{title_tag}')
out_dir = os.path.join(meta_dire, 'nobel_disp_compare_v2')
os.makedirs(out_dir, exist_ok=True)
out_csv = os.path.join(out_dir, f'nobel_disp_compare_v2_{param_suffix}.csv')
fig_dir = os.path.join(BASE_DIR, 'figures', 'nobel_disp_compare_v2')
os.makedirs(fig_dir, exist_ok=True)
cache_path = os.path.join(fig_dir, f'nobel_disp_compare_v2_cache_{param_suffix}.pkl')
print("param_suffix:", param_suffix)
print("out_csv:", out_csv)
print("cache:", cache_path)

# Everything that changes a cached number or curve; param_suffix already covers the filters.
cache_key = {
    'schema': 'nobel_disp_compare_v2_cells_v1',
    'param_suffix': param_suffix,
    'metrics': [m[0] for m in METRICS],
    'value_cols': [m[1] for m in METRICS],
    'n_kde_points': N_KDE_POINTS,
    'draw_others': DRAW_OTHERS,
    'min_cell_size': MIN_CELL_SIZE,
    'require_unique_match': require_unique_match,
    'use_title_fallback': USE_TITLE_FALLBACK,
    'min_title_chars': MIN_TITLE_CHARS,
    'nobel_files': [c['file'] for c in NOBEL_FILES],
}


def identify_nobel_pids():
    """{pid: 'doi' | 'title'} for every Nobel paper the records can be matched to.

    DOI first, then title for the rows the DOI could not place -- one prize record at a time, so
    the order is per PAPER rather than per key set."""
    print("\nLoading pid2doi ...")
    pid2doi_path = os.path.join(meta_dire, 'pid2doi.pkl')
    assert os.path.exists(pid2doi_path), (
        "%s\nRun wos/get_meta_data_doi.py first -- WoS has no DOI map of its own."
        % pid2doi_path)
    with open(pid2doi_path, 'rb') as f:
        pid2doi = pickle.load(f)
    doi2pid_list = defaultdict(list)
    for pid, doi in pid2doi.items():
        doi2pid_list[norm_doi(doi)].append(pid)
    print(f"  pid2doi size: {len(pid2doi)} -> {len(doi2pid_list)} distinct lower-cased DOIs")

    title2pid_list = defaultdict(list)
    if USE_TITLE_FALLBACK:
        pid2title_path = os.path.join(meta_dire, 'pid2title.pkl')
        assert os.path.exists(pid2title_path), (
            "%s\nRun wos/get_meta_data.py first, or set USE_TITLE_FALLBACK = False."
            % pid2title_path)
        with open(pid2title_path, 'rb') as f:
            pid2title = pickle.load(f)
        for pid, title in pid2title.items():
            key = norm_title(title)
            if key:
                title2pid_list[key].append(pid)
        n_amb = sum(1 for v in title2pid_list.values() if len(v) > 1)
        print(f"  pid2title size: {len(pid2title)} -> {len(title2pid_list)} distinct normalised "
              f"titles ({n_amb} shared by >1 paper)")
        del pid2title

    pid_source = {}
    for cfg in NOBEL_FILES:
        domain = cfg['domain']
        fpath = os.path.join(nobel_prize_dire, cfg['file'])
        if not os.path.exists(fpath):
            print(f"  [nobel/{domain}] file not found, skipping: {fpath}")
            continue
        use_cols = ['Is prize-winning paper', 'DOI'] + (['Title'] if USE_TITLE_FALLBACK else [])
        df_n = pd.read_csv(fpath, usecols=use_cols, encoding='latin1')
        prize_rows = df_n[df_n['Is prize-winning paper'] == 'YES']

        by_doi, by_title, unplaced = set(), set(), 0
        for row in prize_rows.itertuples(index=False):
            doi_key = norm_doi(row.DOI) if pd.notna(row.DOI) else None
            pids = resolve_ids(doi_key, doi2pid_list) if doi_key else []
            if pids:
                by_doi.update(pids)
                continue
            if USE_TITLE_FALLBACK:
                title_key = norm_title(row.Title) if pd.notna(row.Title) else None
                pids = resolve_ids(title_key, title2pid_list) if title_key else []
                if pids:
                    by_title.update(pids)
                    continue
            unplaced += 1

        by_title -= by_doi          # found by both -> credited to the route that placed it first
        for pid in by_doi:
            pid_source.setdefault(pid, 'doi')
        for pid in by_title:
            pid_source.setdefault(pid, 'title')
        print(f"  [nobel/{domain}] {len(prize_rows)} prize-winning rows -> "
              f"{len(by_doi)} pids by DOI + {len(by_title)} by title = "
              f"{len(by_doi | by_title)}; {unplaced} row(s) placed by neither")

    assert pid_source, (
        "no Nobel paper identified -- check the Nobel records, pid2doi.pkl and pid2title.pkl")
    return pid_source


def build_curves():
    """The full pipeline: Nobel papers -> qualifying population -> (year, venue) cells ->
    within-cell quantiles -> per-metric curves and statistics.

    Returns (curves, nobel_papers, cells): the {short: {...}} ready to cache and draw, the
    one-row-per-Nobel-paper table (taken BEFORE any cut, so the dropped ones stay visible, each
    with the reason), and the per-cell table."""
    pid_source = identify_nobel_pids()
    nobel_pids = set(pid_source)
    n_via_doi = sum(1 for v in pid_source.values() if v == 'doi')
    n_via_title = len(nobel_pids) - n_via_doi

    # ---- the population: disp_full, with venues attached ----
    disp_col = f'disp_y{off}'
    disp_year_tag = '' if off == 5 else f'_y{off}'      # arrange_disruption.py's own convention
    disp_path = os.path.join(disp_dire, f'disp_full{disp_year_tag}{data_suffix}{k_suffix}.csv')
    assert os.path.exists(disp_path), (
        f"{disp_path}\nRun get_disruption.py then arrange_disruption.py first "
        f"(match year_offset / use_with_review / k_require_after_focal).")
    count_cols = [f'I_count_y{off}', f'J_count_y{off}', f'K_count_y{off}']
    read_cols = ['pid', 'pub_year', 'up_refer_count', citation_col, disp_col]
    have = set(pd.read_csv(disp_path, nrows=0).columns)
    if 'title' in have:
        read_cols.insert(1, 'title')
    count_cols = [c for c in count_cols if c in have]
    read_cols += count_cols
    # pid MUST stay a string: WoS ids are digit-like and would lose their leading zeros
    df = pd.read_csv(disp_path, usecols=read_cols, dtype={'pid': str})
    print("\nloaded disp_full:", os.path.basename(disp_path), "rows:", len(df))

    venue_path = os.path.join(meta_dire, f'pid2source_title{data_suffix}.pkl')
    assert os.path.exists(venue_path), (
        f"{venue_path}\nRun wos/get_meta_data_get_venue.py first, with the same "
        f"use_with_review={use_with_review}.")
    with open(venue_path, 'rb') as f:
        pid2source_title = pickle.load(f)
    print(f"  pid2source_title size: {len(pid2source_title):,}")
    df['venue'] = df['pid'].map(pid2source_title)
    df['venue_key'] = df['venue'].map(norm_venue)
    del pid2source_title

    # ---- the two Foundation ratios (see the note in the settings) ----
    efg_path = os.path.join(efg_dire, f'gen_V3_y{off}{data_suffix}.csv')
    assert os.path.exists(efg_path), (
        f"{efg_path}\nRun get_efg.py then get_efg_s2_merge.py first -- the Foundation metrics "
        f"need {f_count_col} and its own {citation_col}.")
    df_f = pd.read_csv(efg_path, usecols=['pid', f_count_col, citation_col],
                       dtype={'pid': str})
    print(f"  loaded gen_V3: {os.path.basename(efg_path)} rows: {len(df_f):,}")
    # C = 0 would be a division by zero; those papers get NaN and drop out of this metric only
    cit = pd.to_numeric(df_f[citation_col], errors='coerce')
    fcnt = pd.to_numeric(df_f[f_count_col], errors='coerce')
    df_f[f'f_over_c_y{off}'] = np.where(cit > 0, fcnt / cit, np.nan)
    n_nan = int(df_f[f'f_over_c_y{off}'].isna().sum())
    print(f"  F/C computed for {len(df_f) - n_nan:,} of {len(df_f):,} gen_V3 rows "
          f"({n_nan:,} have no F count or C = 0)")
    # LEFT join: gen_V3 never adds or removes a paper here, it only supplies columns. The F
    # COUNT rides along too, because F/(C+K) divides it by disp_full's own C and K.
    df = df.merge(df_f[['pid', f'f_over_c_y{off}', f_count_col]], on='pid', how='left')
    print(f"  papers with an F/C value: {int(df[f'f_over_c_y{off}'].notna().sum()):,} "
          f"of {len(df):,}")
    del df_f

    # F/(C+K): the F count over the DISRUPTION denominator. C and K both come from disp_full,
    # so C + K is the very I + J + K that disp_y{off} is divided by.
    k_col = f'K_count_y{off}'
    assert k_col in df.columns, (
        f"{k_col} is not in {os.path.basename(disp_path)}, so F/(C+K) cannot be formed. "
        f"Re-run arrange_disruption.py so the table carries the I/J/K counts, or drop the "
        f"'f_over_ck' entry from METRICS.")
    denom_ck = (pd.to_numeric(df[citation_col], errors='coerce')
                + pd.to_numeric(df[k_col], errors='coerce'))
    df[f'f_over_ck_y{off}'] = np.where(
        denom_ck > 0, pd.to_numeric(df[f_count_col], errors='coerce') / denom_ck, np.nan)
    print(f"  papers with an F/(C+K) value: "
          f"{int(df[f'f_over_ck_y{off}'].notna().sum()):,} of {len(df):,} "
          f"(C+K == 0 cannot occur while citation_threshold >= 1)")
    print(f"  papers with a known venue: {int(df['venue_key'].notna().sum()):,} of {len(df):,} "
          f"({df['venue_key'].nunique():,} distinct venues)")

    # the Nobel rows BEFORE any cut, so the per-paper CSV can carry a reason for each drop
    nobel_papers = df[df['pid'].isin(nobel_pids)].copy()
    nobel_papers['matched_by'] = nobel_papers['pid'].map(pid_source)
    print(f"distinct Nobel pids: {len(nobel_pids)} ({n_via_doi} by DOI, {n_via_title} by title)"
          f" | found in disp_full: {len(nobel_papers)}")

    # ---- 1. the usual thresholds ----
    n0 = len(df)
    df = df[
        (df['up_refer_count'] >= reference_threshold) &
        (df[citation_col] >= citation_threshold) &
        (df['pub_year'] < max_year)
    ].copy()
    print(f"threshold filter: {n0} -> {len(df)} papers (removed {n0 - len(df)})")
    assert len(df), "the qualifying population is empty -- check the thresholds and disp_full"

    # ---- 2. a paper with no venue names no cell ----
    n1 = len(df)
    df = df[df['venue_key'].notna()].copy()
    print(f"venue known    : {n1} -> {len(df)} papers (removed {n1 - len(df)} with no "
          f"source_title)")

    # ---- 3. keep only the cells a Nobel paper sits in ----
    # A cell is one (pub_year, venue). Every 'other' paper left after this shares BOTH with some
    # Nobel paper, which is the whole point: nothing is compared across years or across venues.
    df['is_nobel'] = df['pid'].isin(nobel_pids)
    cell_cols = ['pub_year', 'venue_key']
    # membership through a MultiIndex rather than python tuples: the corpus is millions of rows
    cell_idx = pd.MultiIndex.from_frame(df[cell_cols])
    nobel_cell_idx = pd.MultiIndex.from_frame(
        df.loc[df['is_nobel'], cell_cols].drop_duplicates())
    n2 = len(df)
    df = df[cell_idx.isin(nobel_cell_idx)].copy()
    print(f"Nobel cells    : {n2} -> {len(df)} papers in {len(nobel_cell_idx):,} "
          f"(pub_year, venue) cell(s) holding a Nobel paper")
    assert len(df), "no (pub_year, venue) cell holds a Nobel paper"

    # ---- 4. drop the cells too small for a rank to mean anything ----
    sizes = df.groupby(cell_cols)['pid'].transform('size')
    df['cell_size'] = sizes
    n3 = len(df)
    n_nobel_before = int(df['is_nobel'].sum())
    small = sizes < MIN_CELL_SIZE
    n_small_cells = int(df.loc[small, cell_cols].drop_duplicates().shape[0])
    df = df[~small].copy()
    n_cells = int(df[cell_cols].drop_duplicates().shape[0])
    print(f"cell >= {MIN_CELL_SIZE:<4}  : {n3} -> {len(df)} papers; {n_cells:,} cell(s) kept, "
          f"{n_small_cells:,} dropped for holding < {MIN_CELL_SIZE} papers "
          f"(Nobel papers: {n_nobel_before} -> {int(df['is_nobel'].sum())})")

    # How many Nobel papers are being ranked against almost nobody. A paper alone in its cell
    # scores 1.0 by definition -- no comparison took place -- so these counts are the caveat the
    # reported mean has to be read against. See MIN_CELL_SIZE in the settings.
    nob_sizes = df.loc[df['is_nobel'], 'cell_size']
    n_nobel_alone = int((nob_sizes == 1).sum())
    n_nobel_tiny = int((nob_sizes < 5).sum())
    if n_nobel_alone or n_nobel_tiny:
        print(f"  NOTE {n_nobel_alone} Nobel paper(s) are ALONE in their cell (quantile 1.0 by "
              f"definition, no comparison made) and {n_nobel_tiny} sit in a cell of fewer than "
              f"5 papers; raise MIN_CELL_SIZE to exclude them")
    assert len(df), (
        f"every cell holds fewer than MIN_CELL_SIZE={MIN_CELL_SIZE} papers; lower it.")

    # ---- 5. the within-cell quantiles, and the two distributions ----
    xs = np.linspace(0.0, 1.0, N_KDE_POINTS)
    curves = {}
    quant_cols = {}
    for short, col, label, _short_label, _colour in METRICS:
        vals = pd.to_numeric(df[col], errors='coerce')
        # a paper with no value is out of BOTH its own quantile and its cell-mates' ranks
        ok = vals.notna()
        sub = df[ok]
        q = vals[ok].groupby([sub['pub_year'], sub['venue_key']]).rank(pct=True,
                                                                      method='average')
        qcol = f'{short}_cell_quantile'
        quant_cols[qcol] = q
        is_nob = sub['is_nobel'].to_numpy()
        nob_q = q.to_numpy()[is_nob]
        oth_q = q.to_numpy()[~is_nob]
        assert len(nob_q), f"no Nobel paper has a finite {col}"

        rng = np.random.default_rng(KDE_SEED)
        nob_ys, nob_how = kde_curve(nob_q, xs)
        if len(oth_q) == 0 or not DRAW_OTHERS:
            # not drawn -> not computed: the others' KDE is the expensive half of this script
            oth_ys, oth_how, n_oth_kde = np.zeros_like(xs), 'not drawn', 0
        else:
            kde_src = oth_q
            if KDE_MAX_POINTS is not None and len(oth_q) > KDE_MAX_POINTS:
                kde_src = rng.choice(oth_q, size=KDE_MAX_POINTS, replace=False)
                print(f"  [{short}] the 'others' curve is built from a {KDE_MAX_POINTS:,}-value "
                      f"subsample of {len(oth_q):,} (every statistic still uses all of them)")
            n_oth_kde = len(kde_src)
            oth_ys, oth_how = kde_curve(kde_src, xs)

        # Mann-Whitney U: are the Nobel ranks stochastically larger than the others'? Reported
        # two-sided, with the rank-biserial effect size (2*AUC-1, i.e. 0 = no separation).
        if len(oth_q):
            u_stat, p_two = mannwhitneyu(nob_q, oth_q, alternative='two-sided')
            auc = float(u_stat) / (len(nob_q) * len(oth_q))
        else:
            u_stat, p_two, auc = float('nan'), float('nan'), float('nan')

        curves[short] = {
            'metric': short, 'metric_label': label, 'value_col': col,
            'xs': xs,
            'nobel_ys': nob_ys, 'nobel_kde_mode': nob_how,
            'other_ys': oth_ys, 'other_kde_mode': oth_how, 'n_other_kde_used': int(n_oth_kde),
            'n_nobel': int(len(nob_q)), 'n_other': int(len(oth_q)),
            'n_cells': n_cells, 'min_cell_size': MIN_CELL_SIZE,
            'n_small_cells_dropped': n_small_cells,
            # the caveat the mean must be read against when MIN_CELL_SIZE is low
            'n_nobel_alone_in_cell': n_nobel_alone, 'n_nobel_cell_under_5': n_nobel_tiny,
            'n_nobel_pids_by_doi': int(n_via_doi), 'n_nobel_pids_by_title': int(n_via_title),
            # the funnel, so the closing summary can show it on the cached path too
            'n_nobel_identified': int(len(nobel_pids)),
            'n_nobel_in_disp_full': int(len(nobel_papers)),
            'nobel_mean': float(nob_q.mean()), 'nobel_median': float(np.median(nob_q)),
            'nobel_std': float(nob_q.std(ddof=1)) if len(nob_q) > 1 else float('nan'),
            'other_mean': float(oth_q.mean()) if len(oth_q) else float('nan'),
            'other_median': float(np.median(oth_q)) if len(oth_q) else float('nan'),
            'other_std': (float(oth_q.std(ddof=1)) if len(oth_q) > 1 else float('nan')),
            'nobel_share_above_0_5': float((nob_q > 0.5).mean()),
            'nobel_share_above_0_9': float((nob_q > 0.9).mean()),
            'other_share_above_0_5': float((oth_q > 0.5).mean()) if len(oth_q) else float('nan'),
            'other_share_above_0_9': float((oth_q > 0.9).mean()) if len(oth_q) else float('nan'),
            'mannwhitney_u': float(u_stat), 'mannwhitney_p_two_sided': float(p_two),
            'auc_nobel_over_other': auc,
            'rank_biserial': 2.0 * auc - 1.0 if auc == auc else float('nan'),
        }

    for qcol, q in quant_cols.items():
        df[qcol] = q

    # ---- the per-cell table ----
    # Built with plain aggregations and merges rather than groupby().apply(): apply's
    # include_groups argument only exists in recent pandas, and this has to run on the server's.
    main_q = f'{METRICS[0][0]}_cell_quantile'
    cells = df.groupby(cell_cols, as_index=False).agg(
        n_papers=('pid', 'size'), n_nobel=('is_nobel', 'sum'))
    # the venue's most common raw spelling, for reading
    names = (df.groupby(cell_cols)['venue']
               .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else '')
               .rename('venue').reset_index())
    nob_mean = (df[df['is_nobel']].groupby(cell_cols)[main_q].mean()
                .rename('nobel_mean_quantile').reset_index())
    oth_mean = (df[~df['is_nobel']].groupby(cell_cols)[main_q].mean()
                .rename('other_mean_quantile').reset_index())
    for part in (names, nob_mean, oth_mean):
        cells = cells.merge(part, on=cell_cols, how='left')
    cells['n_nobel'] = cells['n_nobel'].astype(int)
    cells = (cells.sort_values(['n_nobel', 'n_papers'], ascending=False)
                  .reset_index(drop=True))
    print(f"\n{len(cells)} cell(s); the 10 holding the most Nobel papers:")
    print(cells.head(10).to_string(index=False))

    # ---- the Nobel per-paper table: quantile where there is one, reason where there is not ----
    assert df['pid'].is_unique, "disp_full holds duplicate pids -- the per-paper join is unsafe"
    kept = df.set_index('pid')
    nobel_papers['cell_size'] = nobel_papers['pid'].map(kept['cell_size'])
    for short, col, *_rest in METRICS:
        qcol = f'{short}_cell_quantile'
        nobel_papers[qcol] = nobel_papers['pid'].map(kept[qcol])
    nobel_papers['dropped_reason'] = _drop_reasons(nobel_papers, set(kept.index))
    n_in = int(nobel_papers['dropped_reason'].eq('').sum())
    print(f"\nNobel papers in disp_full: {len(nobel_papers)} | in the analysis: {n_in}")
    print(nobel_papers['dropped_reason'].value_counts().to_string())

    return curves, nobel_papers, cells


def _drop_reasons(frame, kept_pids):
    """Why each Nobel paper is not in the analysis -- '' for the ones that are.

    The cuts are tested in the order the pipeline applies them, so a paper that fails several
    is reported under the FIRST one, not an arbitrary one. Vectorised on purpose: itertuples
    mangles a column named 'citation_count_y-1' into a positional name.
    """
    reason = pd.Series('', index=frame.index, dtype=object)
    todo = ~frame['pid'].isin(kept_pids)
    for test, text in [
        (frame['up_refer_count'] < reference_threshold, 'below reference_threshold'),
        (frame[citation_col] < citation_threshold, 'below citation_threshold'),
        (frame['pub_year'] >= max_year, f'pub_year >= {max_year}'),
        (frame['venue_key'].isna(), 'no venue'),
    ]:
        hit = todo & test.fillna(True)
        reason[hit] = text
        todo = todo & ~hit
    reason[todo] = f'cell smaller than {MIN_CELL_SIZE}'
    return reason


# ==================================================
# 1. the curves + statistics: load the cache, or build and save it
# ==================================================
curves = nobel_papers = cells = None
if not FORCE_RECOMPUTE and os.path.exists(cache_path):
    with open(cache_path, 'rb') as f:
        cc = pickle.load(f)
    if cc.get('key') == cache_key:
        curves, nobel_papers, cells = cc['curves'], cc['nobel_papers'], cc['cells']
        print("\nloaded cache -> redraw only (no table is read):", cache_path)
    else:
        print("\ncache found but its key differs -> rebuilding:", cache_path)

if curves is None:
    curves, nobel_papers, cells = build_curves()
    with open(cache_path, 'wb') as f:
        pickle.dump({'key': cache_key, 'curves': curves, 'nobel_papers': nobel_papers,
                     'cells': cells}, f, pickle.HIGHEST_PROTOCOL)
    print("\nwrote cache:", cache_path)


def _ordered(frame):
    """The per-paper columns in a fixed, readable order, whichever of them the frame has."""
    lead = [c for c in ['pid', 'title', 'venue', 'pub_year', 'up_refer_count', citation_col,
                        f'disp_y{off}', f'f_count_y{off}', f'f_over_c_y{off}',
                        f'f_over_ck_y{off}', f'I_count_y{off}',
                        f'J_count_y{off}', f'K_count_y{off}', 'cell_size'] +
            [f'{m[0]}_cell_quantile' for m in METRICS] +
            ['matched_by', 'dropped_reason']
            if c in frame.columns]
    rest = [c for c in frame.columns if c not in lead and c != 'venue_key']
    return frame[lead + rest]


nobel_csv = os.path.join(out_dir, f'nobel_papers_{param_suffix}.csv')
_ordered(nobel_papers).sort_values('pid').to_csv(nobel_csv, index=False)
print("saved per-paper Nobel table:", nobel_csv, f"({len(nobel_papers)} rows)")

cells_csv = os.path.join(out_dir, f'cells_{param_suffix}.csv')
cells.to_csv(cells_csv, index=False)
print("saved the cell table:", cells_csv, f"({len(cells)} cells)")

missing = [m[0] for m in METRICS if m[0] not in curves]
assert not missing, f"cache {cache_path} has no curve for {missing}; delete it to rebuild"

# ==================================================
# 2. report, draw and tabulate
# ==================================================
rows = []
for short, col, label, short_label, colour in METRICS:
    c = curves[short]

    print("\n" + "=" * 78)
    print(f"{label}   [{col}]   ranked within (pub_year, venue)")
    print("=" * 78)
    print(f"  cells used                     : {c['n_cells']:,} "
          f"(>= {c['min_cell_size']} papers each; {c['n_small_cells_dropped']:,} dropped)")
    if c.get('n_nobel_alone_in_cell') is not None:
        print(f"  Nobel alone in cell / cell < 5 : {c['n_nobel_alone_in_cell']:,} / "
              f"{c['n_nobel_cell_under_5']:,}   (a paper alone in its cell scores 1.0 by "
              f"definition)")
    print(f"  Nobel papers                   : {c['n_nobel']:,}")
    print(f"  other papers in the same cells : {c['n_other']:,}")
    print(f"  Nobel  mean / median quantile  : {c['nobel_mean']:.4f} / {c['nobel_median']:.4f}")
    print(f"  others mean / median quantile  : {c['other_mean']:.4f} / {c['other_median']:.4f}")
    print(f"    (the others are near-uniform by construction -- ranks inside a cell fill the "
          f"unit interval whatever the data says; that is the baseline, not a finding)")
    print(f"  Nobel share above 0.5 / 0.9    : {c['nobel_share_above_0_5']:.4f} / "
          f"{c['nobel_share_above_0_9']:.4f}")
    print(f"  others      above 0.5 / 0.9    : {c['other_share_above_0_5']:.4f} / "
          f"{c['other_share_above_0_9']:.4f}")
    print(f"  Mann-Whitney U (two-sided)     : U={c['mannwhitney_u']:,.0f}  "
          f"p={c['mannwhitney_p_two_sided']:.3e}")
    print(f"    P(Nobel > other)             : {c['auc_nobel_over_other']:.4f}  "
          f"(rank-biserial {c['rank_biserial']:+.4f})")
    print("=" * 78)

    row = {k: v for k, v in c.items() if k not in ('xs', 'nobel_ys', 'other_ys')}
    row.update({
        'reference_threshold': reference_threshold,
        'citation_threshold': citation_threshold,
        'max_year': max_year,
        'year_offset': year_offset,
        'use_with_review': use_with_review,
        'k_require_after_focal': k_require_after_focal,
        'require_unique_match': require_unique_match,
        'use_title_fallback': USE_TITLE_FALLBACK,
        'min_title_chars': MIN_TITLE_CHARS,
    })
    rows.append(row)

# ==================================================
# 3. ONE FIGURE PER METRIC: that metric's own Nobel quantile density, marked with its own
#    Nobel mean (dashed) and median (dash-dotted), drawn in that metric's colour. The panels
#    share the [0, 1] quantile axis, so they can still be read against each other side by side.
# ==================================================
for short, col, label, short_label, colour in METRICS:
    c = curves[short]
    xs = np.asarray(c['xs'])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(xs, np.asarray(c['nobel_ys']), color=colour, linewidth=2.0)

    series = [
        {'name': f"{short_label} mean (n={c['n_nobel']:,})", 'color': colour,
         'mean': c['nobel_mean'], 'style': '--', 'width': 2.0},
        {'name': f'{short_label} median', 'color': colour,
         'mean': c['nobel_median'], 'style': '-.', 'width': 1.6},
    ]
    if DRAW_OTHERS and c['n_other']:
        ax.plot(xs, np.asarray(c['other_ys']), color=OTHER_COLOR, linewidth=2.0)
        series.append({'name': f"Others (n={c['n_other']:,})", 'color': OTHER_COLOR,
                       'mean': c['other_mean'], 'style': '--', 'width': 1.2})
    for sr in series:
        ax.axvline(sr['mean'], color=sr['color'], linestyle=sr['style'], linewidth=sr['width'],
                   alpha=0.9)
    ax.set_xlim(0, 1)

    # No legend: each marker is named by coloured text beside its own line, the way
    # award_quantile_analysis.py's draw_dist_textlabel() does it -- the smallest value puts its
    # text to the LEFT of its line, the rest to the RIGHT.
    #
    # Two additions that matter here but not there. (a) These are QUANTILES, so a marker can sit
    # at 0.95 and its right-hand text would run straight off the axes: a side with less than
    # LABEL_ROOM of the axis left is flipped to the other one. (b) A flip can put several labels
    # on the same side, so each gets its own horizontal band and they cannot overlap whatever
    # the values are.
    LABEL_ROOM = 0.30                              # axis fraction a label needs beside it
    BANDS = [0.97, 0.80, 0.63, 0.46]
    assert len(series) <= len(BANDS), (
        f"{len(series)} labels but only {len(BANDS)} bands -- add bands")
    order = sorted(range(len(series)), key=lambda k: series[k]['mean'])
    for rank, (band, k) in enumerate(zip(BANDS, order)):
        sr = series[k]
        side = 'left' if rank == 0 else 'right'    # smallest value left, the rest right
        if side == 'right' and (1.0 - sr['mean']) < LABEL_ROOM:
            side = 'left'
        elif side == 'left' and sr['mean'] < LABEL_ROOM:
            side = 'right'
        ha, dx = ('left', 0.012) if side == 'right' else ('right', -0.012)
        ax.text(sr['mean'] + dx, band, f"{sr['name']}\n{sr['mean']:.3f}",
                transform=ax.get_xaxis_transform(), color=sr['color'],
                fontsize=MARKER_LABEL_FONTSIZE, fontweight='bold', ha=ha, va='top')

    # The axis is the quantile the markers live on, and that is all it says. Which metric this
    # panel is about is in its marker labels, its filename, the printout and the CSV.
    ax.set_xlabel('Quantile within (publication year, venue)', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel('Density', fontsize=AXIS_LABEL_FONTSIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    fpath = os.path.join(fig_dir, f'nobel_disp_compare_v2_{short}_{param_suffix}.svg')
    fig.savefig(fpath, format='svg', transparent=True)
    fig.savefig(fpath[:-4] + '.png', dpi=150)
    plt.close(fig)
    print(f"\nsaved figure [{short}]:", fpath)
    print("  markers:", ", ".join(f"{sr['name'].split(' (')[0]}={sr['mean']:.3f}"
                                  for sr in series))

pd.DataFrame(rows).to_csv(out_csv, index=False)
print("\nsaved:", out_csv)

# ---- the group sizes, last thing on the screen, on the cached path too ----
print("\n" + "=" * 78)
print("POPULATION  (papers sharing a publication year AND a venue with a Nobel paper)")
for short, col, label, short_label, colour in METRICS:
    c = curves[short]
    print(f"  [{short}]  Nobel prize papers: {c['n_nobel']:,}   "
          f"other papers: {c['n_other']:,}   "
          f"total: {c['n_nobel'] + c['n_other']:,}   cells: {c['n_cells']:,}")
    # where the Nobel papers went, for the ones that did not make it this far. Guarded with
    # .get() so a cache written before these fields existed still redraws instead of failing.
    n_id, n_disp = c.get('n_nobel_identified'), c.get('n_nobel_in_disp_full')
    if n_id is not None and n_disp is not None:
        # the cell-size clause is only a real cut when MIN_CELL_SIZE > 1
        cell_clause = (f"sat in a cell smaller than {c['min_cell_size']}, "
                       if c['min_cell_size'] > 1 else "")
        print(f"           Nobel funnel: {n_id:,} identified from the prize records -> "
              f"{n_disp:,} found in disp_full -> {c['n_nobel']:,} in the analysis "
              f"(the rest failed the thresholds, had no venue, {cell_clause}"
              f"or have no {col} value of their own; see the per-paper CSV's dropped_reason, "
              f"which reports the cuts common to every metric)")
print("=" * 78)
print("finished")
