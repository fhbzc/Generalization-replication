"""
Where does the AVERAGE Nobel paper's disruption sit in the whole WoS corpus?

One number, in four steps:
  1. build the qualifying population from the disruption table (disp_full) alone -- every
     paper in it that passes up_refer_count >= reference_threshold,
     citation_count_y{off} >= citation_threshold and pub_year < max_year;
  2. with RESTRICT_TO_NOBEL_VENUES on (the default), narrow that population to the papers
     published in the SAME VENUES as the Nobel papers themselves -- if the prize papers come
     out of 10 journals, only those 10 journals' papers stay. See the setting for why, and for
     what happens to a Nobel paper whose venue is unknown;
  3. average the raw disruption index D = disp_y{off} over the Nobel-prize papers among them;
  4. report the PERCENTILE RANK of that single average within the disp distribution of all the
     papers from step 2 -- the fraction of them whose disp is at or below it -- and DRAW it:
     a Gaussian KDE of ALL those papers, the area at or below the Nobel mean shaded, and four
     vertical markers: the Nobel mean and median, and the whole-population mean and median.

The four marker labels are laid out on separate horizontal bands, in left-to-right order of
their values, so two markers that land on nearly the same x still cannot overlap in text.

disp_full carries all three filter columns itself -- pub_year, up_refer_count and
citation_count_y{off} (= I + J, the papers citing the focal one inside the same window that
produced disp) -- so nothing else has to be read. In particular gen_V3 is NOT read: it would
contribute only its own citation_count_y{off}, computed by get_efg.py over ITS year_offset,
which is a different window whenever the two runs disagree. Taking the filter columns from
disp_full keeps the filter and the index on one and the same citation window.

The ranking is computed from raw disp, so none of compute_efgd_quantile.py's precomputed
quantile columns is needed either, and its whole disp_f branch is irrelevant (hence no
use_disp_f setting below). The filters are applied HERE, and the settings that select the input
file (use_with_review, k_require_after_focal, year_offset) have to match the get_disruption.py /
arrange_disruption.py run you want.

NOTE this population is disp_full's, not compute_efgd_quantile.py's: that script filters gen_V3
and then inner-joins disp_full, so its rows are the INTERSECTION of the two tables. The
difference is whatever disp_full holds that gen_V3 does not -- the two scripts guard the focal
year slightly differently (get_efg.py skips pub_year >= max_year, get_disruption.py skips
pub_year > 2025) -- so the counts here can be a little larger than that table's.

EVERYTHING the figure and the CSV need -- the KDE curve AND every statistic -- is cached to a
small pkl keyed by all of the settings below. A re-run with an unchanged key loads that and only
redraws: it never touches disp_full, pid2doi or the Nobel records again, which is the
expensive half. Delete the pkl, or set FORCE_RECOMPUTE, to rebuild.

Note what step 3 is and is not. It locates ONE summary value (the Nobel mean) on the corpus-wide
disp axis; it is NOT the mean of the Nobel papers' own percentile ranks, which is a different
statistic and is what award_quantile_analysis.py plots. Both are reported so the two can never be
confused with each other (the per-paper ranks use the same rank(pct=True, method='average')
compute_efgd_quantile.py uses for its _quantile_global column, over the same population, so they
agree with it).

Nobel papers are identified from the rows of the publication records flagged
'Is prize-winning paper' == 'YES', by TWO keys in order:

  1. DOI  -- canonicalised by norm_doi, looked up in pid2doi.pkl. Preferred, because a DOI is
             an exact identifier.
  2. TITLE -- canonicalised by norm_title (lower-cased, accents folded, punctuation stripped,
             whitespace collapsed), looked up in pid2title.pkl. Tried ONLY for a row whose DOI
             found nothing.

The title fallback exists because DOI alone barely works on WoS for this question: the
prize-winning papers have a MEDIAN publication year around 1957, while WoS carries a DOI
essentially only from ~1998 on (Crossref launched in 2000), so a DOI-only join can reach only
the ~30 prize-winning papers published since 1998 -- everything older is invisible to it.
Set USE_TITLE_FALLBACK = False to get the DOI-only behaviour back; its outputs are tagged
'_doionly' so the two never overwrite each other.

Both lookups honour require_unique_match (below): with it off, a key claimed by several pids
contributes ALL of them instead of being dropped, and every output name gains a '_multi_match'
tag. Titles are far more collision-prone than DOIs, so leaving require_unique_match ON is what
keeps a generic title ("Nuclear magnetic resonance") from dragging in unrelated papers; titles
shorter than MIN_TITLE_CHARS are never used as a key at all. How many papers each route
contributed is printed per file and recorded in the output CSV.

Papers with no disp value are already excluded by the inner join; any remaining non-finite value
is dropped from BOTH the Nobel mean and the reference distribution, so the two always describe
the same population. The venue restriction of step 2 is applied to that same single population,
after the thresholds and before any statistic is computed, so the Nobel group, the reference
distribution, the KDE curve, the per-cohort figure and the 'other smallest' tail all describe
the venue-restricted set together -- there is no version of it that mixes the two.

Input  : META_DIRE/disruption/disp_full[_y{off}]{ds}{ks}.csv   (the _y tag only when off != 5)
             (arrange_disruption.py; supplies BOTH the disruption index disp_y{off} and the
              three filter columns pub_year / up_refer_count / citation_count_y{off})
         META_DIRE/pid2doi.pkl        (from wos/get_meta_data_doi.py; RUN IT FIRST)
         META_DIRE/pid2title.pkl      (from wos/get_meta_data.py; the title fallback's index)
         META_DIRE/pid2source_title{ds}.pkl
                                      (from wos/get_meta_data_get_venue.py, RUN IT FIRST with
                                       the same use_with_review; only when
                                       RESTRICT_TO_NOBEL_VENUES is on)
         NOBEL_DATA_DIRE/<nobel csv>  (EXTERNAL Nobel publication records; NOBEL_DATA_DIRE is
                                       META_DIRE/nobel_prize, see utils.py)
Cache  : ./figures/nobel_disp_mean_quantile/
           nobel_disp_kde_cache_{param_suffix}.pkl   (the KDE curves + every statistic)
Output : META_DIRE/nobel_disp_mean_quantile/
           nobel_disp_mean_quantile_{param_suffix}.csv
         one row per metric, holding the Nobel mean, its percentile rank, the sample sizes and
         the corpus-wide reference statistics.
           nobel_papers_{param_suffix}.csv
         one row per IDENTIFIED Nobel paper: pid, title, pub_year, up_refer_count,
         citation_count_y{off}, disp_y{off}, the I/J/K counts D is built from, matched_by
         (doi | title) and passed_filter -- the last saying whether the paper survived the
         rt/ct/pub_year thresholds and so entered the statistics and the figure. Rows are
         taken before the filter, so the dropped ones are visible too.
           nobel_venues_{param_suffix}.csv
         (only when RESTRICT_TO_NOBEL_VENUES is on) one row per venue the population was cut
         down to: its name, how many papers it contributes to the reference distribution, and
         how many of those are Nobel papers. This is the list the restriction is built from.
           other_smallest{N_SMALLEST_OTHER}_{param_suffix}.csv
         the N_SMALLEST_OTHER most negative-disruption papers among the OTHERS (the non-Nobel
         half of the population the figure draws), same columns minus matched_by /
         passed_filter. The two files are meant to be read side by side: one is the group the
         red curve is made of, the other is the far tail of the blue one.
           nobel_disp_cohorts_{metric}_{param_suffix}.csv
         one row per publication-year cohort: its year range, sizes, the cohort's own mean and
         median, and the Nobel mean/median inside it together with their percentile AMONG THAT
         COHORT's papers.
         ./figures/nobel_disp_mean_quantile/
           nobel_disp_mean_quantile_{metric}_{param_suffix}.svg  (+ .png)
           nobel_disp_cohorts_{metric}_{param_suffix}.svg        (+ .png)
         the second figure being one subplot per cohort (COHORT_WIDTH years each, the last one
         short if the data ends mid-cohort), each drawn like the main figure but restricted to
         that cohort -- so the Nobel papers are read against their own contemporaries.
         {param_suffix} = y{off}{ds}{ks}_rt_{rt}_ct_{ct}_ybefore_{max_year}{venue_tag}
                          {match_tag}{title_tag}
         -- venue_tag is '_nobelvenue' when the population is restricted to the Nobel papers'
         own venues and '' otherwise, so a restricted run never overwrites a whole-corpus one

Usage  : python nobel_disp_mean_quantile.py
"""

import os
import re
import sys
import pickle
import unicodedata
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
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

# ---- must match the runs that produced the two raw tables ----
# There is deliberately NO use_disp_f here: that flag only selects which of compute_efgd_
# quantile.py's outputs carries the disp_f columns, and this script reads neither of them.
year_offset = -1
reference_threshold = 1
citation_threshold = 1
# must match get_disruption.py: read the '_with_review' variant (the
# doctypes=='review'-inclusive network) when True. Part of the disp_full filename.
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

# ---- Nobel-prize publication-record files (same list as award_quantile_analysis.py) ----
NOBEL_FILES = [
    {'file': 'Medicine publication record.csv',  'domain': 'Medicine'},
    {'file': 'Chemistry publication record.csv', 'domain': 'Chemistry'},
    {'file': 'Physics publication record.csv',   'domain': 'Physics'},
]

# The metric(s) to locate. The question is about disruption, so D is the only entry; the list
# shape means another disp_full column (I/J/K-derived, say) is a one-line addition.
METRICS = [
    ('disp', f'disp_y{off}', 'Disruption D = (I-J)/(I+J+K)'),
]

# ---- figure / KDE ----
# The distribution is drawn as a Gaussian KDE, exactly as award_quantile_analysis.py draws its
# panels, so the two figures are directly comparable.
N_KDE_POINTS = 400                 # grid points the density is evaluated at, across X_LIM
X_LIM = (-1.0, 1.0)                # D = (I-J)/(I+J+K) is bounded by [-1, 1] by construction
# gaussian_kde costs O(n_points x N_KDE_POINTS) and the corpus is large, so the CURVE is built
# from at most this many values (drawn once, with KDE_SEED). Every NUMBER -- the Nobel mean, its
# percentile, all the reference statistics -- is always computed on the FULL data, so this
# affects only the drawn line, and at plotting resolution a few million values are
# indistinguishable from all of them. Set to None to use every value.
KDE_MAX_POINTS = 2_000_000
KDE_SEED = 20260906
USE_LOG_Y = False                  # True -> log density axis. The KDE smooths disp's mass at
                                   # exactly 0 into a finite bump, so the linear axis stays
                                   # readable AND keeps area proportional to probability (the
                                   # shaded region then literally IS the percentile). Switch it
                                   # on only to read the far tails.
# marker line styles: mean dashed, median dash-dotted, in each group's colour
# Main-figure type sizes. With no legend and no title the panel carries only three pieces of
# text -- the two axis labels and the two Nobel annotations -- so both are set well above the
# matplotlib default, and both are here rather than inline so they are easy to retune.
AXIS_LABEL_FONTSIZE = 16           # the x and y axis labels
MARKER_LABEL_FONTSIZE = 14         # the 'Nobel mean' / 'Nobel median' annotations

# ---- the per-cohort figure ----
# A second figure splits the population into publication-year cohorts and repeats the whole
# reading inside each one: that decade's density, that decade's Nobel mean and median, and the
# percentile of each WITHIN THAT DECADE. It answers a different question from the main figure --
# not "where do Nobel papers sit in the corpus" but "where do they sit among their own
# contemporaries" -- which matters because disp drifts with publication year.
COHORT_START_YEAR = 1945           # get_disruption.py already drops anything earlier; any
                                   # stragglers before this fold into the first cohort
COHORT_WIDTH = 10                  # years per cohort; the last one is however much is left
COHORT_NCOLS = 3                   # subplot grid width
ALL_COLOR = 'tab:blue'             # the whole qualifying population (the drawn curve)
NOBEL_COLOR = 'tab:red'            # the Nobel papers' markers
FORCE_RECOMPUTE = False            # ignore any existing cache and rebuild from the raw tables

# Per-paper CSVs (see Output below): every Nobel paper, and the N most negative-disruption
# papers among the OTHERS -- the tail of the blue curve, for eyeballing against the Nobel rows.
N_SMALLEST_OTHER = 100

# ---- how a prize record is matched to a WoS paper ----
# DOI first, then title for the rows the DOI could not place. See the header for why the title
# route is not optional in practice on WoS.
USE_TITLE_FALLBACK = True
title_tag = '' if USE_TITLE_FALLBACK else '_doionly'
# a normalised title shorter than this is never used as a key: too generic to identify a paper
MIN_TITLE_CHARS = 10

# ---- id-matching strictness --------------------------------------------------------------
# A Nobel paper is located through its DOI, and ONE DOI can be claimed by SEVERAL pids
# (duplicate records, or the same work indexed twice).
#   require_unique_match = True  (default) -> only a ONE-TO-ONE DOI counts; a DOI claimed by
#                                            several pids is DROPPED, so no paper is admitted
#                                            on a guess.
#   require_unique_match = False           -> EVERY pid claiming the DOI is taken, so ambiguous
#                                            DOIs contribute all of their papers.
# match_tag is baked into every output name so the two variants never overwrite each other.
require_unique_match = False
match_tag = '' if require_unique_match else '_multi_match'

# ---- which papers the Nobel ones are ranked AGAINST --------------------------------------
# By default the reference population is every qualifying WoS paper, i.e. a prize paper is read
# against the whole corpus -- including the vast majority of venues no prize-winning work has
# ever appeared in, whose papers therefore say little about how unusual a Nobel paper is.
#
# With RESTRICT_TO_NOBEL_VENUES on, the population is cut down to the papers published in the
# SAME venues (WoS 'source_title') as the Nobel papers themselves: if the prize papers come out
# of 10 journals, only those 10 journals' papers are ranked against. The percentile then
# answers "where does this sit among the papers that could plausibly have been the prize
# paper", rather than "among all of WoS".
#
# The venue set is taken from the Nobel papers that SURVIVE the rt/ct/pub_year filter, so every
# one of them is inside the restricted population by construction -- the cut never removes a
# Nobel paper. The one exception is a Nobel paper whose source_title is missing in the WoS
# parquet: it names no venue, so it is dropped together with the other venue-less papers and
# the count is printed.
#
# Venues come from wos/get_meta_data_get_venue.py -> pid2source_title{data_suffix}.pkl; ITS
# use_with_review has to match this script's, since data_suffix selects the file.
RESTRICT_TO_NOBEL_VENUES = True
venue_tag = '_nobelvenue' if RESTRICT_TO_NOBEL_VENUES else ''
print("RESTRICT_TO_NOBEL_VENUES", RESTRICT_TO_NOBEL_VENUES, "| venue_tag", repr(venue_tag))
print("require_unique_match", require_unique_match, "| match_tag", repr(match_tag))
print("USE_TITLE_FALLBACK", USE_TITLE_FALLBACK, "| title_tag", repr(title_tag),
      "| MIN_TITLE_CHARS", MIN_TITLE_CHARS)


# ---- DOI normalisation (identical to award_quantile_analysis.py) --------------------------
# DOIs are case-INSENSITIVE by spec, but the two sides of the join disagree in practice: the
# Nobel CSVs carry mixed-case DOIs while get_meta_data_doi.py stores them lower-cased. Both
# sides go through norm_doi() before they meet.
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
# Titles are compared after folding away everything that is not information: case, accents
# (NFKD then ASCII, so 'Schrodinger' and 'Schrödinger' meet), every punctuation and symbol
# (replaced by a space, so 'X-ray' == 'x ray'), and runs of whitespace.
_TITLE_NON_ALNUM = re.compile(r'[^a-z0-9]+')


def norm_title(title):
    """Canonical form for title comparison, or None when the title is too short to identify a
    paper (see MIN_TITLE_CHARS) -- a one-word title would match half the corpus."""
    s = unicodedata.normalize('NFKD', str(title))
    s = s.encode('ascii', 'ignore').decode('ascii').lower()
    s = _TITLE_NON_ALNUM.sub(' ', s).strip()
    return s if len(s) >= MIN_TITLE_CHARS else None


# ---- venue normalisation ----
# Two records of the same journal differ in case and in stray whitespace far more often than in
# anything meaningful, so venues are compared folded: lower-cased, whitespace collapsed. The RAW
# spelling is kept alongside for display.
def norm_venue(venue):
    """Canonical form for venue comparison, or None for a missing / empty source_title."""
    if venue is None or (isinstance(venue, float) and pd.isna(venue)):
        return None
    s = ' '.join(str(venue).split()).lower()
    return s or None


def resolve_ids(key, index):
    """The pids an external id resolves to, honouring require_unique_match.

    `index` is a {canonical id: [pid, ...]} map. Returns [] for an id that is absent, and -- when
    require_unique_match is on -- also for one claimed by several pids."""
    pids = index.get(key) or []
    if require_unique_match and len(pids) != 1:
        return []
    return [str(p) for p in pids]


def kde_curve(vals, xs):
    """Gaussian-KDE density of `vals` on grid `xs` (same helper as award_quantile_analysis.py).

    Returns (ys, how) where `how` records whether the smooth estimate or the histogram fallback
    was used -- gaussian_kde is undefined for fewer than two points or a zero-variance sample."""
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
                f'ybefore_{max_year}{venue_tag}{match_tag}{title_tag}')
out_dir = os.path.join(meta_dire, 'nobel_disp_mean_quantile')
os.makedirs(out_dir, exist_ok=True)
out_csv = os.path.join(out_dir, f'nobel_disp_mean_quantile_{param_suffix}.csv')
# figures + the KDE cache live next to this script, as in award_quantile_analysis.py
fig_dir = os.path.join(BASE_DIR, 'figures', 'nobel_disp_mean_quantile')
os.makedirs(fig_dir, exist_ok=True)
cache_path = os.path.join(fig_dir, f'nobel_disp_kde_cache_{param_suffix}.pkl')
print("param_suffix:", param_suffix)
print("out_csv:", out_csv)
print("fig_dir:", fig_dir)
print("cache:", cache_path)

# Everything that changes a cached number or curve. A mismatch rebuilds rather than silently
# redrawing a stale curve; param_suffix already covers the filters, so these are the rest.
cache_key = {
    'schema': 'nobel_disp_kde_v8_venue',
    'param_suffix': param_suffix,
    'metrics': [m[0] for m in METRICS],
    'value_cols': [m[1] for m in METRICS],
    'n_kde_points': N_KDE_POINTS,
    'x_lim': list(X_LIM),
    'kde_max_points': KDE_MAX_POINTS,
    'kde_seed': KDE_SEED,
    'require_unique_match': require_unique_match,
    'restrict_to_nobel_venues': RESTRICT_TO_NOBEL_VENUES,
    'cohort_start_year': COHORT_START_YEAR,
    'cohort_width': COHORT_WIDTH,
    'use_title_fallback': USE_TITLE_FALLBACK,
    'min_title_chars': MIN_TITLE_CHARS,
    'nobel_files': [c['file'] for c in NOBEL_FILES],
}


def build_curves():
    """The full pipeline: Nobel papers -> qualifying population -> per-metric statistics and
    KDE curves. Returns (curves, nobel_papers, other_smallest, venue_stats): {short: {...}}
    ready to cache, draw and tabulate, plus the one-row-per-Nobel-paper table, the smallest-disp
    slice of the others, and -- when RESTRICT_TO_NOBEL_VENUES is on -- the per-venue table the
    population was cut down to (None otherwise)."""
    # ---- 1. the Nobel papers (before any table is read: this needs none of them) ----
    print("\nLoading pid2doi ...")
    pid2doi_path = os.path.join(meta_dire, 'pid2doi.pkl')
    assert os.path.exists(pid2doi_path), (
        "%s\nRun wos/get_meta_data_doi.py first -- WoS has no DOI map of its own."
        % pid2doi_path)
    with open(pid2doi_path, 'rb') as f:
        pid2doi = pickle.load(f)
    # keys are norm_doi()-canonicalised, so the CSV side must be canonicalised too
    doi2pid_list = defaultdict(list)
    for pid, doi in pid2doi.items():
        doi2pid_list[norm_doi(doi)].append(pid)
    print(f"  pid2doi size: {len(pid2doi)} -> {len(doi2pid_list)} distinct lower-cased DOIs")

    # the title index, used only for the rows the DOI cannot place
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
        n_amb_title = sum(1 for v in title2pid_list.values() if len(v) > 1)
        print(f"  pid2title size: {len(pid2title)} -> {len(title2pid_list)} distinct normalised "
              f"titles ({n_amb_title} shared by >1 paper)")
        del pid2title

    # pid -> which route placed it ('doi' wins; 'title' only fills what DOI could not)
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

        # one prize record at a time, so the DOI-first / title-second order is per PAPER rather
        # than per key set: a row only falls through to its title when its DOI placed nothing
        by_doi, by_title, unplaced = set(), set(), 0
        n_doi_keys, n_title_keys = set(), set()
        for row in prize_rows.itertuples(index=False):
            doi_key = norm_doi(row.DOI) if pd.notna(row.DOI) else None
            pids = []
            if doi_key:
                n_doi_keys.add(doi_key)
                pids = resolve_ids(doi_key, doi2pid_list)
            if pids:
                by_doi.update(pids)
                continue
            if USE_TITLE_FALLBACK:
                title_key = norm_title(row.Title) if pd.notna(row.Title) else None
                if title_key:
                    n_title_keys.add(title_key)
                    pids = resolve_ids(title_key, title2pid_list)
                if pids:
                    by_title.update(pids)
                    continue
            unplaced += 1

        # a paper found by both routes counts as DOI: that is the route that placed it first
        by_title -= by_doi
        for pid in by_doi:
            pid_source.setdefault(pid, 'doi')
        for pid in by_title:
            pid_source.setdefault(pid, 'title')
        print(f"  [nobel/{domain}] {len(prize_rows)} prize-winning rows "
              f"({len(n_doi_keys)} distinct DOIs, {len(n_title_keys)} distinct fallback titles)"
              f" -> {len(by_doi)} pids by DOI + {len(by_title)} by title"
              f" = {len(by_doi | by_title)}; {unplaced} row(s) placed by neither")

    nobel_pids = set(pid_source)
    n_via_doi = sum(1 for v in pid_source.values() if v == 'doi')
    n_via_title = len(nobel_pids) - n_via_doi
    assert nobel_pids, (
        "no Nobel paper identified -- check the Nobel records, pid2doi.pkl and pid2title.pkl")

    # ---- 2. the qualifying population: disp_full, filtered ----
    # disp_full carries the disruption index AND the three filter columns, all computed over the
    # same citation window, so it is the only table this script needs.
    disp_col = f'disp_y{off}'
    # arrange_disruption.py tags the window only when it is not the default 5
    disp_year_tag = '' if off == 5 else f'_y{off}'
    disp_path = os.path.join(disp_dire, f'disp_full{disp_year_tag}{data_suffix}{k_suffix}.csv')
    assert os.path.exists(disp_path), (
        f"{disp_path}\nRun get_disruption.py then arrange_disruption.py first "
        f"(match year_offset / use_with_review / k_require_after_focal).")
    # pid MUST stay a string: WoS ids are digit-like and would otherwise lose their leading
    # zeros and stop matching the pid2doi / pid2title keys.
    # the three counts D = (I - J) / (I + J + K) is built from, so the per-paper CSVs below
    # show where a disp value comes from rather than just its value
    count_cols = [f'I_count_y{off}', f'J_count_y{off}', f'K_count_y{off}']
    read_cols = ['pid', 'pub_year', 'up_refer_count', citation_col, disp_col]
    have = set(pd.read_csv(disp_path, nrows=0).columns)
    if 'title' in have:                       # get_disruption.py writes it; nice to eyeball
        read_cols.insert(1, 'title')
    count_cols = [c for c in count_cols if c in have]
    read_cols += count_cols
    df = pd.read_csv(disp_path, usecols=read_cols, dtype={'pid': str})
    print("\nloaded disp_full:", os.path.basename(disp_path), "rows:", len(df))

    # the venue of every paper, attached BEFORE anything is filtered so the per-paper Nobel
    # table below carries it too (that table is taken pre-filter on purpose)
    venue_stats = None
    venue_path = os.path.join(meta_dire, f'pid2source_title{data_suffix}.pkl')
    if RESTRICT_TO_NOBEL_VENUES:
        assert os.path.exists(venue_path), (
            f"{venue_path}\nRun wos/get_meta_data_get_venue.py first (with the same "
            f"use_with_review={use_with_review}), or set RESTRICT_TO_NOBEL_VENUES = False.")
        with open(venue_path, 'rb') as f:
            pid2source_title = pickle.load(f)
        print(f"  pid2source_title size: {len(pid2source_title):,}")
        df['venue'] = df['pid'].map(pid2source_title)
        df['venue_key'] = df['venue'].map(norm_venue)
        del pid2source_title
        print(f"  papers with a known venue: {int(df['venue_key'].notna().sum()):,} "
              f"of {len(df):,} ({df['venue_key'].nunique():,} distinct venues)")

    # every identified Nobel paper's row, taken BEFORE the filter, so the per-paper CSV can
    # show the ones the thresholds drop as well as the ones that stay
    nobel_papers = df[df['pid'].isin(nobel_pids)].copy()

    n_before = len(df)
    df = df[
        (df['up_refer_count'] >= reference_threshold) &
        (df[citation_col] >= citation_threshold) &
        (df['pub_year'] < max_year)
    ].copy()
    print(f"threshold filter: {n_before} -> {len(df)} papers (removed {n_before - len(df)})")
    assert len(df), "the qualifying population is empty -- check the thresholds and disp_full"

    # ---- 2b. cut the population down to the Nobel papers' own venues ----
    # Done AFTER the thresholds so the venue set is read off the Nobel papers that actually
    # enter the statistics: every one of them is then inside the restricted population and the
    # cut can only ever remove NON-Nobel papers (plus any Nobel paper with no venue at all).
    if RESTRICT_TO_NOBEL_VENUES:
        nob_rows = df[df['pid'].isin(nobel_pids)]
        venue_keys = set(nob_rows['venue_key'].dropna())
        n_nobel_no_venue = int(nob_rows['venue_key'].isna().sum())
        assert venue_keys, (
            f"no Nobel paper in the qualifying population has a known venue -- check "
            f"{venue_path}: it has to come from the same use_with_review run as disp_full.")
        n_before_v = len(df)
        df = df[df['venue_key'].isin(venue_keys)].copy()
        print(f"venue restriction: the {len(nob_rows)} Nobel paper(s) in the population come "
              f"from {len(venue_keys)} distinct venue(s)"
              + (f" ({n_nobel_no_venue} Nobel paper(s) have no venue and are dropped with them)"
                 if n_nobel_no_venue else "")
              + f"; population {n_before_v:,} -> {len(df):,} papers "
                f"(removed {n_before_v - len(df):,})")
        assert len(df), "the venue-restricted population is empty"

        # one row per venue: its most common raw spelling, how many papers it contributes to
        # the reference distribution, and how many of them are Nobel papers
        rows_v = []
        for key, grp in df.groupby('venue_key'):
            names = grp['venue'].mode()
            rows_v.append({
                'venue': names.iloc[0] if len(names) else key,
                'venue_key': key,
                'n_papers': int(len(grp)),
                'n_nobel_papers': int(grp['pid'].isin(nobel_pids).sum()),
            })
        venue_stats = (pd.DataFrame(rows_v)
                       .sort_values(['n_nobel_papers', 'n_papers'], ascending=False)
                       .reset_index(drop=True))
        print(venue_stats.head(20).to_string(index=False))
        if len(venue_stats) > 20:
            print(f"  ... and {len(venue_stats) - 20} more venue(s); see the venues CSV")

    is_nobel = df['pid'].isin(nobel_pids).to_numpy()
    kept = set(df['pid'])
    nobel_papers['matched_by'] = nobel_papers['pid'].map(pid_source)
    nobel_papers['passed_filter'] = nobel_papers['pid'].isin(kept)
    print(f"distinct Nobel pids: {len(nobel_pids)} ({n_via_doi} by DOI, {n_via_title} by "
          f"title) | found in disp_full: {len(nobel_papers)} | passing the filter: "
          f"{int(is_nobel.sum())}")

    # the far tail of the 'other' group: the N most negative-disruption non-Nobel papers.
    # Ties are broken by pid so a re-run gives the same rows in the same order.
    other_smallest = (df[~is_nobel]
                      .dropna(subset=[disp_col])
                      .sort_values([disp_col, 'pid'], kind='mergesort')
                      .head(N_SMALLEST_OTHER)
                      .copy())
    print(f"other (non-Nobel) papers: {int((~is_nobel).sum()):,}; keeping the "
          f"{len(other_smallest)} smallest by {disp_col} "
          f"({other_smallest[disp_col].min():+.6f} .. {other_smallest[disp_col].max():+.6f})")

    # ---- 3. per metric: the statistics (on ALL values) and the two KDE curves ----
    xs = np.linspace(X_LIM[0], X_LIM[1], N_KDE_POINTS)
    rng = np.random.default_rng(KDE_SEED)
    curves = {}
    for short, col, label in METRICS:
        vals = pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=float)
        keep = np.isfinite(vals)
        # any non-finite value drops out of BOTH groups, so the mean and the reference
        # distribution always describe the same population
        all_v = vals[keep]
        nob_v = vals[keep & is_nobel]
        assert len(nob_v) > 0, f"no Nobel paper has a finite {col}"

        nobel_mean = float(nob_v.mean())
        # percentile rank of that ONE value among all qualifying papers. '<=' is the usual
        # convention; '<' is reported too, and the two differ only by exact ties, which a
        # continuous mean essentially never has.
        q_le = float((all_v <= nobel_mean).mean())
        q_lt = float((all_v < nobel_mean).mean())
        # the same reading for the Nobel MEDIAN: where that single value sits on the
        # corpus-wide axis. Ties matter more here -- a median can land exactly on a value many
        # papers share (disp = 0 above all) -- so the '<=' convention is what is reported.
        nobel_median = float(np.median(nob_v))
        q_med = float((all_v <= nobel_median).mean())

        # the OTHER statistic, for contrast: the mean of the Nobel papers' OWN percentile ranks.
        # Same population and the same rank(pct=True, method='average') compute_efgd_quantile.py
        # uses for its _quantile_global column, so this matches what that column would give.
        ranks = pd.Series(all_v).rank(pct=True, method='average').to_numpy()
        mean_of_ranks = float(ranks[is_nobel[keep]].mean())
        del ranks

        # the non-Nobel remainder: not drawn any more, but its mean/n stay in the CSV
        oth_v = vals[keep & ~is_nobel]

        # the drawn curve is the WHOLE population (see KDE_MAX_POINTS for the subsample that
        # only affects the line, never any of the numbers above)
        kde_src = all_v
        n_kde_used = len(all_v)
        if KDE_MAX_POINTS is not None and len(all_v) > KDE_MAX_POINTS:
            kde_src = rng.choice(all_v, size=KDE_MAX_POINTS, replace=False)
            n_kde_used = KDE_MAX_POINTS
            print(f"  [{short}] KDE built from a {KDE_MAX_POINTS:,}-value subsample of "
                  f"{len(all_v):,} (every statistic still uses all of them)")
        print(f"  [{short}] evaluating the KDE on {n_kde_used:,} values ...")
        ys, how = kde_curve(kde_src, xs)

        curves[short] = {
            'metric': short, 'metric_label': label, 'value_col': col,
            'xs': xs, 'ys': ys, 'kde_mode': how, 'n_kde_used': int(n_kde_used),
            'n_all': int(len(all_v)), 'n_nobel': int(len(nob_v)), 'n_other': int(len(oth_v)),
            'n_nobel_pids_by_doi': int(n_via_doi), 'n_nobel_pids_by_title': int(n_via_title),
            'nobel_mean': nobel_mean,
            'quantile_of_nobel_mean': q_le,
            'quantile_of_nobel_mean_strict': q_lt,
            'nobel_median': nobel_median,
            'quantile_of_nobel_median': q_med,
            'nobel_std': float(nob_v.std(ddof=1)) if len(nob_v) > 1 else float('nan'),
            'other_mean': float(oth_v.mean()) if len(oth_v) else float('nan'),
            'other_std': float(oth_v.std(ddof=1)) if len(oth_v) > 1 else float('nan'),
            'all_mean': float(all_v.mean()),
            'all_median': float(np.median(all_v)),
            'all_std': float(all_v.std(ddof=1)) if len(all_v) > 1 else float('nan'),
            'quantile_of_all_mean': float((all_v <= all_v.mean()).mean()),
            'mean_of_nobel_own_quantiles': mean_of_ranks,
            'n_outside_xlim': int(((all_v < X_LIM[0]) | (all_v > X_LIM[1])).sum()),
            'restricted_to_nobel_venues': bool(RESTRICT_TO_NOBEL_VENUES),
            'n_venues': int(len(venue_stats)) if venue_stats is not None else 0,
            'cohorts': _build_cohorts(df, keep, all_v, is_nobel, xs, rng, short),
        }
    return curves, nobel_papers, other_smallest, venue_stats


def _build_cohorts(df, keep, all_v, is_nobel, xs, rng, short):
    """Split the population into publication-year cohorts and read the same statistics inside
    each one. Returns a list of dicts, oldest first; a cohort with no Nobel paper is kept (its
    density is still worth seeing) with NaN Nobel statistics."""
    years = df['pub_year'].to_numpy()[keep]
    nob_keep = is_nobel[keep]
    y_max = int(np.nanmax(years))
    starts = list(range(COHORT_START_YEAR, y_max + 1, COHORT_WIDTH))
    assert starts, f"no cohort covers pub_year <= {y_max}; check COHORT_START_YEAR"

    out = []
    for i, st in enumerate(starts):
        en = st + COHORT_WIDTH                      # exclusive
        in_c = (years >= st) & (years < en)
        if i == 0:
            in_c |= years < st                      # fold anything earlier into the first one
        last = (i == len(starts) - 1)
        cv = all_v[in_c]
        nv = all_v[in_c & nob_keep]
        if len(cv) == 0:
            continue

        # the label states the closed year range actually covered, so a short last cohort is
        # not mistaken for a full decade
        hi = min(en - 1, y_max) if last else en - 1
        entry = {
            'start': int(st), 'end_exclusive': int(en), 'last_partial': bool(last and hi < en - 1),
            'label': f'{st}-{hi}', 'n_all': int(len(cv)), 'n_nobel': int(len(nv)),
            'all_mean': float(cv.mean()), 'all_median': float(np.median(cv)),
        }
        if len(nv):
            entry['nobel_mean'] = float(nv.mean())
            entry['nobel_median'] = float(np.median(nv))
            entry['quantile_of_nobel_mean'] = float((cv <= entry['nobel_mean']).mean())
            entry['quantile_of_nobel_median'] = float((cv <= entry['nobel_median']).mean())
        else:
            for k in ('nobel_mean', 'nobel_median',
                      'quantile_of_nobel_mean', 'quantile_of_nobel_median'):
                entry[k] = float('nan')

        kde_src = cv
        if KDE_MAX_POINTS is not None and len(cv) > KDE_MAX_POINTS:
            kde_src = rng.choice(cv, size=KDE_MAX_POINTS, replace=False)
        entry['xs'] = xs
        entry['ys'], entry['kde_mode'] = kde_curve(kde_src, xs)
        entry['n_kde_used'] = int(len(kde_src))
        out.append(entry)

    print(f"  [{short}] cohorts: {len(out)} of {COHORT_WIDTH} years from "
          f"{COHORT_START_YEAR} ({sum(c['n_nobel'] for c in out)} Nobel papers placed)")
    return out


# ==================================================
# 1. the curves + statistics: load the cache, or build and save it
# ==================================================
curves = nobel_papers = other_smallest = venue_stats = None
if not FORCE_RECOMPUTE and os.path.exists(cache_path):
    with open(cache_path, 'rb') as f:
        cc = pickle.load(f)
    if cc.get('key') == cache_key:
        curves = cc['curves']
        nobel_papers, other_smallest = cc['nobel_papers'], cc['other_smallest']
        venue_stats = cc.get('venue_stats')
        print("\nloaded KDE cache -> redraw only (no table is read):", cache_path)
    else:
        print("\ncache found but its key differs -> rebuilding:", cache_path)

if curves is None:
    curves, nobel_papers, other_smallest, venue_stats = build_curves()
    with open(cache_path, 'wb') as f:
        pickle.dump({'key': cache_key, 'curves': curves, 'nobel_papers': nobel_papers,
                     'other_smallest': other_smallest, 'venue_stats': venue_stats},
                    f, pickle.HIGHEST_PROTOCOL)
    print("\nwrote KDE cache:", cache_path)


def _ordered(frame):
    """The per-paper columns in a fixed, readable order, whichever of them the frame has."""
    lead = [c for c in ['pid', 'title', 'venue', 'pub_year', 'up_refer_count', citation_col,
                        f'disp_y{off}', f'I_count_y{off}', f'J_count_y{off}',
                        f'K_count_y{off}', 'matched_by', 'passed_filter']
            if c in frame.columns]
    # venue_key is the internal fold of 'venue'; the readable spelling is what belongs in a CSV
    rest = [c for c in frame.columns if c not in lead and c != 'venue_key']
    return frame[lead + rest]


# ---- the two per-paper tables. Both are cached alongside the curves, so they are rewritten
#      even on a cache hit. For the Nobel one, 'passed_filter' says whether the paper is in the
#      population the statistics and the figure are computed on (rows are taken BEFORE the
#      filter, so the dropped ones are visible) and 'matched_by' says which key found it. ----
nobel_csv = os.path.join(out_dir, f'nobel_papers_{param_suffix}.csv')
_ordered(nobel_papers).sort_values('pid').to_csv(nobel_csv, index=False)
print("saved per-paper Nobel table:", nobel_csv,
      f"({len(nobel_papers)} rows, {int(nobel_papers['passed_filter'].sum())} passing "
      f"the filter)")

other_csv = os.path.join(out_dir,
                         f'other_smallest{N_SMALLEST_OTHER}_{param_suffix}.csv')
_ordered(other_smallest).to_csv(other_csv, index=False)
print("saved smallest-disp 'other' papers:", other_csv, f"({len(other_smallest)} rows)")

if venue_stats is not None:
    venue_csv = os.path.join(out_dir, f'nobel_venues_{param_suffix}.csv')
    venue_stats.to_csv(venue_csv, index=False)
    print("saved the venue table:", venue_csv,
          f"({len(venue_stats)} venue(s), "
          f"{int(venue_stats['n_papers'].sum()):,} papers in the reference population)")

missing = [m[0] for m in METRICS if m[0] not in curves]
assert not missing, f"cache {cache_path} has no curve for {missing}; delete it to rebuild"

# ==================================================
# 2. report, draw and tabulate
# ==================================================
rows = []
for short, col, label in METRICS:
    c = curves[short]
    nobel_mean = c['nobel_mean']
    q_le, q_lt = c['quantile_of_nobel_mean'], c['quantile_of_nobel_mean_strict']

    print("\n" + "=" * 78)
    print(f"{label}   [{col}]")
    print("=" * 78)
    if c.get('restricted_to_nobel_venues'):
        print(f"  population RESTRICTED to the {c['n_venues']} venue(s) the Nobel papers "
              f"themselves appear in")
    print(f"  qualifying papers with a value : {c['n_all']:,}")
    print(f"  ... of them Nobel              : {c['n_nobel']:,}")
    if 'n_nobel_pids_by_doi' in c:
        print(f"      (matched {c['n_nobel_pids_by_doi']} by DOI, "
              f"{c['n_nobel_pids_by_title']} by title, before this population's filters)")
    q_med = c['quantile_of_nobel_median']
    print(f"  Nobel MEAN   disp              : {nobel_mean:+.6f}")
    print(f"    -> its percentile among all  : {q_le:.6f}  ({q_le * 100:.3f}%)")
    print(f"       (strict '<' convention    : {q_lt:.6f})")
    print(f"  Nobel MEDIAN disp              : {c['nobel_median']:+.6f}")
    print(f"    -> its percentile among all  : {q_med:.6f}  ({q_med * 100:.3f}%)")
    print(f"  corpus mean / median / sd      : {c['all_mean']:+.6f} / "
          f"{c['all_median']:+.6f} / {c['all_std']:.6f}")
    print(f"  where the CORPUS mean sits     : {c['quantile_of_all_mean']:.6f} "
          f"(reference point -- disp is skewed, so this is not 0.5)")
    print(f"  [different statistic] mean of the Nobel papers' OWN percentile ranks: "
          f"{c['mean_of_nobel_own_quantiles']:.6f}")
    print(f"  other (non-Nobel) papers       : {c['n_other']:,}  "
          f"mean {c['other_mean']:+.6f}  (not drawn; in the CSV only)")
    print(f"  curve: {c['kde_mode']} on {c['n_kde_used']:,} values, "
          f"{len(c['xs'])} grid points")
    if c['n_outside_xlim']:
        print(f"  NOTE: {c['n_outside_xlim']} value(s) fall outside {X_LIM} and are off the "
              f"drawn range (they DO count in every number above)")
    print("=" * 78)

    # ---- the figure: the whole population's density, with four markers ----
    xs, ys = np.asarray(c['xs']), np.asarray(c['ys'])
    fig, ax = plt.subplots(figsize=(7.5, 5))

    # the density, then the part at or below the Nobel mean shaded darker. On the linear axis
    # the area under a density up to x IS the fraction of papers at or below x, so the shading
    # is the percentile made visible; with USE_LOG_Y on it only marks the region. Either way
    # the exact figure is the annotated/printed number.
    ax.fill_between(xs, ys, color=ALL_COLOR, alpha=0.25, linewidth=0)
    below = xs <= nobel_mean
    if below.any():
        ax.fill_between(xs[below], ys[below], color=ALL_COLOR, alpha=0.50, linewidth=0)
    ax.plot(xs, ys, color=ALL_COLOR, linewidth=1.8)

    if USE_LOG_Y:
        ax.set_yscale('log')
    ax.set_xlim(*X_LIM)

    # All four markers are DRAWN -- the Nobel mean and median in red, the whole population's
    # mean and median in blue for reference -- but only the two NOBEL ones carry text: with no
    # legend and no title, the figure states exactly the two numbers it is about.
    # Each label gets its OWN horizontal band, assigned in left-to-right order of the labelled
    # markers, so two sitting on nearly the same x still cannot overlap; within a band the label
    # goes to the right of its line unless that would run off the axes. y is in axes fractions
    # (get_xaxis_transform: x data, y axes).
    # The percentile is stated bare. WHICH population it is taken within is a property of the
    # whole run, not of one marker -- it is in the printed summary, in every output filename
    # (venue_tag) and in the CSV column -- so repeating it on both labels only adds clutter.
    markers = [
        (nobel_mean, NOBEL_COLOR, '--', 2.0,
         f"Nobel mean {nobel_mean:+.3f}\n"
         f"{q_le * 100:.1f}th pct"),
        (c['nobel_median'], NOBEL_COLOR, '-.', 1.6,
         f"Nobel median {c['nobel_median']:+.3f}\n"
         f"{c['quantile_of_nobel_median'] * 100:.1f}th pct"),
        (c['all_mean'], ALL_COLOR, '--', 1.5, None),
        (c['all_median'], ALL_COLOR, '-.', 1.5, None),
    ]
    # two-line labels, so the bands need room for two lines rather than one
    BANDS = [0.97, 0.79]
    span = X_LIM[1] - X_LIM[0]
    for x, colr, ls, lw, _text in markers:
        ax.axvline(x, color=colr, linestyle=ls, linewidth=lw)
    labelled = sorted([m for m in markers if m[4] is not None], key=lambda m: m[0])
    for band, (x, colr, ls, lw, text) in zip(BANDS, labelled):
        room_right = (X_LIM[1] - x) / span > 0.34      # enough width for the label
        dx = 0.015 * span * (1 if room_right else -1)
        ax.text(x + dx, band, text, transform=ax.get_xaxis_transform(),
                ha='left' if room_right else 'right', va='top',
                fontsize=MARKER_LABEL_FONTSIZE, color=colr, fontweight='bold',
                linespacing=1.35)

    ax.set_xlabel(label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel('Density (log scale)' if USE_LOG_Y else 'Density',
                  fontsize=AXIS_LABEL_FONTSIZE)
    # no title and no legend: which population this is, and how big it is, is in the printed
    # summary and in every output CSV, so the panel itself stays clean
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    fpath = os.path.join(fig_dir, f'nobel_disp_mean_quantile_{short}_{param_suffix}.svg')
    fig.savefig(fpath, format='svg', transparent=True)
    fig.savefig(fpath[:-4] + '.png', dpi=150)
    plt.close(fig)
    print("  saved figure:", fpath)

    # ---- the per-cohort figure: one subplot per publication-year cohort, each repeating the
    #      reading inside that cohort only (its own density, its own Nobel mean/median, and the
    #      percentile of each among that cohort's papers) ----
    cohorts = c.get('cohorts') or []
    if cohorts:
        ncols = min(COHORT_NCOLS, len(cohorts))
        nrows = int(np.ceil(len(cohorts) / ncols))
        cfig, caxes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows),
                                   squeeze=False, sharex=True)
        print(f"  cohort figure: {len(cohorts)} cohorts in a {nrows}x{ncols} grid")
        for ax_i, (cax, ch) in enumerate(zip(caxes.ravel(), cohorts)):
            cxs, cys = np.asarray(ch['xs']), np.asarray(ch['ys'])
            cax.fill_between(cxs, cys, color=ALL_COLOR, alpha=0.25, linewidth=0)
            cax.plot(cxs, cys, color=ALL_COLOR, linewidth=1.4)
            if USE_LOG_Y:
                cax.set_yscale('log')
            cax.set_xlim(*X_LIM)
            cax.spines['top'].set_visible(False)
            cax.spines['right'].set_visible(False)
            cax.set_title(f"{ch['label']}   (n={ch['n_all']:,}, Nobel {ch['n_nobel']})",
                          fontsize=10)

            if ch['n_nobel'] and np.isfinite(ch['nobel_mean']):
                cax.axvline(ch['nobel_mean'], color=NOBEL_COLOR, linestyle='--', linewidth=1.7)
                cax.axvline(ch['nobel_median'], color=NOBEL_COLOR, linestyle='-.',
                            linewidth=1.4)
                # the text goes on whichever side the markers are NOT, so it never sits on them
                mid = 0.5 * (X_LIM[0] + X_LIM[1])
                left = ch['nobel_mean'] > mid
                cax.text(0.03 if left else 0.97, 0.95,
                         f"mean {ch['nobel_mean']:+.3f}\n"
                         f"  {ch['quantile_of_nobel_mean'] * 100:.1f}th pct\n"
                         f"median {ch['nobel_median']:+.3f}\n"
                         f"  {ch['quantile_of_nobel_median'] * 100:.1f}th pct",
                         transform=cax.transAxes, ha='left' if left else 'right', va='top',
                         fontsize=8.5, color=NOBEL_COLOR, fontweight='bold', linespacing=1.35)
            else:
                cax.text(0.5, 0.5, 'no Nobel paper', transform=cax.transAxes,
                         ha='center', va='center', fontsize=9, color='0.5')
        for cax in caxes.ravel()[len(cohorts):]:      # blank out any unused cell
            cax.axis('off')
        cfig.suptitle(f'{label} by publication-year cohort -- Nobel mean / median and their '
                      f'percentile WITHIN each cohort', fontsize=12)
        cfig.supxlabel(label, fontsize=11)
        cfig.supylabel('Density (log scale)' if USE_LOG_Y else 'Density', fontsize=11)
        cfig.tight_layout(rect=[0.01, 0.01, 1, 0.97])
        cpath = os.path.join(fig_dir, f'nobel_disp_cohorts_{short}_{param_suffix}.svg')
        cfig.savefig(cpath, format='svg', transparent=True)
        cfig.savefig(cpath[:-4] + '.png', dpi=150)
        plt.close(cfig)
        print("  saved cohort figure:", cpath)

        ccsv = os.path.join(out_dir, f'nobel_disp_cohorts_{short}_{param_suffix}.csv')
        pd.DataFrame([{k: v for k, v in ch.items() if k not in ('xs', 'ys')}
                      for ch in cohorts]).to_csv(ccsv, index=False)
        print("  saved cohort table :", ccsv)

    row = {k: v for k, v in c.items() if k not in ('xs', 'ys')}
    row.update({
        'require_unique_match': require_unique_match,
        'reference_threshold': reference_threshold,
        'citation_threshold': citation_threshold,
        'max_year': max_year,
        'year_offset': year_offset,
        'use_with_review': use_with_review,
        'k_require_after_focal': k_require_after_focal,
        'use_title_fallback': USE_TITLE_FALLBACK,
        'min_title_chars': MIN_TITLE_CHARS,
        'restrict_to_nobel_venues': RESTRICT_TO_NOBEL_VENUES,
    })
    rows.append(row)

pd.DataFrame(rows).to_csv(out_csv, index=False)
print("\nsaved:", out_csv)

# ---- the group sizes, last thing on the screen. Read off `curves`, which is whatever this run
#      used -- freshly built or loaded from the cache -- so the numbers appear either way. ----
print("\n" + "=" * 78)
print("POPULATION" + (f" (restricted to the Nobel papers' own venues)"
                      if any(curves[m[0]].get('restricted_to_nobel_venues') for m in METRICS)
                      else " (all qualifying papers)"))
for short, col, label in METRICS:
    c = curves[short]
    print(f"  [{short}]  Nobel prize papers: {c['n_nobel']:,}   "
          f"other papers: {c['n_other']:,}   total: {c['n_all']:,}")
    if c.get('restricted_to_nobel_venues'):
        print(f"           venues: {c['n_venues']:,}")
print("=" * 78)
print("finished")
