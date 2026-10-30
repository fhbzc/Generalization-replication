"""
Country-level scatter: average academic age vs. average E/F/G index.

For the valid authors (>= MIN_PAPERS papers, kept by get_author_academic_age.py), attribute
each (paper, author) INSTANCE to that author's AFFILIATION country ON THAT PAPER -- the same
country attribution used by plot_academic_age_bins_person_top_efg_by_country.py (so the country
follows the author's academic age). Then aggregate over ALL such instances per country:

    * average academic age = mean over instances of academic_age(author, paper)
                             (academic_age(paper) = pub_year(paper) - author's first pub_year)
    * average F / E / G     = mean over instances of the paper's raw F / E / G index

The figure has three panels (Foundation / Extension / Generalization). Each POINT is a country;
x = the country's average academic age, y = the country's average index. Countries with fewer
than MIN_INSTANCES author-paper instances are dropped so the scatter stays readable.

Every instance uses the SAME edge subset as the sibling script: gen_V3 papers passing the
reference + citation thresholds and pub_year < SELECT_MAX_YEAR (which defaults to MAX_YEAR;
set it lower to analyse a shorter period -- see the settings block).

Input : META_DIRE/author_paper_academic_age.pkl   (author -> {pid: academic_age})   [get_author_academic_age.py]
        META_DIRE/pid2author_list.pkl              (pid -> [author, ...])
        META_DIRE/pid2author_country_list.pkl      (pid -> [[country, ...] per author])
        META_DIRE/efg_computation/gen_V3_y{off}.csv (pid, pub_year, up_refer_count, citation, e/f/g)
        META_DIRE/disruption/disp_full[_k_no_lower_bound].csv (pid, ..., disp_y{off})  [arrange_disruption.py]
            -> per-country avg disruption drawn as an inset on the Generalization panel.
        META_DIRE/efg_computation/gen_V3_disp_f_y{off}.csv (pid, ..., disruption_f_y{off})  [get_efg_v2_disp_f.py]
            -> when USE_DISP_F is True, per-country avg disp_f drawn as an inset on the Foundation panel.

A SECOND, separate 1x4 figure (only when PLOT_IJK) uses the same country attribution and edge
subset to scatter avg academic age against the country's average of each (i+j+k)-denominator
ratio -- F/(C+K), Disp(F)/(C+K), I/(C+K), (I-J)/(C+K), where C = I+J -- reading the raw COUNTS
(f_count / disruption_f_count / I / J / K counts + disp_y{off}) rather than the shares. Its
panels, regression fits and caches are wholly independent of the F/E/G figure above.

Input : (1x4 figure) META_DIRE/efg_computation/gen_V3_y{off}.csv          (f_count_y{off})
        (1x4 figure) META_DIRE/efg_computation/gen_V3_disp_f_y{off}.csv   (disruption_f_count_y{off})
        (1x4 figure) META_DIRE/disruption/disp_full[_k_no_lower_bound].csv (I/J/K counts, disp_y{off})
Cache : ./figures/country_avg_age_vs_efg/country_avg_age_efg_agg_<params>.csv
        ./figures/country_avg_age_vs_efg/country_avg_disp_agg_<params>.csv
        ./figures/country_avg_age_vs_efg/country_avg_dispf3_agg_<params>.csv  (only when USE_DISP_F; 3 disp_f variants)
        ./figures/country_avg_age_vs_efg/country_avg_ijk_agg_<params>.csv     (only when PLOT_IJK; 4 ratios)
        ./figures/country_avg_age_vs_efg/country_avg_age_vs_ijk_regression_<params>.csv (1x4 fits)
Output: ./figures/country_avg_age_vs_efg/country_avg_age_vs_efg_<params>.svg / .png
        ./figures/country_avg_age_vs_efg/country_avg_age_vs_ijk_<params>.svg / .png  (the 1x4)
"""

import os
import sys
import glob
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless / cluster-safe
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)               # utils.py is in this same directory
from utils import META_DIRE  # noqa: E402

meta_dire = META_DIRE
print("meta_dire", meta_dire)

# ---- must match compute_efg_quantile_by_domain.py / get_efg_s2_merge.py ----
year_offset = -1
reference_threshold = 1
citation_threshold = 5

# Only papers with pub_year < MAX_YEAR are counted. MAG's data horizon is 2020 (get_efg.py /
# get_disruption.py use max_year = 2021 as the first EXCLUDED year), and the rule is that a
# paper's whole citation window must fit inside it:
#     pub_year + year_offset <= 2020   <=>   pub_year < 2021 - year_offset
# year_offset == -1 is the no-window SENTINEL, so there is no window to leave room for and the
# condition degenerates to "the paper itself is within the horizon": pub_year <= 2020, i.e.
# < 2021. Applying the formula literally would give 2022 -- past MAG's data, and claiming 2021
# papers that get_efg.py never computed.
MAX_YEAR = 2021 if year_offset == -1 else 2021 - year_offset

# SELECT_MAX_YEAR narrows the analysed period further: only papers with pub_year <
# SELECT_MAX_YEAR are counted, the same strict-less-than MAX_YEAR uses. Leave it equal to
# MAX_YEAR for no extra narrowing.
#
#   * it must be <= MAX_YEAR: MAX_YEAR is the window the citation counts were built for, and
#     admitting later papers would mean counting incomplete citation windows;
#   * on the USE_QUANTILE path the quantiles are NOT recomputed -- they come from
#     compute_efg_quantile_by_domain.py, which ranked papers over the pre-narrowing population.
#     That stays valid only for a ranking done WITHIN a publication year, i.e.
#     QUANTILE_GROUP = 'year_domain' (its groups are (domain, pub_year)): every surviving paper
#     is still ranked against its own same-year peers. A 'global' rank was computed against the
#     whole table INCLUDING the years now dropped, so it no longer describes the plotted
#     population; the assert further down (once QUANTILE_GROUP exists) refuses that pair.
#     With USE_QUANTILE = False nothing is ranked and any narrowing is fine.
#   * every cache and figure name carries SELECT_MAX_YEAR through agg_suffix, so a narrowed run
#     never reuses or overwrites the full one.
SELECT_MAX_YEAR = 2010
assert SELECT_MAX_YEAR <= MAX_YEAR, (
    "SELECT_MAX_YEAR (%s) must be <= MAX_YEAR (%s)." % (SELECT_MAX_YEAR, MAX_YEAR))
MIN_INSTANCES = 500              # drop a country with fewer author-paper instances than this
ANNOTATE = True                  # write the country code next to each point
force_recompute = False
print("year_offset", year_offset, "rt", reference_threshold, "ct", citation_threshold,
      "MAX_YEAR", MAX_YEAR, "SELECT_MAX_YEAR", SELECT_MAX_YEAR,
      "MIN_INSTANCES", MIN_INSTANCES)

# NOTE maxyear carries SELECT_MAX_YEAR (the period actually plotted), not MAX_YEAR (the
# window the inputs were built for), so a narrowed run keeps its own caches and figures. With
# no narrowing the two are equal and every existing name is unchanged.
agg_suffix = (f'y{year_offset}_rt_{reference_threshold}_ct_{citation_threshold}_'
              f'maxyear{SELECT_MAX_YEAR}')

efg_dire = os.path.join(meta_dire, 'efg_computation')
gen_v3_path = os.path.join(efg_dire, f'gen_V3_y{year_offset}.csv')

citation_col = f'citation_count_y{year_offset}'
e_col, f_col, g_col = (f'e_y{year_offset}', f'f_y{year_offset}', f'g_y{year_offset}')

save_dire = os.path.join(BASE_DIR, 'figures', 'country_avg_age_vs_efg')
os.makedirs(save_dire, exist_ok=True)
# NOTE the cache paths are defined further down, once USE_QUANTILE / value_tag are known.

# ---- disruption source (arrange_disruption.py output) for the Generalization-panel inset ----
# arrange_disruption.py writes disp_full{year_tag}[_k_no_lower_bound].csv to META_DIRE/disruption/.
# Which K-window variant of the disruption run to read. MUST match get_disruption.py /
# arrange_disruption.py: True -> K counts only the references' citers published on/after the
# focal paper (untagged file); False -> the lower bound is dropped, K also counts earlier ones
# ('_k_no_lower_bound'). The two variants are different files, so a mismatch here reads the
# wrong disruption -- or, more often, no file at all.
k_require_after_focal = True
disp_k_suffix = '' if k_require_after_focal else '_k_no_lower_bound'
print("k_require_after_focal", k_require_after_focal, "| disp_k_suffix", repr(disp_k_suffix))
disp_col = f'disp_y{year_offset}'
# arrange_disruption.py tags its output with year_tag ('' at the default 5), so a run with
# another citation window must ask for THAT file -- otherwise this script would pair
# gen_V3_y{year_offset} with a DIFFERENT window's disruption and never say so.
disp_year_tag = '' if year_offset == 5 else f'_y{year_offset}'
disp_full_path = os.path.join(meta_dire, 'disruption',
                              f'disp_full{disp_year_tag}{disp_k_suffix}.csv')

# ---- disp_f source (get_efg_v2_disp_f.py -> get_efg_s2_merge.py with use_disp_f=True). When
#      USE_DISP_F is True, a SECOND ROW of three panels is added below the F/E/G row, one per
#      disp_f variant (down>0 / down>1 / down>2), each a country scatter of avg academic age vs
#      that variant's average share. The three variant share columns live in gen_V3_disp_f_y{off}.csv
#      (produced by the updated get_efg_v2_disp_f.py). ----
USE_DISP_F = False
# (short key, source column in gen_V3_disp_f, panel title). The short key is reused as the
# aggregate/cache column, the per-country map key, and the regression-stats key.
DISPF_VARIANTS = [
    ('dispf0', f'disruption_f_y{year_offset}',          'Disp_f (down>0)'),
    ('dispf1', f'disruption_f_down_gt1_y{year_offset}', 'Disp_f (down>1)'),
    ('dispf2', f'disruption_f_down_gt2_y{year_offset}', 'Disp_f (down>2)'),
]
dispf_cols = [col for _, col, _ in DISPF_VARIANTS]
dispf_full_path = os.path.join(efg_dire, f'gen_V3_disp_f_y{year_offset}.csv')
# tag so the regression cache + figure names don't collide between the disp_f-on / -off variants
# ('_dispf3' = the three-variant era, so the old single-variant '_dispf' caches are never reused)
dispf_tag = '_dispf3' if USE_DISP_F else ''
print("USE_DISP_F", USE_DISP_F, "| dispf_full_path", dispf_full_path)
print("DISPF_VARIANTS", DISPF_VARIANTS)

# ---- the four (i+j+k)-denominator ratios -> a SEPARATE 1x4 figure (see section 5), one panel
#      per ratio, each a country scatter of avg academic age (x) vs the country's average ratio
#      (y). Definitions are taken verbatim from compute_efgd_quantile.py so the scripts agree:
#        i+j+k          = I_count + J_count + K_count   (k = citations of the focal paper's
#                         references that do NOT cite the focal paper itself)
#        f/(i+j+k)      = f_count_y{off} / (i+j+k)              <- the raw F COUNT, not share f_y
#        disp_f/(i+j+k) = disruption_f_count_y{off} / (i+j+k)   <- the raw disp_f_0 COUNT, not share
#        i/(i+j+k)      = I_count / (i+j+k)                     <- numerator i ALONE, not i-j
#        (i-j)/(i+j+k)  = disp_y{off}, read straight from disp_full (it IS the disruption index)
#      The counts live in three files: f_count in gen_V3, disruption_f_count in gen_V3_disp_f,
#      and i/j/k + disp_y in disp_full. C = I+J (the paper's own citations), so the panel labels
#      write the denominator as C+K -- same notation as the award_quantile_analysis.py 1x4.
PLOT_IJK = True
# (short key used in the aggregate/cache columns and per-country maps, panel label)
IJK_VARIANTS = [
    ('f_ijk',     'F / (C+K)'),
    ('dispf_ijk', 'Disp(F) / (C+K)'),
    ('i_ijk',     'I / (C+K)'),
    ('imj_ijk',   '(I-J) / (C+K)'),
]
f_count_col = f'f_count_y{year_offset}'                  # raw F count      (gen_V3)
dispf0_count_col = f'disruption_f_count_y{year_offset}'  # raw disp_f_0 cnt (gen_V3_disp_f)
i_count_col = f'I_count_y{year_offset}'
j_count_col = f'J_count_y{year_offset}'
k_count_col = f'K_count_y{year_offset}'
print("PLOT_IJK", PLOT_IJK)

# ---- Output value: RAW index or QUANTILE ------------------------------------
# USE_QUANTILE picks what every panel averages -- the figures are otherwise identical:
#   False -> the raw per-paper index (f_y5, disp_y5, the i+j+k ratios, ...). Reads gen_V3 /
#            disp_full / gen_V3_disp_f and derives the (i+j+k) ratios from the counts here.
#   True  -> each paper's QUANTILE (its percentile rank among its peers) from
#            compute_efg_quantile_by_domain.py. Every metric already has a quantile column in
#            that output, so none of the raw sources are read and no I/J/K arithmetic is done.
# The quantile figures/caches carry a '_quantile_{QUANTILE_GROUP}' tag and an 'Index quantile'
# axis label, so they never overwrite the raw ones.
USE_QUANTILE = True

# Which ranking to average (compute_efg_quantile_by_domain.py writes both):
#   'year_domain' -> {col}_quantile         (rank within the paper's (domain, pub_year) peers)
#   'global'      -> {col}_quantile_global  (rank over all threshold-passing papers, no grouping)
QUANTILE_GROUP = 'year_domain'
_QUANTILE_SUFFIX = {'year_domain': '_quantile', 'global': '_quantile_global'}
assert QUANTILE_GROUP in _QUANTILE_SUFFIX, QUANTILE_GROUP
qs = _QUANTILE_SUFFIX[QUANTILE_GROUP]

# see SELECT_MAX_YEAR above: dropping publication years after the fact leaves a within-year
# rank intact but invalidates a global one
assert (SELECT_MAX_YEAR == MAX_YEAR or not USE_QUANTILE
        or QUANTILE_GROUP == 'year_domain'), (
    "SELECT_MAX_YEAR (%s) != MAX_YEAR (%s) narrows the population AFTER the quantiles were "
    "computed, which only stays valid for a WITHIN-YEAR ranking. QUANTILE_GROUP is %r: set it "
    "to 'year_domain', or re-run compute_efg_quantile_by_domain.py over the narrowed period."
    % (SELECT_MAX_YEAR, MAX_YEAR, QUANTILE_GROUP))

# compute_efg_quantile_by_domain.py's output dir. Built from THIS script's rt/ct so a threshold
# mismatch fails loudly: the quantile ranks papers among the peers that script kept, which must
# be the same population plotted here.
quantile_dire = os.path.join(
    efg_dire,
    f'efg_within_year_domain_quantile_y{year_offset}_'
    f'rt_{reference_threshold}_ct_{citation_threshold}')

# a paper in several level-0 domains has one {col}_quantile PER domain; its per-paper value is
# the MEAN over those domains. (For 'global' the per-domain values are identical, so the mean
# is a no-op.)
# this figure's panel key -> the column compute_efg_quantile_by_domain.py quantiles under.
# (i-j)/(i+j+k) IS the disruption index, so 'imj_ijk' reads disp_y{off}, same as the inset.
IJK_QUANTILE_BASE = {'f_ijk': f'f_over_ijk_y{year_offset}',
                     'dispf_ijk': f'dispf_over_ijk_y{year_offset}',
                     'i_ijk': f'i_over_ijk_y{year_offset}',
                     'imj_ijk': disp_col}

# every cache / figure name carries this, so the raw and quantile runs coexist
value_tag = f'_quantile_{QUANTILE_GROUP}' if USE_QUANTILE else ''
# the shared y-axis label of both figures follows the plotted value
y_axis_label = 'Index quantile' if USE_QUANTILE else 'Average index'
print("USE_QUANTILE", USE_QUANTILE,
      ("| QUANTILE_GROUP %s -> %r | %s" % (QUANTILE_GROUP, qs, quantile_dire))
      if USE_QUANTILE else "(raw values)")

# The raw (non-quantile), disp_f-off figure at the paper's default settings IS the Extended Data
# deliverable, so its figure + caches are named after the panel; every other combination keeps its
# params in the filename instead. NOTE MIN_INSTANCES is deliberately NOT part of this gate.
ED_FIG_NAME = 'Extended Data Fig. 10b'
# ... and a narrowed period is not the deliverable either: the deliverable is the full one.
# Without this, a SELECT_MAX_YEAR run at the paper's other settings would write its caches and
# figure straight over 'Extended Data Fig. 10b', which every other name avoids via agg_suffix.
is_ed_fig = (not USE_DISP_F and not USE_QUANTILE and year_offset == 5
             and reference_threshold == 1 and citation_threshold == 5
             and SELECT_MAX_YEAR == MAX_YEAR)
print("is_ed_fig", is_ed_fig, ("-> %s" % ED_FIG_NAME) if is_ed_fig else "")

if is_ed_fig:
    cache_path = os.path.join(save_dire, f'{ED_FIG_NAME}.csv')
    disp_cache_path = os.path.join(save_dire, f'{ED_FIG_NAME}_disp.csv')
else:
    cache_path = os.path.join(save_dire, f'country_avg_age_efg_agg_{agg_suffix}{value_tag}.csv')
    disp_cache_path = os.path.join(
        save_dire, f'country_avg_disp_agg_{agg_suffix}{disp_k_suffix}{value_tag}.csv')
dispf_cache_path = os.path.join(save_dire, f'country_avg_dispf3_agg_{agg_suffix}{value_tag}.csv')
ijk_cache_path = os.path.join(
    save_dire, f'country_avg_ijk_agg_{agg_suffix}{disp_k_suffix}{value_tag}.csv')
print("cache_path", cache_path)
print("disp_cache_path", disp_cache_path)
print("dispf_cache_path", dispf_cache_path)
print("ijk_cache_path", ijk_cache_path)


_QPID = None


def level0_domains():
    """The real level-0 field names, taken straight from pid2field_of_study.pkl -- the same map
    compute_efg_quantile_by_domain.py explodes by, so it IS the authoritative list.

    Needed because that script also writes its COMBINE_DOMAINS merged domains ('Social science',
    ...) as their own per-domain CSVs: an Economics paper therefore appears BOTH in the Economics
    file and in the Social science one, ranked among a DIFFERENT peer set. Averaging both would
    count such a paper twice and mix two incompatible rankings. A merged domain is a made-up
    pooled name, so it is absent from this map and gets dropped -- no hand-maintained list to
    drift out of sync with COMBINE_DOMAINS."""
    with open(os.path.join(meta_dire, 'pid2field_of_study.pkl'), 'rb') as fh:
        pid2fos = pickle.load(fh)
    names = set()
    for v in pid2fos.values():
        names.update(v)
    return names


def quantile_pid_frame():
    """pid -> one quantile per metric, read from compute_efg_quantile_by_domain.py's per-domain
    CSVs (memoised; every aggregate below shares the one load).

    The table is one row per (pid, domain), so a multi-domain paper is averaged over ITS domains
    into a single per-paper value. Only real level-0 domains are kept (see level0_domains), and
    pub_year < SELECT_MAX_YEAR is applied here -- that script applies rt/ct but no year
    window.
    Filtering the year afterwards is safe for 'year_domain', whose ranks are computed within a
    single year anyway. Columns come back under their BASE names (f_y5, disp_y5, ...), so each
    aggregate block can read them exactly as it reads the raw table."""
    global _QPID
    if _QPID is not None:
        return _QPID
    assert os.path.isdir(quantile_dire), (
        f"{quantile_dire}\nRun MAG/compute_efg_quantile_by_domain.py with "
        f"reference_threshold={reference_threshold}, citation_threshold={citation_threshold} "
        f"(and use_disp_f=True) so its output matches this script's thresholds.")
    files = sorted(glob.glob(os.path.join(quantile_dire, '*.csv')))
    assert files, f"no per-domain CSVs in {quantile_dire}"

    real_domains = level0_domains()
    print(f"level-0 domains in pid2field_of_study: {len(real_domains)}")

    need_base = sorted({e_col, f_col, g_col, disp_col}
                       | {c for _, c, _ in DISPF_VARIANTS}
                       | set(IJK_QUANTILE_BASE.values()))
    qcols = [f'{c}{qs}' for c in need_base]

    frames, dropped = [], {}
    for p in files:
        head = pd.read_csv(p, nrows=0).columns
        missing = [c for c in ['pid', 'pub_year', 'domain'] + qcols if c not in head]
        assert not missing, (
            f"{p} is missing {missing}.\nRe-run compute_efg_quantile_by_domain.py with "
            f"use_disp_f=True so every metric is quantiled.")
        d = pd.read_csv(p, usecols=['pid', 'pub_year', 'domain'] + qcols)
        keep = d['domain'].isin(real_domains)
        for nm, cnt in d.loc[~keep, 'domain'].value_counts().items():
            dropped[nm] = dropped.get(nm, 0) + int(cnt)
        d = d[keep & (d['pub_year'] < SELECT_MAX_YEAR)]
        if len(d):
            frames.append(d[['pid'] + qcols])
    if dropped:
        print("skipped non-level-0 (merged/pooled) domains:", dropped)
    assert frames, f"no rows left in {quantile_dire} after the level-0 / pub_year filters"
    allq = pd.concat(frames, ignore_index=True)
    print(f"quantile table: {len(allq)} (pid, level-0 domain) rows from {len(files)} files")
    _QPID = (allq.groupby('pid')[qcols].mean()
             .rename(columns={f'{c}{qs}': c for c in need_base}))
    print(f"quantile table -> {len(_QPID)} unique papers (multi-domain papers averaged)")
    return _QPID

# shared plot styling (repo convention)
try:
    plt.rcParams['font.family'] = 'Gill Sans MT'
except Exception:
    pass
plt.rcParams['axes.unicode_minus'] = False
panels = [('Foundation', 'f'), ('Extension', 'e'), ('Generalization', 'g')]

# ---- figure size + font sizes: kept consistent with
#      openalex/author_country_analysis/author_count_feg_agg.py ----
FIGSIZE    = (16, 4)   # matches author_count_feg_agg.py
tick_size  = 13        # x/y tick-label size
label_size = 15
axis_label_size = label_size + 2   # supx/supy axis labels, a touch larger than the base

# major countries to call out by readable name next to their point.
# the raw data uses ISO 3166 alpha-2 codes (US, CN, GB, ...), so map code -> label.
COUNTRY_LABELS = {
    'CN': 'China', 'IN': 'India', 'US': 'United States', 'GB': 'United Kingdom',
    'JP': 'Japan', 'DE': 'Germany', 'FR': 'France', 'KR': 'South Korea',
}
# short forms used only in the cramped disruption inset (overrides the full names above)
INSET_LABEL_OVERRIDE = {'US': 'US', 'GB': 'UK'}
# Generalization-panel disruption inset placement (axes fraction: x0, y0, w, h). Tucked hard
# into the upper-right corner. Also used to keep the main labels out of the inset region.
INSET_RECT = [0.57, 0.59, 0.42, 0.40]

# point/label colours. Un-labelled countries are drawn grey; the called-out (labelled)
# countries are highlighted per panel: Foundation=red, Extension=orange, Generalization
# (and the disruption inset)=blue.
GREY = '#BBBBBB'
HILITE = {'f': '#C44E52', 'e': '#DD8452', 'g': '#4C72B0'}   # red / orange / blue
DISP_HILITE = '#4C72B0'                                     # blue (disruption inset)
DISPF_HILITE = HILITE['f']                                  # red  (disp_f inset, Foundation panel)

# ---------------------------------------------------------------------------
# 1. per-country aggregates over all author-paper instances attributed to the
#    country (same attribution + edge subset for both):
#      (a) instance count + sum of academic age + sum of e/f/g            -> agg
#      (b) instance count + sum of raw disruption (G-panel inset)         -> disp_agg
# ---------------------------------------------------------------------------
ACC_KEYS = ('n', 'age', 'e', 'f', 'g')
DISP_KEYS = ('n', 'd')
# disp_f accumulator: instance count + one running sum per disp_f variant (keyed by short key)
DISPF_KEYS = ('n',) + tuple(k for k, _, _ in DISPF_VARIANTS)
# (i+j+k)-ratio accumulator: 'n' counts instances for the three always-present ratios, while
# 'n_dispf' counts only instances whose paper HAS a disp_f count -- papers lacking one must drop
# out of the disp_f panel alone, not out of the other three.
IJK_KEYS = ('n', 'n_dispf') + tuple(k for k, _ in IJK_VARIANTS)

NEED_MAIN = force_recompute or not os.path.exists(cache_path)
NEED_DISP = force_recompute or not os.path.exists(disp_cache_path)
NEED_DISPF = USE_DISP_F and (force_recompute or not os.path.exists(dispf_cache_path))
NEED_IJK = PLOT_IJK and (force_recompute or not os.path.exists(ijk_cache_path))

if NEED_MAIN or NEED_DISP or NEED_DISPF or NEED_IJK:
    print("loading author/country/age pkls ...")
    with open(os.path.join(meta_dire, 'author_paper_academic_age.pkl'), 'rb') as fh:
        author_paper_academic_age = pickle.load(fh)
    with open(os.path.join(meta_dire, 'pid2author_list.pkl'), 'rb') as fh:
        pid2author_list = pickle.load(fh)
    with open(os.path.join(meta_dire, 'pid2author_country_list.pkl'), 'rb') as fh:
        pid2author_country_list = pickle.load(fh)
    print("authors (>=3 papers):", len(author_paper_academic_age),
          "| pid2author_list:", len(pid2author_list))

    def iter_age_country(pid):
        """Yield (academic_age, country) for each valid author-paper-country instance,
        deduping so one paper counts once per country per author (same as the main agg)."""
        authors = pid2author_list.get(pid)
        if not authors:
            return
        clists = pid2author_country_list.get(pid)
        if not clists:
            return
        for author, countries in zip(authors, clists):
            if not countries:
                continue
            age_map = author_paper_academic_age.get(author)   # None unless a valid (>=3 papers) author
            if not age_map:
                continue
            age = age_map.get(pid)
            if age is None or age < 0:
                continue
            for country in set(countries):     # dedupe: one paper counts once per country per author
                yield age, country

# ---- (a) main e/f/g aggregate ----
if not NEED_MAIN:
    agg = pd.read_csv(cache_path, dtype={'country': str})
    print("loaded cache:", cache_path, "| countries:", len(agg))
else:
    print("cache not found (or force_recompute) -> computing per-country e/f/g aggregates ...")

    # --- paper -> (e, f, g) on the shared edge subset (thresholds + pub_year window) ---
    if USE_QUANTILE:
        # the quantile frame is already thresholded (upstream) + year-windowed, and its columns
        # carry the same base names, so the accumulation below is unchanged
        dfr = quantile_pid_frame().reset_index()
    else:
        assert os.path.exists(gen_v3_path), gen_v3_path
        dfr = pd.read_csv(gen_v3_path,
                          usecols=['pid', 'pub_year', 'up_refer_count', citation_col, e_col, f_col, g_col])
        dfr = dfr[(dfr['up_refer_count'] >= reference_threshold)
                  & (dfr[citation_col] >= citation_threshold)
                  & (dfr['pub_year'] < SELECT_MAX_YEAR)]
    print("papers after filter:", len(dfr))
    _pids = dfr['pid'].to_numpy()
    _es, _fs, _gs = dfr[e_col].to_numpy(), dfr[f_col].to_numpy(), dfr[g_col].to_numpy()
    pid2raw = {int(_pids[i]): (_es[i], _fs[i], _gs[i]) for i in range(len(dfr))}
    del dfr

    acc = {}   # country -> {'n','age','e','f','g'}

    def get_acc(c):
        a = acc.get(c)
        if a is None:
            a = {k: 0.0 for k in ACC_KEYS}
            acc[c] = a
        return a

    for pid, (e, f, g) in pid2raw.items():
        for age, country in iter_age_country(pid):
            a = get_acc(country)
            a['n'] += 1
            a['age'] += age
            a['e'] += e;  a['f'] += f;  a['g'] += g
    print("countries accumulated:", len(acc))

    rows = [(c, a['n'], a['age'], a['e'], a['f'], a['g']) for c, a in acc.items()]
    agg = pd.DataFrame(rows, columns=['country'] + list(ACC_KEYS))
    agg.to_csv(cache_path, index=False)
    print("wrote cache:", cache_path)

# ---- (b) disruption aggregate (Generalization-panel inset) ----
# Disruption is a raw continuous per-paper index; accumulate sum + instance count per
# country over the SAME attribution / threshold / pub_year subset, then avg at plot time.
if not NEED_DISP:
    disp_agg = pd.read_csv(disp_cache_path, dtype={'country': str})
    print("loaded disp cache:", disp_cache_path, "| countries:", len(disp_agg))
else:
    print("computing per-country disruption aggregates ...")
    if USE_QUANTILE:
        dfd = quantile_pid_frame().reset_index()
    else:
        assert os.path.exists(disp_full_path), disp_full_path
        dfd = pd.read_csv(disp_full_path,
                          usecols=['pid', 'pub_year', 'up_refer_count', citation_col, disp_col])
        dfd = dfd[(dfd['up_refer_count'] >= reference_threshold)
                  & (dfd[citation_col] >= citation_threshold)
                  & (dfd['pub_year'] < SELECT_MAX_YEAR)]
    print("disp papers after filter:", len(dfd))
    _dp = dfd['pid'].to_numpy()
    _dd = dfd[disp_col].to_numpy()
    pid2disp = {int(_dp[i]): _dd[i] for i in range(len(dfd))}
    del dfd

    acc_disp = {}   # country -> {'n','d'}

    def get_disp(c):
        a = acc_disp.get(c)
        if a is None:
            a = {k: 0.0 for k in DISP_KEYS}
            acc_disp[c] = a
        return a

    for pid, d in pid2disp.items():
        for age, country in iter_age_country(pid):
            a = get_disp(country)
            a['n'] += 1
            a['d'] += d
    print("disp: countries accumulated:", len(acc_disp))

    rows = [(c, a['n'], a['d']) for c, a in acc_disp.items()]
    disp_agg = pd.DataFrame(rows, columns=['country'] + list(DISP_KEYS))
    disp_agg.to_csv(disp_cache_path, index=False)
    print("wrote disp cache:", disp_cache_path)

# ---- (c) disp_f aggregate (second row of panels; only when USE_DISP_F) ----
# Each disp_f variant is a raw per-paper share; accumulate a running sum PER VARIANT plus the
# instance count per country over the SAME attribution / threshold / pub_year subset, then avg
# at plot time. All three variant columns are read from gen_V3_disp_f in one pass.
dispf_agg = None
_dispf_keys = [k for k, _, _ in DISPF_VARIANTS]
if USE_DISP_F:
    if not NEED_DISPF:
        dispf_agg = pd.read_csv(dispf_cache_path, dtype={'country': str})
        print("loaded disp_f cache:", dispf_cache_path, "| countries:", len(dispf_agg))
    else:
        print("computing per-country disp_f aggregates ...")
        if USE_QUANTILE:
            dff = quantile_pid_frame().reset_index()
        else:
            assert os.path.exists(dispf_full_path), dispf_full_path
            dff = pd.read_csv(
                dispf_full_path,
                usecols=['pid', 'pub_year', 'up_refer_count', citation_col] + dispf_cols)
            dff = dff[(dff['up_refer_count'] >= reference_threshold)
                      & (dff[citation_col] >= citation_threshold)
                      & (dff['pub_year'] < SELECT_MAX_YEAR)]
        print("disp_f papers after filter:", len(dff))
        _fp = dff['pid'].to_numpy()
        # pid -> tuple of the three variant shares, in DISPF_VARIANTS order
        _fvals = {col: dff[col].to_numpy() for col in dispf_cols}
        pid2dispf = {int(_fp[i]): tuple(_fvals[col][i] for col in dispf_cols)
                     for i in range(len(dff))}
        del dff

        acc_dispf = {}   # country -> {'n', 'dispf0', 'dispf1', 'dispf2'}

        def get_dispf(c):
            a = acc_dispf.get(c)
            if a is None:
                a = {k: 0.0 for k in DISPF_KEYS}
                acc_dispf[c] = a
            return a

        for pid, dvals in pid2dispf.items():
            for age, country in iter_age_country(pid):
                a = get_dispf(country)
                a['n'] += 1
                for k, dv in zip(_dispf_keys, dvals):
                    a[k] += dv
        print("disp_f: countries accumulated:", len(acc_dispf))

        rows = [tuple([c] + [a[k] for k in DISPF_KEYS]) for c, a in acc_dispf.items()]
        dispf_agg = pd.DataFrame(rows, columns=['country'] + list(DISPF_KEYS))
        dispf_agg.to_csv(dispf_cache_path, index=False)
        print("wrote disp_f cache:", dispf_cache_path)

# ---- (d) (i+j+k)-ratio aggregate (the separate 1x4 figure; only when PLOT_IJK) ----
# Each ratio is a raw per-paper number; accumulate a running sum PER RATIO plus the instance
# count per country over the SAME attribution / threshold / pub_year subset, then avg at plot
# time. disp_full is the base (it carries pub_year / up_refer_count / citations); the f count is
# INNER-joined and the disp_f count LEFT-joined, so a paper without a disp_f value still feeds
# the other three ratios (its disp_f contribution is skipped via the separate 'n_dispf' count).
ijk_agg = None
_ijk_keys = [k for k, _ in IJK_VARIANTS]
if PLOT_IJK:
    if not NEED_IJK:
        ijk_agg = pd.read_csv(ijk_cache_path, dtype={'country': str})
        print("loaded ijk cache:", ijk_cache_path, "| countries:", len(ijk_agg))
    else:
        print("computing per-country (i+j+k)-ratio aggregates ...")
        if USE_QUANTILE:
            # the ratios are already quantiled columns upstream -- just rename them to the
            # figure's keys; no counts, no joins, no arithmetic.
            dfi = quantile_pid_frame().reset_index()
            for _key, _base in IJK_QUANTILE_BASE.items():
                dfi[_key] = dfi[_base]
            print("ijk: papers after filter:", len(dfi))
        else:
            assert os.path.exists(disp_full_path), disp_full_path
            dfi = pd.read_csv(disp_full_path,
                              usecols=['pid', 'pub_year', 'up_refer_count', citation_col,
                                       disp_col, i_count_col, j_count_col, k_count_col])
            dfi = dfi[(dfi['up_refer_count'] >= reference_threshold)
                      & (dfi[citation_col] >= citation_threshold)
                      & (dfi['pub_year'] < SELECT_MAX_YEAR)]
            print("ijk: disp_full papers after filter:", len(dfi))

            # raw F count (inner join: the f/(i+j+k) panel needs it)
            assert os.path.exists(gen_v3_path), gen_v3_path
            dfi = dfi.merge(pd.read_csv(gen_v3_path, usecols=['pid', f_count_col]),
                            on='pid', how='inner')
            print("ijk: after join with", f_count_col, ":", len(dfi))

            # raw disp_f_0 count (LEFT join: missing -> NaN -> skipped by the disp_f panel only)
            assert os.path.exists(dispf_full_path), dispf_full_path
            dfi = dfi.merge(pd.read_csv(dispf_full_path, usecols=['pid', dispf0_count_col]),
                            on='pid', how='left')
            print("ijk: papers with a", dispf0_count_col, ":",
                  int(dfi[dispf0_count_col].notna().sum()))

            # the four ratios. denom > 0 always holds under citation_threshold >= 5 (i+j IS the
            # citation count), but guard anyway: a 0 denominator gives NaN, so a degenerate paper
            # drops out of the mean instead of dragging it toward 0.
            _denom_ijk = dfi[i_count_col] + dfi[j_count_col] + dfi[k_count_col]
            dfi['f_ijk']     = (dfi[f_count_col] / _denom_ijk).where(_denom_ijk > 0)
            dfi['dispf_ijk'] = (dfi[dispf0_count_col] / _denom_ijk).where(_denom_ijk > 0)
            dfi['i_ijk']     = (dfi[i_count_col] / _denom_ijk).where(_denom_ijk > 0)
            dfi['imj_ijk']   = dfi[disp_col]  # (i-j)/(i+j+k): the disruption index, as-is

        for _k in _ijk_keys:
            print(f"  {_k} stats:", dfi[_k].describe().to_dict())

        _ip = dfi['pid'].to_numpy()
        _ivals = {k: dfi[k].to_numpy() for k in _ijk_keys}
        pid2ijk = {int(_ip[i]): tuple(_ivals[k][i] for k in _ijk_keys) for i in range(len(dfi))}
        del dfi

        acc_ijk = {}   # country -> {'n', 'n_dispf', 'f_ijk', 'dispf_ijk', 'i_ijk', 'imj_ijk'}

        def get_ijk(c):
            a = acc_ijk.get(c)
            if a is None:
                a = {k: 0.0 for k in IJK_KEYS}
                acc_ijk[c] = a
            return a

        for pid, ivals in pid2ijk.items():
            for age, country in iter_age_country(pid):
                a = get_ijk(country)
                a['n'] += 1
                for k, iv in zip(_ijk_keys, ivals):
                    if k == 'dispf_ijk':
                        if not np.isnan(iv):      # only papers that HAVE a disp_f count
                            a['n_dispf'] += 1
                            a[k] += iv
                    else:
                        a[k] += iv
        print("ijk: countries accumulated:", len(acc_ijk))

        rows = [tuple([c] + [a[k] for k in IJK_KEYS]) for c, a in acc_ijk.items()]
        ijk_agg = pd.DataFrame(rows, columns=['country'] + list(IJK_KEYS))
        ijk_agg.to_csv(ijk_cache_path, index=False)
        print("wrote ijk cache:", ijk_cache_path)

# ---------------------------------------------------------------------------
# 2. per-country means (one point per country)
# ---------------------------------------------------------------------------
agg = agg[agg['n'] >= MIN_INSTANCES].copy()
print(f"countries with >= {MIN_INSTANCES} instances: {len(agg)}")
agg['avg_age'] = agg['age'] / agg['n']
for key in ('e', 'f', 'g'):
    agg['avg_' + key] = agg[key] / agg['n']
agg = agg.sort_values('n', ascending=False).reset_index(drop=True)

# ---- figure sample sizes: unique papers / authors WITHIN the qualified (>= MIN_INSTANCES)
#      countries, plus the qualified-country count. Recomputed here (never cached) so the numbers
#      print on every run, including cache-load runs where the accumulation loop was skipped.
#      Reuses the SAME main-scatter pid subset + attribution rule as iter_age_country, so a paper
#      is counted iff at least one of its author-paper-country instances fed a qualified country's
#      mean, and an author iff at least one of their such instances did. ----
qualified_countries = set(agg['country'].astype(str))
# the main-scatter pid subset (same source as pid2raw above): the quantile frame when USE_QUANTILE
# (already thresholded / year-windowed / one row per unique pid), else the threshold+year-filtered
# gen_V3 pids.
if USE_QUANTILE:
    _stat_pid_set = {int(p) for p in quantile_pid_frame().index.to_numpy()}
else:
    _sdf = pd.read_csv(gen_v3_path, usecols=['pid', 'pub_year', 'up_refer_count', citation_col])
    _sdf = _sdf[(_sdf['up_refer_count'] >= reference_threshold)
                & (_sdf[citation_col] >= citation_threshold)
                & (_sdf['pub_year'] < SELECT_MAX_YEAR)]
    _stat_pid_set = {int(p) for p in _sdf['pid'].to_numpy()}
    del _sdf
# the three attribution pkls are already in memory when anything was recomputed above; load them
# here only if every cache short-circuited that (so a pure cache-load run can still count).
if 'pid2author_list' not in globals():
    with open(os.path.join(meta_dire, 'author_paper_academic_age.pkl'), 'rb') as fh:
        author_paper_academic_age = pickle.load(fh)
    with open(os.path.join(meta_dire, 'pid2author_list.pkl'), 'rb') as fh:
        pid2author_list = pickle.load(fh)
    with open(os.path.join(meta_dire, 'pid2author_country_list.pkl'), 'rb') as fh:
        pid2author_country_list = pickle.load(fh)

_uniq_papers, _uniq_authors = set(), set()
for _pid in _stat_pid_set:
    _authors = pid2author_list.get(_pid)
    if not _authors:
        continue
    _clists = pid2author_country_list.get(_pid)
    if not _clists:
        continue
    for _author, _countries in zip(_authors, _clists):
        if not _countries:
            continue
        _age_map = author_paper_academic_age.get(_author)   # None unless a valid (>=3 papers) author
        if not _age_map:
            continue
        _age = _age_map.get(_pid)
        if _age is None or _age < 0:
            continue
        if any(c in qualified_countries for c in set(_countries)):
            _uniq_papers.add(_pid)
            _uniq_authors.add(_author)
print(f"[figure sample | qualified countries only] unique papers: {len(_uniq_papers)} | "
      f"unique authors: {len(_uniq_authors)} | "
      f"qualified unique countries (>= {MIN_INSTANCES} instances): {len(qualified_countries)}")

# marker size scaled by instance count (sqrt so a few huge countries don't dominate);
# kept small so points/labels don't crowd the compact (16x4) panels
n_arr = agg['n'].to_numpy(dtype=float)
sizes = 12 + 90 * np.sqrt(n_arr / n_arr.max())

# per-country mean disruption for the Generalization-panel inset (same MIN_INSTANCES cut;
# aligned to the main scatter's countries via the country code)
disp_agg = disp_agg[disp_agg['n'] >= MIN_INSTANCES].copy()
disp_agg['avg_disp'] = disp_agg['d'] / disp_agg['n']
disp_map = dict(zip(disp_agg['country'].astype(str), disp_agg['avg_disp']))
print(f"countries with disruption + >= {MIN_INSTANCES} instances: {len(disp_agg)}")

# per-country mean disp_f for each variant (same MIN_INSTANCES cut); only when USE_DISP_F.
# dispf_maps[short_key] = {country -> avg share for that variant}, aligned via the country code.
dispf_maps = {k: {} for k, _, _ in DISPF_VARIANTS}
if USE_DISP_F and dispf_agg is not None:
    dispf_agg = dispf_agg[dispf_agg['n'] >= MIN_INSTANCES].copy()
    for k, _, _ in DISPF_VARIANTS:
        dispf_agg['avg_' + k] = dispf_agg[k] / dispf_agg['n']
        dispf_maps[k] = dict(zip(dispf_agg['country'].astype(str), dispf_agg['avg_' + k]))
    print(f"countries with disp_f + >= {MIN_INSTANCES} instances: {len(dispf_agg)}")

# per-country mean of each (i+j+k) ratio (same MIN_INSTANCES cut on the instance count); only
# when PLOT_IJK. ijk_maps[short_key] = {country -> avg ratio}, aligned via the country code.
# disp_f/(i+j+k) averages over its OWN instance count (n_dispf), since papers without a disp_f
# count contributed nothing to its sum; a country with no such paper gets no disp_f point.
ijk_maps = {k: {} for k, _ in IJK_VARIANTS}
if PLOT_IJK and ijk_agg is not None:
    ijk_agg = ijk_agg[ijk_agg['n'] >= MIN_INSTANCES].copy()
    for k, _ in IJK_VARIANTS:
        denom = ijk_agg['n_dispf'] if k == 'dispf_ijk' else ijk_agg['n']
        ijk_agg['avg_' + k] = (ijk_agg[k] / denom).where(denom > 0)
        ijk_maps[k] = dict(zip(ijk_agg['country'].astype(str), ijk_agg['avg_' + k]))
        ijk_maps[k] = {c: v for c, v in ijk_maps[k].items() if not np.isnan(v)}
    print(f"countries with (i+j+k) ratios + >= {MIN_INSTANCES} instances: {len(ijk_agg)}")

from scipy import stats  # noqa: E402  (t-distribution for the regression CI band)


def compute_regression(x, y):
    """OLS fit + everything needed to (a) report and (b) later REDRAW the line and its
    95% CI band WITHOUT the raw data: slope/intercept/r/p/CI plus the band parameters
    (n, residual std error, x-mean, Sxx, x-range). Returns None if n < 3."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    dof = n - 2
    resid = y - (slope * x + intercept)
    s_err = float(np.sqrt(np.sum(resid ** 2) / dof))   # residual std error
    xmean = float(x.mean())
    sxx = float(np.sum((x - xmean) ** 2))
    tval = float(stats.t.ppf(0.975, dof))
    r = float(np.corrcoef(x, y)[0, 1])
    slope_se = s_err / np.sqrt(sxx) if sxx > 0 else np.nan   # std error of the slope
    tstat = slope / slope_se if slope_se > 0 else np.nan
    p_value = float(2.0 * stats.t.sf(abs(tstat), dof)) if slope_se > 0 else np.nan
    return {
        'n': int(n),
        'slope': float(slope),
        'intercept': float(intercept),
        'r': r,
        'r2': r * r,
        'p_value': p_value,
        'slope_se': float(slope_se),
        'slope_ci_low': float(slope - tval * slope_se),
        'slope_ci_high': float(slope + tval * slope_se),
        's_err': s_err,
        'x_mean': xmean,
        'sxx': sxx,
        'x_min': float(x.min()),
        'x_max': float(x.max()),
    }


def draw_regression(ax, st, color='#444444', lw=1.8):
    """Reconstruct and draw the best-fit line + 95% CI band purely from a cached stats
    dict (as produced by compute_regression) -- no raw x/y needed, so redraws are free."""
    n = int(st['n'])
    dof = n - 2
    xs = np.linspace(st['x_min'], st['x_max'], 100)
    ys = st['slope'] * xs + st['intercept']
    tval = float(stats.t.ppf(0.975, dof)) if dof > 0 else 0.0
    ci = tval * st['s_err'] * np.sqrt(1.0 / n + (xs - st['x_mean']) ** 2 / st['sxx'])
    ax.fill_between(xs, ys - ci, ys + ci, color=color, alpha=0.15, linewidth=0, zorder=1)
    ax.plot(xs, ys, color=color, lw=lw, zorder=3)


def annotate_r_beside_line(ax, st, avoid_anns=(), fontsize=15, color='#444444', avoid_boxes=(),
                           prefer_left=False):
    """Place 'r = ..' beside the fitted line, on the sparser side (below-right for a
    rising line, above-right for a falling one). Try several spots along the line and
    keep the first one whose text box does NOT overlap any of `avoid_anns` (the country
    labels) nor any `avoid_boxes` (display-coord Bboxes, e.g. the disruption inset), so r
    never lands on another label or inside the inset. `prefer_left` scans the line from the
    left end first, so r lands on the left side (used for the cramped disruption inset)."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    avoid = [a.get_window_extent(renderer) for a in avoid_anns] + list(avoid_boxes)
    x0, x1 = st['x_min'], st['x_max']
    rising = st['slope'] >= 0
    txt = f"r = {st['r']:.2f}"
    fracs = ((0.05, 0.14, 0.24, 0.36, 0.48, 0.60, 0.72) if prefer_left
             else (0.72, 0.60, 0.84, 0.50, 0.90, 0.40, 0.30, 0.20, 0.95))
    pref = -1 if rising else 1     # preferred (sparser) side: below a rising line, above a falling one
    # candidates ordered NEAREST-THE-LINE FIRST: for each small->large offset, scan the whole line
    # on the preferred side then the other side, so r hugs the line and only drifts away if every
    # close spot overlaps a label / reserved box.
    cands = []
    for dmag in (8, 13, 20, 28, 38):
        cands += [(f, pref * dmag) for f in fracs]
        cands += [(f, -pref * dmag) for f in fracs]
    placed = None
    for frac, dy in cands:
        xr = x0 + frac * (x1 - x0)
        yr = st['slope'] * xr + st['intercept']
        va = 'bottom' if dy >= 0 else 'top'
        ann = ax.annotate(txt, (xr, yr), xytext=(6, dy), textcoords='offset points',
                          fontsize=fontsize, color=color, ha='left', va=va, zorder=4)
        fig.canvas.draw()
        if not any(ann.get_window_extent(renderer).overlaps(b) for b in avoid):
            placed = ann
            break
        ann.remove()
    if placed is None:      # fallback: closest preferred spot even if it overlaps
        xr = x0 + fracs[0] * (x1 - x0)
        yr = st['slope'] * xr + st['intercept']
        placed = ax.annotate(txt, (xr, yr), xytext=(6, pref * 8), textcoords='offset points',
                             fontsize=fontsize, color=color, ha='left',
                             va=('bottom' if pref > 0 else 'top'), zorder=4)
    return placed


def add_labels_no_overlap(ax, xs, ys, texts, fontsize, color, dx=5, dy=4,
                          step=1.2, iters=400, line_color='#999999', line_width=0.7,
                          draw_leader=True, max_off=60, avoid_box=None):
    """Annotate points and iteratively push the text boxes apart so labels that
    would land on top of each other separate vertically. A thin neutral-grey leader
    line links each label to its point unless draw_leader=False.

    `max_off` caps the vertical displacement (in points) of any label from its point,
    so a crowded cluster can't fling a label off the panel and leave a long stray
    leader line stretching across it (which happened once the canvas shrank to 16x4).

    `avoid_box` is an optional display-coord Bbox (e.g. the disruption inset) that no
    label may sit on: any label overlapping it is pushed straight down until it clears,
    with extra downward travel allowed so it always ends up below the reserved region."""
    fig = ax.figure
    arrow = (dict(arrowstyle='-', color=line_color, lw=line_width, shrinkA=0, shrinkB=0)
             if draw_leader else None)
    anns, offs = [], []
    for x, y, t in zip(xs, ys, texts):
        ann = ax.annotate(t, (x, y), xytext=(dx, dy), textcoords='offset points',
                          fontsize=fontsize, fontweight='bold', color=color,
                          ha='left', va='bottom', zorder=5,
                          arrowprops=arrow)
        anns.append(ann)
        offs.append([float(dx), float(dy)])
    if not anns:
        return anns
    # per-label upper bound on the offset. Normally max_off; once a label is pushed out of
    # the reserved region its ceiling is pinned just below it so the mutual-overlap step can't
    # lift it back in (that tug-of-war otherwise never converges -> the loop ran every iter).
    ceil = [float(max_off)] * len(anns)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    best_ov, stale = None, 0
    for _ in range(iters):
        boxes = [a.get_window_extent(renderer) for a in anns]
        moved = False
        n_ov = 0                             # overlaps still present this pass
        # keep every label out of the reserved region (inset): push it straight down,
        # allowing more travel than the mutual-overlap cap so it fully clears the inset.
        if avoid_box is not None:
            for i in range(len(anns)):
                if boxes[i].overlaps(avoid_box):
                    moved = True; n_ov += 1
                    offs[i][1] = max(-2.0 * max_off, offs[i][1] - step)
                    ceil[i] = offs[i][1]     # pin the ceiling below the inset
                    anns[i].set_position(tuple(offs[i]))
        for i in range(len(anns)):
            for j in range(i + 1, len(anns)):
                if boxes[i].overlaps(boxes[j]):
                    moved = True; n_ov += 1
                    if boxes[i].y0 <= boxes[j].y0:
                        offs[i][1] -= step; offs[j][1] += step
                    else:
                        offs[i][1] += step; offs[j][1] -= step
                    offs[i][1] = max(-max_off, min(ceil[i], offs[i][1]))
                    offs[j][1] = max(-max_off, min(ceil[j], offs[j][1]))
                    anns[i].set_position(tuple(offs[i]))
                    anns[j].set_position(tuple(offs[j]))
        if not moved:
            break
        # a larger max_off lets labels ping-pong instead of pinning at the cap, so the loop
        # can run every iteration (a full redraw each) without ever settling. Stop once the
        # remaining-overlap count stops improving -- the layout is as separated as it will get.
        if best_ov is None or n_ov < best_ov:
            best_ov, stale = n_ov, 0
        else:
            stale += 1
            if stale >= 30:
                break
        fig.canvas.draw()
    return anns


# ---------------------------------------------------------------------------
# 3. regression stats -> CACHE. Computed once from the per-country means and saved;
#    on any later redraw they are loaded straight from CSV (no recompute, and the
#    line + CI band are reconstructed from the stored parameters by draw_regression).
# ---------------------------------------------------------------------------
REG_COLS = ['panel', 'index', 'x', 'y', 'n', 'slope', 'intercept', 'r', 'r2',
            'p_value', 'slope_se', 'slope_ci_low', 'slope_ci_high',
            's_err', 'x_mean', 'sxx', 'x_min', 'x_max']
reg_cache_path = os.path.join(
    save_dire,
    f'{ED_FIG_NAME}_regression.csv' if is_ed_fig else
    f'country_avg_age_vs_efg_regression_{agg_suffix}_mi{MIN_INSTANCES}{dispf_tag}{value_tag}.csv')
print("reg_cache_path", reg_cache_path)

xvals = agg['avg_age'].to_numpy()
_codes = agg['country'].astype(str).to_numpy()
# inset (disruption) x/y: the subset of countries that also have a disruption mean
disp_x = np.array([a for a, c in zip(xvals, _codes) if c in disp_map], dtype=float)
disp_y = np.array([disp_map[c] for c in _codes if c in disp_map], dtype=float)
# disp_f variant x/y: per variant, the subset of countries that have that variant's mean
dispf_xy = {}
for _k, _, _ in DISPF_VARIANTS:
    _m = dispf_maps[_k]
    dispf_xy[_k] = (
        np.array([a for a, c in zip(xvals, _codes) if c in _m], dtype=float),
        np.array([_m[c] for c in _codes if c in _m], dtype=float),
    )

_reg_cache_ok = False
if (not force_recompute) and os.path.exists(reg_cache_path):
    reg_df = pd.read_csv(reg_cache_path)
    # an old cache may lack the band-reconstruction columns -> treat it as stale and recompute
    _missing = [c for c in REG_COLS if c not in reg_df.columns]
    if _missing:
        print("regression cache is stale (missing %s) -> recomputing" % _missing)
    else:
        reg_stats = {str(row['index']): row.to_dict() for _, row in reg_df.iterrows()}
        print("loaded regression cache:", reg_cache_path, "| fits:", len(reg_stats))
        _reg_cache_ok = True
if not _reg_cache_ok:
    print("regression cache not found (or force_recompute) -> computing fits ...")
    reg_stats = {}
    for title, key in panels:
        st = compute_regression(xvals, agg['avg_' + key].to_numpy())
        if st is not None:
            reg_stats[key] = {'panel': title, 'index': key,
                              'x': 'avg_age', 'y': 'avg_' + key, **st}
    if len(disp_x) >= 3:
        st = compute_regression(disp_x, disp_y)
        if st is not None:
            reg_stats['disp'] = {'panel': 'Disruption (inset)', 'index': 'disp',
                                 'x': 'avg_age', 'y': 'avg_disp', **st}
    if USE_DISP_F:
        for _k, _, _title in DISPF_VARIANTS:
            _dx, _dy = dispf_xy[_k]
            if len(_dx) >= 3:
                st = compute_regression(_dx, _dy)
                if st is not None:
                    reg_stats[_k] = {'panel': _title, 'index': _k,
                                     'x': 'avg_age', 'y': 'avg_' + _k, **st}
    reg_df = pd.DataFrame([reg_stats[k] for k in reg_stats])[REG_COLS]
    reg_df.to_csv(reg_cache_path, index=False)
    print("wrote regression cache:", reg_cache_path)
    print(reg_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 4. figure: top row = F/E/G (x = avg academic age, y = avg index). When USE_DISP_F, a second
#    row of three panels is added, one per disp_f variant (down>0 / down>1 / down>2).
# ---------------------------------------------------------------------------
if USE_DISP_F:
    fig, axes2d = plt.subplots(2, 3, figsize=(FIGSIZE[0], FIGSIZE[1] * 2))
    axes = axes2d[0]           # top row: F / E / G
    dispf_axes = axes2d[1]     # bottom row: the three disp_f variants
else:
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
    dispf_axes = None
for ax, (title, key) in zip(axes, panels):
    yvals = agg['avg_' + key].to_numpy()
    hilite = HILITE[key]
    # labelled (called-out) countries in the panel highlight colour; the rest grey
    pt_colors = [hilite if c in COUNTRY_LABELS else GREY for c in _codes]
    ax.scatter(xvals, yvals, s=sizes, c=pt_colors, alpha=0.7,
               edgecolors='white', linewidths=0.5, zorder=2)
    # best-fit regression line + 95% CI band (reconstructed from cached stats)
    st = reg_stats.get(key)
    if st is not None:
        draw_regression(ax, st)
    # the Generalization panel reserves its upper-right for the disruption inset -> keep BOTH
    # the country labels and the 'r = ..' annotation out of that region.
    avoid_box = None
    if key == 'g':
        x0f, y0f, wf, hf = INSET_RECT
        avoid_box = Bbox([ax.transAxes.transform((x0f, y0f)),
                          ax.transAxes.transform((x0f + wf, y0f + hf))])
    # draw the country labels first, then place r beside the line AVOIDING those labels
    label_anns = []
    if ANNOTATE:
        lx, ly, lt = [], [], []
        for xc, yc, c in zip(xvals, yvals, _codes):
            name = COUNTRY_LABELS.get(c)
            if name is not None:   # only the called-out major countries get a label
                lx.append(xc); ly.append(yc); lt.append(name)
        # all panels draw a thin leader line from each point to its label
        label_anns = add_labels_no_overlap(ax, lx, ly, lt, fontsize=8, color=hilite,
                                           draw_leader=True, avoid_box=avoid_box)
    if st is not None:
        annotate_r_beside_line(ax, st, avoid_anns=label_anns, fontsize=tick_size,
                               avoid_boxes=([avoid_box] if avoid_box is not None else ()))
    # panel name in the bottom-left corner (matches author_count_feg_agg.py) instead of a top title
    ax.text(0.03, 0.03, title, transform=ax.transAxes, fontsize=label_size + 7,
            ha='left', va='bottom', zorder=4)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # inset on the Generalization panel: country avg academic age vs country avg DISRUPTION
    # (same scatter layout as the main panels, one point per country)
    if key == 'g':
        dx, dy, ds, dcodes = [], [], [], []
        for xc, c, sz in zip(xvals, agg['country'].astype(str), sizes):
            if c in disp_map:
                dx.append(xc); dy.append(disp_map[c]); ds.append(sz); dcodes.append(c)
        if len(dx) >= 2:
            dx = np.asarray(dx); dy = np.asarray(dy); ds = np.asarray(ds)
            # upper-right of the G panel (matches author_count_feg_agg.py), with an opaque white
            # background so the inset reads as its own panel over the main scatter
            axins = ax.inset_axes(INSET_RECT)
            axins.patch.set_facecolor('white')
            axins.patch.set_alpha(1.0)
            axins.set_zorder(5)
            dcolors = [DISP_HILITE if c in COUNTRY_LABELS else GREY for c in dcodes]
            axins.scatter(dx, dy, s=ds * 0.5, c=dcolors, alpha=0.7,
                          edgecolors='white', linewidths=0.5, zorder=2)
            dst = reg_stats.get('disp')
            if dst is not None:
                draw_regression(axins, dst, lw=1.4)
            # labels first, then r beside the line avoiding them
            dlabel_anns = []
            if ANNOTATE:
                lx, ly, lt = [], [], []
                for xc, yc, c in zip(dx, dy, dcodes):
                    name = COUNTRY_LABELS.get(c)
                    if name is not None:   # only the called-out major countries
                        name = INSET_LABEL_OVERRIDE.get(c, name)   # short form in the inset
                        lx.append(xc); ly.append(yc); lt.append(name)
                dlabel_anns = add_labels_no_overlap(axins, lx, ly, lt, fontsize=label_size - 10,
                                                    color=DISP_HILITE, dx=3, dy=2, step=1.0,
                                                    draw_leader=True, line_width=0.5, max_off=12)
            if dst is not None:
                # r on the LEFT of the inset, hugging the fitted line, skipping label overlaps
                annotate_r_beside_line(axins, dst, avoid_anns=dlabel_anns, fontsize=label_size - 6,
                                       prefer_left=True)
            axins.tick_params(axis='both', labelsize=tick_size - 4)
            axins.spines['top'].set_visible(False)
            axins.spines['right'].set_visible(False)
            # inset caption as text in its lower-left corner (was a title)
            axins.text(0.03, 0.03, 'Disruption', transform=axins.transAxes,
                       fontsize=label_size - 5, fontweight='bold', ha='left', va='bottom',
                       zorder=4)

# ---- second row (only when USE_DISP_F): one full panel per disp_f variant ----
# Each panel is a country scatter of avg academic age (x) vs that variant's average disp_f
# share (y), with the same regression line / CI band / country labels / 'r =' annotation as the
# top-row panels. All three share the Foundation red highlight (disp_f = foundation-disruption).
if USE_DISP_F and dispf_axes is not None:
    for ax, (vkey, _vcol, vtitle) in zip(dispf_axes, DISPF_VARIANTS):
        vmap = dispf_maps[vkey]
        # keep only countries that have this variant's mean; align x / size / code to them
        fx, fy, fsz, fcodes = [], [], [], []
        for xc, c, sz in zip(xvals, _codes, sizes):
            if c in vmap:
                fx.append(xc); fy.append(vmap[c]); fsz.append(sz); fcodes.append(c)
        fx = np.asarray(fx, dtype=float); fy = np.asarray(fy, dtype=float)
        fsz = np.asarray(fsz, dtype=float)
        pt_colors = [DISPF_HILITE if c in COUNTRY_LABELS else GREY for c in fcodes]
        ax.scatter(fx, fy, s=fsz, c=pt_colors, alpha=0.7,
                   edgecolors='white', linewidths=0.5, zorder=2)
        st = reg_stats.get(vkey)
        if st is not None:
            draw_regression(ax, st)
        # country labels first, then r beside the line avoiding them
        label_anns = []
        if ANNOTATE:
            lx, ly, lt = [], [], []
            for xc, yc, c in zip(fx, fy, fcodes):
                name = COUNTRY_LABELS.get(c)
                if name is not None:   # only the called-out major countries get a label
                    lx.append(xc); ly.append(yc); lt.append(name)
            label_anns = add_labels_no_overlap(ax, lx, ly, lt, fontsize=8, color=DISPF_HILITE,
                                               draw_leader=True)
        if st is not None:
            annotate_r_beside_line(ax, st, avoid_anns=label_anns, fontsize=tick_size)
        # panel name in the bottom-left corner (matches the top-row panels)
        ax.text(0.03, 0.03, vtitle, transform=ax.transAxes, fontsize=label_size + 7,
                ha='left', va='bottom', zorder=4)
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

_sxl = fig.supxlabel('Average academic age (years since author\'s first paper)',
                     fontsize=axis_label_size, y=0.04)
_syl = fig.supylabel(y_axis_label, fontsize=axis_label_size, x=0.013)
fig.tight_layout()
# the sup-labels centre on the whole figure, but the y-axis labels on the left and the x-label at
# the bottom shift the panels' true centre. Re-centre each label on the panels' plotting area.
# With the disp_f second row, span the y-label across BOTH rows (top row top .. bottom row bottom).
_p0 = axes[0].get_position()
_p2 = axes[2].get_position()
_p_bottom = (dispf_axes[0].get_position() if (USE_DISP_F and dispf_axes is not None) else _p0)
_syl.set_y(0.5 * (_p_bottom.y0 + _p0.y1))   # y-label centred across all panel rows
_sxl.set_x(0.5 * (_p0.x0 + _p2.x1))         # x-label on the panels' horizontal centre

out_base = os.path.join(
    save_dire,
    ED_FIG_NAME if is_ed_fig else
    f'country_avg_age_vs_efg_{agg_suffix}_mi{MIN_INSTANCES}{dispf_tag}{value_tag}')
fig.savefig(out_base + '.svg', format='svg')
fig.savefig(out_base + '.png', dpi=150)
plt.close()
print("saved", out_base + '.svg', "and .png")

# ---------------------------------------------------------------------------
# 5. SECOND figure: 1x4, one panel per (i+j+k)-denominator ratio. Same scatter layout as the
#    F/E/G panels above (one point per country, regression line + 95% CI band, called-out
#    country labels, 'r = ..' beside the line), x = the country's average academic age.
#    Foundation-family ratios take the Foundation red, disruption-family ones the disruption
#    blue. Its regression fits get their OWN cache file: sharing the F/E/G one would mean an
#    existing cache (which has no ijk rows) silently yields no line for these panels.
# ---------------------------------------------------------------------------
IJK_HILITE = {'f_ijk': HILITE['f'], 'dispf_ijk': HILITE['f'],
              'i_ijk': DISP_HILITE, 'imj_ijk': DISP_HILITE}

if PLOT_IJK:
    ijk_reg_cache_path = os.path.join(
        save_dire,
        f'country_avg_age_vs_ijk_regression_{agg_suffix}_mi{MIN_INSTANCES}'
        f'{disp_k_suffix}{value_tag}.csv')
    print("ijk_reg_cache_path", ijk_reg_cache_path)

    # per-ratio x/y: the subset of countries that have that ratio's mean
    ijk_xy = {}
    for _k, _ in IJK_VARIANTS:
        _m = ijk_maps[_k]
        ijk_xy[_k] = (
            np.array([a for a, c in zip(xvals, _codes) if c in _m], dtype=float),
            np.array([_m[c] for c in _codes if c in _m], dtype=float),
        )

    _ijk_reg_ok = False
    if (not force_recompute) and os.path.exists(ijk_reg_cache_path):
        _ijk_reg_df = pd.read_csv(ijk_reg_cache_path)
        _missing = [c for c in REG_COLS if c not in _ijk_reg_df.columns]
        if _missing:
            print("ijk regression cache is stale (missing %s) -> recomputing" % _missing)
        else:
            ijk_reg_stats = {str(row['index']): row.to_dict()
                             for _, row in _ijk_reg_df.iterrows()}
            print("loaded ijk regression cache:", ijk_reg_cache_path,
                  "| fits:", len(ijk_reg_stats))
            _ijk_reg_ok = True
    if not _ijk_reg_ok:
        print("ijk regression cache not found (or force_recompute) -> computing fits ...")
        ijk_reg_stats = {}
        for _k, _title in IJK_VARIANTS:
            _x, _y = ijk_xy[_k]
            if len(_x) >= 3:
                st = compute_regression(_x, _y)
                if st is not None:
                    ijk_reg_stats[_k] = {'panel': _title, 'index': _k,
                                         'x': 'avg_age', 'y': 'avg_' + _k, **st}
        _ijk_reg_df = pd.DataFrame([ijk_reg_stats[k] for k in ijk_reg_stats])[REG_COLS]
        _ijk_reg_df.to_csv(ijk_reg_cache_path, index=False)
        print("wrote ijk regression cache:", ijk_reg_cache_path)
        print(_ijk_reg_df.to_string(index=False))

    # width scaled from the 3-panel FIGSIZE so each panel keeps the same aspect
    fig2, axes2 = plt.subplots(1, 4, figsize=(FIGSIZE[0] / 3 * 4, FIGSIZE[1]))
    for ax, (vkey, vtitle) in zip(axes2, IJK_VARIANTS):
        vmap = ijk_maps[vkey]
        # keep only countries that have this ratio's mean; align x / size / code to them
        fx, fy, fsz, fcodes = [], [], [], []
        for xc, c, sz in zip(xvals, _codes, sizes):
            if c in vmap:
                fx.append(xc); fy.append(vmap[c]); fsz.append(sz); fcodes.append(c)
        fx = np.asarray(fx, dtype=float); fy = np.asarray(fy, dtype=float)
        fsz = np.asarray(fsz, dtype=float)
        hilite = IJK_HILITE[vkey]
        pt_colors = [hilite if c in COUNTRY_LABELS else GREY for c in fcodes]
        ax.scatter(fx, fy, s=fsz, c=pt_colors, alpha=0.7,
                   edgecolors='white', linewidths=0.5, zorder=2)
        st = ijk_reg_stats.get(vkey)
        if st is not None:
            draw_regression(ax, st)
        # country labels first, then r beside the line avoiding them
        label_anns = []
        if ANNOTATE:
            lx, ly, lt = [], [], []
            for xc, yc, c in zip(fx, fy, fcodes):
                name = COUNTRY_LABELS.get(c)
                if name is not None:   # only the called-out major countries get a label
                    lx.append(xc); ly.append(yc); lt.append(name)
            label_anns = add_labels_no_overlap(ax, lx, ly, lt, fontsize=8, color=hilite,
                                               draw_leader=True)
        if st is not None:
            annotate_r_beside_line(ax, st, avoid_anns=label_anns, fontsize=tick_size)
        # ratio name in the bottom-left corner. Smaller than the F/E/G panels' label_size+7:
        # these are formulas, not single words, and would otherwise run across the panel.
        ax.text(0.03, 0.03, vtitle, transform=ax.transAxes, fontsize=label_size + 2,
                ha='left', va='bottom', zorder=4)
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    _sxl2 = fig2.supxlabel('Average academic age (years since author\'s first paper)',
                           fontsize=axis_label_size, y=0.04)
    _syl2 = fig2.supylabel(y_axis_label, fontsize=axis_label_size, x=0.013)
    fig2.tight_layout()
    # re-centre the sup-labels on the panels' plotting area (same recipe as the figure above)
    _q0 = axes2[0].get_position()
    _q3 = axes2[-1].get_position()
    _syl2.set_y(0.5 * (_q0.y0 + _q0.y1))
    _sxl2.set_x(0.5 * (_q0.x0 + _q3.x1))

    ijk_out_base = os.path.join(
        save_dire,
        f'country_avg_age_vs_ijk_{agg_suffix}_mi{MIN_INSTANCES}{disp_k_suffix}{value_tag}')
    fig2.savefig(ijk_out_base + '.svg', format='svg')
    fig2.savefig(ijk_out_base + '.png', dpi=150)
    plt.close()
    print("saved", ijk_out_base + '.svg', "and .png")

print("finished")
