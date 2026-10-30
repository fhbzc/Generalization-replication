"""
WoS version of the OpenAlex venue_validation/plot_annual_review_efgd_distribution.py.

Distribution of E/F/G/D and I/(I+J) (raw value AND within-(year,domain) quantile)
for the two annual-review paper sets:
    * REVIEW papers     -> annual_review_papers.pkl            (review papers themselves)
    * REFERENCE papers  -> annual_review_paper_references.pkl  (papers they cite)
both from get_annual_review_papers_and_refs.py.

The per-paper metrics come from ../compute_efgd_quantile.py:
    META_DIRE/efg_computation/
        efgd_within_year_domain_quantile_y{off}{data_suffix}_rt_{rt}_ct_{ct}_ybefore_{max_year}.csv
which holds, per pid, the raw indices  e_y{off}, f_y{off}, g_y{off}, disp_y{off},
ij_ratio_y{off}  and their quantiles under all three groupings (_quantile,
_quantile_by_year, _quantile_global). QUANTILE_GROUP selects which quantile to plot.

The main grid has one index per ROW (F, E, G, D) x four COLUMNS, REVIEW vs REFERENCE
overlaid -> a 4 x 4 grid of histograms:
    (a) raw value                         (ALL papers)
    (b) quantile, no grouping / global    (ALL papers)
    (c) quantile, within publication year (ALL papers)
    (d) quantile, within (year, domain)   (papers WITH a domain only)
Only column (d) restricts to domain papers; (a)-(c) use all papers (the global / by-year
quantiles are defined for every paper). Panels carry NO title; only the mean is drawn as a
dashed vertical line; the legend (group name, n, mean AND median) sits in the top-right
corner; the top/right spines are dropped; per-panel y-labels are replaced by a single
'density' label on the far left, vertically centred over all rows. (The separate 2 x 2
figure below still uses the QUANTILE_GROUP quantile and draws its own mean annotation.)

Columns (a)-(c) need raw indices / quantiles for papers that have no domain too. The raw
values come from rebuilding the upstream E/F/G and disruption tables WITHOUT the domain
filter (same reference / citation / year filters); the global / by-year quantiles already
cover every paper in the compute_efgd_quantile.py table (only the (year, domain) quantile
is NaN for no-/multi-domain papers).

To avoid recomputing on every redraw, the per-paper values for the two groups are
cached to a CSV in ./figures; if that cache exists it is loaded directly and the
pkl/metric-table inputs are not touched (delete the cache to force a refresh).

Set k_require_after_focal to match compute_efgd_quantile.py / get_disruption.py: it
selects the disruption (D + I/J) variant. False -> the dropped-lower-bound K tagged
'_k_no_lower_bound' (k_suffix); True -> default K with no tag. It changes which disp_full
and quantile table are read, and the cache/figure names, so the two variants never collide.

detection_mode selects which review/reference pid sets are used:
    venue / title : the two get_annual_review_papers_and_refs.py pkls below.
    type          : NOT pkls -- the review set is derived in-script as the document-type-
                    review papers (set(pid2pub_year_with_review) - set(pid2pub_year)) and
                    their references (union over citing_pid2cited_pids_list_with_review);
                    'type' forces the '_with_review' metric tables.

Input  : META_DIRE/annual_review_papers{data_suffix}{mode_suffix}.pkl          (venue/title)
         META_DIRE/annual_review_paper_references{data_suffix}{mode_suffix}.pkl (venue/title)
         META_DIRE/pid2pub_year{_with_review,}.pkl                              (type: review-set diff)
         META_DIRE/citing_pid2cited_pids_list_with_review.pkl                   (type: references)
         META_DIRE/efg_computation/efgd_within_year_domain_quantile_y{off}{data_suffix}{k_suffix}{dispf_suffix}_*.csv   (quantiles, domain papers; dispf_suffix='_disp_f' when QUANTILE_TABLE_DISP_F, matching compute_efgd_quantile.py's use_disp_f -- that table is a superset, so it is read regardless of INCLUDE_DISP_F)
         META_DIRE/efg_computation/gen_V3_y{off}{data_suffix}.csv           (raw E/F/G, all papers)
         META_DIRE/disruption/disp_full[_y{year_offset}]{data_suffix}{k_suffix}.csv  (raw D + I/J, all papers;
                                                                        the _y tag only when year_offset != 5)
Cache  : venue_validation/figures/annual_review_efgd_ijratio/
             annual_review_efgd_ijratio_cache_y{off}{data_suffix}{k_suffix}_rt_{rt}_ct_{ct}_ybefore_{max_year}{mode_suffix}_withalldomain_allq.csv
             (raw values + all three quantile groupings; independent of QUANTILE_GROUP)
Output : venue_validation/figures/annual_review_efgd_ijratio/
             annual_review_efgd_ijratio_distribution_y{off}{data_suffix}_rt_{rt}_ct_{ct}_ybefore_{max_year}.svg
             (the full 4 x 4 grid)
         venue_validation/figures/annual_review_efgd_ijratio/subplots_{param_suffix}/
             {raw_col}_{panel_key}_{param_suffix}.svg
             (each grid cell saved standalone, no title; panel_key in
              {raw_all, quantile_global, quantile_by_year, quantile_year_domain};
              quantile_global is absent whenever select_max_year < max_year (see below);
              subdir keyed by the threshold params)
         venue_validation/figures/annual_review_efgd_ijratio/
             annual_review_dfeg_quantile_distribution_y{off}{data_suffix}_rt_{rt}_ct_{ct}_ybefore_{max_year}.svg
             (extra 2 x 2 grid, no titles: D / E (top) and F / G (bottom)
              within-(year,domain) quantile distributions)
             -> at the deliverable setting (year_offset=5, rt=1, ct=5, use_with_review=True,
                detection_mode='venue', k_require_after_focal=False,
                QUANTILE_GROUP='year_domain', INCLUDE_DISP_F=False) this 2 x 2 figure and the
                data table behind it are instead 'Extended Data Fig. 4.svg' + '.csv'
                (per panel and group: the plotted KDE curve plus that group's n/mean/median)
         venue_validation/figures/annual_review_efgd_ijratio/disp_f_quantile/
             annual_review_f_dispf_ijratio_quantile_distribution_y{off}{data_suffix}..._q_{QUANTILE_GROUP}.svg
             (extra 1 x 4 grid, no titles: Foundation / disp_f(>0) / I(I+J) / Generalization
              QUANTILE_GROUP quantile distributions, Review vs Reference overlaid;
              needs compute_efgd_quantile.py run with use_disp_f=True)
         venue_validation/figures/annual_review_efgd_ijratio/ijk_ratio_quantile/
             annual_review_ijk_ratio_quantile_distribution_y{off}{data_suffix}..._q_{QUANTILE_GROUP}.svg
             (extra 1 x 4 grid, no titles: f/(i+j+k) / disp_f/(i+j+k) / i/(i+j+k) /
              (i-j)/(i+j+k)=Disruption QUANTILE_GROUP quantile distributions, Review vs
              Reference overlaid; the three ratio panels need compute_efgd_quantile.py run
              with the new (i+j+k) ratios (use_disp_f=True for the disp_f one))
         venue_validation/figures/annual_review_efgd_ijratio/
             annual_review_gap_bootstrap_..._q_{QUANTILE_GROUP}.csv
             (no figure of its own: one row per metric with the Reference/Review means, the
              reference-minus-review gap and a bootstrap 95% CI of that gap -- read by
              combine_wos_openalex/plot_validation_quantile_gap.py. Covers every metric the
              cache carries, including ones no panel draws.)
         (transparent background)
The cache/figure file names carry the filter parameters so different settings
do not overwrite each other.
"""

import os
import sys
import json
import pickle

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

# all text uses Gill Sans MT
plt.rcParams['font.family'] = 'Gill Sans MT'
# Gill Sans MT lacks the Unicode minus glyph (U+2212), which renders as a box -> use the
# plain ASCII hyphen for negative tick labels instead.
plt.rcParams['axes.unicode_minus'] = False
# enlarge all text EXCEPT the x/y tick labels (those keep the matplotlib default).
plt.rcParams['axes.labelsize'] = 15    # x/y axis labels
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 16

# utils.py lives in the parent wos/ directory
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
WOS_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, WOS_DIR)
from utils import META_DIRE  # noqa: E402

# ---- parameters: must match compute_efgd_quantile.py to find its output file ----
year_offset = -1
reference_threshold = 1
citation_threshold = 5
# see wos/compute_efgd_quantile.py: at year_offset == -1 there is no elapsed window, so it
# writes the every-paper-before-2025 table and tags it '_full_cover'
if year_offset == -1:
    max_year = 2025
    input_name_tag = '_full_cover'
else:
    max_year = 2025 - year_offset
    input_name_tag = ''

# select_max_year narrows the LOADED tables further: only papers with pub_year <
# select_max_year are kept, the same strict-less-than they were built with. Set it equal to
# max_year for no extra narrowing.
#
#   * it must be <= max_year: papers the tables never held cannot be recovered here;
#   * the QUANTILES ARE NOT RECOMPUTED, so this is only sound for a ranking done WITHIN a
#     publication year -- QUANTILE_GROUP 'year_domain' or 'year'. Then every surviving paper's
#     rank is relative to its own same-year peers, all of which are either kept too or
#     irrelevant to it. A 'global' rank was computed against the whole table INCLUDING the
#     years now dropped, so it no longer describes the plotted population; the assert below
#     refuses that combination. (The 4x4 grid draws a '_quantile_global' row regardless of
#     QUANTILE_GROUP -- see the NOTE printed when narrowing is on.)
#   * the CACHE and every OUTPUT name carry select_max_year, not max_year, so a narrowed run
#     never reuses or overwrites the full one. The INPUT filenames keep max_year: that is the
#     table being read.
select_max_year = 2015
assert select_max_year <= max_year, (
    "select_max_year (%s) must be <= max_year (%s): the tables only hold papers with "
    "pub_year < %s." % (select_max_year, max_year, max_year))

# must match compute_efgd_quantile.py: read the '_with_review' variant when True
use_with_review = True

# how the review paper set is built (selects the inputs / cache / figure names):
#   'venue' -> annual_review_papers{data_suffix}.pkl                 (annual-review venues)
#   'title' -> annual_review_papers{data_suffix}_title_review.pkl    (title contains 'review')
#             (both from get_annual_review_papers_and_refs.py)
#   'type'  -> NOT a pkl: the review set is derived in-script as the document-type-review
#             papers = the pids that appear only once reviews are included, i.e.
#                 set(pid2pub_year_with_review) - set(pid2pub_year)
#             and their references = the union of citing_pid2cited_pids_list_with_review
#             over those pids. Because those pids only exist in the '_with_review' tables,
#             'type' forces use_with_review=True for the E/F/G, disruption and quantile reads.
# Only the review/reference pid sets differ between modes; the metric tables are otherwise
# mode-independent. mode_suffix is added to the cache/figure names so the modes never
# overwrite each other.
detection_mode = 'venue'   # 'venue', 'title' or 'type'
assert detection_mode in ('venue', 'title', 'type'), detection_mode
_MODE_SUFFIX = {'venue': '', 'title': '_title_review', 'type': '_type_review'}
mode_suffix = _MODE_SUFFIX[detection_mode]

# 'type' derives its review set from the with_review-vs-base pid difference, so its metrics
# only live in the '_with_review' tables: force use_with_review for that mode.
if detection_mode == 'type' and not use_with_review:
    print("detection_mode 'type' needs the _with_review tables; forcing use_with_review=True")
    use_with_review = True
data_suffix = '_with_review' if use_with_review else ''

# must match compute_efgd_quantile.py / get_disruption.py / arrange_disruption.py: the
# K-window disruption variant. False -> dropped-lower-bound K -> '_k_no_lower_bound';
# True -> default K -> ''. Affects the disp_full read, the quantile-table name and the
# cache/figure names so the two variants never collide.
k_require_after_focal = True
k_suffix = '' if k_require_after_focal else '_k_no_lower_bound'

# ---- which quantile to plot: choose the grouping the percentile rank was computed
#      within (all three are columns in the compute_efgd_quantile.py table). ----
#   'year_domain' -> {col}_quantile          (rank within pub_year x top_concept)
#   'year'        -> {col}_quantile_by_year   (rank within pub_year only)
#   'global'      -> {col}_quantile_global    (rank across all papers, no grouping)
QUANTILE_GROUP = 'year_domain'
_QUANTILE_SUFFIX = {
    'year_domain': '_quantile',
    'year': '_quantile_by_year',
    'global': '_quantile_global',
}
assert QUANTILE_GROUP in _QUANTILE_SUFFIX, QUANTILE_GROUP
quantile_suffix = _QUANTILE_SUFFIX[QUANTILE_GROUP]

# see select_max_year above: dropping publication years after the fact leaves a within-year
# rank intact but invalidates a global one
assert select_max_year == max_year or QUANTILE_GROUP in ('year_domain', 'year'), (
    "select_max_year (%s) != max_year (%s) narrows the population AFTER the quantiles were "
    "computed, which only stays valid for a WITHIN-YEAR ranking. QUANTILE_GROUP is %r: either "
    "set it to 'year_domain' / 'year', or re-run ../compute_efgd_quantile.py with "
    "max_year = %s and read that table." % (select_max_year, max_year, QUANTILE_GROUP,
                                            select_max_year))
if select_max_year != max_year:
    print("select_max_year != max_year: the '_quantile_global' column is DROPPED from the grid "
          "(see main_grid_cols below). A global rank was computed over the UN-narrowed table, "
          "so it cannot be drawn for this population -- and QUANTILE_GROUP='global' is refused "
          "outright by the assert above.")

print("==== hyper-parameters ====")
print("year_offset", year_offset)
print("reference_threshold", reference_threshold)
print("citation_threshold", citation_threshold)
print("max_year", max_year)
print("select_max_year", select_max_year)
print("use_with_review", use_with_review)
print("detection_mode", detection_mode)
print("mode_suffix", repr(mode_suffix))
print("k_require_after_focal", k_require_after_focal)
print("k_suffix", repr(k_suffix))
print("QUANTILE_GROUP", QUANTILE_GROUP, "-> column suffix", repr(quantile_suffix))
print("==========================")
# (INCLUDE_DISP_F / QUANTILE_TABLE_DISP_F are printed below, where they are defined)

# metrics to plot: (raw column, human label)
metrics = [
    (f'e_y{year_offset}',        'E (Extension)'),
    (f'f_y{year_offset}',        'F (Foundation)'),
    (f'g_y{year_offset}',        'G (Generalization)'),
    (f'disp_y{year_offset}',     'D (Disruption)'),
    (f'ij_ratio_y{year_offset}', 'I/(I+J)'),
]
label_by_raw = {raw_col: label for raw_col, label in metrics}

# the two overlaid paper sets: (label, colour). The actual pid lists are resolved per
# detection_mode when the cache is built (venue/title read pkls; type derives a pid diff).
groups = [
    ('Review papers',    'tab:red'),
    ('Reference papers', 'tab:blue'),
]
group_labels = [g[0] for g in groups]
group_color = {label: color for label, color in groups}
# short names shown on the legend (the internal group keys above stay unchanged so the
# cache 'group' column keeps matching).
group_display = {'Review papers': 'Reviews', 'Reference papers': 'References'}

# the raw index columns. The cache carries ALL THREE quantile groupings per index
# (year_domain / by_year / global) so the main grid can show them side by side; the 2x2
# figure and any single-group lookup still pick the QUANTILE_GROUP column out of these.
raw_cols = [raw_col for raw_col, _ in metrics]
ALL_QUANTILE_SUFFIXES = ['_quantile', '_quantile_by_year', '_quantile_global']
quant_cols = [f'{raw_col}{suf}' for raw_col in raw_cols for suf in ALL_QUANTILE_SUFFIXES]

# ---- disp_f (disruptive-foundation) quantiles, for the extra disp_f figure below ----
# Draw the extra 7-panel disp_f quantile figure. Needs compute_efgd_quantile.py to have been
# run with use_disp_f=True so its quantile table carries the three disp_f share columns +
# quantiles; the disp_f *quantile* columns are joined into the cache alongside E/F/G/D.
INCLUDE_DISP_F = True
# Which quantile TABLE to READ (input filename only). compute_efgd_quantile.py tags its
# output '_disp_f' when it was run with use_disp_f=True; that table is a SUPERSET of the
# untagged one (identical rows -- the disp_f columns come from a left join -- plus the
# disp_f columns), so it is safe to read even with INCLUDE_DISP_F=False: the extra columns
# are simply not selected. Set this to match the file that actually exists in
# META_DIRE/efg_computation/. It affects NOTHING else -- not what is plotted, not the
# cache/figure names, not the Extended Data Fig. 4 gate.
QUANTILE_TABLE_DISP_F = True
# drawing the disp_f panels obviously requires reading the disp_f table
assert QUANTILE_TABLE_DISP_F or not INCLUDE_DISP_F, \
    "INCLUDE_DISP_F=True needs QUANTILE_TABLE_DISP_F=True (the disp_f quantile table)"
print("INCLUDE_DISP_F", INCLUDE_DISP_F,
      "| QUANTILE_TABLE_DISP_F", QUANTILE_TABLE_DISP_F,
      "-> input table tag", repr('_disp_f' if QUANTILE_TABLE_DISP_F else ''))
dispf_raw_cols = [f'disruption_f_y{year_offset}',            # disp_f_0: down_count > 0
                  f'disruption_f_down_gt1_y{year_offset}',   # disp_f_1: down_count > 1
                  f'disruption_f_down_gt2_y{year_offset}']   # disp_f_2: down_count > 2
if INCLUDE_DISP_F:
    quant_cols = quant_cols + [f'{c}{suf}' for c in dispf_raw_cols
                               for suf in ALL_QUANTILE_SUFFIXES]

# ---- (i+j+k)-denominator ratio quantiles, for the extra 1x4 figure below ----
# The three new (i+j+k)-denominator ratios from compute_efgd_quantile.py:
#   f/(i+j+k) -> f_over_ijk, i/(i+j+k) -> i_over_ijk, disp_f/(i+j+k) -> dispf_over_ijk
#   (the last needs compute_efgd_quantile.py run with use_disp_f=True).
# The 4th panel of that figure, (i-j)/(i+j+k), IS the disruption index D and is already
# carried via disp_y{off}, so it is not re-listed here.
INCLUDE_IJK_RATIO = True
ijk_ratio_raw_cols = [f'f_over_ijk_y{year_offset}',
                      f'i_over_ijk_y{year_offset}',
                      f'dispf_over_ijk_y{year_offset}']
if INCLUDE_IJK_RATIO:
    quant_cols = quant_cols + [f'{c}{suf}' for c in ijk_ratio_raw_cols
                               for suf in ALL_QUANTILE_SUFFIXES]

# ---- OPTIONAL columns: no figure uses them; they exist only for the gap/bootstrap CSV
# (section 7) and are skipped there when the quantile table has not got them yet.
# E/(I+J+K) is the E counterpart of f_over_ijk (e_count_y{off} over the disruption
# denominator) and only appears once compute_efgd_quantile.py has been re-run with that
# column added; until then the quantile join reindexes them to all-NaN and the CSV drops them.
# NOTE a cache written before the column existed does not carry it at all -- after re-running
# compute_efgd_quantile.py, delete the cache (cache_path below) to pick it up.
OPTIONAL_RAW_COLS = [f'e_over_ijk_y{year_offset}']
quant_cols = quant_cols + [f'{c}{suf}' for c in OPTIONAL_RAW_COLS
                           for suf in ALL_QUANTILE_SUFFIXES]

# source-file column names, mirroring compute_efgd_quantile.py, used to rebuild the
# raw indices for papers that have NO domain (and therefore no quantile).
citation_col = f'citation_count_y{year_offset}'
disp_col = f'disp_y{year_offset}'
efg_cols = [f'e_y{year_offset}', f'f_y{year_offset}', f'g_y{year_offset}']
ratio_col = f'ij_ratio_y{year_offset}'
j_count_col = f'J_count_y{year_offset}'
i_count_col = f'I_count_y{year_offset}'

# filter-parameter suffix shared by the metric table, cache and figure names.
# k_suffix sits right after data_suffix to match compute_efgd_quantile.py's output name.
# NOTE ybefore_ carries select_max_year (the population actually plotted), not max_year (the
# tables that were read), so two runs off the same tables but narrowed differently keep their
# own cache and figures. With no narrowing the two are equal and every existing name is kept.
param_suffix = (f'y{year_offset}{data_suffix}{k_suffix}_'
                f'rt_{reference_threshold}_'
                f'ct_{citation_threshold}_'
                f'ybefore_{select_max_year}')
# two output-name suffixes, both carrying mode_suffix so venue/title/type never collide:
#   full_param_suffix : params + mode only. Used by the CACHE and the FULL 4x4 grid (+ its
#                       standalone subplots), which draw ALL THREE quantile groupings and
#                       are therefore INDEPENDENT of QUANTILE_GROUP.
#   out_param_suffix  : the above PLUS '_q_{QUANTILE_GROUP}'. Used only by the 2x2 figure
#                       (and the shared legend JSON), which plot the single selected group.
# The INPUT metric table is independent of both and keeps using param_suffix.
full_param_suffix = param_suffix + mode_suffix
out_param_suffix = full_param_suffix + f'_q_{QUANTILE_GROUP}'

# figures (and the redraw cache) go into a dedicated subdirectory under ./figures
fig_dire = os.path.join(THIS_DIR, 'figures', 'annual_review_efgd_ijratio')
os.makedirs(fig_dire, exist_ok=True)
# the cache holds raw values + ALL THREE quantile groupings for the no-domain-inclusive
# paper set, so it is independent of QUANTILE_GROUP -> keyed by full_param_suffix.
# '_withalldomain' marks the schema that also carries the no-domain papers (raw value
# present, year/domain quantile NaN); '_allq' marks the all-quantile-groupings schema so
# it never loads an older single-group cache.
# '_dispf' marks the schema that also carries the three disp_f quantile columns, so a
# pre-disp_f cache is never reused (it would be missing those columns).
_dispf_cache_tag = '_dispf' if INCLUDE_DISP_F else ''
# '_ijk' marks the schema that also carries the three (i+j+k)-ratio quantile columns, so a
# pre-ijk cache is never reused (it would be missing those columns).
_ijk_cache_tag = '_ijk' if INCLUDE_IJK_RATIO else ''
cache_path = os.path.join(
    fig_dire,
    f'annual_review_efgd_ijratio_cache_{full_param_suffix}_withalldomain_allq{_dispf_cache_tag}{_ijk_cache_tag}.csv')


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# ==================================================
# 1. obtain the long-form cache (group, pid, <raw_cols>, <quant_cols>):
#    load from CSV cache if present, otherwise build it from the inputs and save.
#
#    Two tables feed the cache:
#      * all_raw : the compute_efgd_quantile.py pipeline WITHOUT the domain filter, so
#                  it keeps papers that have no domain (and hence no quantile). Gives
#                  the raw E/F/G/D + I/(I+J) index for EVERY paper passing the
#                  reference / citation / year filters.  -> the "all papers" panel.
#      * quant   : the domain-filtered quantile table from compute_efgd_quantile.py,
#                  carrying the *_quantile columns. Its pids are a subset of all_raw's.
#    A group's papers are taken from all_raw, then the quantile columns are joined on
#    (NaN where the paper has no domain).
# ==================================================
value_cache_hit = os.path.exists(cache_path)   # False -> value cache rebuilt -> curves stale
if value_cache_hit:
    # pid kept as str (WoS ids are digit-like and would otherwise lose leading zeros /
    # be read as mixed int/float -> DtypeWarning)
    cache = pd.read_csv(cache_path, dtype={'pid': str})
    print("loaded cache:", cache_path, cache.shape)
    # a cache written before an optional column existed simply has not got it: add it as NaN
    # so every lookup below sees a uniform column set (the consumers skip all-NaN columns).
    absent = [c for c in quant_cols if c not in cache.columns]
    if absent:
        for c in absent:
            cache[c] = np.nan
        print(f"  NOTE: {len(absent)} quantile column(s) absent from this cache -> NaN "
              f"(e.g. {absent[:3]}). Delete the cache to rebuild them from the quantile table.")
else:
    print("cache not found, building from inputs ...")

    # ---- all_raw: raw indices for every paper (no domain filter) ----
    # pid kept as str: WoS ids are digit-like and would otherwise lose leading zeros.
    efg_path = os.path.join(META_DIRE, 'efg_computation',
                            f'gen_V3_y{year_offset}{data_suffix}.csv')
    assert os.path.exists(efg_path), efg_path
    df_efg = pd.read_csv(efg_path, dtype={'pid': str})
    df_efg = df_efg[['pid', 'pub_year', 'up_refer_count', citation_col] + efg_cols].copy()
    # same reference / citation / max_year filter as compute_efgd_quantile.py
    df_efg = df_efg[
        (df_efg['up_refer_count'] >= reference_threshold) &
        (df_efg[citation_col] >= citation_threshold) &
        (df_efg['pub_year'] < select_max_year)      # select_max_year == max_year unless narrowed
    ].copy()

    # arrange_disruption.py tags the window only when it is not the default 5
    disp_year_tag = '' if year_offset == 5 else f'_y{year_offset}'
    disp_path = os.path.join(META_DIRE, 'disruption',
                             f'disp_full{disp_year_tag}{data_suffix}{k_suffix}.csv')
    assert os.path.exists(disp_path), disp_path
    df_disp = pd.read_csv(disp_path,
                          usecols=['pid', disp_col, j_count_col, i_count_col],
                          dtype={'pid': str})
    all_raw = df_efg.merge(df_disp, on='pid', how='inner')   # keep papers that also have D
    # R = I / (I + J); guard against a 0 denominator
    denom_ij = all_raw[i_count_col] + all_raw[j_count_col]
    all_raw[ratio_col] = (all_raw[i_count_col] / denom_ij).where(denom_ij > 0, 0.0)
    all_raw = all_raw.set_index('pid')[raw_cols]
    print("all_raw (no domain filter) shape:", all_raw.shape)

    # ---- quant: the domain-filtered quantile table ----
    # compute_efgd_quantile.py appends a '_disp_f' suffix to this filename when it was run with
    # use_disp_f=True (the disp_f-inclusive quantile table). QUANTILE_TABLE_DISP_F selects which
    # of the two to read -- the disp_f one is a superset, so it works for any INCLUDE_DISP_F.
    # The suffix goes in the INPUT filename ONLY; param_suffix (used for the cache/figure names)
    # is left unchanged.
    input_dispf_suffix = '_disp_f' if QUANTILE_TABLE_DISP_F else ''
    quant_dire = os.path.join(META_DIRE, 'efg_computation')
    csv_path = os.path.join(
        quant_dire,
        f'efgd_within_year_domain_quantile_y{year_offset}{data_suffix}{k_suffix}{input_dispf_suffix}_'
        f'rt_{reference_threshold}_ct_{citation_threshold}_'
        f'ybefore_{max_year}{input_name_tag}.csv'
    )
    if not os.path.exists(csv_path):
        # the parameters here must match the compute_efgd_quantile.py run that wrote the table;
        # list what IS there so the mismatching parameter is obvious (a '_disp_f' tag on disk
        # means that run had use_disp_f=True -> set QUANTILE_TABLE_DISP_F accordingly).
        found = sorted(f for f in os.listdir(quant_dire)
                       if f.startswith('efgd_within_year_domain_quantile_')) \
            if os.path.isdir(quant_dire) else []
        raise FileNotFoundError(
            f'quantile table not found:\n  {csv_path}\n'
            f'quantile tables present in {quant_dire}:\n  '
            + ('\n  '.join(found) if found else '(none -- run ../compute_efgd_quantile.py first)'))
    quant_full = pd.read_csv(csv_path, dtype={'pid': str}).set_index('pid')
    # the extra pub_year cut, matching the one applied to df_efg above. The quantile columns
    # are left exactly as they are -- see select_max_year in the settings for why that is sound
    # only for a within-year ranking.
    if select_max_year != max_year:
        assert 'pub_year' in quant_full.columns, (
            f"select_max_year ({select_max_year}) needs a pub_year column, which "
            f"{os.path.basename(csv_path)} does not have.")
        n_before_sel = len(quant_full)
        quant_full = quant_full[quant_full['pub_year'] < select_max_year]
        print(f"select_max_year={select_max_year}: quantile table {n_before_sel} -> "
              f"{len(quant_full)} papers (dropped {n_before_sel - len(quant_full)} published "
              f"in {select_max_year}-{max_year - 1}); quantiles NOT recomputed")
        assert len(quant_full), (
            f"no paper is left with pub_year < {select_max_year}; check select_max_year.")
    missing_q = [c for c in quant_cols if c not in quant_full.columns]
    if missing_q:
        print(f"WARNING: quantile table is missing {len(missing_q)} column(s), filled with NaN "
              f"(e.g. {missing_q[:3]}). For disp_f, re-run compute_efgd_quantile.py with "
              f"use_disp_f=True.")
    # reindex (not [quant_cols]) so absent columns become NaN instead of raising KeyError
    quant = quant_full.reindex(columns=quant_cols)
    print("loaded quantile table (domain papers):", quant.shape)

    # ---- resolve each group's pid list for the chosen detection_mode ----
    #   venue / title : read the two get_annual_review_papers_and_refs.py pkls.
    #   type          : derive in-script -- review papers = the document-type-review pids
    #                   that appear only once reviews are included
    #                       set(pid2pub_year_with_review) - set(pid2pub_year)
    #                   reference papers = the union of those papers' references from
    #                   citing_pid2cited_pids_list_with_review.pkl.
    if detection_mode == 'type':
        wr_pids = set(load_pkl(os.path.join(META_DIRE, 'pid2pub_year_with_review.pkl')))
        base_pids = set(load_pkl(os.path.join(META_DIRE, 'pid2pub_year.pkl')))
        review_native = sorted(wr_pids - base_pids)   # the extra (review-type) papers
        print(f"type mode: {len(wr_pids)} with_review pids - {len(base_pids)} base pids "
              f"-> {len(review_native)} review-type papers")
        citing2cited = load_pkl(
            os.path.join(META_DIRE, 'citing_pid2cited_pids_list_with_review.pkl'))
        ref_native = set()
        for pid in review_native:
            refs = citing2cited.get(pid)
            if refs:
                ref_native.update(refs)
        print(f"type mode: their references -> {len(ref_native)} unique pids")
        group_pids = {
            'Review papers':    [str(p) for p in review_native],
            'Reference papers': [str(p) for p in sorted(ref_native)],
        }
    else:
        group_pids = {}
        for label, pkl_name in (
            ('Review papers',    f'annual_review_papers{data_suffix}{mode_suffix}.pkl'),
            ('Reference papers', f'annual_review_paper_references{data_suffix}{mode_suffix}.pkl'),
        ):
            group_pids[label] = [str(p) for p in load_pkl(os.path.join(META_DIRE, pkl_name))]

    parts = []
    for label in group_labels:
        pids = pd.Index(group_pids[label]).unique()
        present = pids.intersection(all_raw.index)
        n_domain = len(pids.intersection(quant.index))
        print(f"{label}: {len(pids)} pids, {len(present)} with a raw value "
              f"(any/no domain), of which {n_domain} have a domain/quantile")
        # raw indices for every present paper, with the quantile columns joined on
        # (NaN where the paper has no domain)
        part = all_raw.loc[present].join(quant)
        part.insert(0, 'group', label)
        part.index.name = 'pid'
        parts.append(part.reset_index())
    cache = pd.concat(parts, ignore_index=True)
    cache.to_csv(cache_path, index=False)
    print("wrote cache:", cache_path, cache.shape)

# ==================================================
# 2. plot: 5 metrics (rows) x 3 panels (cols), REVIEW vs REFERENCE overlaid, with
#    mean (dashed) and median (dotted) vertical lines per group. The three panels are
#      (a) raw value, domain papers only (those that have a quantile)
#      (b) raw value, ALL papers (with or without a domain/quantile)
#      (c) within-(year,domain) quantile (domain papers only)
# ==================================================
# number of points at which the KDE curve is evaluated across the x-range (a smooth
# line replaces the old step histogram; more points -> smoother curve). The ct=5/rt=1/venue
# setting uses a finer grid (512); every other parameter combination uses 100.
if citation_threshold == 5 and reference_threshold == 1 and detection_mode == 'venue':
    n_kde_points = 512
else:
    n_kde_points = 100


def _kde_curve(vals, xs):
    """Gaussian-KDE density of `vals` evaluated on the grid `xs`. Falls back to a
    normalised histogram interpolated onto `xs` when KDE is undefined (fewer than 2
    points or zero variance -> singular covariance)."""
    vals = np.asarray(vals, dtype=float)
    if len(vals) >= 2 and np.ptp(vals) > 0:
        try:
            return gaussian_kde(vals)(xs)
        except np.linalg.LinAlgError:
            pass
    # degenerate fallback: coarse histogram interpolated onto the grid
    counts, edges = np.histogram(vals, bins=min(50, max(1, len(vals))), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return np.interp(xs, centers, counts, left=0.0, right=0.0)


# ---- KDE-curve cache -------------------------------------------------------------------
# The per-panel KDE curves (+ each group's mean / median / n) are the expensive part of a
# redraw -- gaussian_kde runs over the large Reference set every time. Compute each panel ONCE
# and cache it to a pkl keyed by the plot params + grid size, so later redraws replot the cached
# curves instead of recomputing. Panel id = (col, quant_col when domain-filtered else None,
# domain_only); the same id is reused across figures (e.g. the 2x2 and 1x3 share panels). The
# cache is only trusted when the value cache was loaded (not just rebuilt) and its key matches.
curve_cache_path = os.path.join(
    fig_dire, f'annual_review_kde_curves_{full_param_suffix}{_dispf_cache_tag}{_ijk_cache_tag}.pkl')
_curve_key = {'schema': 'kde_v1', 'param': full_param_suffix,
              'dispf': bool(INCLUDE_DISP_F), 'ijk': bool(INCLUDE_IJK_RATIO),
              'n_kde': n_kde_points}
_curve_cache = {}
if value_cache_hit and os.path.exists(curve_cache_path):
    with open(curve_cache_path, 'rb') as f:
        _cc = pickle.load(f)
    if _cc.get('key') == _curve_key:
        _curve_cache = _cc['panels']
        print("loaded KDE-curve cache:", curve_cache_path, "|", len(_curve_cache), "panels")
    else:
        print("KDE-curve cache config changed -> rebuilding:", curve_cache_path)


def _panel_curves(col, quant_col, domain_only):
    """(xs, [(glabel, color, n, mean, median, ys), ...]) for a panel; gaussian_kde runs once and
    the result is cached (keyed by the panel id) so redraws reuse it. Returns (None, []) if empty."""
    pid = (col, quant_col if domain_only else None, bool(domain_only))
    if pid in _curve_cache:
        e = _curve_cache[pid]
        return e['xs'], e['groups']
    sub = cache[cache[quant_col].notna()] if domain_only else cache
    all_vals = pd.to_numeric(sub[col], errors='coerce').dropna()
    if len(all_vals) == 0:
        _curve_cache[pid] = {'xs': None, 'groups': []}
        return None, []
    lo, hi = np.nanpercentile(all_vals, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(all_vals.min()), float(all_vals.max())
    if lo == hi:
        hi = lo + 1.0
    xs = np.linspace(lo, hi, n_kde_points)
    groups = []
    for glabel in group_labels:
        vals = pd.to_numeric(sub.loc[sub['group'] == glabel, col],
                             errors='coerce').dropna().values
        if len(vals) == 0:
            continue
        groups.append((glabel, group_color[glabel], int(len(vals)),
                       float(np.mean(vals)), float(np.median(vals)), _kde_curve(vals, xs)))
    _curve_cache[pid] = {'xs': xs, 'groups': groups}
    return xs, groups


def _decimals_to_distinguish(values, base=3, max_dec=4):
    """Smallest number of decimals (>= base) at which all `values` format to DISTINCT
    strings -- i.e. the precision needed to tell the groups apart. With fewer than 2
    finite values nothing needs distinguishing, so `base` is returned. max_dec caps the
    precision so near-/exactly-equal values (e.g. two medians both = 2/3) don't blow the
    label up to 15 decimals; if they still match at max_dec, max_dec is returned."""
    vals = [v for v in values if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return base
    d = base
    while d < max_dec:
        formatted = [f'{v:.{d}f}' for v in vals]
        if len(set(formatted)) == len(formatted):
            break
        d += 1
    return d


def draw_panel(ax, col, quant_col, domain_only, title, xlabel, show_ylabel=True,
               annotate_mean=False, draw_median=True, ylabel='density',
               annotate_fontsize=8, legend_loc=None, legend_fontsize=None,
               annotate_group_name=False, annotate_median=True):
    """Draw one REVIEW-vs-REFERENCE overlaid histogram onto ax. When title is None no
    title is set (used for the standalone subplot files). xlabel is the metric name shown
    on the x-axis (E/F/G/D/...). show_ylabel controls whether the 'density' y-axis label
    is drawn (used to label only the leftmost column of a grid). annotate_mean adds a
    coloured 'mean=...' text (with '(median=...)' on the line below it) beside each
    group's mean vertical line. draw_median controls
    whether the dotted median line is drawn (the median is always recorded in the returned
    legend entry regardless). legend_loc, when not None (e.g. 'lower left'), draws a
    matplotlib legend of the per-group (n, mean, median) labels at that location; left None
    no legend is drawn on the figure. annotate_group_name (only meaningful with
    annotate_mean) prepends each group's name (Reviews / References) on a line above its
    mean/median text, in that group's colour (red for Reviews, blue for References).
    Returns the list of per-group legend entries (empty when there is no data)."""
    # per-panel KDE curves (+ mean/median/n per group), computed once and cached across redraws
    xs, drawn = _panel_curves(col, quant_col, domain_only)
    if not drawn:                    # no data for this panel
        if title is not None:
            ax.set_title(title)
        return []
    # drawn: list of (glabel, color, n, mean_v, median_v, ys)

    # print how many Review / Reference papers pass this panel's filter (the domain_only
    # panels keep only papers whose quantile is not NaN; the others use every paper).
    _flt = f'domain-only ({quant_col} not NaN)' if domain_only else 'all papers'
    _counts = ', '.join(f'{group_display.get(gl, gl)}={n:,}'
                        for gl, _c, n, _m, _md, _ys in drawn)
    print(f"[panel col={col} | filter={_flt}] {_counts}")

    # decimals needed to tell Review from Reference apart, independently for mean / median
    mean_dec = _decimals_to_distinguish([m for _, _, _, m, _, _ in drawn])
    median_dec = _decimals_to_distinguish([md for _, _, _, _, md, _ in drawn])

    # second pass: draw with the chosen precision. The legend text is NOT drawn on the
    # figure (legends are suppressed); instead each entry is collected and returned so the
    # caller can persist it to JSON for lookup.
    # to place the 'mean=...' labels in empty space rather than on top of the lines, the
    # group with the smaller mean gets its label to the LEFT of its line and the larger
    # to the RIGHT, so the two labels open away from each other (and off the dashed lines)
    means_sorted = sorted(m for _, _, _, m, _, _ in drawn)

    # the empty vertical band near the mean lines differs per panel, so pick the label y
    # per panel: look at the curve heights in the bins spanning the two means and put the
    # labels in whichever band (above or below the curves) has more room.
    ann_y = {}  # gi -> (y_frac in axes coords, vertical alignment)
    if annotate_mean and drawn:
        dens = [ys for _, _, _, _, _, ys in drawn]  # KDE curves on the grid xs
        ymax = max((d.max() for d in dens), default=0.0) or 1.0
        means_all = [m for _, _, _, m, _, _ in drawn]
        band = (xs >= min(means_all)) & (xs <= max(means_all))
        if not band.any():  # means fall between grid points -> use the nearest point
            band[int(np.argmin(np.abs(xs - means_all[0])))] = True
        local_max = max(d[band].max() for d in dens) / ymax  # top of curves in the band
        local_min = min(d[band].min() for d in dens) / ymax  # bottom of curves in the band
        if (1.0 - local_max) >= local_min:        # more room above the curves
            base, step, va = min(local_max + 0.05, 0.97), 0.07, 'bottom'
        else:                                     # more room below: drop to the bottom,
            base, step, va = 0.06, 0.07, 'bottom'  # the whole band under the curve is empty
        for gi in range(len(drawn)):
            ann_y[gi] = (min(max(base + gi * step, 0.02), 0.98), va)

    legend_entries = []
    for gi, (glabel, color, n, mean_v, median_v, ys) in enumerate(drawn):
        legend_label = (f'{group_display.get(glabel, glabel)} (n={n:,}, '
                        f'mean={mean_v:.{mean_dec}f}, median={median_v:.{median_dec}f})')
        # smooth KDE curve in place of the old step histogram
        ax.plot(xs, ys, linewidth=1.8, color=color, label=legend_label)
        # mean = dashed, median = dotted, in the group colour
        ax.axvline(mean_v, color=color, linestyle='--', linewidth=1.3, alpha=0.9)
        if draw_median:
            ax.axvline(median_v, color=color, linestyle=':', linewidth=1.6, alpha=0.9)
        # optional coloured 'mean=...' text beside each group's mean line, offset to the
        # emptier side and staggered vertically so the two never overlap or sit on a line
        if annotate_mean:
            on_right = mean_v >= means_sorted[-1]  # the larger mean opens to the right
            ha = 'left' if on_right else 'right'
            dx = 5 if on_right else -5             # points away from the line
            y_frac, va = ann_y[gi]                 # empty band chosen per panel
            # mean on the first line, median in parentheses on the line below it;
            # optionally the group name (Reviews / References) on a line above them,
            # placed in the empty space beside that group's coloured mean line.
            if annotate_median:
                ann_text = f'mean={mean_v:.{mean_dec}f}\n(median={median_v:.{median_dec}f})'
            else:
                ann_text = f'mean={mean_v:.{mean_dec}f}'
            if annotate_group_name:
                ann_text = f'{group_display.get(glabel, glabel)}\n' + ann_text
            ax.annotate(ann_text,
                        xy=(mean_v, y_frac),
                        xycoords=blended_transform_factory(ax.transData, ax.transAxes),
                        xytext=(dx, 0), textcoords='offset points',
                        color=color, fontsize=annotate_fontsize, ha=ha, va=va,
                        clip_on=False)
        legend_entries.append({
            'group': glabel,
            'color': color,
            'n': n,
            'mean': mean_v,
            'median': median_v,
            'mean_str': f'{mean_v:.{mean_dec}f}',
            'median_str': f'{median_v:.{median_dec}f}',
            'legend_label': legend_label,
        })

    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    if show_ylabel:
        ax.set_ylabel(ylabel)
    # optional on-figure legend (the per-group n/mean/median labels) at legend_loc;
    # otherwise legends are not drawn (the info is still returned / persisted to JSON)
    if legend_loc is not None and drawn:
        # legend_fontsize=None -> use the rcParams 'legend.fontsize' default
        ax.legend(loc=legend_loc, fontsize=legend_fontsize, frameon=False)
    return legend_entries


# standalone subplots (cells of the full 4x4 grid) go into their own subdirectory keyed by
# the threshold params + mode; they draw all three quantile groupings, so the directory is
# independent of QUANTILE_GROUP (full_param_suffix, no '_q_').
subplot_dire = os.path.join(fig_dire, f'subplots_{full_param_suffix}')
os.makedirs(subplot_dire, exist_ok=True)

# legend text is no longer drawn on the figures; collect it here so it can be written to
# a JSON sidecar for lookup. Keyed by figure -> panel -> list of per-group entries.
legend_records = {'main_grid': {}, 'dfeg_quantile_2x2': {}, 'fdg_quantile_1x4': {},
                  'ijk_quantile_1x4': {}}

# main grid: one index per ROW (F, E, G, D order) x four COLUMNS:
#   raw value | quantile (no group / global) | quantile (by year) | quantile (year/domain)
# Only the last column (year/domain) restricts to papers that have a domain; the other
# three use ALL papers (the global / by-year quantiles are defined for every paper).
main_grid_rows = [
    f'f_y{year_offset}',     # F (Foundation)
    f'e_y{year_offset}',     # E (Extension)
    f'g_y{year_offset}',     # G (Generalization)
    f'disp_y{year_offset}',  # D (Disruption)
]
# (column suffix -- None means the raw value, domain-filter?, panel label, filename key)
main_grid_cols = [
    (None,                False, 'raw value',              'raw_all'),
    ('_quantile_global',  False, 'quantile (no group)',    'quantile_global'),
    ('_quantile_by_year', False, 'quantile (by year)',     'quantile_by_year'),
    ('_quantile',         True,  'quantile (by year, domain)', 'quantile_year_domain'),
]
# A global rank was computed against the whole table INCLUDING the years select_max_year drops,
# so once the population is narrowed that column describes a different set of papers from every
# other panel. It is REMOVED rather than drawn with a caveat -- a figure should not contain a
# panel its own caption has to disown. The grid becomes 4x3, which the layout below picks up
# from len(main_grid_cols), and the Disruption global standalone subplot disappears with it.
# (QUANTILE_GROUP='global' is a separate case: the assert in the settings refuses it outright.)
if select_max_year != max_year:
    _n_before = len(main_grid_cols)
    main_grid_cols = [c for c in main_grid_cols if c[0] != '_quantile_global']
    assert len(main_grid_cols) == _n_before - 1, main_grid_cols
    print(f"select_max_year={select_max_year}: dropped the '_quantile_global' column -> the "
          f"grid is {len(main_grid_rows)}x{len(main_grid_cols)}")

fig, axes = plt.subplots(len(main_grid_rows), len(main_grid_cols),
                         figsize=(5.5 * len(main_grid_cols), 3.3 * len(main_grid_rows)))

def _despine(ax):
    """Drop the top and right border on a panel."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


for r, raw_col in enumerate(main_grid_rows):
    label = label_by_raw[raw_col]
    # the year/domain quantile column marks domain membership (used only when domain_only)
    domain_quant_col = f'{raw_col}_quantile'
    for c, (suf, domain_only, kind, panel_key) in enumerate(main_grid_cols):
        col = raw_col if suf is None else f'{raw_col}{suf}'
        # x-axis shows the metric name (F/E/G/D); quantile columns append the grouping
        xlabel = label if suf is None else f'{label} {kind}'
        # (a) panel inside the shared grid -- no title; only the mean line is drawn, and the
        #     legend (group name, n, mean, median) sits in the top-right corner. Per-panel
        #     y-labels are suppressed in favour of one shared 'density' label (set below).
        entries = draw_panel(axes[r, c], col, domain_quant_col, domain_only,
                             None, xlabel, show_ylabel=False,
                             draw_median=False, legend_loc='upper right',
                             legend_fontsize=9)
        _despine(axes[r, c])
        legend_records['main_grid'][f'{raw_col}_{panel_key}'] = {
            'metric': label, 'panel': kind, 'xlabel': xlabel, 'entries': entries,
        }

        # (b) the same panel saved on its own -- identical styling, distinguished by filename.
        #     Only the Disruption raw-value and the Disruption quantile (no group / global)
        #     panels are exported standalone; all other cells live in the 4x4 grid only.
        if (raw_col == f'disp_y{year_offset}'
                and panel_key in ('raw_all', 'quantile_global')):
            fig_single, ax_single = plt.subplots(figsize=(6, 4))
            draw_panel(ax_single, col, domain_quant_col, domain_only, None, xlabel,
                       show_ylabel=False, draw_median=False, legend_loc='upper right',
                       legend_fontsize=9)
            _despine(ax_single)
            # smaller x/y labels on the standalone subplots
            ax_single.set_xlabel(xlabel, fontsize=11)
            fig_single.tight_layout(rect=[0.05, 0, 1, 1])
            # single shared 'density' label on the left, vertically centred
            fig_single.text(0.02, 0.5, 'Density', rotation=90, va='center', ha='center',
                            fontsize=12)
            single_path = os.path.join(
                subplot_dire, f'{raw_col}_{panel_key}_{full_param_suffix}.svg')
            fig_single.savefig(single_path, format='svg', transparent=True)
            plt.close(fig_single)

# no figure-level title; one shared 'density' y-label on the far left, vertically centred
# over all rows (per-panel y-labels are suppressed above).
fig.tight_layout(rect=[0.03, 0, 1, 1])
fig.text(0.015, 0.5, 'Density', rotation=90, va='center', ha='center', fontsize=20)

# ==================================================
# 3. save
# ==================================================
save_path = os.path.join(
    fig_dire, f'annual_review_efgd_ijratio_distribution_{full_param_suffix}.svg')
fig.savefig(save_path, format='svg', transparent=True)
print("saved figure to:", save_path)
print("saved single subplots to:", subplot_dire)

# ==================================================
# 4. extra figure: the within-(year,domain) quantile distributions for
#    D, F, E, G drawn together in a single 2 x 2 grid (REVIEW vs REFERENCE
#    overlaid, domain papers only). Top row: D, E; bottom row: F, G. No titles.
# ==================================================
quantile_order = [
    f'disp_y{year_offset}',   # D (Disruption)  -> top-left
    f'e_y{year_offset}',      # E (Extension)   -> top-right
    f'f_y{year_offset}',      # F (Foundation)  -> bottom-left
    f'g_y{year_offset}',      # G (Generalization) -> bottom-right
]

fig_q, axes_q = plt.subplots(2, 2, figsize=(12, 9))
for idx, (ax_q, raw_col) in enumerate(zip(axes_q.flat, quantile_order)):
    label = label_by_raw[raw_col]
    quant_col = f'{raw_col}{quantile_suffix}'
    # 2x2 figure: only the mean line (no median), with a coloured 'mean=...' annotation.
    # per-axis x/y labels are suppressed in favour of a single shared label (set below).
    # every panel labels the two coloured mean lines with the group name -- 'Reviews' beside
    # the red line, 'References' beside the blue line (each drawn in its own group colour, in
    # the empty space next to that line).
    entries = draw_panel(ax_q, quant_col, quant_col, True, None,
                         '', show_ylabel=False,
                         annotate_mean=True, draw_median=False, annotate_fontsize=15,
                         annotate_group_name=True)
    ax_q.tick_params(labelsize=14)
    # drop the top and right borders
    ax_q.spines['top'].set_visible(False)
    ax_q.spines['right'].set_visible(False)
    # metric name (the word in parentheses, e.g. 'Disruption') in the top-right corner
    metric_name = label.split('(')[-1].rstrip(')').strip()
    ax_q.text(0.97, 0.97, metric_name, transform=ax_q.transAxes,
              ha='right', va='top', fontsize=20)
    legend_records['dfeg_quantile_2x2'][quant_col] = {
        'metric': label, 'panel': f'{QUANTILE_GROUP} quantile',
        'xlabel': 'Index quantile', 'entries': entries,
    }

# single shared axis labels: x centred under the bottom row, y centred on the left
fig_q.supxlabel('Index quantile', fontsize=20)
fig_q.supylabel('Density', fontsize=20)
fig_q.tight_layout()

# This D/E/F/G 2x2 at the paper's default settings IS the Extended Data deliverable, so it and
# the data table behind it are named after the figure; every other setting (and every other
# figure this script draws) keeps its descriptive name. The gate lists every parameter that
# changes what the panels show, so no other run can land on this name.
ED_FIG_NAME = 'Extended Data Fig. 4'
is_ed_fig = (year_offset == 5 and reference_threshold == 1 and citation_threshold == 5
             and use_with_review and detection_mode == 'venue'
             and not k_require_after_focal and QUANTILE_GROUP == 'year_domain'
             and not INCLUDE_DISP_F)
print("is_ed_fig", is_ed_fig, ("-> %s" % ED_FIG_NAME) if is_ed_fig else "")

q_stem = (ED_FIG_NAME if is_ed_fig else
          f'annual_review_dfeg_quantile_distribution_{out_param_suffix}')
save_path_q = os.path.join(fig_dire, q_stem + '.svg')
fig_q.savefig(save_path_q, format='svg', transparent=True)
print("saved D/F/E/G quantile figure to:", save_path_q)

# the data table behind the 2x2: for every panel and group, the plotted KDE curve plus that
# group's n / mean / median (the same numbers the panel annotates), so the figure can be
# redrawn or checked from the CSV alone. Reuses _panel_curves, so nothing is recomputed.
q_rows = []
for raw_col in quantile_order:
    quant_col_q = f'{raw_col}{quantile_suffix}'
    xs_q, groups_q = _panel_curves(quant_col_q, quant_col_q, True)
    if xs_q is None:
        continue
    for glabel_q, _color_q, n_q, mean_q, median_q, ys_q in groups_q:
        for x_q, y_q in zip(xs_q, ys_q):
            q_rows.append({
                'metric': label_by_raw[raw_col], 'value_col': quant_col_q,
                'group': glabel_q, 'n': n_q, 'mean': mean_q, 'median': median_q,
                'x': float(x_q), 'density': float(y_q),
            })
q_csv_path = os.path.join(fig_dire, q_stem + '.csv')
pd.DataFrame(q_rows).to_csv(q_csv_path, index=False)
print("saved D/F/E/G quantile figure data to:", q_csv_path)

# ==================================================
# 4b. extra figure: Foundation / disp_f(>0) / I(I+J) / Generalization within-(year,domain)
#     quantile distributions drawn together in a single 1 x 4 grid (REVIEW vs REFERENCE
#     overlaid, domain papers only). Same styling as the 2x2 figure above: only the mean line,
#     an inline coloured 'mean=...' annotation, and the metric name in the top-right corner.
#     Needs compute_efgd_quantile.py to have been run with use_disp_f=True (for the disp_f
#     quantile column); the Generalization panel always exists (it is the G index).
# ==================================================
if INCLUDE_DISP_F:
    qs = quantile_suffix
    # (raw column, metric name shown in the corner) in panel order
    fdg_metrics = [
        (f'f_y{year_offset}',            'Foundation'),
        (f'disruption_f_y{year_offset}', 'Disruptive Foundation'),
        (f'ij_ratio_y{year_offset}',     'Disruption Ratio'),
        (f'g_y{year_offset}',            'Generalization'),
    ]
    fdg_quant_sel = [f'{c}{qs}' for c, _ in fdg_metrics]
    have_fdg = all(
        qc in cache.columns and pd.to_numeric(cache[qc], errors='coerce').notna().any()
        for qc in fdg_quant_sel)
    if not have_fdg:
        print("WARNING: F/disp_f/I(I+J) quantile columns are missing/empty in the cache; skipping the "
              "F/disp_f/I(I+J) figure. Re-run compute_efgd_quantile.py with use_disp_f=True, then delete "
              f"the cache to rebuild:\n  {cache_path}")
    else:
        # dedicated subdirectory for the disp_f figure
        dispf_dire = os.path.join(fig_dire, 'disp_f_quantile')
        os.makedirs(dispf_dire, exist_ok=True)

        # figure size + label/tick sizes match openalex/award_analysis/award_quantile_analysis.py
        # (width scaled from the old 1x3 layout: 19 wide for 3 panels -> ~6.33/panel)
        fig_df, axes_df = plt.subplots(1, len(fdg_metrics),
                                       figsize=(6.325 * len(fdg_metrics), 5))
        for idx, (ax_df, (raw_col, metric_name)) in enumerate(zip(axes_df.flat, fdg_metrics)):
            quant_col = f'{raw_col}{qs}'
            # REVIEW vs REFERENCE overlaid, domain papers only; every panel labels the two
            # coloured mean lines with the group name.
            entries = draw_panel(ax_df, quant_col, quant_col, True, None, '',
                                 show_ylabel=False, annotate_mean=True, draw_median=False,
                                 annotate_fontsize=13, annotate_group_name=True,
                                 annotate_median=False)
            ax_df.tick_params(labelsize=14)
            ax_df.set_xlim(0, 1)
            _despine(ax_df)
            # metric name in the top-right corner
            ax_df.text(0.97, 0.97, metric_name, transform=ax_df.transAxes,
                       ha='right', va='top', fontsize=18)
            legend_records['fdg_quantile_1x4'][quant_col] = {
                'metric': metric_name, 'panel': f'{QUANTILE_GROUP} quantile',
                'xlabel': 'Index quantile', 'entries': entries,
            }

        # single shared axis labels: x centred under the row, y centred on the left.
        # supxlabel eats space at the bottom, which would push a figure-centred supylabel
        # below the panels; place the y-label at the panels' true vertical centre instead
        # (computed from the axes' bbox after tight_layout so it sits dead centre).
        # reserve margin at the left/bottom (tight_layout otherwise leaves none for the
        # shared labels, so they collide with the tick numbers), then place both labels well
        # out in that reserved band: the x-label sits farther below the x-axis than the
        # y-label sits left of the y-axis. The y-label is centred on the panels' bbox.
        fig_df.tight_layout(rect=[0.02, 0.075, 1, 1])
        fig_df.supxlabel('Index quantile', fontsize=18, y=0.035)
        _pos = axes_df.flat[0].get_position()
        fig_df.text(0.016, 0.5 * (_pos.y0 + _pos.y1), 'Density', rotation=90,
                    va='center', ha='center', fontsize=18)
        save_path_df = os.path.join(
            dispf_dire, f'annual_review_f_dispf_ijratio_quantile_distribution_{out_param_suffix}.svg')
        fig_df.savefig(save_path_df, format='svg', transparent=True)
        plt.close(fig_df)
        print("saved F/disp_f/I(I+J)/G quantile figure to:", save_path_df)

# ==================================================
# 4c. extra figure: f/(i+j+k), disp_f/(i+j+k), i/(i+j+k) and (i-j)/(i+j+k) [= the disruption
#     index D] within-(year,domain) quantile distributions drawn together in a single 1 x 4
#     grid (REVIEW vs REFERENCE overlaid, domain papers only). Same styling as the 4b figure
#     above: only the mean line, an inline coloured 'mean=...' annotation, the metric name in
#     the top-right corner. The three ratio panels need compute_efgd_quantile.py run with the
#     new (i+j+k) ratios (and use_disp_f=True for the disp_f one); the D panel always exists.
# ==================================================
if INCLUDE_IJK_RATIO:
    qs = quantile_suffix
    # (raw column, metric name shown in the corner) in panel order.
    # C = I+J (the focal paper's own citations), so C+K is the I+J+K denominator.
    ijk_fig_metrics = [
        (f'f_over_ijk_y{year_offset}',     'F/(C+K)'),
        (f'dispf_over_ijk_y{year_offset}', 'Disp(F)/(C+K)'),
        (f'i_over_ijk_y{year_offset}',     'I/(C+K)'),
        (f'disp_y{year_offset}',           '(I-J)/(C+K)'),
    ]
    ijk_quant_sel = [f'{c}{qs}' for c, _ in ijk_fig_metrics]
    have_ijk = all(
        qc in cache.columns and pd.to_numeric(cache[qc], errors='coerce').notna().any()
        for qc in ijk_quant_sel)
    if not have_ijk:
        print("WARNING: f/(i+j+k), disp_f/(i+j+k), i/(i+j+k) or disruption quantile columns are "
              "missing/empty in the cache; skipping the 1x4 (i+j+k)-ratio figure. Re-run "
              "compute_efgd_quantile.py (with use_disp_f=True), then delete the cache to rebuild:\n"
              f"  {cache_path}")
    else:
        # dedicated subdirectory for the (i+j+k)-ratio figure
        ijk_dire = os.path.join(fig_dire, 'ijk_ratio_quantile')
        os.makedirs(ijk_dire, exist_ok=True)

        # width scaled from the 1x3 figure (19 wide for 3 panels -> ~6.33/panel)
        fig_ijk, axes_ijk = plt.subplots(1, 4, figsize=(25.3, 5))
        for idx, (ax_ijk, (raw_col, metric_name)) in enumerate(zip(axes_ijk.flat, ijk_fig_metrics)):
            quant_col = f'{raw_col}{qs}'
            # REVIEW vs REFERENCE overlaid, domain papers only; every panel labels the two
            # coloured mean lines with the group name.
            entries = draw_panel(ax_ijk, quant_col, quant_col, True, None, '',
                                 show_ylabel=False, annotate_mean=True, draw_median=False,
                                 annotate_fontsize=13, annotate_group_name=True,
                                 annotate_median=False)
            ax_ijk.tick_params(labelsize=14)
            ax_ijk.set_xlim(0, 1)
            _despine(ax_ijk)
            # metric name in the top-right corner
            ax_ijk.text(0.97, 0.97, metric_name, transform=ax_ijk.transAxes,
                        ha='right', va='top', fontsize=18)
            legend_records['ijk_quantile_1x4'][quant_col] = {
                'metric': metric_name, 'panel': f'{QUANTILE_GROUP} quantile',
                'xlabel': 'Index quantile', 'entries': entries,
            }

        # single shared axis labels: x centred under the row, y centred on the panels' bbox
        # (same layout recipe as the 4b figure above).
        fig_ijk.tight_layout(rect=[0.02, 0.075, 1, 1])
        fig_ijk.supxlabel('Index quantile', fontsize=18, y=0.035)
        _pos_ijk = axes_ijk.flat[0].get_position()
        fig_ijk.text(0.016, 0.5 * (_pos_ijk.y0 + _pos_ijk.y1), 'Density', rotation=90,
                     va='center', ha='center', fontsize=18)
        save_path_ijk = os.path.join(
            ijk_dire, f'annual_review_ijk_ratio_quantile_distribution_{out_param_suffix}.svg')
        fig_ijk.savefig(save_path_ijk, format='svg', transparent=True)
        plt.close(fig_ijk)
        print("saved f/(i+j+k), disp_f/(i+j+k), i/(i+j+k), disruption quantile figure to:",
              save_path_ijk)

# ==================================================
# 5. legend sidecar: the per-group (n, mean, median) info that used to live in the figure
#    legends is written to JSON so it can still be looked up after legends were removed.
# ==================================================
legend_path = os.path.join(
    fig_dire, f'annual_review_efgd_ijratio_legend_{out_param_suffix}.json')
with open(legend_path, 'w', encoding='utf-8') as f:
    json.dump(legend_records, f, indent=2, ensure_ascii=False)
print("saved legend info to:", legend_path)

# ==================================================
# 6. persist the KDE-curve cache so the next redraw skips gaussian_kde entirely
# ==================================================
with open(curve_cache_path, 'wb') as f:
    pickle.dump({'key': _curve_key, 'panels': _curve_cache}, f, pickle.HIGHEST_PROTOCOL)
print("saved KDE-curve cache:", curve_cache_path, "|", len(_curve_cache), "panels")


# ==================================================
# 7. the reference-minus-review GAP in mean index quantile, with a bootstrap 95% CI.
#    Draws nothing: it is written purely for
#    combine_wos_openalex/plot_validation_quantile_gap.py, which puts this gap on one axis
#    next to the award and ForeCite validations' gaps. One row per metric:
#         gap = mean quantile(Reference papers) - mean quantile(Review papers)
#    The REFERENCES are the desired class here (the reviews' cited papers), which is why the
#    difference is taken that way round. Every metric the cache carries gets a row, including
#    ones no panel draws (Extension, and E/(I+J+K) once the quantile table has it).
# ==================================================
BOOT_N = 2000              # bootstrap replicates
BOOT_SEED = 20260801       # fixed, so re-running reproduces the same CI
BOOT_EXACT_MAX = 200_000   # a group larger than this uses the normal approximation below


def _boot_means(v, rng, n_boot=BOOT_N):
    """(replicate means, how they were obtained) for the 1-D array `v`.
    A small group is resampled exactly (with replacement, in blocks so the n_boot x n index
    matrix never materialises at once). For a very large group exact resampling would cost
    n x n_boot draws for a bootstrap distribution that is by then indistinguishable from
    N(mean, s^2/n) (the bootstrap mean is asymptotically normal), so beyond BOOT_EXACT_MAX the
    replicate means are drawn from that normal instead."""
    n = len(v)
    if n > BOOT_EXACT_MAX:
        return rng.normal(v.mean(), v.std(ddof=1) / np.sqrt(n), size=n_boot), 'normal'
    block = max(1, int(2e7 // max(n, 1)))         # cap the index matrix at ~2e7 entries
    out = np.empty(n_boot, dtype=float)
    for s in range(0, n_boot, block):
        k = min(block, n_boot - s)
        out[s:s + k] = v[rng.integers(0, n, size=(k, n))].mean(axis=1)
    return out, 'resample'


def bootstrap_gap(des, und):
    """One CSV row's worth of stats for gap = mean(des) - mean(und). The two classes are
    INDEPENDENT samples, so each is resampled on its own and a replicate gap is the difference
    of the two replicate means; the CI is the 2.5/97.5 percentiles of those replicate gaps.
    Non-finite values are dropped first, exactly as the quantile panels do."""
    des = np.asarray(des, dtype=float)
    des = des[np.isfinite(des)]
    und = np.asarray(und, dtype=float)
    und = und[np.isfinite(und)]
    row = {'desired_mean': float(des.mean()) if len(des) else float('nan'),
           'undesired_mean': float(und.mean()) if len(und) else float('nan'),
           'desired_std': float(des.std(ddof=1)) if len(des) > 1 else float('nan'),
           'undesired_std': float(und.std(ddof=1)) if len(und) > 1 else float('nan'),
           'n_desired': int(len(des)), 'n_undesired': int(len(und)),
           'n_boot': BOOT_N, 'seed': BOOT_SEED}
    row['gap'] = row['desired_mean'] - row['undesired_mean']
    if len(des) < 2 or len(und) < 2:              # no CI is definable from < 2 values
        row.update({'gap_ci_low': float('nan'), 'gap_ci_high': float('nan'),
                    'boot_desired': '', 'boot_undesired': ''})
        return row
    rng = np.random.default_rng(BOOT_SEED)
    md, mode_d = _boot_means(des, rng)
    mu, mode_u = _boot_means(und, rng)
    lo, hi = np.percentile(md - mu, [2.5, 97.5])
    row.update({'gap_ci_low': float(lo), 'gap_ci_high': float(hi),
                'boot_desired': mode_d, 'boot_undesired': mode_u})
    return row


# (canonical metric key the combine script uses, raw column, label). The canonical keys are
# shared with the award and ForeCite gap CSVs, so the three tables concatenate cleanly.
GAP_METRICS = [
    ('f',              f'f_y{year_offset}',               'Foundation'),
    ('e',              f'e_y{year_offset}',               'Extension'),
    ('disp_f',         f'disruption_f_y{year_offset}',    'Disruptive Foundation'),
    ('ij_ratio',       f'ij_ratio_y{year_offset}',        'Disruption Ratio'),
    ('g',              f'g_y{year_offset}',               'Generalization'),
    ('f_over_ijk',     f'f_over_ijk_y{year_offset}',      'F/(C+K)'),
    ('e_over_ijk',     f'e_over_ijk_y{year_offset}',      'E/(C+K)'),
    ('dispf_over_ijk', f'dispf_over_ijk_y{year_offset}',  'Disp(F)/(C+K)'),
    ('i_over_ijk',     f'i_over_ijk_y{year_offset}',      'I/(C+K)'),
    ('disp',           f'disp_y{year_offset}',            '(I-J)/(C+K)'),
]
GAP_COLUMNS = ['source', 'group_id', 'group', 'metric', 'metric_label',
               'desired_class', 'undesired_class', 'desired_mean', 'undesired_mean',
               'gap', 'gap_ci_low', 'gap_ci_high', 'n_desired', 'n_undesired',
               'desired_std', 'undesired_std', 'n_boot', 'seed',
               'boot_desired', 'boot_undesired', 'quantile_group', 'value_col']

gap_rows, gap_skipped = [], []
_des_mask = (cache['group'] == 'Reference papers').values
_und_mask = (cache['group'] == 'Review papers').values
for mkey, raw_col, label in GAP_METRICS:
    quant_col = f'{raw_col}{quantile_suffix}'
    vals = (pd.to_numeric(cache[quant_col], errors='coerce').values
            if quant_col in cache.columns else None)
    if vals is None or not np.isfinite(vals).any():
        gap_skipped.append(mkey)     # column absent from / empty in the quantile table
        continue
    row = bootstrap_gap(vals[_des_mask], vals[_und_mask])
    row.update({'source': 'annual_review', 'group_id': 'annual_review',
                'group': 'Annual Review', 'metric': mkey, 'metric_label': label,
                'desired_class': 'Reference papers', 'undesired_class': 'Review papers',
                'quantile_group': QUANTILE_GROUP, 'value_col': quant_col})
    gap_rows.append(row)
    print(f"  [{mkey}] gap={row['gap']:+.4f} "
          f"95%CI [{row['gap_ci_low']:+.4f}, {row['gap_ci_high']:+.4f}] "
          f"(n {row['n_desired']:,} refs vs {row['n_undesired']:,} reviews)")

gap_csv = os.path.join(fig_dire, f'annual_review_gap_bootstrap_{out_param_suffix}.csv')
pd.DataFrame(gap_rows)[GAP_COLUMNS].to_csv(gap_csv, index=False)
print("saved gap + bootstrap CI CSV:", gap_csv)
if gap_skipped:
    print(f"  skipped {gap_skipped} (column missing/empty). Re-run compute_efgd_quantile.py "
          f"with it, then delete the cache to rebuild:\n  {cache_path}")

print("finished")
