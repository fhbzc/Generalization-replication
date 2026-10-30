"""
One bar chart per metric family summarising every "desired vs undesired" validation.

Three separate validation exercises each compare a DESIRED class of papers against an
UNDESIRED one, and each already reduces that comparison to a single number per metric:

        gap = mean quantile(desired) - mean quantile(undesired)

plus a bootstrap 95% CI of that gap. This script does no computation of its own -- it reads
those three scripts' gap CSVs and puts every gap on one axis. A positive bar means the desired
class sits higher on that index; a negative bar means it sits lower (Generalization and the
disruption ratio are expected to go the other way, which is the point of drawing them). Each
bar carries its bootstrap CI as an error bar, so a bar whose interval clears 0 is the visible
claim; the CI numbers are written to the values CSV as well.

The CI is computed WHERE THE PER-PAPER VALUES ARE (in the three source scripts): the two
classes are independent samples, so each is resampled with replacement 2000 times and the CI
is the 2.5/97.5 percentiles of mean(desired*) - mean(undesired*). A class larger than 200k
papers (always the "undesired" whole-corpus baseline) has its replicate means drawn from
N(mean, s^2/n) instead -- at that size the bootstrap mean is indistinguishable from it. Each
row records which of the two was used ('boot_desired' / 'boot_undesired').

Comparison groups (desired / undesired), and where each one's numbers come from:

  Nobel Prize        Award-winning / Others
  Major award        Award-winning / Others
      <- openalex/award_analysis/award_quantile_analysis.py
         figures/award_quantile_disp_f/q_{QG}/award_quantile_gap_bootstrap_*.csv

  Annual Review      Reference papers / Review papers
      <- wos/venue_validation/plot_annual_review_efgd_distribution.py
         figures/annual_review_efgd_ijratio/annual_review_gap_bootstrap_*.csv
         NOTE reference is the DESIRED side here, so the gap is Reference - Review.

  New-concept papers Papers of new concept / Others
      <- semantic_scholar/academic_keyword_analysis/forecite_central_papers_efg_s2_plot.py
         figures/forecite_gt_{threshold}/gap_bootstrap_{value_tag}.csv
         (one 'group_id' per central-paper definition -- FORECITE_DEFINITIONS below picks
          which ones become bars)

All three CSVs share one schema (source, group_id, group, metric, metric_label,
desired/undesired class + mean + n + std, gap, gap_ci_low, gap_ci_high, n_boot, seed,
boot_desired, boot_undesired, quantile_group, value_col) and one set of canonical metric keys,
so they simply concatenate.

Both figures put the COMPARISON GROUP on x and encode the METRIC by colour, so one group's
metric bars are clustered together at its tick and can be read against each other. The two
figures differ only in which metrics that is:
  fig 1 (fdg) : Foundation / Disruptive Foundation / Extension / Disruption Ratio /
                Generalization
  fig 2 (ijk) : F/(C+K) / E/(C+K) / Disp(F)/(C+K) / I/(C+K) / (I-J)/(C+K)     [C = I+J]
Two toggles drop a whole metric family from BOTH figures: INCLUDE_DISP_F = False removes the
disruptive-foundation bars (Disruptive Foundation + Disp(F)/(C+K)) and INCLUDE_DISP = False
removes the disruption-index bars (Disruption Ratio + (I-J)/(C+K)); the dropped metrics leave
the figures and the values CSV, whose filenames then carry a "_nodispf" / "_nodisp" tag. A
metric that none of the sources reports yet (E/(C+K) until compute_*_quantile.py has been
re-run with that column) is dropped from its figure rather than left as an empty slot; the
colours are bound to the metric key, so dropping one does not recolour the others.

All three sources are read at the SAME quantile grouping (QUANTILE_GROUP), so the gaps are
on one common scale: every value is a within-(pub_year x domain) percentile rank in [0, 1],
hence every gap lies in [-1, 1] and is directly comparable across metrics and groups.

Input  : the three scripts' ./figures gap CSVs listed above (repo-local, not META_DIRE, so
         this script needs no metadata root at all). Each source path is resolved at its
         canonical location first, then by a recursive search for the same basename under
         that project's figures/ directory (so files copied elsewhere are still found).
Output : ./figures/validation_quantile_gap/   (at the deliverable setting year_offset=5 and
             QUANTILE_GROUP='year_domain', the two panels and their shared values CSV carry the
             figure's name; otherwise every file keeps its descriptive one)
             (tag = y{year_offset}_ybefore_{select_max_year}_q_{QUANTILE_GROUP}
                    {_nodispf}{_nodisp})
             quantile_gap_fdg_{tag}.svg/.png     -> 'Extended Data Fig. 5a.svg/.png'
             quantile_gap_ijk_{tag}.svg/.png     -> 'Extended Data Fig. 5b.svg/.png'
             quantile_gap_values_{tag}.csv       -> 'Extended Data Fig. 5.csv'
                 (every bar's two means, n's, gap and 95% CI, for BOTH panels -- the rows carry
                  a 'family' column, so one file covers 5a and 5b)

Usage  : edit the settings block below (they must match the three source scripts' own
         parameters, since they only serve to rebuild those scripts' filenames), then
         `python plot_validation_quantile_gap.py`.
"""

import glob
import os
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REBUILD_DIR = os.path.dirname(BASE_DIR)

# repo plotting convention; fall back silently if the font is unavailable (e.g. Linux)
try:
    plt.rcParams['font.family'] = 'Gill Sans MT'
except Exception:
    pass
# Gill Sans MT lacks the Unicode minus glyph (U+2212), which renders as a box -> ASCII hyphen
plt.rcParams['axes.unicode_minus'] = False


# ==================================================
# settings -- these do not change any computation, they only rebuild the three source
# scripts' OUTPUT FILENAMES, so each block must match that script's own parameters.
# ==================================================
year_offset = -1

# select_max_year applies to EVERY group: all three sources are read at the same publication
# window, which is the only way their gaps stay on one scale. Set it to the value the three
# source scripts were run with; leave it at max_year for no narrowing.
#
# max_year here is the shared cutoff the sources use -- 2025 - year_offset, or 2025 when
# year_offset == -1 (no elapsed window to leave room for). Each source still decides whether
# ITS filename carries a narrowing tag by comparing select_max_year against its OWN max_year,
# because that is the comparison the source script itself makes; those per-source max_years are
# derived in the three blocks below.
max_year = 2025 if year_offset == -1 else 2025 - year_offset
select_max_year = 2015
assert select_max_year <= max_year, (
    "select_max_year (%s) must be <= max_year (%s)." % (select_max_year, max_year))

# which quantile grouping every source is read at. Must be the QUANTILE_GROUP the award and
# annual-review scripts ran with, and must match the ForeCite script's quantile_kind
# ('' -> year_domain, '_by_year' -> year, '_global' -> global). Every gap CSV records the
# grouping it was built at, and a mismatch is reported below rather than silently mixed.
QUANTILE_GROUP = 'year_domain'
assert QUANTILE_GROUP in ('year_domain', 'year', 'global'), QUANTILE_GROUP
# the ForeCite script names its files by value_tag instead of a quantile group
_FORECITE_VALUE_TAG = {'year_domain': 'quantile_year_domain', 'year': 'quantile_by_year',
                       'global': 'quantile_global'}[QUANTILE_GROUP]


def _forecite_value_tag():
    """The ForeCite script's value_tag, including the narrowing suffix when it applies.

    Defined as a function because forecite_select_max_year is set further down, with the rest
    of that source's block; it is called where the path is built.
    """
    return (_FORECITE_VALUE_TAG if forecite_select_max_year == forecite_max_year
            else f'{_FORECITE_VALUE_TAG}_ybefore_{forecite_select_max_year}')

# ---- openalex/award_analysis/award_quantile_analysis.py ----
award_rt = 1
award_ct = 5
award_with_review = False
award_data_suffix = '_with_review' if award_with_review else ''
award_max_year = 2025 if year_offset == -1 else 2025 - year_offset
award_name_tag = '_full_cover' if year_offset == -1 else ''
# award_quantile_analysis.py puts select_max_year -- not max_year -- in 'ybefore_'
award_select_max_year = select_max_year
assert award_select_max_year <= award_max_year, (award_select_max_year, award_max_year)
# ... and it appends '_multi_match' when it ran with require_unique_match = False
award_require_unique_match = False
award_match_tag = '' if award_require_unique_match else '_multi_match'

# ---- wos/venue_validation/plot_annual_review_efgd_distribution.py ----
wos_rt = 1
wos_ct = 5
# same '_full_cover' rule as wos/compute_efgd_quantile.py -- but note the venue script's
# OUTPUT names never carried that tag, only its INPUT filename does, so there is no
# wos_name_tag here.
wos_max_year = 2025 if year_offset == -1 else 2025 - year_offset
# plot_annual_review_efgd_distribution.py puts select_max_year in its 'ybefore_'
wos_select_max_year = select_max_year
assert wos_select_max_year <= wos_max_year, (wos_select_max_year, wos_max_year)
wos_use_with_review = True
wos_data_suffix = '_with_review' if wos_use_with_review else ''
wos_k_require_after_focal = True
wos_k_suffix = '' if wos_k_require_after_focal else '_k_no_lower_bound'
wos_detection_mode = 'venue'                    # 'venue' / 'title' / 'type'
wos_mode_suffix = {'venue': '', 'title': '_title_review', 'type': '_type_review'}[wos_detection_mode]

# ---- semantic_scholar/academic_keyword_analysis/forecite_central_papers_efg_s2_plot.py ----
FORECITE_THRESHOLD = 0.5
# which central-paper definitions become their own bar. The source script writes all four
# (max_score / top5_score / earliest / earliest_above_mean) into one CSV; listing more than one
# here adds one comparison group -- and so one bar per metric -- for each.
FORECITE_DEFINITIONS = ['max_score']
# forecite_central_papers_efg_s2_plot.py names everything by value_tag, which gains an
# '_ybefore_{select_max_year}' suffix ONLY when that script narrows (select_max_year <
# max_year); with no narrowing its names are unchanged. Its max_year is 2025 - year_offset.
forecite_max_year = 2025 - year_offset
forecite_select_max_year = select_max_year
assert forecite_select_max_year <= forecite_max_year, (forecite_select_max_year,
                                                       forecite_max_year)

# True -> stop with the missing paths listed rather than quietly dropping a comparison group
# (a dropped bar is invisible in the finished figure, so silence is the dangerous default).
# A missing METRIC (a column a source has not got yet) is only reported, never fatal.
STRICT_MISSING = True

# print each bar's value above/below it. Off by default: 20+ numbers crowd the panel and every
# value is written to the CSV next to the figure anyway.
ANNOTATE_VALUES = False

# draw each bar's bootstrap 95% CI as an error bar
DRAW_CI = True

# ---- which metric families to draw -------------------------------------------------------
# Each toggle drops ONE metric from EACH of the two figures (both figures show the same index
# in their own denominator), in the figure AND in the values CSV:
#   INCLUDE_DISP_F = False -> drop 'disp_f' (Disruptive Foundation) and 'dispf_over_ijk'
#                             (Disp(F)/(C+K))
#   INCLUDE_DISP   = False -> drop 'ij_ratio' (Disruption Ratio) and 'disp' ((I-J)/(C+K))
# The remaining bars keep their colours: METRIC_COLOR is sampled over the FULL declared family
# below, exactly as when a metric is dropped for being absent from a source. Outputs carry a
# "_nodispf" / "_nodisp" filename tag so the variants coexist.
INCLUDE_DISP_F = True
INCLUDE_DISP = True
DISP_F_METRICS = {'disp_f', 'dispf_over_ijk'}
DISP_METRICS = {'ij_ratio', 'disp'}
EXCLUDED_METRICS = ((set() if INCLUDE_DISP_F else DISP_F_METRICS)
                    | (set() if INCLUDE_DISP else DISP_METRICS))
NODISPF_SUFFIX = '' if INCLUDE_DISP_F else '_nodispf'
NODISP_SUFFIX = '' if INCLUDE_DISP else '_nodisp'

print("year_offset", year_offset)
print("QUANTILE_GROUP", QUANTILE_GROUP)
print("max_year", max_year, "| select_max_year", select_max_year,
      "(applies to every group)")
print("FORECITE_THRESHOLD", FORECITE_THRESHOLD, "| definitions", FORECITE_DEFINITIONS)
print("STRICT_MISSING", STRICT_MISSING, "| DRAW_CI", DRAW_CI)
print("INCLUDE_DISP_F", INCLUDE_DISP_F, "| INCLUDE_DISP", INCLUDE_DISP,
      ("| dropped metrics: %s" % sorted(EXCLUDED_METRICS)) if EXCLUDED_METRICS else "")

off = year_offset

# ==================================================
# the two metric families, in bar order. The keys are the canonical ones every gap CSV uses.
# C = I+J (the focal paper's own citations), so C+K is the I+J+K denominator.
# ==================================================
METRIC_FAMILIES = {
    'fdg': [
        # (canonical key, x-axis label)
        ('f',          'Foundation'),
        ('disp_f',     'Disruptive Foundation'),
        ('e',          'Extension'),
        ('ij_ratio',   'Disruption Ratio'),
        ('g',          'Generalization'),
    ],
    'ijk': [
        ('f_over_ijk',     'F/(C+K)'),
        ('dispf_over_ijk', 'Disp(F)/(C+K)'),
        ('e_over_ijk',     'E/(C+K)'),
        ('i_over_ijk',     'I/(C+K)'),
        ('disp',           '(I-J)/(C+K)'),
    ],
}
FAMILY_TITLE = {'fdg': 'Foundation / disruption / extension / generalization indices',
                'ijk': '(I+J+K)-denominator ratios'}
METRIC_LABEL = {k: lbl for fam in METRIC_FAMILIES.values() for k, lbl in fam}

# metric colours: a dark red -> light red -> light blue -> dark blue ramp, sampled evenly
# across a family's metrics, so the bars inside one cluster read as an ordered gradient. The
# anchors are ColorBrewer's 4-class RdBu. With an ODD family the middle metric lands on the
# ramp's centre, where the light red and light blue meet in a neutral taupe -- that slot is
# Extension in one family and E/(C+K) in the other, so the neutral consistently marks the E
# index in both.
# The colour is bound to a FIXED metric key (never cycled) and sampled over the FULL declared
# family, so a metric keeps its colour no matter which comparison groups are drawn and whether
# a sibling metric had to be dropped. The two families' matching slots share a colour: they
# live in SEPARATE figures and never appear together, so the pair reads as one system.
METRIC_RAMP_ANCHORS = ['#CA0020',    # dark red
                       '#F4A582',    # light red
                       '#92C5DE',    # light blue
                       '#0571B0']    # dark blue
_metric_ramp = LinearSegmentedColormap.from_list('darkred_darkblue', METRIC_RAMP_ANCHORS)
METRIC_COLOR = {}
for _fam in METRIC_FAMILIES.values():
    _n = len(_fam)
    for _i, (_k, _l) in enumerate(_fam):
        METRIC_COLOR[_k] = to_hex(_metric_ramp(_i / (_n - 1) if _n > 1 else 0.0))


# ==================================================
# path resolution: canonical location first, then a recursive search for the same basename
# under the owning project's figures/ dir (the SVG/CSV outputs are sometimes copied around)
# ==================================================
missing_sources = []


def resolve_path(primary, search_root, what):
    if os.path.exists(primary):
        return primary
    hits = sorted(glob.glob(os.path.join(search_root, '**', os.path.basename(primary)),
                            recursive=True))
    if hits:
        print(f"  NOTE: {what} is not at its canonical path; using {hits[0]}")
        return hits[0]
    missing_sources.append(f'{what}\n      expected: {primary}')
    print(f"  MISSING: {what}\n      expected: {primary}")
    return None


# every gap CSV must carry these; the rest of the schema is passed through to the values CSV
GAP_REQUIRED_COLS = ['group_id', 'metric', 'desired_mean', 'undesired_mean', 'gap',
                     'gap_ci_low', 'gap_ci_high', 'n_desired', 'n_undesired']


def load_gap_csv(primary, search_root, what):
    """One source's gap CSV as a DataFrame (None when it could not be resolved). The quantile
    grouping it was built at is checked against QUANTILE_GROUP: mixing groupings would put
    percentile ranks computed within different reference sets on one axis."""
    path = resolve_path(primary, search_root, what)
    if path is None:
        return None
    df = pd.read_csv(path)
    miss = [c for c in GAP_REQUIRED_COLS if c not in df.columns]
    assert not miss, (f"{path}\n  is missing {miss}. Re-run the owning script -- its gap CSV "
                      f"predates this schema.")
    groups_in = set(df.get('quantile_group', pd.Series(dtype=str)).dropna().unique())
    if groups_in and groups_in != {QUANTILE_GROUP}:
        print(f"  WARNING: {what} was built at quantile grouping {sorted(groups_in)}, not "
              f"'{QUANTILE_GROUP}'. Re-run it with a matching QUANTILE_GROUP.")
    print(f"  loaded {what}: {os.path.basename(path)} ({len(df)} rows)")
    return df


# ==================================================
# 1. award_quantile_analysis.py -> one group per award type
# ==================================================
award_fig_root = os.path.join(REBUILD_DIR, 'openalex', 'award_analysis', 'figures')
award_param_suffix = (f'y{year_offset}{award_data_suffix}_'
                      f'rt_{award_rt}_ct_{award_ct}_'
                      f'ybefore_{award_select_max_year}{award_name_tag}_'
                      f'q_{QUANTILE_GROUP}{award_match_tag}')

# ==================================================
# 2. plot_annual_review_efgd_distribution.py -> the Reference-vs-Review group
# ==================================================
wos_fig_root = os.path.join(REBUILD_DIR, 'wos', 'venue_validation', 'figures')
wos_out_param_suffix = (f'y{year_offset}{wos_data_suffix}{wos_k_suffix}_'
                        f'rt_{wos_rt}_ct_{wos_ct}_ybefore_{wos_select_max_year}'
                        f'{wos_mode_suffix}_q_{QUANTILE_GROUP}')

# ==================================================
# 3. forecite_central_papers_efg_s2_plot.py -> one group per central-paper definition
# ==================================================
forecite_fig_root = os.path.join(REBUILD_DIR, 'semantic_scholar',
                                 'academic_keyword_analysis', 'figures')
# the source script builds this directory name from a float, so 0.5 -> 'forecite_gt_0.5';
# an integer threshold has been written both ways ('2' and '2.0') -> try both spellings.
_forecite_candidates = [os.path.join(forecite_fig_root, f'forecite_gt_{t}',
                                     f'gap_bootstrap_{_forecite_value_tag()}.csv')
                        for t in dict.fromkeys([f'{FORECITE_THRESHOLD:g}',
                                                f'{FORECITE_THRESHOLD}'])]
_forecite_primary = next((c for c in _forecite_candidates if os.path.exists(c)),
                         _forecite_candidates[0])

print("\nreading the three sources' gap CSVs ...")
sources = [
    load_gap_csv(os.path.join(award_fig_root, 'award_quantile_disp_f', f'q_{QUANTILE_GROUP}',
                              f'award_quantile_gap_bootstrap_{award_param_suffix}.csv'),
                 award_fig_root, 'award gap CSV'),
    load_gap_csv(os.path.join(wos_fig_root, 'annual_review_efgd_ijratio',
                              f'annual_review_gap_bootstrap_{wos_out_param_suffix}.csv'),
                 wos_fig_root, 'annual-review gap CSV'),
    load_gap_csv(_forecite_primary, forecite_fig_root, 'ForeCite gap CSV'),
]
df_src = pd.concat([d for d in sources if d is not None], ignore_index=True) \
    if any(d is not None for d in sources) else pd.DataFrame(columns=GAP_REQUIRED_COLS)


# ==================================================
# assemble every (group x metric) gap into one long table
# ==================================================
# (group id in the source CSVs, x-axis label) -- drawn in this order. The label is this
# figure's own Title Case wording; the class names come from the source rows unchanged.
GROUPS = [
    ('nobel', 'Nobel Prize'),
    ('major', 'Major Award'),
    ('annual_review', 'Annual Review'),
]
for _i, _defn in enumerate(FORECITE_DEFINITIONS):
    # the definition name stays verbatim -- it is the source script's own setting key
    _label = ('Papers of New Concept' if len(FORECITE_DEFINITIONS) == 1
              else f'New Concept ({_defn})')
    GROUPS.append((f'forecite_{_defn}', _label))

for _gid, _lbl in GROUPS:
    if not (df_src['group_id'] == _gid).any():
        missing_sources.append(f"comparison group '{_gid}' is absent from the gap CSVs")
        print(f"  MISSING: comparison group '{_gid}' has no rows in the gap CSVs")

if missing_sources:
    msg = "could not resolve %d source file(s)/group(s):\n    - %s" % (
        len(missing_sources), "\n    - ".join(missing_sources))
    if STRICT_MISSING:
        raise SystemExit(
            msg + "\n\nRun the owning script (with the parameters in this script's settings "
                  "block), or set STRICT_MISSING = False to draw the groups that DID resolve.")
    print("\nWARNING:", msg)

rows = []
absent_metrics = {}
for family, fam_metrics in METRIC_FAMILIES.items():
    for key, _lbl in fam_metrics:
        if key in EXCLUDED_METRICS:      # switched off by INCLUDE_DISP_F / INCLUDE_DISP
            continue
        for gid, glabel in GROUPS:
            sub = df_src[(df_src['group_id'] == gid) & (df_src['metric'] == key)]
            if not len(sub):
                absent_metrics.setdefault(key, []).append(gid)
                continue
            r = sub.iloc[0]
            rows.append({'family': family, 'group_id': gid, 'group': glabel,
                         'metric': key, 'metric_label': METRIC_LABEL[key],
                         'desired_class': r.get('desired_class', ''),
                         'undesired_class': r.get('undesired_class', ''),
                         'desired_mean': float(r['desired_mean']),
                         'undesired_mean': float(r['undesired_mean']),
                         'gap': float(r['gap']),
                         'gap_ci_low': float(r['gap_ci_low']),
                         'gap_ci_high': float(r['gap_ci_high']),
                         'n_desired': int(r['n_desired']), 'n_undesired': int(r['n_undesired']),
                         'n_boot': r.get('n_boot', np.nan), 'seed': r.get('seed', np.nan),
                         'boot_desired': r.get('boot_desired', ''),
                         'boot_undesired': r.get('boot_undesired', ''),
                         'value_col': r.get('value_col', '')})

if absent_metrics:
    print("\nmetrics with no row in some source (their bars are dropped, the others keep "
          "their colour):")
    for k, gids in absent_metrics.items():
        print(f"  {k:<16} missing for: {', '.join(gids)}")
    print("  E/(C+K) needs compute_*_quantile.py re-run with the e_over_ijk column (and the "
          "owning script's per-paper cache deleted so it picks the column up).")

assert rows, "nothing loaded -- none of the three gap CSVs resolved."
df_gap = pd.DataFrame(rows)

fig_dire = os.path.join(BASE_DIR, 'figures', 'validation_quantile_gap')
os.makedirs(fig_dire, exist_ok=True)
# Both settings that define the population are in the name: year_offset (the citation window)
# and select_max_year (the publication cutoff). A narrowed run is a different comparison, so it
# must never land on the un-narrowed run's files.
tag = (f'y{year_offset}_ybefore_{select_max_year}_q_{QUANTILE_GROUP}'
       f'{NODISPF_SUFFIX}{NODISP_SUFFIX}')

# both bar charts at the paper's default settings ARE the Extended Data deliverable: the fdg
# family is panel a, the ijk family panel b. The values CSV carries BOTH families (it has a
# 'family' column), so it is named for the figure as a whole rather than for either panel.
ED_FIG_NAME = 'Extended Data Fig. 5'
ED_FIG_PANEL = {'fdg': f'{ED_FIG_NAME}a', 'ijk': f'{ED_FIG_NAME}b'}
# the deliverable shows every metric, so a run with either family switched off keeps the
# descriptive names rather than landing on the Extended Data name
# ... and a narrowed run is not the deliverable either: the deliverable is the full period
is_ed_fig = (year_offset == 5 and QUANTILE_GROUP == 'year_domain'
             and INCLUDE_DISP_F and INCLUDE_DISP and select_max_year == max_year)
print("is_ed_fig", is_ed_fig, ("-> %s" % ED_FIG_PANEL) if is_ed_fig else "")

csv_path = os.path.join(
    fig_dire,
    f'{ED_FIG_NAME}.csv' if is_ed_fig else f'quantile_gap_values_{tag}.csv')
df_gap.to_csv(csv_path, index=False)
print("\nwrote values:", csv_path)
print(df_gap[['family', 'group', 'metric_label', 'desired_mean', 'undesired_mean',
              'gap', 'gap_ci_low', 'gap_ci_high', 'n_desired',
              'n_undesired']].to_string(index=False))


# ==================================================
# draw: one grouped bar chart per metric family. x = comparison group, one bar per metric
# (colour = metric), so a group's bars sit together and are read against each other.
# y = the desired-minus-undesired gap in mean index quantile (so 0 = no difference), with the
# bootstrap 95% CI as a black error bar on each bar.
# ==================================================
label_fontsize = 17
tick_fontsize = 14
legend_fontsize = 19


def draw_family(family):
    sub_all = df_gap[df_gap['family'] == family]
    # only metrics that at least one group reports (an unreported metric is dropped rather
    # than left as an empty slot in the middle of every cluster)
    metrics = [k for k, _l in METRIC_FAMILIES[family]
               if k not in EXCLUDED_METRICS and (sub_all['metric'] == k).any()]
    # only groups that actually produced a bar in this family, in the GROUPS order
    groups = [(gid, glabel) for gid, glabel in GROUPS if (sub_all['group_id'] == gid).any()]
    if not metrics or not groups:
        print(f"[{family}] nothing resolved -- skipping the figure")
        return

    x = np.arange(len(groups), dtype=float)
    # 0.58 of the slot is bars, the rest is the gap separating one comparison group's cluster
    # from the next -- a narrow cluster leaves visibly more whitespace between groups. Bars are
    # drawn at the FULL bar width, so the metrics inside one cluster butt against each other
    # with no gap: the only whitespace on the x-axis separates comparison groups, which is what
    # makes a cluster read as one group.
    slot = 0.58
    width = slot / len(metrics)

    # one bar series per metric, so a series spans the groups and the legend keys on colour
    fig, ax = plt.subplots(figsize=(max(6.5, 2.15 * len(groups) + 1.2), 5.6))
    by_gid = {gid: sub_all[sub_all['group_id'] == gid].set_index('metric')
              for gid, _lbl in groups}
    for mi, mkey in enumerate(metrics):
        def _col(name, _mkey=mkey):
            """one column of this metric's row, per group, NaN where the group has no row"""
            return np.array([by_gid[gid][name].get(_mkey, np.nan) for gid, _lbl in groups],
                            dtype=float)
        vals = _col('gap')
        pos = x - slot / 2 + width * (mi + 0.5)
        ax.bar(pos, vals, width=width, color=METRIC_COLOR[mkey],
               label=METRIC_LABEL[mkey], zorder=3)
        if DRAW_CI:
            lo, hi = _col('gap_ci_low'), _col('gap_ci_high')
            # asymmetric error bars around the point estimate; clip at 0 so a CI that rounds
            # to the wrong side of the estimate cannot become a negative bar length
            yerr = np.vstack([np.clip(vals - lo, 0, None), np.clip(hi - vals, 0, None)])
            ax.errorbar(pos, vals, yerr=yerr, fmt='none', ecolor='black',
                        elinewidth=1.0, capsize=3, capthick=1.0, zorder=5)
        if ANNOTATE_VALUES:
            for px, v in zip(pos, vals):
                if not np.isfinite(v):
                    continue
                ax.annotate(f'{v:+.3f}', (px, v), textcoords='offset points',
                            xytext=(0, 3 if v >= 0 else -3), ha='center',
                            va='bottom' if v >= 0 else 'top', fontsize=9, rotation=90)

    # 0 is the "no difference" reference, so it gets a real line rather than a grid tick
    ax.axhline(0.0, color='black', linewidth=1.0, zorder=4)
    ax.set_xticks(x)
    # wrap rather than rotate, so the labels stay horizontal under their own cluster. The width
    # is set so every default group name ('Papers of New Concept' is the longest at 21 chars)
    # stays on ONE line; only the longer multi-definition names wrap. break_long_words=False
    # keeps a single long token (e.g. 'earliest_above_mean') intact rather than splitting it.
    ax.set_xticklabels([textwrap.fill(lbl, 24, break_long_words=False)
                        for _gid, lbl in groups], fontsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.set_ylabel('Mean index quantile difference', fontsize=label_fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # legend in the panel's top-right corner. Reserve headroom above the tallest bar (error
    # bars included -- they are part of the data limits) first, so the legend sits in empty
    # space rather than on top of the last group's positive bars.
    lo_y, hi_y = ax.get_ylim()
    ax.set_ylim(lo_y, hi_y + 0.30 * (hi_y - lo_y))
    ax.legend(fontsize=legend_fontsize, frameon=False, ncol=1, loc='upper right')

    fig.tight_layout()
    stem = os.path.join(
        fig_dire,
        ED_FIG_PANEL[family] if is_ed_fig else f'quantile_gap_{family}_{tag}')
    fig.savefig(f'{stem}.svg', format='svg', transparent=True)
    fig.savefig(f'{stem}.png', dpi=300)
    plt.close(fig)
    print(f"saved [{family}] {FAMILY_TITLE[family]}:", f'{stem}.svg')


for _family in METRIC_FAMILIES:
    draw_family(_family)

print("finished")
