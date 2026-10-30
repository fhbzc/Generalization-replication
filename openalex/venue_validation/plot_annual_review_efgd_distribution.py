"""
Distribution of E/F/G/D and I/(I+J) (raw value AND quantiles) for the two annual-review
paper sets:
    * REVIEW papers     -> annual_review_papers.pkl            (review papers themselves)
    * REFERENCE papers  -> annual_review_paper_references.pkl  (papers they cite)
both from get_annual_review_papers_and_refs.py.

The per-paper metrics come from ../compute_efgd_quantile.py:
    <META>/efg_computation/
        efgd_within_year_domain_quantile_y{off}{NAME_SUFFIX}_rt_{rt}_ct_{ct}_ybefore_{max_year}.csv
which holds, per pid, the raw indices  e_y{off}, f_y{off}, g_y{off}, disp_y{off},
ij_ratio_y{off}  AND their quantiles under all three groupings (_quantile,
_quantile_by_year, _quantile_global), for EVERY paper passing the reference / citation /
year filters (no-/multi-domain papers are kept with the (year,domain) quantile = NaN).
QUANTILE_GROUP selects which single quantile the extra 2x2 figure draws.

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
'Density' label on the far left, vertically centred over all rows. (The separate 2 x 2
figure below still uses the QUANTILE_GROUP quantile and draws its own mean annotation.)

The raw value and the global / by-year quantiles are present for every paper in the
compute_efgd_quantile.py table; only the (year, domain) quantile is NaN for no-/multi-
domain papers (column (d) filters on that being present).

To avoid recomputing on every redraw, the per-paper values for the two groups are
cached to a CSV in ./figures; if that cache exists it is loaded directly and the
pkl/metric-table inputs are not touched (delete the cache to force a refresh).

USE_REVIEW_META (must match find_annual_review_venues.py / get_annual_review_papers_and_refs.py
/ compute_efgd_quantile.py): True -> read the '_with_review' meta tree (its paper-set pkls
and the '_with_review' quantile table, whose domain comes from
get_meta_data_preprint_review_s2_simple_concept.py); False -> the plain META_DIRE tree.

detection_mode selects how the review/reference pid sets are identified:
    venue : papers in the annual-review venues -> the two get_annual_review_papers_and_refs.py
            pkls below.
    title : a paper is a review iff its title (lower-cased) contains 'review'; derived in-script
            from <META>/pid2title.pkl, references = union over <META>/citing_pid2cited_pids_list.pkl.
    type  : the document-type-review papers = set(pid2pub_year in META_DIRE_with_review)
            - set(pid2pub_year in META_DIRE), references = union over the with_review
            citing_pid2cited_pids_list; 'type' forces USE_REVIEW_META=True.
mode_suffix ('' / '_title_review' / '_type_review') is added to the cache/figure names so
the three modes never overwrite each other. The INPUT quantile table is mode-independent.

Input  : <META>/annual_review_papers{NAME_SUFFIX}.pkl            (venue)
         <META>/annual_review_paper_references{NAME_SUFFIX}.pkl  (venue)
         <META>/pid2title.pkl                                    (title)
         META_DIRE{,_with_review}/pid2pub_year.pkl               (type: review-set diff)
         <META>/citing_pid2cited_pids_list.pkl                   (title/type: references)
         <META>/efg_computation/efgd_within_year_domain_quantile_y{off}{NAME_SUFFIX}_rt_{rt}_ct_{ct}_ybefore_{max_year}.csv
Cache  : venue_validation/figures/annual_review_efgd_ijratio/
             annual_review_efgd_ijratio_cache_y{off}{NAME_SUFFIX}_rt_{rt}_ct_{ct}_ybefore_{max_year}{mode_suffix}_withalldomain_allq.csv
             (raw values + all three quantile groupings; independent of QUANTILE_GROUP)
Output : venue_validation/figures/annual_review_efgd_ijratio/
             annual_review_efgd_ijratio_distribution_y{off}{NAME_SUFFIX}_rt_{rt}_ct_{ct}_ybefore_{max_year}{mode_suffix}.svg
             (the full 4 x 4 grid)
         venue_validation/figures/annual_review_efgd_ijratio/subplots_{full_param_suffix}/
             {raw_col}_{panel_key}_{full_param_suffix}.svg
             (each grid cell saved standalone, no title; panel_key in
              {raw_all, quantile_global, quantile_by_year, quantile_year_domain})
         venue_validation/figures/annual_review_efgd_ijratio/
             annual_review_dfeg_quantile_distribution_y{off}{NAME_SUFFIX}_rt_{rt}_ct_{ct}_ybefore_{max_year}_q_{QUANTILE_GROUP}.svg
             (extra 2 x 2 grid, no titles: D / E (top) and F / G (bottom)
              QUANTILE_GROUP quantile distributions)
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

# utils.py lives in the parent openalex/ directory
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OPENALEX_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, OPENALEX_DIR)
from utils import META_DIRE  # noqa: E402

# how the review paper set is identified (selects the inputs / cache / figure names):
#   'venue' -> papers in the annual-review venues: read the two
#              get_annual_review_papers_and_refs.py pkls (annual_review_papers{NAME_SUFFIX}.pkl
#              and annual_review_paper_references{NAME_SUFFIX}.pkl) from META.
#   'title' -> a paper is a review iff its title (lower-cased) CONTAINS REVIEW_TITLE_KEYWORD
#              ('review'). Derived in-script from META/pid2title.pkl; the references are the
#              union of META/citing_pid2cited_pids_list.pkl over those papers.
#   'type'  -> the document-type-review papers = the pids that appear only once reviews are
#              included, i.e. set(pid2pub_year in META_DIRE_with_review) - set(pid2pub_year in
#              META_DIRE); their references are the union of the with_review
#              citing_pid2cited_pids_list over them. Because those pids only exist in the
#              '_with_review' tree, 'type' forces USE_REVIEW_META=True.
# Only the review/reference pid sets differ between modes; the metric table is otherwise
# mode-independent. mode_suffix is added to the cache/figure names so the modes never
# overwrite each other.
detection_mode = 'type'   # 'venue', 'title' or 'type'
assert detection_mode in ('venue', 'title', 'type'), detection_mode
_MODE_SUFFIX = {'venue': '', 'title': '_title_review', 'type': '_type_review'}
mode_suffix = _MODE_SUFFIX[detection_mode]
REVIEW_TITLE_KEYWORD = 'review'

# ---- option: read the review-inclusive meta (must match find_annual_review_venues.py /
#      get_annual_review_papers_and_refs.py / compute_efgd_quantile.py). True -> the
#      '_with_review' tree (paper-set pkls + the '_with_review' quantile table); False ->
#      the plain META_DIRE tree. ----
USE_REVIEW_META = True
# 'type' derives its review set from the with_review-vs-base pid difference, so its metrics
# only live in the '_with_review' tree: force USE_REVIEW_META for that mode.
if detection_mode == 'type' and not USE_REVIEW_META:
    print("detection_mode 'type' needs the _with_review tree; forcing USE_REVIEW_META=True")
    USE_REVIEW_META = True
META = META_DIRE + '_with_review' if USE_REVIEW_META else META_DIRE
# suffix matching the upstream '_with_review' file/dir naming so the two variants never collide
NAME_SUFFIX = '_with_review' if USE_REVIEW_META else ''
print("detection_mode:", detection_mode, "| USE_REVIEW_META:", USE_REVIEW_META,
      "| META dir:", META)

# ---- parameters: must match compute_efgd_quantile.py to find its output file ----
year_offset = 5
reference_threshold = 1
citation_threshold = 1
max_year = 2025 - year_offset

# ---- which quantile the extra 2x2 figure draws: choose the grouping the percentile rank
#      was computed within (all three are columns in the compute_efgd_quantile.py table).
#      The main 4x4 grid is INDEPENDENT of this -- it always draws all three. ----
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

print("==== hyper-parameters ====")
print("year_offset", year_offset)
print("reference_threshold", reference_threshold)
print("citation_threshold", citation_threshold)
print("max_year", max_year)
print("detection_mode", detection_mode)
print("mode_suffix", repr(mode_suffix))
print("USE_REVIEW_META", USE_REVIEW_META)
print("NAME_SUFFIX", repr(NAME_SUFFIX))
print("QUANTILE_GROUP", QUANTILE_GROUP, "-> column suffix", repr(quantile_suffix))
print("==========================")

# metrics to plot: (raw column, human label)
metrics = [
    (f'e_y{year_offset}',        'E (Extension)'),
    (f'f_y{year_offset}',        'F (Foundation)'),
    (f'g_y{year_offset}',        'G (Generalization)'),
    (f'disp_y{year_offset}',     'D (Disruption)'),
    (f'ij_ratio_y{year_offset}', 'I/(I+J)'),
]
label_by_raw = {raw_col: label for raw_col, label in metrics}

# the two overlaid paper sets: (internal label, colour). The actual pid lists are resolved
# per detection_mode when the cache is built (venue reads pkls; title/type derive in-script).
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

# filter-parameter suffix shared by the input quantile table, cache and figure names.
# NAME_SUFFIX sits right after y{off} to match compute_efgd_quantile.py's output name.
param_suffix = (f'y{year_offset}{NAME_SUFFIX}_'
                f'rt_{reference_threshold}_'
                f'ct_{citation_threshold}_'
                f'ybefore_{max_year}')
# two output-name suffixes, both carrying mode_suffix so venue/title/type never collide:
#   full_param_suffix : params + mode. Used by the CACHE and the FULL 4x4 grid (+ its
#                       standalone subplots), which draw ALL THREE quantile groupings and
#                       are therefore INDEPENDENT of QUANTILE_GROUP.
#   out_param_suffix  : the above PLUS '_q_{QUANTILE_GROUP}'. Used only by the 2x2 figure
#                       (and the shared legend JSON), which plot the single selected group.
# The INPUT metric table is independent of both and keeps using param_suffix (it is
# mode-independent).
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
cache_path = os.path.join(
    fig_dire,
    f'annual_review_efgd_ijratio_cache_{full_param_suffix}_withalldomain_allq.csv')


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# ==================================================
# 1. obtain the long-form cache (group, pid, <raw_cols>, <quant_cols>):
#    load from CSV cache if present, otherwise build it from the inputs and save.
#
#    The compute_efgd_quantile.py table already carries, for EVERY paper passing the
#    reference / citation / year filters, the raw indices AND all three quantile groupings
#    (no-/multi-domain papers are kept with the (year, domain) quantile = NaN). So a
#    group's rows are simply taken from that table by pid.
# ==================================================
if os.path.exists(cache_path):
    cache = pd.read_csv(cache_path, dtype={'pid': str})
    print("loaded cache:", cache_path, cache.shape)
else:
    print("cache not found, building from inputs ...")

    csv_path = os.path.join(
        META, 'efg_computation',
        f'efgd_within_year_domain_quantile_{param_suffix}.csv'
    )
    assert os.path.exists(csv_path), csv_path
    df = pd.read_csv(csv_path, dtype={'pid': str}).set_index('pid')
    print("loaded metric table:", csv_path, df.shape)
    cols_needed = raw_cols + quant_cols
    # the (year, domain) quantile of any index marks domain membership (NaN if no domain)
    domain_marker_col = f'disp_y{year_offset}_quantile'

    # ---- resolve each group's pid list for the chosen detection_mode ----
    #   venue : read the two get_annual_review_papers_and_refs.py pkls from META.
    #   title : review papers = pids whose title (lower-cased) contains REVIEW_TITLE_KEYWORD
    #           (from META/pid2title.pkl); references = union of their cited pids.
    #   type  : review papers = the document-type-review pids that appear only once reviews
    #           are included = set(pid2pub_year in META_DIRE_with_review)
    #           - set(pid2pub_year in META_DIRE); references = union of their cited pids
    #           (from the with_review citing map). USE_REVIEW_META is forced True above.
    if detection_mode == 'venue':
        group_pids = {}
        for label, pkl_name in (
            ('Review papers',    f'annual_review_papers{NAME_SUFFIX}.pkl'),
            ('Reference papers', f'annual_review_paper_references{NAME_SUFFIX}.pkl'),
        ):
            group_pids[label] = [str(p) for p in load_pkl(os.path.join(META, pkl_name))]
    elif detection_mode == 'title':
        pid2title = load_pkl(os.path.join(META, 'pid2title.pkl'))
        review_native = [pid for pid, title in pid2title.items()
                         if title and REVIEW_TITLE_KEYWORD in title.lower()]
        print(f"title mode: {len(review_native)} papers whose title contains "
              f"{REVIEW_TITLE_KEYWORD!r}")
        citing2cited = load_pkl(os.path.join(META, 'citing_pid2cited_pids_list.pkl'))
        ref_native = set()
        for pid in review_native:
            refs = citing2cited.get(pid)
            if refs:
                ref_native.update(refs)
        print(f"title mode: their references -> {len(ref_native)} unique pids")
        group_pids = {
            'Review papers':    [str(p) for p in review_native],
            'Reference papers': [str(p) for p in sorted(ref_native)],
        }
    else:  # 'type'
        wr_pids = set(load_pkl(os.path.join(META_DIRE + '_with_review', 'pid2pub_year.pkl')))
        base_pids = set(load_pkl(os.path.join(META_DIRE, 'pid2pub_year.pkl')))
        review_native = sorted(wr_pids - base_pids)   # the extra (review-type) papers
        print(f"type mode: {len(wr_pids)} with_review pids - {len(base_pids)} base pids "
              f"-> {len(review_native)} review-type papers")
        citing2cited = load_pkl(os.path.join(META_DIRE + '_with_review',
                                             'citing_pid2cited_pids_list.pkl'))
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

    parts = []
    for label in group_labels:
        pids = pd.Index(group_pids[label]).unique()
        present = pids.intersection(df.index)
        n_domain = int(df.loc[present, domain_marker_col].notna().sum())
        print(f"{label}: {len(pids)} pids, {len(present)} present in metric table "
              f"(after ref/citation/year filters), of which {n_domain} have a domain/quantile")
        part = df.loc[present, cols_needed].copy()
        part.insert(0, 'group', label)
        part.index.name = 'pid'
        parts.append(part.reset_index())
    cache = pd.concat(parts, ignore_index=True)
    cache.to_csv(cache_path, index=False)
    print("wrote cache:", cache_path, cache.shape)

# ==================================================
# 2. plotting helpers
# ==================================================
# number of points at which the KDE curve is evaluated across the x-range (a smooth
# line replaces the old step histogram; more points -> smoother curve)
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
               annotate_fontsize=8, legend_loc=None, legend_fontsize=None):
    """Draw one REVIEW-vs-REFERENCE overlaid histogram onto ax. When title is None no
    title is set (used for the standalone subplot files). xlabel is the metric name shown
    on the x-axis (E/F/G/D/...). show_ylabel controls whether the 'density' y-axis label
    is drawn (used to label only the leftmost column of a grid). annotate_mean adds a
    coloured 'mean=...' text beside each group's mean vertical line. draw_median controls
    whether the dotted median line is drawn (the median is always recorded in the returned
    legend entry regardless). legend_loc, when not None (e.g. 'lower left'), draws a
    matplotlib legend of the per-group (n, mean, median) labels at that location; left None
    no legend is drawn on the figure. Returns the list of per-group legend entries (empty
    when there is no data)."""
    # restrict to papers that have a quantile (i.e. a domain) when asked
    sub = cache[cache[quant_col].notna()] if domain_only else cache

    # common bin edges across both groups so the histograms are comparable
    all_vals = pd.to_numeric(sub[col], errors='coerce').dropna()
    if len(all_vals) == 0:
        if title is not None:
            ax.set_title(title)
        return []
    lo, hi = np.nanpercentile(all_vals, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(all_vals.min()), float(all_vals.max())
    if lo == hi:
        hi = lo + 1.0
    # fine grid on which each group's KDE curve is evaluated (a smooth line replaces the
    # old step histogram)
    xs = np.linspace(lo, hi, n_kde_points)

    # first pass: gather each group's values + mean/median + KDE curve so the label
    # precision can be chosen AFTER both groups are known (3 decimals by default, more if
    # the two groups' mean -- or median -- still round to the same string, until they differ)
    drawn = []  # (glabel, color, vals, mean_v, median_v, ys)
    for glabel in group_labels:
        color = group_color[glabel]
        vals = pd.to_numeric(
            sub.loc[sub['group'] == glabel, col], errors='coerce'
        ).dropna().values
        if len(vals) == 0:
            continue
        ys = _kde_curve(vals, xs)
        drawn.append((glabel, color, vals, float(np.mean(vals)),
                      float(np.median(vals)), ys))

    # decimals needed to tell Review from Reference apart, independently for mean / median
    mean_dec = _decimals_to_distinguish([m for _, _, _, m, _, _ in drawn])
    median_dec = _decimals_to_distinguish([md for _, _, _, _, md, _ in drawn])

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
    for gi, (glabel, color, vals, mean_v, median_v, ys) in enumerate(drawn):
        legend_label = (f'{group_display.get(glabel, glabel)} (n={len(vals):,}, '
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
            ax.annotate(f'mean={mean_v:.{mean_dec}f}',
                        xy=(mean_v, y_frac),
                        xycoords=blended_transform_factory(ax.transData, ax.transAxes),
                        xytext=(dx, 0), textcoords='offset points',
                        color=color, fontsize=annotate_fontsize, ha=ha, va=va,
                        clip_on=False)
        legend_entries.append({
            'group': glabel,
            'color': color,
            'n': int(len(vals)),
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


# legend text is no longer drawn on the figures; collect it here so it can be written to
# a JSON sidecar for lookup. Keyed by figure -> panel -> list of per-group entries.
legend_records = {'main_grid': {}, 'dfeg_quantile_2x2': {}}

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
    (None,                False, 'raw value',                  'raw_all'),
    ('_quantile_global',  False, 'quantile (no group)',        'quantile_global'),
    ('_quantile_by_year', False, 'quantile (by year)',         'quantile_by_year'),
    ('_quantile',         True,  'quantile (by year, domain)', 'quantile_year_domain'),
]

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
        #     y-labels are suppressed in favour of one shared 'Density' label (set below).
        entries = draw_panel(axes[r, c], col, domain_quant_col, domain_only,
                             None, xlabel, show_ylabel=False,
                             draw_median=False, legend_loc='upper right',
                             legend_fontsize=9)
        _despine(axes[r, c])
        legend_records['main_grid'][f'{raw_col}_{panel_key}'] = {
            'metric': label, 'panel': kind, 'xlabel': xlabel, 'entries': entries,
        }
        # standalone single-panel exports are intentionally NOT produced here -- only the
        # full 4x4 grid (and the 2x2 quantile figure below) are saved.

# no figure-level title; one shared 'Density' y-label on the far left, vertically centred
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

# ==================================================
# 4. extra figure: the QUANTILE_GROUP quantile distributions for D, F, E, G drawn together
#    in a single 2 x 2 grid (REVIEW vs REFERENCE overlaid, domain papers only). Top row:
#    D, E; bottom row: F, G. No titles.
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
    entries = draw_panel(ax_q, quant_col, quant_col, True, None,
                         '', show_ylabel=False,
                         annotate_mean=True, draw_median=False, annotate_fontsize=15)
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

save_path_q = os.path.join(
    fig_dire, f'annual_review_dfeg_quantile_distribution_{out_param_suffix}.svg')
fig_q.savefig(save_path_q, format='svg', transparent=True)
print("saved D/F/E/G quantile figure to:", save_path_q)

# ==================================================
# 5. legend sidecar: the per-group (n, mean, median) info that used to live in the figure
#    legends is written to JSON so it can still be looked up after legends were removed.
# ==================================================
legend_path = os.path.join(
    fig_dire, f'annual_review_efgd_ijratio_legend_{out_param_suffix}.json')
with open(legend_path, 'w', encoding='utf-8') as f:
    json.dump(legend_records, f, indent=2, ensure_ascii=False)
print("saved legend info to:", legend_path)
print("finished")
