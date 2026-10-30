"""
STEP 2 of 2: draw the central-vs-non-central index distributions.

Step 1 (forecite_central_papers_efg_s1_split.py) decided WHICH papers are central and cached the
sets; this script only reads that cache, looks the papers' indices up in the S2AG quantile table,
and draws. It never rebuilds the central sets -- if the cache is missing it tells you to run
step 1 (with the same FORECITE_THRESHOLD / TOP_K_PAPERS / out_tag, which are what key it).

For each of step 1's four central-paper definitions (max_score / top5_score / earliest /
earliest_above_mean) TWO figures are drawn, comparing the central papers (one class) against all
the other papers (the other class):
    1x4: Foundation / Disruptive Foundation / Disruption Ratio / Generalization
         (F, disp_f, I/(I+J), G)
    1x4: F/(C+K) / Disp(F)/(C+K) / I/(C+K) / (I-J)/(C+K)         (the (i+j+k)-denominator ratios
         plus the disruption index D; C = I+J, so C+K is the I+J+K denominator)

Both mirror openalex/award_analysis/award_quantile_analysis.py's combined figures (same font
sizes, same panel titles): overlaid KDE curves with a dashed mean line per class and a coloured
in-place "<class>\nmean=..." text beside each mean (no boxed legend; the central class is
labelled "Papers of new concept" vs "Others"), a per-panel subtitle, top+right spines dropped,
one shared "Index quantile" x-label under the row, and one shared "Density" y-label at the left
edge, vertically centred on the panels.

Every plotted value comes from the S2AG quantile table (compute_efg_quantile.py); by default the
within-year quantiles are used.

Input  : ./figures/forecite_gt_<threshold>/central_sets_<out_tag>.pkl   [step 1]
         META_DIRE/efg_computation/efg_within_year_domain_quantile_y{off}{network_suffix}_rt_{rt}_ct_{ct}_ybefore_{max_year}{name_tag}.csv
             (compute_efg_quantile.py; F/disp_f/ij_ratio, the three (i+j+k) ratios and the
              disruption index D, raw + quantile columns, per pid)
Output : ./figures/forecite_gt_<threshold>/
             dist_<definition>_<value_tag>.svg/.png      (one per definition; the F/disp_f/ratio/G 1x4)
             dist_ijk_<definition>_<value_tag>.svg/.png  (one per definition; the 1x4)
             sample_counts_<value_tag>.csv            (per setting x metric x group: sample count n + mean)
             gap_bootstrap_<value_tag>.csv            (no figure: per setting x metric, the
                 central/rest means, their difference and a bootstrap 95% CI of it -- read by
                 combine_wos_openalex/plot_validation_quantile_gap.py. Covers more metrics than
                 the figures: Extension, and E/(C+K) when the quantile table has it.)
             gap_bootstrap_key_<value_tag>.pkl        (that CSV's config key, so a redraw with
                 the same settings does not re-read the quantile table)
             hist_cache_<value_tag>.pkl               (KDE curves, cached PER METRIC: adding a
                 metric reuses the existing curves and only builds the new ones)

Usage  : python forecite_central_papers_efg_s2_plot.py [FORECITE_THRESHOLD]
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
from utils import META_DIRE  # noqa: E402

meta_dire = META_DIRE
print("meta_dire", meta_dire)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
# ---- these three MUST match forecite_central_papers_efg_s1_split.py: together they key the
#      central-sets cache this script loads (a mismatch is caught by an assert below). ----
FORECITE_THRESHOLD = 0.5     # keep keywords whose ForeCite score is > this; CLI override below
if len(sys.argv) > 1:
    FORECITE_THRESHOLD = float(sys.argv[1])

TOP_K_PAPERS = 5             # 'top5_score': top-K scoring papers per keyword

PMI_bigram = 0
begin_year = 1951
finish_year = 2025
MIN_TERM_DF = 10
MIN_CENTRAL_PAPERS = 10
tag = f'pmib_{PMI_bigram}_df_{MIN_TERM_DF}_b_{begin_year}_e_{finish_year}'
out_tag = f'{tag}_mcp_{MIN_CENTRAL_PAPERS}'

# ---- efg quantile table (must match compute_efg_quantile.py) ----
year_offset = -1
use_openalex_augmented = False
network_suffix = '_openalex_augmented' if use_openalex_augmented else ''
reference_threshold = 1       # rt
citation_threshold = 5        # ct
max_year = 2025 - year_offset               # (2025 - year_offset, '') complete window | (2025, '_full_cover') all
name_tag = ''


# which value to plot: the within-year quantile by default. Set USE_QUANTILE=False for the raw
# F/G/disp_f fractions. quantile_kind: '' (year,domain) | '_by_year' | '_global'.
USE_QUANTILE = True
quantile_kind = ''
# select_max_year narrows the LOADED table further: only papers with pub_year <
# select_max_year are kept, the same strict-less-than the table itself used. Set it equal to
# max_year for no extra narrowing.
#
#   * it must be <= max_year: papers the table never held cannot be recovered here;
#   * the quantiles are NOT recomputed, so with USE_QUANTILE this is only sound when the
#     ranking was done WITHIN a publication year -- quantile_kind '' (year, domain) or
#     '_by_year'. Then every surviving paper's rank is relative to its own same-year peers, all
#     of which are either kept too or irrelevant to it. A '_global' rank was computed against
#     the whole table INCLUDING the years now dropped, so it would no longer describe the
#     plotted population; the assert below refuses that combination. With USE_QUANTILE = False
#     nothing is ranked and any narrowing is fine.
#   * every OUTPUT name gains an '_ybefore_{select_max_year}' tag WHEN narrowed, so a narrowed
#     run never overwrites the full one; with no narrowing the names are unchanged.
select_max_year = 2015
assert select_max_year <= max_year, (
    "select_max_year (%s) must be <= max_year (%s): the quantile table only holds papers with "
    "pub_year < %s." % (select_max_year, max_year, max_year))
assert (select_max_year == max_year or not USE_QUANTILE
        or quantile_kind in ('', '_by_year')), (
    "select_max_year (%s) != max_year (%s) narrows the population AFTER the quantiles were "
    "computed, which only stays valid for a WITHIN-YEAR ranking. quantile_kind is %r: set it "
    "to '' or '_by_year', or re-run compute_efg_quantile.py with max_year = %s."
    % (select_max_year, max_year, quantile_kind, select_max_year))

# "rest" class universe: 'all_efg' = every other paper in the efg table (compare central papers
# to the general population); 'scored' = only papers that score some threshold keyword, minus
# the central set (compare central vs non-central WITHIN the ForeCite-scored papers).
REST_UNIVERSE = 'all_efg'

N_KDE_POINTS = 200            # points at which each KDE density curve is evaluated (award style)
WIN_COLOR, OTHER_COLOR = 'tab:red', 'tab:blue'   # central (red) vs rest (blue), as in award
FORCE_RECOMPUTE = False       # ignore any existing KDE-curve cache and rebuild

off = year_offset
value_tag = ('quantile' + (quantile_kind if quantile_kind else '_year_domain')) if USE_QUANTILE else 'raw'
# a narrowed run gets its own cache, CSVs and figures; an un-narrowed one keeps every name
if select_max_year != max_year:
    value_tag += f'_ybefore_{select_max_year}'
# base index columns; the plotted column is the raw base or its quantile.
#   FDG figure: F / disp_f / I/(I+J) = ij_ratio / G (compute_efg_quantile.py takes ij_ratio
#               straight from the disp_full I/J counts).
#   IJK figure: the (i+j+k)-denominator ratios + the disruption index D, mirroring
#               award_quantile_analysis.py's 1x4. All four come from compute_efg_quantile.py.
metric_base = {'f': f'f_y{off}', 'disp_f': f'disruption_f_y{off}', 'ij_ratio': f'ij_ratio_y{off}',
               'g': f'g_y{off}',
               'f_over_ijk': f'f_over_ijk_y{off}', 'dispf_over_ijk': f'dispf_over_ijk_y{off}',
               'i_over_ijk': f'i_over_ijk_y{off}', 'disp': f'disp_y{off}'}
metric_col = ({k: f'{v}_quantile{quantile_kind}' for k, v in metric_base.items()}
              if USE_QUANTILE else dict(metric_base))

print("FORECITE_THRESHOLD", FORECITE_THRESHOLD, "| out_tag", out_tag)
print("value_tag", value_tag, "| metric columns", metric_col)
print("REST_UNIVERSE", REST_UNIVERSE)
print("max_year", max_year, "| select_max_year", select_max_year)

# output dir: ./figures/forecite_gt_<threshold>/  (shared with step 1)
save_dir = os.path.join(BASE_DIR, 'figures', f'forecite_gt_{FORECITE_THRESHOLD}')
os.makedirs(save_dir, exist_ok=True)
cache_path = os.path.join(save_dir, f'hist_cache_{value_tag}.pkl')
print("save_dir", save_dir)

# the per-paper quantile table every value comes from (compute_efg_quantile.py). Read by the
# KDE builder (only for curves it has to build) and by the gap/bootstrap section at the end.
efg_csv = os.path.join(
    meta_dire, 'efg_computation',
    f'efg_within_year_domain_quantile_y{off}{network_suffix}_'
    f'rt_{reference_threshold}_ct_{citation_threshold}_ybefore_{max_year}{name_tag}.csv')

# Config the KDE-curve cache must match to be reusable ('schema' bump invalidates the old
# histogram-schema caches so they are never loaded by this KDE version).
# NOTE metric_col is deliberately NOT part of the key: the curves are cached PER METRIC, and each
# cached entry records the column it was built from, so adding a metric (e.g. the 1x4 figure's
# (i+j+k) ratios) reuses the curves already computed for the others and only builds the new ones.
# A rest-class KDE runs over the whole corpus, so recomputing all of them would be slow.
cache_key = {'schema': 'kde_v1',
             'threshold': FORECITE_THRESHOLD, 'out_tag': out_tag, 'network_suffix': network_suffix,
             'rt': reference_threshold, 'ct': citation_threshold, 'max_year': max_year,
             'select_max_year': select_max_year,
             'name_tag': name_tag, 'value_tag': value_tag,
             'N_KDE_POINTS': N_KDE_POINTS, 'REST_UNIVERSE': REST_UNIVERSE,
             'TOP_K_PAPERS': TOP_K_PAPERS, 'year_offset': year_offset}

# each definition -> readable description (kept for logging; titles are NOT drawn). Must match
# the keys forecite_central_papers_efg_s1_split.py stores in its central_sets dict.
DEFINITIONS = [
    ('max_score', 'MAX-score paper per keyword'),
    ('top5_score', f'TOP-{TOP_K_PAPERS} score papers per keyword'),
    ('earliest', 'EARLIEST (>0 score) paper per keyword'),
    ('earliest_above_mean', 'EARLIEST paper among those scoring > keyword mean'),
]


def apply_select_max_year(df, what):
    """The extra pub_year cut, in the table's own strict-less-than convention.

    A no-op unless select_max_year < max_year. The quantile columns are left exactly as they
    are -- see select_max_year in the settings for why that is sound only for a within-year
    ranking.
    """
    if select_max_year == max_year:
        return df
    assert 'pub_year' in df.columns, (
        f"select_max_year ({select_max_year}) needs a pub_year column, which "
        f"{os.path.basename(efg_csv)} does not have.")
    n_before = len(df)
    df = df[df['pub_year'] < select_max_year]
    print(f"  select_max_year={select_max_year}: {what} {n_before} -> {len(df)} papers "
          f"(dropped {n_before - len(df)} published in {select_max_year}-{max_year - 1}); "
          f"quantiles NOT recomputed")
    assert len(df), f"no paper is left with pub_year < {select_max_year}; check select_max_year."
    return df


def _key_of(cached):
    """The comparable key of a loaded cache: older caches embedded metric_col in the key, so drop
    it before comparing -- their per-metric curves are still reusable (matched by 'col' below)."""
    return {k: v for k, v in (cached.get('key') or {}).items() if k != 'metric_col'}


def metric_xs(mkey):
    """KDE grid for one metric. Quantiles (and the F/disp_f/ratio shares) live in [0,1]; the raw
    disruption index D = (I-J)/(I+J+K) is the one column spanning [-1,1]."""
    lo = -1.0 if (not USE_QUANTILE and mkey == 'disp') else 0.0
    return np.linspace(lo, 1.0, N_KDE_POINTS)


def kde_curve(vals, xs):
    """Gaussian-KDE density of `vals` on grid `xs` (award-style); histogram fallback when KDE
    is undefined (fewer than 2 points or zero variance)."""
    vals = np.asarray(vals, dtype=float)
    if len(vals) >= 2 and np.ptp(vals) > 0:
        try:
            return gaussian_kde(vals)(xs)
        except np.linalg.LinAlgError:
            pass
    counts, edges = np.histogram(vals, bins=min(50, max(1, len(vals))), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return np.interp(xs, centers, counts, left=0.0, right=0.0)


# ===========================================================================
# The central sets come from step 1 -- this script never builds them.
# ===========================================================================
central_cache_path = os.path.join(save_dir, f'central_sets_{out_tag}.pkl')
# must match forecite_central_papers_efg_s1_split.py's central_key exactly, 'defs' included --
# it versions the definition RULES, so a cache built before a rule changed is rejected here.
central_key = {'threshold': FORECITE_THRESHOLD, 'out_tag': out_tag,
               'TOP_K_PAPERS': TOP_K_PAPERS, 'defs': 'v2_earliest_above_mean'}


def load_central_sets():
    """Return (central_sets, scored_pids, n_keywords) from step 1's cache. Fails loudly (rather
    than silently rebuilding) if it is absent or was built with different parameters."""
    assert os.path.exists(central_cache_path), (
        "%s\nRun forecite_central_papers_efg_s1_split.py first (with FORECITE_THRESHOLD=%g)."
        % (central_cache_path, FORECITE_THRESHOLD))
    with open(central_cache_path, 'rb') as f:
        cc = pickle.load(f)
    assert cc.get('key') == central_key, (
        "central-sets cache %s was built with %s but this script expects %s.\nRe-run "
        "forecite_central_papers_efg_s1_split.py with matching FORECITE_THRESHOLD / TOP_K_PAPERS "
        "/ out_tag." % (central_cache_path, cc.get('key'), central_key))
    print("loaded central sets:", central_cache_path, "(%d keywords)" % cc['n_keywords'])
    return cc['central_sets'], cc['scored_pids'], cc['n_keywords']


# ===========================================================================
# Build (or load) the per-definition, per-metric KDE curves
# ===========================================================================
def _reusable(entry, mcol):
    """True if a cached per-metric entry was built from column `mcol` on the current KDE grid."""
    return (entry is not None and entry.get('col') == mcol
            and len(np.asarray(entry.get('xs', []))) == N_KDE_POINTS)


def build_hist_data(prior_hist=None):
    """Compute the central-vs-rest KDE curves for every (definition, metric), REUSING any curve
    already present in prior_hist (matched by the column it was built from) so that adding a
    metric only builds the missing curves. Returns the full cache payload."""
    prior_hist = prior_hist or {}

    # ---- the four unique central-paper sets (built by step 1) ----
    central_sets, scored_pids, n_keywords = load_central_sets()
    for name, s in central_sets.items():
        print("  central set %-11s: %d unique papers" % (name, len(s)))

    # ---- work out which (definition, metric) curves are missing ----
    todo = {}
    for name in central_sets:
        prev = (prior_hist.get(name) or {}).get('metrics', {})
        miss = [mk for mk, mc in metric_col.items() if not _reusable(prev.get(mk), mc)]
        if miss:
            todo[name] = miss
    n_reused = len(central_sets) * len(metric_col) - sum(len(v) for v in todo.values())
    print("KDE curves: %d reused from cache, %d to build" % (n_reused, sum(len(v) for v in todo.values())))

    if not todo:      # everything cached -> no need to touch the efg table at all
        return {'key': cache_key, 'metric_col': dict(metric_col), 'n_keywords': n_keywords,
                'hist': {n: prior_hist[n] for n in central_sets}}

    # ---- efg quantile table: pid + only the columns whose curves must be (re)built ----
    assert os.path.exists(efg_csv), "%s\nRun compute_efg_quantile.py first." % efg_csv
    need_cols = sorted({metric_col[mk] for mks in todo.values() for mk in mks})
    _sel = [] if select_max_year == max_year else ['pub_year']
    df = pd.read_csv(efg_csv, usecols=['pid'] + _sel + need_cols)
    print("loaded efg table:", os.path.basename(efg_csv), "shape", df.shape,
          "| columns read:", need_cols)
    df = apply_select_max_year(df, 'efg table')

    pid_vals = df['pid'].values
    in_scored = (np.isin(pid_vals, np.array(list(scored_pids)))
                 if (REST_UNIVERSE == 'scored' and scored_pids) else None)

    # ---- per-definition, per-metric KDE density curves for central (a) vs rest (b) ----
    hist_data = {}
    for name, cset in tqdm(central_sets.items(), total=len(central_sets), desc='kde curves'):
        prev = (prior_hist.get(name) or {}).get('metrics', {})
        per_metric = {mk: prev[mk] for mk in metric_col if mk not in todo.get(name, [])}
        missing = todo.get(name, [])
        if not missing:
            hist_data[name] = {'n_central_unique': len(cset),
                               'n_central_in_efg': prior_hist[name]['n_central_in_efg'],
                               'metrics': per_metric}
            continue

        central_mask = (np.isin(pid_vals, np.array(list(cset))) if cset
                        else np.zeros(len(pid_vals), dtype=bool))
        if REST_UNIVERSE == 'scored':
            rest_mask = in_scored & ~central_mask
        else:
            rest_mask = ~central_mask
        n_central_in = int(central_mask.sum())
        print("  [%s] central papers with an efg value: %d / %d (rest: %d)"
              % (name, n_central_in, len(cset), int(rest_mask.sum())))
        for mkey in tqdm(missing, total=len(missing), desc='  %s metrics' % name, leave=False):
            mcol = metric_col[mkey]
            xs = metric_xs(mkey)
            col = df[mcol].values
            a = col[central_mask]
            a = a[np.isfinite(a)]
            b = col[rest_mask]
            b = b[np.isfinite(b)]
            per_metric[mkey] = {
                'xs': xs, 'col': mcol,
                'a_ys': kde_curve(a, xs) if len(a) else np.zeros_like(xs),
                'a_n': int(len(a)), 'a_mean': float(a.mean()) if len(a) else float('nan'),
                'b_ys': kde_curve(b, xs) if len(b) else np.zeros_like(xs),
                'b_n': int(len(b)), 'b_mean': float(b.mean()) if len(b) else float('nan'),
            }
        hist_data[name] = {'n_central_unique': len(cset), 'n_central_in_efg': n_central_in,
                           'metrics': per_metric}
    return {'key': cache_key, 'metric_col': dict(metric_col), 'n_keywords': n_keywords,
            'hist': hist_data}


# ---- load cache or (re)build. A cache whose config still matches has its per-metric curves
#      reused even when the metric set changed (build_hist_data only fills the gaps). ----
prior = None
if not FORCE_RECOMPUTE and os.path.exists(cache_path):
    with open(cache_path, 'rb') as f:
        cached = pickle.load(f)
    if _key_of(cached) == cache_key:
        prior = cached.get('hist')
        print("loaded KDE-curve cache:", cache_path)
    else:
        print("cache config changed -> rebuilding:", cache_path)

payload = build_hist_data(prior)
with open(cache_path, 'wb') as f:
    pickle.dump(payload, f, pickle.HIGHEST_PROTOCOL)
print("saved KDE-curve cache:", cache_path)

hist = payload['hist']
n_keywords = payload['n_keywords']

# ===========================================================================
# Draw TWO figures per definition, both central vs rest, both in award_quantile_analysis.py style:
# overlaid KDE curves + dashed mean line + inline coloured "<class>\nmean=..." text (no legend),
# a per-panel subtitle, top/right spines dropped, one shared "Index quantile" x-label under the
# row and one shared "Density" y-label at the left edge, vertically centred on the panels.
#   FDG 1x4: Foundation / Disruptive Foundation / Disruption Ratio / Generalization
#   IJK 1x4: F/(C+K) / Disp(F)/(C+K) / I/(C+K) / (I-J)/(C+K)   [C = I+J, so C+K is the I+J+K denominator]
# ===========================================================================
METRICS_FDG = ['f', 'disp_f', 'ij_ratio', 'g']
TITLES_FDG = {'f': 'Foundation', 'disp_f': 'Disruptive Foundation',
              'ij_ratio': 'Disruption Ratio', 'g': 'Generalization'}
METRICS_IJK = ['f_over_ijk', 'dispf_over_ijk', 'i_over_ijk', 'disp']
TITLES_IJK = {'f_over_ijk': 'F/(C+K)', 'dispf_over_ijk': 'Disp(F)/(C+K)',
              'i_over_ijk': 'I/(C+K)', 'disp': '(I-J)/(C+K)'}

GROUP_LABELS = {'a': 'Papers of new concept', 'b': 'Others'}
sample_rows = []   # per (definition/setting, metric, group): sample count n (+ mean)


def draw_metric_row(name, hd, metric_order, titles, out_stem):
    """Draw + save one 1xN figure (one panel per metric in metric_order) for definition `name`,
    recording each panel's per-group n/mean into sample_rows."""
    fig, axes = plt.subplots(1, len(metric_order), figsize=(6.33 * len(metric_order), 5))
    for ax, mkey in zip(axes, metric_order):
        m = hd['metrics'][mkey]
        xs = np.asarray(m['xs'])
        # record the per-group sample counts for this (setting, metric)
        for gk in ('a', 'b'):
            sample_rows.append({'setting': name, 'metric': mkey, 'group': GROUP_LABELS[gk],
                                'n': m[f'{gk}_n'], 'mean': m[f'{gk}_mean']})
        # print each cohort's sample size passing this panel's filter (finite values in the
        # metric column: central class 'a' vs rest class 'b')
        print("  [%s | %s (%s)] %s=%s, %s=%s"
              % (name, mkey, m['col'], GROUP_LABELS['a'], format(m['a_n'], ','),
                 GROUP_LABELS['b'], format(m['b_n'], ',')))
        # overlay each class's KDE curve + its dashed mean line
        series = []
        for ys, n, mean, color, cls in [
                (m['a_ys'], m['a_n'], m['a_mean'], WIN_COLOR, GROUP_LABELS['a']),
                (m['b_ys'], m['b_n'], m['b_mean'], OTHER_COLOR, GROUP_LABELS['b'])]:
            if not n:
                continue
            ax.plot(xs, np.asarray(ys), color=color, linewidth=2.0)
            ax.axvline(mean, color=color, linestyle='--', linewidth=1.2, alpha=0.9)
            series.append({'name': cls, 'color': color, 'mean': mean})
        ax.set_title(titles.get(mkey, mkey), fontsize=17)
        ax.set_xlim(xs[0], xs[-1])    # [0,1] everywhere except the raw D, which spans [-1,1]
        ax.tick_params(labelsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # coloured text beside each mean line: smaller mean -> left of its line, larger -> right
        if series:
            order = sorted(range(len(series)), key=lambda k: series[k]['mean'])
            placement = {order[0]: ('right', -0.012)}
            if len(order) > 1:
                placement[order[-1]] = ('left', 0.012)
            for k, s in enumerate(series):
                ha, dx = placement[k]
                ax.text(s['mean'] + dx, 0.97, f"{s['name']}\nmean={s['mean']:.3f}",
                        transform=ax.get_xaxis_transform(),
                        color=s['color'], fontsize=10, fontweight='bold', ha=ha, va='top')
    # one shared x-label centred under the row + one shared y-label at the left edge, centred on
    # the panels' bbox; reserve margins so the panels/ticks shift clear of both labels.
    fig.tight_layout(rect=[0.02, 0.06, 1, 1])
    fig.supxlabel('Index quantile', fontsize=18, y=0.01)
    pos = axes[0].get_position()      # after tight_layout, so the panel box is final
    fig.text(0.016, 0.5 * (pos.y0 + pos.y1), 'Density', rotation=90,
             va='center', ha='center', fontsize=18)
    out_svg = os.path.join(save_dir, f'{out_stem}.svg')
    fig.savefig(out_svg, format='svg', transparent=True)
    fig.savefig(out_svg[:-4] + '.png', dpi=150)
    plt.close(fig)
    print("saved:", out_svg)


for name, desc in tqdm(DEFINITIONS, desc='draw figures'):
    hd = hist[name]
    draw_metric_row(name, hd, METRICS_FDG, TITLES_FDG, f'dist_{name}_{value_tag}')
    draw_metric_row(name, hd, METRICS_IJK, TITLES_IJK, f'dist_ijk_{name}_{value_tag}')

# per-setting, per-group sample-count table (one row per definition x metric x group)
sample_csv = os.path.join(save_dir, f'sample_counts_{value_tag}.csv')
pd.DataFrame(sample_rows).to_csv(sample_csv, index=False)
print("saved sample-count CSV:", sample_csv)


# ===========================================================================
# The central-minus-rest GAP in mean index quantile, with a bootstrap 95% CI.
# Draws nothing: it is written purely for
# combine_wos_openalex/plot_validation_quantile_gap.py, which puts this gap on one axis next
# to the award and annual-review validations' gaps. One row per (definition x metric):
#      gap = mean quantile(Papers of new concept) - mean quantile(Others)
# It covers MORE metrics than the figures above -- Extension, and E/(C+K) once the quantile
# table carries it -- so it reads the quantile table itself rather than the KDE cache (which
# only ever stores curves + means). The result is cached: the CSV is rebuilt only when its
# config key changed, so a pure redraw does not re-read the table.
# ===========================================================================
BOOT_N = 2000              # bootstrap replicates
BOOT_SEED = 20260801       # fixed, so re-running reproduces the same CI
BOOT_EXACT_MAX = 200_000   # a group larger than this uses the normal approximation below


def _boot_means(v, rng, n_boot=BOOT_N):
    """(replicate means, how they were obtained) for the 1-D array `v`.
    A small group is resampled exactly (with replacement, in blocks so the n_boot x n index
    matrix never materialises at once). The rest class is the whole corpus, where exact
    resampling would cost n x n_boot draws for a bootstrap distribution that is by then
    indistinguishable from N(mean, s^2/n) (the bootstrap mean is asymptotically normal), so
    beyond BOOT_EXACT_MAX the replicate means are drawn from that normal instead."""
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
    Non-finite values are dropped first, exactly as the KDE panels do."""
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


# the metrics reported in the gap CSV: the eight drawn above PLUS Extension, PLUS E/(C+K)
# when the quantile table has it (it only appears once compute_efg_quantile.py has been re-run
# with that column). The keys are the canonical ones the combine script uses, shared with the
# award and annual-review gap CSVs so the three tables concatenate cleanly. Adding a metric
# here costs one more column read + one bootstrap -- no KDE curve, so no corpus-wide gaussian_kde.
GAP_METRIC_BASE = dict(metric_base)
GAP_METRIC_BASE['e'] = f'e_y{off}'
GAP_METRIC_BASE['e_over_ijk'] = f'e_over_ijk_y{off}'          # optional; dropped when absent
GAP_METRIC_LABEL = {'f': 'Foundation', 'e': 'Extension', 'disp_f': 'Disruptive Foundation',
                    'ij_ratio': 'Disruption Ratio', 'g': 'Generalization',
                    'f_over_ijk': 'F/(C+K)', 'e_over_ijk': 'E/(C+K)',
                    'dispf_over_ijk': 'Disp(F)/(C+K)', 'i_over_ijk': 'I/(C+K)',
                    'disp': '(I-J)/(C+K)'}
GAP_METRIC_COL = ({k: f'{v}_quantile{quantile_kind}' for k, v in GAP_METRIC_BASE.items()}
                  if USE_QUANTILE else dict(GAP_METRIC_BASE))
GAP_COLUMNS = ['source', 'group_id', 'group', 'metric', 'metric_label',
               'desired_class', 'undesired_class', 'desired_mean', 'undesired_mean',
               'gap', 'gap_ci_low', 'gap_ci_high', 'n_desired', 'n_undesired',
               'desired_std', 'undesired_std', 'n_boot', 'seed',
               'boot_desired', 'boot_undesired', 'quantile_group', 'value_col']

# this script names quantile groupings by value_tag; the gap CSV records the same grouping
# under the name the other two validations (and the combine script) use for it.
GAP_QUANTILE_GROUP = ({'': 'year_domain', '_by_year': 'year', '_global': 'global'}[quantile_kind]
                      if USE_QUANTILE else 'raw')

gap_csv = os.path.join(save_dir, f'gap_bootstrap_{value_tag}.csv')
gap_key_path = os.path.join(save_dir, f'gap_bootstrap_key_{value_tag}.pkl')
gap_key = {**cache_key, 'schema': 'gap_v1', 'metrics': sorted(GAP_METRIC_COL.values()),
           'n_boot': BOOT_N, 'seed': BOOT_SEED, 'boot_exact_max': BOOT_EXACT_MAX}
_gap_prior = None
if not FORCE_RECOMPUTE and os.path.exists(gap_csv) and os.path.exists(gap_key_path):
    with open(gap_key_path, 'rb') as f:
        _gap_prior = pickle.load(f)

if _gap_prior == gap_key:
    print("gap CSV already matches this config -- skipping the bootstrap:", gap_csv)
else:
    assert os.path.exists(efg_csv), "%s\nRun compute_efg_quantile.py first." % efg_csv
    have = set(pd.read_csv(efg_csv, nrows=0).columns)
    gap_use = {k: c for k, c in GAP_METRIC_COL.items() if c in have}
    gap_skipped = sorted(set(GAP_METRIC_COL) - set(gap_use))
    if gap_skipped:
        print("gap CSV: skipping %s -- absent from %s (re-run compute_efg_quantile.py with "
              "the column to include it)." % (gap_skipped, os.path.basename(efg_csv)))

    central_sets, scored_pids, _ = load_central_sets()
    need_cols = sorted(set(gap_use.values()))
    _sel = [] if select_max_year == max_year else ['pub_year']
    df_gap = pd.read_csv(efg_csv, usecols=['pid'] + _sel + need_cols)
    print("gap CSV: loaded efg table", df_gap.shape, "| columns read:", need_cols)
    df_gap = apply_select_max_year(df_gap, 'gap efg table')
    pid_vals = df_gap['pid'].values
    in_scored = (np.isin(pid_vals, np.array(list(scored_pids)))
                 if (REST_UNIVERSE == 'scored' and scored_pids) else None)

    gap_rows = []
    for name, cset in tqdm(central_sets.items(), total=len(central_sets), desc='gap bootstrap'):
        central_mask = (np.isin(pid_vals, np.array(list(cset))) if cset
                        else np.zeros(len(pid_vals), dtype=bool))
        rest_mask = (in_scored & ~central_mask) if REST_UNIVERSE == 'scored' else ~central_mask
        for mkey, col in gap_use.items():
            vals = pd.to_numeric(df_gap[col], errors='coerce').values
            row = bootstrap_gap(vals[central_mask], vals[rest_mask])
            row.update({'source': 'forecite', 'group_id': f'forecite_{name}', 'group': name,
                        'metric': mkey, 'metric_label': GAP_METRIC_LABEL.get(mkey, mkey),
                        'desired_class': GROUP_LABELS['a'],
                        'undesired_class': GROUP_LABELS['b'],
                        'quantile_group': GAP_QUANTILE_GROUP, 'value_col': col})
            gap_rows.append(row)
            print(f"  [{name} | {mkey}] gap={row['gap']:+.4f} "
                  f"95%CI [{row['gap_ci_low']:+.4f}, {row['gap_ci_high']:+.4f}] "
                  f"(n {row['n_desired']:,} vs {row['n_undesired']:,})")

    pd.DataFrame(gap_rows)[GAP_COLUMNS].to_csv(gap_csv, index=False)
    with open(gap_key_path, 'wb') as f:
        pickle.dump(gap_key, f, pickle.HIGHEST_PROTOCOL)
    print("saved gap + bootstrap CI CSV:", gap_csv)

print("finished")
