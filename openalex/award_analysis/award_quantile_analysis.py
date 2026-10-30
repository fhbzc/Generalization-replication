"""
award_quantile_analysis.py

For every MAJOR-AWARD winning paper and every NOBEL-PRIZE winning paper, look up its
quantile of  f, e, g, disp, disp_f>0, disp_f>1, disp_f>2, I/(I+J), and the three (i+j+k)-
denominator ratios  f/(i+j+k), disp_f/(i+j+k), i/(i+j+k)  from compute_efgd_quantile.py's
per-paper quantile table, write them to CSV, AND draw the quantile DISTRIBUTIONS (KDE, in the
style of venue_validation/plot_annual_review_efgd_distribution.py).

The per-paper data feeding the figures is cached: on the first run the script computes it
(identify award papers -> look up quantiles), writes the winner CSVs, and saves the plotting
cache; on later runs it loads the cache directly and only redraws (delete the cache to refresh).

Figures: one per (award type x metric) = 2 x 8 = 16. Each overlays AWARD-WINNING vs OTHER
papers' quantile distribution for that metric:
  * Nobel figure(s)       : Nobel winners vs other papers
  * Major-award figure(s) : major-award winners vs other papers
"Other" = EVERY non-winner paper in the quantile table (the whole corpus, not domain-
restricted), so both award types share the SAME baseline; the comparison stays fair because
the quantile is domain-relative -- each paper is ranked within its own domain, so pooling all
domains does not mix scales.

The winner quantiles are written to CSV kept SEPARATE BY DOMAIN: each award file is tied to a
domain (major-prize files -> Medicine / Biology / Chemistry; Nobel files -> Medicine /
Chemistry / Physics). A combined CSV plus one CSV per domain are written (when computing).

Identification (mirrors the regression scripts):
  * MAJOR awards : PRIZE_DATA_DIRE Excel files. Nobel rows excluded. PMIDs gathered from every
                   'PMID*' column, mapped to a pid via META_DIRE/pid2pmid.pkl. The individual
                   award name comes from each file's prize_col.
  * NOBEL        : NOBEL_DATA_DIRE CSV files. Rows with 'Is prize-winning paper' == 'YES';
                   DOI mapped to a pid via META_DIRE/pid2doi.pkl (unique DOI mapping only).

Input  : compute_efgd_quantile.py output (must match its parameters below):
           META_DIRE/efg_computation/
             efgd_within_year_domain_quantile_y{off}{ds}_rt_{rt}_ct_{ct}_ybefore_{max_year}{tag}.csv
         META_DIRE/pid2pmid.pkl                                   (major-award PMID -> pid)
         META_DIRE/pid2doi.pkl                                    (Nobel DOI -> pid)
         PRIZE_DATA_DIRE/<prize xls/xlsx>, NOBEL_DATA_DIRE/<nobel csv>   (external; see utils.py)
Cache  : ./figures/award_quantile_disp_f/q_{QUANTILE_GROUP}/cache/{cache_stem}.csv
           (winners + their-domain peers, with the quantile columns + is_nobel/is_major flags)
         ./figures/award_quantile_disp_f/q_{QUANTILE_GROUP}/award_quantile_curves_{cache_stem}.pkl
           (the KDE curves; lets a re-run redraw without recomputing)
Output : every filename below carries
           {param_suffix} = y{off}{ds}_rt_{rt}_ct_{ct}_ybefore_{max_year}{tag}_q_{QUANTILE_GROUP}
         so runs with different thresholds / windows / quantile groupings keep their own CSVs,
         figures and caches side by side instead of overwriting each other. (The paths are also
         still nested under a q_{QUANTILE_GROUP} subdirectory.)
         META_DIRE/award_quantile_analysis/q_{QUANTILE_GROUP}/
           award_quantiles_all_{param_suffix}.csv                 (winners, one row per paper)
           by_domain/award_quantiles_{domain}_{param_suffix}.csv
         ./figures/award_quantile_disp_f/q_{QUANTILE_GROUP}/
           award_quantile_dist_{award_type}_{metric}_{param_suffix}.svg  (per-metric)
           combined_f_dispf_ijratio/
             award_quantile_dist_{award_type}_f_dispf_ijratio_{param_suffix}.svg
             (combined 1x4: Foundation / disp_f(>0) / I(I+J) / Generalization, one per award type)
           combined_ijk_ratio/
             award_quantile_dist_{award_type}_ijk_ratio_{param_suffix}.svg
             (combined 1x4: f/(i+j+k) / disp_f/(i+j+k) / i/(i+j+k) / (i-j)/(i+j+k)=Disruption,
              one per award type; award-winning vs other overlaid)
         Each multi-panel figure's mean/n CSV is written next to it in the same subfolder.
           award_quantile_gap_bootstrap_{param_suffix}.csv
             (no figure of its own: one row per award type x metric with the winner/other
              means, the gap between them, and a bootstrap 95% CI of that gap -- read by
              combine_wos_openalex/plot_validation_quantile_gap.py. Covers EVERY metric,
              including ones no panel draws.)
"""

import os
import re
import sys
import pickle
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
from utils import META_DIRE, PRIZE_DATA_DIRE, NOBEL_DATA_DIRE  # noqa: E402

meta_dire = META_DIRE
efg_dire = os.path.join(meta_dire, 'efg_computation')

# ---- must match compute_efgd_quantile.py so its output filename resolves ----
year_offset = 5
reference_threshold = 1
citation_threshold = 5
with_review = False
data_suffix = '_with_review' if with_review else ''
# which pub_year cutoff / variant of the quantile table to read (see compute_efgd_quantile.py):
#   max_year = 2025 - year_offset, tag = ''            -> complete-window papers only
#   max_year = 2025, tag = '_full_cover'               -> ALL papers before 2025
# For year_offset == -1 compute_efgd_quantile.py only produces the '_full_cover' variant, so
# read that one automatically.
if year_offset == -1:
    max_year = 2025
    input_name_tag = '_full_cover'
else:
    max_year = 2025 - year_offset
    input_name_tag = ''

# which quantile grouping to report (a column suffix produced by compute_efgd_quantile.py):
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
qs = _QUANTILE_SUFFIX[QUANTILE_GROUP]

print("year_offset", year_offset, "| rt", reference_threshold, "| ct", citation_threshold,
      "| max_year", max_year, "| tag", repr(input_name_tag),
      "| QUANTILE_GROUP", QUANTILE_GROUP, "->", repr(qs))

# the seven metrics: (short name, raw column, quantile column, display label)
yo = year_offset
METRICS = [
    ('f',        f'f_y{yo}',                     f'f_y{yo}{qs}',                     'Foundation'),
    ('e',        f'e_y{yo}',                     f'e_y{yo}{qs}',                     'Extension'),
    ('g',        f'g_y{yo}',                     f'g_y{yo}{qs}',                     'Generalization'),
    ('disp',     f'disp_y{yo}',                  f'disp_y{yo}{qs}',                  'Disruption'),
    ('disp_f_0', f'disruption_f_y{yo}',          f'disruption_f_y{yo}{qs}',          'Disruptive foundation (>0)'),
    ('disp_f_1', f'disruption_f_down_gt1_y{yo}', f'disruption_f_down_gt1_y{yo}{qs}', 'Disruptive foundation (>1)'),
    ('disp_f_2', f'disruption_f_down_gt2_y{yo}', f'disruption_f_down_gt2_y{yo}{qs}', 'Disruptive foundation (>2)'),
    ('ij_ratio', f'ij_ratio_y{yo}',             f'ij_ratio_y{yo}{qs}',              'I / (I + J)'),
    # the three (i+j+k)-denominator ratios from compute_efgd_quantile.py (used in the 1x4 figure
    # below; the 4th panel of that figure, (i-j)/(i+j+k), IS the disruption index 'disp' above).
    ('f_over_ijk',     f'f_over_ijk_y{yo}',     f'f_over_ijk_y{yo}{qs}',     'F / (I+J+K)'),
    ('dispf_over_ijk', f'dispf_over_ijk_y{yo}', f'dispf_over_ijk_y{yo}{qs}', 'disp_f / (I+J+K)'),
    ('i_over_ijk',     f'i_over_ijk_y{yo}',     f'i_over_ijk_y{yo}{qs}',     'I / (I+J+K)'),
]
raw_cols = [m[1] for m in METRICS]
quant_cols = [m[2] for m in METRICS]

# OPTIONAL metrics: same 4-tuple shape, but only reported when the quantile table actually
# carries them, so this script runs unchanged against a table produced before they existed.
# E/(I+J+K) is the E counterpart of f_over_ijk (e_count_y{off} over the disruption denominator)
# and only appears once compute_efgd_quantile.py has been re-run with that column added.
# They take part in the gap/bootstrap CSV (section 5) only -- the figures keep their fixed
# panel sets. NOTE the plot cache is written with whatever columns existed when it was built:
# after re-running compute_efgd_quantile.py, delete the cache under fig_dir/cache to pick the
# new column up.
OPTIONAL_METRICS = [
    ('e_over_ijk', f'e_over_ijk_y{yo}', f'e_over_ijk_y{yo}{qs}', 'E / (I+J+K)'),
]

# ---- per-file award config (all domains; missing files are skipped with a warning) ----
PRIZE_FILES = [
    {'file': 'Medicine Prize Winners Cleaned 2.xls',  'domain': 'Medicine',  'prize_col': 'Prize'},
    {'file': 'Biology Prize Winners (12.30).xlsx',    'domain': 'Biology',   'prize_col': 'Prize'},
    {'file': 'Chemistry Prize Winners Cleaned 2.xlsx', 'domain': 'Chemistry', 'prize_col': 'Award'},
]
NOBEL_FILES = [
    {'file': 'Medicine publication record.csv',  'domain': 'Medicine'},
    {'file': 'Chemistry publication record.csv', 'domain': 'Chemistry'},
    {'file': 'Physics publication record.csv',   'domain': 'Physics'},
]

# ---- output + cache locations ----
# CSV outputs stay under META_DIRE; the plotting cache and figures live in ./figures
# (next to this script): both inside the figure's own ./figures/award_quantile_disp_f
# subfolder -- the figures directly in it, and the plotting cache under its ./cache subdir.
# Every output path carries a q_{QUANTILE_GROUP} subdirectory, so each quantile grouping
# keeps its own CSVs / figures / caches and runs with different groupings never mix.
out_dir = os.path.join(meta_dire, 'award_quantile_analysis', f'q_{QUANTILE_GROUP}')
by_domain_dir = os.path.join(out_dir, 'by_domain')
figures_root = os.path.join(BASE_DIR, 'figures')
fig_dir = os.path.join(figures_root, 'award_quantile_disp_f', f'q_{QUANTILE_GROUP}')
cache_dir = os.path.join(fig_dir, 'cache')
# the two multi-panel figures (+ their mean/n CSVs) get their own subfolders, so they stay
# separate from the 16 per-metric figures sitting directly in fig_dir.
combined_fdg_dir = os.path.join(fig_dir, 'combined_f_dispf_ijratio')
combined_ijk_dir = os.path.join(fig_dir, 'combined_ijk_ratio')
for _d in (by_domain_dir, cache_dir, fig_dir, combined_fdg_dir, combined_ijk_dir):
    os.makedirs(_d, exist_ok=True)
# Every output CSV / figure / cache filename carries this suffix, so runs with different
# thresholds (rt / ct), citation windows, or quantile groupings never overwrite each other's
# results -- each parameter set keeps its own files side by side.
param_suffix = (f'y{year_offset}{data_suffix}_'
                f'rt_{reference_threshold}_ct_{citation_threshold}_'
                f'ybefore_{max_year}{input_name_tag}_q_{QUANTILE_GROUP}')
print("output param_suffix:", param_suffix)

# the plotting cache holds every paper in the quantile table (all winners + all non-winners,
# the shared 'other' baseline) with the quantile columns + winner flags; keyed by all parameters
# so settings never clash.
# '_ijk' marks the schema that also carries the three (i+j+k)-ratio quantile columns, so a
# pre-ijk cache (missing those columns) is never reused -- the path changes -> clean rebuild.
# '_allother' marks the whole-corpus 'other' baseline, so a pre-change domain-restricted cache
# (which held only the winners' domains) is never reused.
cache_stem = f'award_quantile_plotdata_{param_suffix}_ijk_allother'
plot_cache_path = os.path.join(cache_dir, f'{cache_stem}.csv')


# ==================================================
# 1. obtain the plotting data: load the cache if present, else COMPUTE it (identify the award
#    papers, look up their quantiles, write the winner CSVs) and save the cache.
# ==================================================
plot_cache_hit = os.path.exists(plot_cache_path)
if plot_cache_hit:
    plot_df = pd.read_csv(plot_cache_path, dtype={'pid': str})
    print("loaded plot cache:", plot_cache_path, plot_df.shape)
    missing = [c for c in quant_cols + ['is_nobel', 'is_major'] if c not in plot_df.columns]
    assert not missing, f"cache {plot_cache_path} is missing {missing}; delete it to rebuild."
    # optional metrics are whatever this cache happens to carry (see OPTIONAL_METRICS)
    OPT_METRICS = [m for m in OPTIONAL_METRICS if m[2] in plot_df.columns]
else:
    print("plot cache not found -- computing from the quantile table + award data ...")

    # ---- load the per-paper quantile table (compute_efgd_quantile.py output) ----
    src_path = os.path.join(
        efg_dire,
        f'efgd_within_year_domain_quantile_y{year_offset}{data_suffix}_'
        f'rt_{reference_threshold}_ct_{citation_threshold}_ybefore_{max_year}{input_name_tag}.csv')
    assert os.path.exists(src_path), f"{src_path}\nRun compute_efgd_quantile.py first."
    df_q = pd.read_csv(src_path)
    print("loaded quantile table shape:", df_q.shape)

    # disp_f quantile columns require compute_efgd_quantile.py's use_disp_f=True
    missing = [c for c in raw_cols + quant_cols if c not in df_q.columns]
    assert not missing, (
        f"quantile table is missing columns {missing}.\n"
        f"Re-run compute_efgd_quantile.py with use_disp_f=True (and matching rt/ct/year).")

    # optional metrics: kept only when the table has BOTH their raw and quantile column
    OPT_METRICS = [m for m in OPTIONAL_METRICS
                   if m[1] in df_q.columns and m[2] in df_q.columns]

    # keep just what we report, indexed by pid (as str, so it matches the pmid/doi -> pid maps)
    extra_cols = [c for c in ['pub_year', 'top_concept', 'group_size'] if c in df_q.columns]
    df_q['pid'] = df_q['pid'].astype(str)
    df_q = df_q.set_index('pid')[extra_cols + raw_cols + quant_cols
                                 + [m[1] for m in OPT_METRICS] + [m[2] for m in OPT_METRICS]]

    # ---- identify award papers -> (award_type, domain, award, pid) records ----
    records = []

    print("\nLoading pid2pmid ...")
    with open(os.path.join(meta_dire, 'pid2pmid.pkl'), 'rb') as f:
        pid2pmid = pickle.load(f)
    pmid2pid = {v: k for k, v in pid2pmid.items()}
    print(f"  pid2pmid size: {len(pid2pmid)}")

    for cfg in PRIZE_FILES:
        domain, prize_col = cfg['domain'], cfg['prize_col']
        fpath = os.path.join(PRIZE_DATA_DIRE, cfg['file'])
        if not os.path.exists(fpath):
            print(f"  [major/{domain}] file not found, skipping: {fpath}")
            continue
        engine = 'xlrd' if cfg['file'].endswith('.xls') else 'openpyxl'
        df_prize = pd.read_excel(fpath, engine=engine)
        n_before = len(df_prize)
        df_prize = df_prize[~df_prize[prize_col].str.lower().str.contains('nobel', na=False)]
        df_prize['_award'] = df_prize[prize_col].astype(str).str.strip()
        df_prize = df_prize[df_prize['_award'] != '']

        pmid_cols = [c for c in df_prize.columns
                     if re.match(r'^\s*PMID\s*\d*\s*$', str(c), re.IGNORECASE)]
        for col in pmid_cols:
            df_prize[col] = pd.to_numeric(df_prize[col], errors='coerce')

        award2pids = defaultdict(set)
        for _, prow in df_prize.iterrows():
            award = prow['_award']
            for col in pmid_cols:
                val = prow[col]
                if pd.isna(val):
                    continue
                pid = pmid2pid.get(str(int(val)))
                if pid is not None:
                    award2pids[award].add(str(pid))
        n_papers = sum(len(v) for v in award2pids.values())
        print(f"  [major/{domain}] non-Nobel rows {n_before}->{len(df_prize)}; "
              f"{len(award2pids)} awards; {n_papers} matched (award, pid) pairs")
        for award, pids in award2pids.items():
            for pid in pids:
                records.append({'award_type': 'major', 'domain': domain,
                                'award': award, 'pid': pid})

    print("\nLoading pid2doi ...")
    with open(os.path.join(meta_dire, 'pid2doi.pkl'), 'rb') as f:
        pid2doi = pickle.load(f)
    doi2pid_list = defaultdict(list)
    for pid, doi in pid2doi.items():
        doi2pid_list[doi].append(pid)
    print(f"  pid2doi size: {len(pid2doi)}")

    for cfg in NOBEL_FILES:
        domain = cfg['domain']
        fpath = os.path.join(NOBEL_DATA_DIRE, cfg['file'])
        if not os.path.exists(fpath):
            print(f"  [nobel/{domain}] file not found, skipping: {fpath}")
            continue
        df_n = pd.read_csv(fpath, usecols=['Is prize-winning paper', 'DOI'], encoding='latin1')
        doi_set = set(df_n.loc[df_n['Is prize-winning paper'] == 'YES', 'DOI'].dropna())
        matched = set()
        for doi in doi_set:
            if doi in doi2pid_list and len(doi2pid_list[doi]) == 1:
                matched.add(str(doi2pid_list[doi][0]))
        print(f"  [nobel/{domain}] {len(doi_set)} prize DOIs -> {len(matched)} unique pids")
        for pid in matched:
            records.append({'award_type': 'nobel', 'domain': domain,
                            'award': f'Nobel Prize ({domain})', 'pid': pid})

    assert records, "no award papers identified -- check the prize/nobel data files."
    df_awards = pd.DataFrame(records)
    print(f"\ntotal identified (award, pid) rows: {len(df_awards)} "
          f"({df_awards['pid'].nunique()} distinct pids)")

    # ---- join the quantiles onto the winners and write the winner CSVs (separate by domain) ----
    out = df_awards.join(df_q, on='pid')
    out['in_quantile_table'] = out[quant_cols[0]].notna()
    print(f"award papers found in the quantile table: {int(out['in_quantile_table'].sum())} "
          f"/ {len(out)} (the rest failed the rt/ct filter or lack a disruption value)")
    lead = ['award_type', 'domain', 'award', 'pid', 'in_quantile_table'] + extra_cols
    out = out[lead + raw_cols + quant_cols
              + [m[1] for m in OPT_METRICS] + [m[2] for m in OPT_METRICS]].sort_values(
        ['award_type', 'domain', 'award', 'pid']).reset_index(drop=True)

    all_path = os.path.join(out_dir, f'award_quantiles_all_{param_suffix}.csv')
    out.to_csv(all_path, index=False)
    print("saved (all):", all_path, out.shape)
    for domain, df_dom in out.groupby('domain'):
        dpath = os.path.join(by_domain_dir, f'award_quantiles_{domain}_{param_suffix}.csv')
        df_dom.to_csv(dpath, index=False)
        print(f"  saved [{domain}]: {dpath}  ({len(df_dom)} rows, "
              f"{int(df_dom['in_quantile_table'].sum())} with quantiles)")

    # ---- build + save the plotting cache: EVERY paper in the quantile table + winner flags ----
    # 'other' is the whole corpus (all non-winners passing the rt/ct/year filter), shared by both
    # award types, so the cache keeps every paper -- no domain restriction.
    nobel_pids = set(df_awards.loc[df_awards['award_type'] == 'nobel', 'pid'])
    major_pids = set(df_awards.loc[df_awards['award_type'] == 'major', 'pid'])
    relevant = df_q.copy()
    relevant['is_nobel'] = relevant.index.isin(nobel_pids).astype(int)
    relevant['is_major'] = relevant.index.isin(major_pids).astype(int)
    keep_cols = (['top_concept'] if 'top_concept' in relevant.columns else []) \
        + quant_cols + [m[2] for m in OPT_METRICS] + ['is_nobel', 'is_major']
    plot_df = relevant.reset_index()[['pid'] + keep_cols]
    plot_df.to_csv(plot_cache_path, index=False)
    print("wrote plot cache:", plot_cache_path, plot_df.shape)


# ==================================================
# 2. KDE curves: precompute each (award type x metric) winner/other curve ONCE and cache them
#    (to a small pkl keyed by the plot params), so redraws just replot the cached arrays instead
#    of re-running gaussian_kde on the large "other" population every time. Figures: one per
#    (award type x metric) = 2 x 8 = 16, overlaying award-winning vs "other".
# ==================================================
N_KDE_POINTS = 200
WIN_COLOR, OTHER_COLOR = 'tab:red', 'tab:blue'
AWARD_TYPES = [('nobel', 'Nobel Prize', 'is_nobel'), ('major', 'Major award', 'is_major')]


def kde_curve(vals, xs):
    """Gaussian-KDE density of `vals` on grid `xs`; histogram fallback when KDE is undefined."""
    vals = np.asarray(vals, dtype=float)
    if len(vals) >= 2 and np.ptp(vals) > 0:
        try:
            return gaussian_kde(vals)(xs)
        except np.linalg.LinAlgError:
            pass
    counts, edges = np.histogram(vals, bins=min(50, max(1, len(vals))), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return np.interp(xs, centers, counts, left=0.0, right=0.0)


def _win_other_masks(flag):
    """(win_mask, other_mask): 'other' = EVERY non-winner paper in the quantile table (the whole
    corpus that passed the rt/ct/year filter), the same baseline for both award types."""
    win_mask = plot_df[flag].astype(bool)
    other_mask = ~win_mask
    return win_mask, other_mask


def build_curves():
    """Per (award type, metric): the winner + other KDE curve on a fixed [0,1] grid, plus each
    group's mean and n. gaussian_kde runs here ONCE per run; the result is cached and reused so
    later redraws never recompute it."""
    xs = np.linspace(0.0, 1.0, N_KDE_POINTS)
    curves = {}
    for atype, _adisplay, flag in AWARD_TYPES:
        win_mask, other_mask = _win_other_masks(flag)
        per_metric = {}
        for short, _raw, quant_col, _lbl in METRICS:
            entry = {'xs': xs}
            for grp, mask in [('win', win_mask), ('other', other_mask)]:
                v = pd.to_numeric(plot_df.loc[mask, quant_col], errors='coerce').values
                v = np.asarray(v, dtype=float)
                # keep only finite quantile values. For QUANTILE_GROUP='year_domain' the
                # _quantile column is NaN for no-/multi-domain papers (they are dropped by
                # compute_efgd_quantile.py's (pub_year, top_concept) groupby), so this also
                # removes every no-domain paper from BOTH the winner and the whole-corpus
                # 'other' baseline -- the year_domain comparison stays domain-paper only.
                v = v[np.isfinite(v)]
                entry[f'{grp}_n'] = int(len(v))
                entry[f'{grp}_mean'] = float(v.mean()) if len(v) else float('nan')
                entry[f'{grp}_ys'] = kde_curve(v, xs) if len(v) else np.zeros_like(xs)
            per_metric[short] = entry
        curves[atype] = per_metric
    return curves


# Load the curve cache when it matches AND the plot data wasn't just recomputed; else (re)build.
# Keyed by the plot-cache stem (so any param change picks a new file) + N_KDE_POINTS + metric set.
curve_cache_path = os.path.join(fig_dir, f'award_quantile_curves_{cache_stem}.pkl')
_curve_key = {'schema': 'curves_v1', 'cache_stem': cache_stem,
              'n_kde': N_KDE_POINTS, 'metrics': [m[0] for m in METRICS]}
_cc = None
if plot_cache_hit and os.path.exists(curve_cache_path):
    with open(curve_cache_path, 'rb') as f:
        _cc = pickle.load(f)
if _cc is not None and _cc.get('key') == _curve_key:
    curves = _cc['curves']
    print("loaded curve cache -> redraw only:", curve_cache_path)
else:
    curves = build_curves()
    with open(curve_cache_path, 'wb') as f:
        pickle.dump({'key': _curve_key, 'curves': curves}, f, pickle.HIGHEST_PROTOCOL)
    print("built + saved curve cache:", curve_cache_path)


def draw_dist(ax, entry, xlabel, title):
    """Plot one panel's cached winner/other curves (+ dashed mean lines + boxed legend)."""
    xs = np.asarray(entry['xs'])
    # sample size of each cohort passing the filter (finite quantile values in this panel)
    print("  [%s] Award-winning n=%s, Others n=%s"
          % (title, format(entry['win_n'], ','), format(entry['other_n'], ',')))
    for grp, color, name in [('win', WIN_COLOR, 'Award-winning'), ('other', OTHER_COLOR, 'Others')]:
        if entry[f'{grp}_n'] == 0:
            continue
        ax.plot(xs, np.asarray(entry[f'{grp}_ys']), color=color, linewidth=2.0,
                label=f"{name} (n={entry[f'{grp}_n']:,}, mean={entry[f'{grp}_mean']:.3f})")
        ax.axvline(entry[f'{grp}_mean'], color=color, linestyle='--', linewidth=1.2, alpha=0.9)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_xlim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=9, frameon=False, loc='upper center')


n_saved = 0
for atype, adisplay, flag in AWARD_TYPES:
    if plot_df[flag].astype(bool).sum() == 0:
        print(f"[{adisplay}] no winners in the plotting data -- skipping its figures.")
        continue
    ac = curves[atype]
    print(f"[{adisplay}] winners: {ac['f']['win_n']}  |  other: {ac['f']['other_n']}")
    for short, _raw, quant_col, disp_label in METRICS:
        fig, ax = plt.subplots(figsize=(7, 5))
        draw_dist(ax, ac[short], f'{disp_label} {QUANTILE_GROUP} quantile',
                  f'{adisplay}: {disp_label}')
        fig.tight_layout()
        fpath = os.path.join(
            fig_dir, f'award_quantile_dist_{atype}_{short}_{param_suffix}.svg')
        fig.savefig(fpath, format='svg', transparent=True)
        fig.savefig(fpath[:-4] + '.png', dpi=150)
        plt.close(fig)
        n_saved += 1

print(f"\nsaved {n_saved} distribution figures to: {fig_dir}")


# ==================================================
# 3. combined 4-panel figure per award type: Foundation (f) + disp_f (>0) + I/(I+J) +
#    Generalization (g) quantiles side by side, each overlaying award-winning vs other. One
#    figure for Nobel, one for major awards. Uses the same plotting cache/quantile columns above.
#
#    Instead of a boxed legend, each series' mean line is annotated in-place with a colored
#    text label (paper type + mean, n omitted): the smaller-mean series is labelled to the
#    LEFT of its line and the larger-mean series to the RIGHT, so the two labels sit on
#    opposite (outer) sides of their vertical lines and never overlap. The per-sample
#    mean and n are additionally written to a CSV saved next to the figure.
# ==================================================
# panels: Foundation, disp_f (>0), I/(I+J), Generalization
COMBINED_METRICS = ['f', 'disp_f_0', 'ij_ratio', 'g']
# panel titles shown above each subplot (override the generic METRICS display labels)
COMBINED_TITLES = {'f': 'Foundation', 'disp_f_0': 'Disruptive Foundation',
                   'ij_ratio': 'Disruption Ratio', 'g': 'Generalization'}
_metric_by_short = {m[0]: m for m in METRICS}


def draw_dist_textlabel(ax, entry, title):
    """Plot one panel's cached winner/other curves + a colored mean-line text label per series
    (paper type + mean) on the outer side of each line. Returns per-series stat dicts
    {name, mean, n} for the accompanying CSV. No per-panel y-label: the multi-panel figures
    draw one shared 'Density' label via add_shared_ylabel()."""
    xs = np.asarray(entry['xs'])
    # sample size of each cohort passing the filter (finite quantile values in this panel)
    print("  [%s] Award-winning n=%s, Others n=%s"
          % (title, format(entry['win_n'], ','), format(entry['other_n'], ',')))
    series = []
    for grp, color, name in [('win', WIN_COLOR, 'Award-winning'), ('other', OTHER_COLOR, 'Others')]:
        if entry[f'{grp}_n'] == 0:
            continue
        ax.plot(xs, np.asarray(entry[f'{grp}_ys']), color=color, linewidth=2.0)
        ax.axvline(entry[f'{grp}_mean'], color=color, linestyle='--', linewidth=1.2, alpha=0.9)
        series.append({'name': name, 'color': color,
                       'mean': entry[f'{grp}_mean'], 'n': entry[f'{grp}_n']})

    ax.set_title(title, fontsize=17)
    # title + tick font sizes match the wos plot_annual_review_efgd_distribution.py 1x3 figure;
    # the per-panel x/y labels are suppressed in favour of one shared label each (see the
    # figure loops below).
    ax.set_xlim(0, 1)
    ax.tick_params(labelsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # colored text beside each mean line: smaller mean -> left of its line, larger -> right.
    if series:
        order = sorted(range(len(series)), key=lambda k: series[k]['mean'])
        placement = {order[0]: ('right', -0.012)}          # smaller mean: text to the left
        if len(order) > 1:
            placement[order[-1]] = ('left', 0.012)         # larger mean: text to the right
        for k, s in enumerate(series):
            ha, dx = placement[k]
            ax.text(s['mean'] + dx, 0.97, f"{s['name']}\nmean={s['mean']:.3f}",
                    transform=ax.get_xaxis_transform(),
                    color=s['color'], fontsize=10, fontweight='bold', ha=ha, va='top')
    return series


def add_shared_ylabel(fig, axes):
    """Draw a single 'Density' label at the left edge of a 1xN panel row, vertically centred on
    the panels' bbox (same recipe as the wos plot_annual_review_efgd_distribution.py figures).
    Call AFTER tight_layout so the panel positions are final."""
    pos = axes[0].get_position()
    fig.text(0.016, 0.5 * (pos.y0 + pos.y1), 'Density', rotation=90,
             va='center', ha='center', fontsize=18)


n_combined = 0
for atype, adisplay, flag in AWARD_TYPES:
    if plot_df[flag].astype(bool).sum() == 0:
        print(f"[{adisplay}] no winners -- skipping combined figure.")
        continue
    ac = curves[atype]

    stat_rows = []
    # width scaled per panel from the original 1x3 layout (19 wide for 3 panels -> ~6.33/panel)
    fig, axes = plt.subplots(1, len(COMBINED_METRICS),
                             figsize=(6.325 * len(COMBINED_METRICS), 5))
    for ax, short in zip(axes, COMBINED_METRICS):
        panel_title = COMBINED_TITLES.get(short, _metric_by_short[short][3])
        series = draw_dist_textlabel(ax, ac[short], panel_title)
        for s in series:
            stat_rows.append({'award_type': adisplay, 'metric': short,
                              'metric_label': panel_title, 'paper_type': s['name'],
                              'mean': s['mean'], 'n': s['n']})
    # one shared x-label centred under the panels + one shared y-label at the left edge;
    # reserve margins so the panels/ticks shift up and right, clear of both labels.
    fig.tight_layout(rect=[0.02, 0.06, 1, 1])
    fig.supxlabel('Index quantile', fontsize=18, y=0.01)
    add_shared_ylabel(fig, axes)
    fpath = os.path.join(
        combined_fdg_dir, f'award_quantile_dist_{atype}_f_dispf_ijratio_{param_suffix}.svg')
    fig.savefig(fpath, format='svg', transparent=True)
    fig.savefig(fpath[:-4] + '.png', dpi=150)
    plt.close(fig)

    csv_path = fpath[:-4] + '.csv'
    pd.DataFrame(stat_rows).to_csv(csv_path, index=False)
    n_combined += 1
    print(f"saved combined 4-panel figure [{adisplay}]: {fpath}")
    print(f"  saved per-sample mean/n CSV: {csv_path}")

print(f"saved {n_combined} combined f+disp_f+I/(I+J)+g figures to: {combined_fdg_dir}")


# ==================================================
# 4. combined 4-panel figure per award type: f/(i+j+k) + disp_f/(i+j+k) + i/(i+j+k) +
#    (i-j)/(i+j+k) [= the disruption index 'disp'] quantiles side by side, each overlaying
#    award-winning vs other. One figure for Nobel, one for major awards. Same style as the
#    section-3 figure above (inline coloured mean-line text labels; shared 'Index quantile'
#    x-label).
# ==================================================
IJK_METRICS = ['f_over_ijk', 'dispf_over_ijk', 'i_over_ijk', 'disp']   # 4 panels, in order
# C = I+J (the focal paper's own citations), so C+K is the I+J+K denominator.
IJK_TITLES = {'f_over_ijk': 'F/(C+K)', 'dispf_over_ijk': 'Disp(F)/(C+K)',
              'i_over_ijk': 'I/(C+K)', 'disp': '(I-J)/(C+K)'}

n_ijk = 0
for atype, adisplay, flag in AWARD_TYPES:
    if plot_df[flag].astype(bool).sum() == 0:
        print(f"[{adisplay}] no winners -- skipping 1x4 (i+j+k)-ratio figure.")
        continue
    ac = curves[atype]

    stat_rows = []
    fig, axes = plt.subplots(1, 4, figsize=(25.3, 5))
    for ax, short in zip(axes, IJK_METRICS):
        panel_title = IJK_TITLES.get(short, _metric_by_short[short][3])
        series = draw_dist_textlabel(ax, ac[short], panel_title)
        for s in series:
            stat_rows.append({'award_type': adisplay, 'metric': short,
                              'metric_label': panel_title, 'paper_type': s['name'],
                              'mean': s['mean'], 'n': s['n']})
    # one shared x-label centred under the four panels + one shared y-label at the left edge;
    # reserve margins so the panels/ticks shift up and right, clear of both labels.
    fig.tight_layout(rect=[0.02, 0.06, 1, 1])
    fig.supxlabel('Index quantile', fontsize=18, y=0.01)
    add_shared_ylabel(fig, axes)
    fpath = os.path.join(
        combined_ijk_dir, f'award_quantile_dist_{atype}_ijk_ratio_{param_suffix}.svg')
    fig.savefig(fpath, format='svg', transparent=True)
    fig.savefig(fpath[:-4] + '.png', dpi=150)
    plt.close(fig)

    csv_path = fpath[:-4] + '.csv'
    pd.DataFrame(stat_rows).to_csv(csv_path, index=False)
    n_ijk += 1
    print(f"saved 1x4 (i+j+k)-ratio figure [{adisplay}]: {fpath}")
    print(f"  saved per-sample mean/n CSV: {csv_path}")

print(f"saved {n_ijk} 1x4 f/(i+j+k)+disp_f/(i+j+k)+i/(i+j+k)+disruption figures to: "
      f"{combined_ijk_dir}")


# ==================================================
# 5. the award-winning-minus-other GAP in mean index quantile, with a bootstrap 95% CI.
#    Draws nothing: it is written purely for
#    combine_wos_openalex/plot_validation_quantile_gap.py, which puts this gap on one axis
#    next to the other two validations' gaps. One row per (award type x metric):
#         gap = mean quantile(Award-winning) - mean quantile(Others)
#    Unlike the figures above, EVERY metric gets a row -- including the ones no panel shows
#    (Extension, and E/(I+J+K) once the quantile table carries it).
# ==================================================
BOOT_N = 2000              # bootstrap replicates
BOOT_SEED = 20260801       # fixed, so re-running reproduces the same CI
BOOT_EXACT_MAX = 200_000   # a group larger than this uses the normal approximation below


def _boot_means(v, rng, n_boot=BOOT_N):
    """(replicate means, how they were obtained) for the 1-D array `v`.
    A small group is resampled exactly (with replacement, in blocks so the n_boot x n index
    matrix never materialises at once). The 'Others' baseline is the whole corpus, where exact
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


# this script's short names -> the canonical metric key the combine script keys on (only
# disp_f_0 differs; it is the disp_f the other two validations call 'disp_f')
GAP_METRIC_KEY = {'disp_f_0': 'disp_f'}
GAP_COLUMNS = ['source', 'group_id', 'group', 'metric', 'metric_label',
               'desired_class', 'undesired_class', 'desired_mean', 'undesired_mean',
               'gap', 'gap_ci_low', 'gap_ci_high', 'n_desired', 'n_undesired',
               'desired_std', 'undesired_std', 'n_boot', 'seed',
               'boot_desired', 'boot_undesired', 'quantile_group', 'value_col']

gap_rows = []
for atype, adisplay, flag in AWARD_TYPES:
    win_mask, other_mask = _win_other_masks(flag)
    win_idx, other_idx = win_mask.values, other_mask.values
    for short, _raw, quant_col, label in METRICS + OPT_METRICS:
        vals = pd.to_numeric(plot_df[quant_col], errors='coerce').values
        row = bootstrap_gap(vals[win_idx], vals[other_idx])
        row.update({'source': 'award', 'group_id': atype, 'group': adisplay,
                    'metric': GAP_METRIC_KEY.get(short, short), 'metric_label': label,
                    'desired_class': 'Award-winning', 'undesired_class': 'Others',
                    'quantile_group': QUANTILE_GROUP, 'value_col': quant_col})
        gap_rows.append(row)
        print(f"  [{atype} | {row['metric']}] gap={row['gap']:+.4f} "
              f"95%CI [{row['gap_ci_low']:+.4f}, {row['gap_ci_high']:+.4f}] "
              f"(n {row['n_desired']:,} vs {row['n_undesired']:,})")

gap_csv = os.path.join(fig_dir, f'award_quantile_gap_bootstrap_{param_suffix}.csv')
pd.DataFrame(gap_rows)[GAP_COLUMNS].to_csv(gap_csv, index=False)
print("saved gap + bootstrap CI CSV:", gap_csv)
if not OPT_METRICS:
    print("  NOTE: no optional metric (%s) in the plot cache. Re-run compute_efgd_quantile.py "
          "with that column, then delete %s to pick it up."
          % (', '.join(m[0] for m in OPTIONAL_METRICS), plot_cache_path))

print("finished")
