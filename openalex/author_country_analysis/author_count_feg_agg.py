import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)             # utils.py is in the parent directory
from utils import META_DIRE

plt.rcParams['font.family']       = 'Gill Sans MT'
plt.rcParams['axes.unicode_minus'] = False

SAVE_DIRE = os.path.join('./figures', 'author_analysis')
os.makedirs(SAVE_DIRE, exist_ok=True)

# ── Settings ──────────────────────────────────────────────────────────────────
reference_threshold    = 1
citation_threshold     = 5
# ---- citation window ----------------------------------------------------------------------
# Names the index columns (f_y{off}, disp_y{off}, ...) AND every input filename, so it must
# match the get_efg / get_disruption / compute_efgd_quantile runs being read. It is part of the
# OUTPUT names only when it is NOT the default 5, so the 5-year run keeps the names it has.
year_offset            = -1
off                    = year_offset
year_tag               = '' if year_offset == 5 else f'_y{year_offset}'
# arrange_disruption.py tags disp_full the same way
disp_year_tag          = '' if year_offset == 5 else f'_y{year_offset}'
# compute_efgd_quantile.py's own rule, and the reason max_year is DERIVED rather than typed
# in: the pub_year cutoff normally leaves room for the citation window (2025 - year_offset),
# but at year_offset == -1 there is no elapsed window to leave room for, so it keeps every
# paper before the data horizon and tags that file '_full_cover'. Getting either half wrong
# makes the load below look for a file that was never written.
# To analyse a narrower period, leave this alone and set select_max_year below.
if year_offset == -1:
    max_year           = 2025
    quantile_name_tag  = '_full_cover'
else:
    max_year           = 2025 - year_offset
    quantile_name_tag  = ''
print("year_offset",year_offset)
# select_max_year narrows the analysed population further: only papers with pub_year <
# select_max_year are kept. It must be <= max_year, and the OUTPUT names carry IT rather than
# max_year, so a narrowed run never overwrites the full one.
#
# On the USE_QUANTILE path the quantiles are NOT recomputed, so the extra cut is only sound
# when the ranking was done WITHIN a publication year ('year_domain' or 'year'): a 'global'
# rank was computed against the whole table including the years now dropped. The assert below
# refuses that combination. On the raw path there is no such issue -- nothing is ranked.
select_max_year        = 2015
assert select_max_year <= max_year, (
    f"select_max_year ({select_max_year}) must be <= max_year ({max_year}).")
n_bootstrap            = 1000
tick_size              = 13
label_size             = 15
legend_size            = 11

# when True, additionally load the three disp_f ("disruptive foundation" share) variants from
# get_efg_v2_disp_f.py -> get_efg_s2_merge.py (use_disp_f=True): disruption_f_y5 (down>0),
# disruption_f_down_gt1_y5 (down>1), disruption_f_down_gt2_y5 (down>2). Instead of an inset, the
# disp_f-on figure adds a SECOND ROW of three panels (disp_f 0/1/2 vs author_count) below the
# F/E/G row. The disp_f-on figure gets a '_dispf' filename tag so it never overwrites the
# disp_f-off deliverable.
use_disp_f = False
dispf_tag  = '_dispf' if use_disp_f else ''

# the three disp_f variants: (short key used in df_agg columns, source column in gen_V3_disp_f_y5.csv,
# panel label). T = 0 / 1 / 2 -> a citing paper shares none of the focal paper's references and
# co-cites more than T of its other citing papers.
dispf_variants = [
    ('dispf0', f'disruption_f_y{off}',          'Disruptive foundation'),
    ('dispf1', f'disruption_f_down_gt1_y{off}', 'Disruptive foundation (>1)'),
    ('dispf2', f'disruption_f_down_gt2_y{off}', 'Disruptive foundation (>2)'),
]

# ── Output value: RAW index or QUANTILE ───────────────────────────────────────
# USE_QUANTILE picks what every panel plots -- the two figures are otherwise identical:
#   False -> the raw per-paper index (f_y5, disp_y5, the i+j+k ratios, ...), averaged per
#            author_count. Reads gen_V3 / disp_full / gen_V3_disp_f and derives the ratios here.
#   True  -> each paper's QUANTILE (its percentile rank among its peers) from
#            compute_efgd_quantile.py, averaged per author_count. Every metric -- E/F/G, D, the
#            disp_f shares and the three (i+j+k) ratios -- already has a quantile column in that
#            ONE table, so this path needs no joins and none of the I/J/K arithmetic: the raw
#            aggregation blocks below are skipped entirely.
# The quantile figures carry a '_quantile_{QUANTILE_GROUP}' filename tag and use an 'Index
# quantile' y-label, so they never overwrite (or get confused with) the raw deliverable.
USE_QUANTILE = True

# Which of compute_efgd_quantile.py's three rankings to average (only used when USE_QUANTILE):
#   'year_domain' -> {col}_quantile          (rank within pub_year x top_concept)
#   'year'        -> {col}_quantile_by_year  (rank within pub_year only)
#   'global'      -> {col}_quantile_global   (rank across all papers, no grouping)
# NOTE 'year_domain' leaves no-/multi-domain papers with a NaN quantile (they have no
# top_concept), so they drop out of the means -- a smaller sample than the raw figures.
QUANTILE_GROUP = 'year_domain'
_QUANTILE_SUFFIX = {'year_domain': '_quantile', 'year': '_quantile_by_year',
                    'global': '_quantile_global'}
assert QUANTILE_GROUP in _QUANTILE_SUFFIX, QUANTILE_GROUP
qs = _QUANTILE_SUFFIX[QUANTILE_GROUP]

# see select_max_year above: dropping years after the fact leaves a within-year rank intact
# but invalidates a global one
assert (not USE_QUANTILE) or select_max_year == max_year or QUANTILE_GROUP in (
    'year_domain', 'year'), (
    f"select_max_year ({select_max_year}) != max_year ({max_year}) narrows the population "
    f"AFTER the quantiles were computed, which only stays valid for a WITHIN-YEAR ranking. "
    f"QUANTILE_GROUP is {QUANTILE_GROUP!r}: set it to 'year_domain' / 'year', or re-run "
    f"compute_efgd_quantile.py with max_year = {select_max_year}.")

# must match compute_efgd_quantile.py's with_review / name_tag so its filename resolves
quantile_with_review = False
quantile_data_suffix = '_with_review' if quantile_with_review else ''
# quantile_name_tag is derived from year_offset up in the citation-window block
# The quantile is a rank WITHIN the peer set compute_efgd_quantile.py kept, so that script must
# have been run with the SAME rt / ct / max_year as this one -- otherwise the papers would be
# ranked against a different population than the one plotted. The path is built from THIS
# script's thresholds precisely so a mismatch fails loudly instead of silently.
quantile_table_path = os.path.join(
    META_DIRE + ('_with_review' if quantile_with_review else ''), 'efg_computation',
    f'efgd_within_year_domain_quantile_y{off}{quantile_data_suffix}_'
    f'rt_{reference_threshold}_ct_{citation_threshold}_'
    f'ybefore_{max_year}{quantile_name_tag}.csv')

print("reference_threshold:", reference_threshold)
print("citation_threshold:", citation_threshold)
print("max_year:", max_year, "| quantile_name_tag:", repr(quantile_name_tag),
      "| select_max_year:", select_max_year)
print("use_disp_f:", use_disp_f)
print("USE_QUANTILE:", USE_QUANTILE,
      ("| QUANTILE_GROUP: %s -> %r" % (QUANTILE_GROUP, qs)) if USE_QUANTILE else "(raw values)")

if (not USE_QUANTILE and not use_disp_f and year_offset == 5
        and reference_threshold == 1 and citation_threshold == 5
        and max_year == 2020 and select_max_year == max_year):
    csv_path = os.path.join(SAVE_DIRE, 'Extended Data Fig. 10a.csv')
else:
    csv_path = os.path.join(SAVE_DIRE,
        f'author_count_feg_agg{year_tag}_rt_{reference_threshold}_ct_{citation_threshold}'
        f'_my_{select_max_year}.csv')

# ── Load or compute ───────────────────────────────────────────────────────────
# Skipped wholesale when USE_QUANTILE: the quantile table carries every metric already, so none
# of the raw sources (gen_V3 / disp_full / gen_V3_disp_f) need to be read or joined.
if USE_QUANTILE:
    df_agg = None
    print("USE_QUANTILE -> skipping the raw E/F/G / disruption / disp_f / (i+j+k) aggregates")

elif os.path.exists(csv_path):
    print(f"Loading cached data from {csv_path}")
    df_agg = pd.read_csv(csv_path)

else:
    # ── 1. Load generalization data and compute F, E, G ──────────────────────
    print("Loading generalization data ...")
    df = pd.read_csv(
        os.path.join(META_DIRE, 'efg_computation', f'gen_V3_y{off}.csv'),
        usecols=['pid', 'pub_year', 'up_refer_count', f'citation_count_y{off}',
                 f'f_y{off}', f'e_y{off}', f'g_y{off}']
    )

    df = df[df['pub_year']          < select_max_year]
    df = df[df['up_refer_count']    >= reference_threshold]
    df = df[df[f'citation_count_y{off}'] >= citation_threshold]

    df['f'] = df[f'f_y{off}']
    df['e'] = df[f'e_y{off}']
    df['g'] = df[f'g_y{off}']

    valid_pids = set(df['pid'])
    print(f"  papers after threshold filter: {len(df)}")

    # ── 2. Load pid2author_list and compute author_count ─────────────────────
    print("Loading pid2author_list ...")
    with open(os.path.join(META_DIRE, 'pid2author_list.pkl'), 'rb') as f:
        pid2author_list = pickle.load(f)

    rows = []
    for pid, author_list in tqdm(pid2author_list.items(), desc='computing author_count'):
        if pid not in valid_pids:
            continue
        rows.append({'pid': pid, 'author_count': len(author_list)})

    df_authors = pd.DataFrame(rows)
    print(f"  matched pids: {len(df_authors)}")

    # ── 3. Merge and filter author_count 1–8 ─────────────────────────────────
    df = df.merge(df_authors, on='pid', how='inner')
    df = df[(df['author_count'] >= 1) & (df['author_count'] <= 8)]

    print(f"  papers with author_count 1-8: {len(df)}")

    # ── 4. Bootstrap 95% CI per author_count ─────────────────────────────────
    rng = np.random.default_rng(42)
    result_rows = []
    for ac, grp in tqdm(df.groupby('author_count'), desc='bootstrap'):
        # 'n_group' is the group size; '{col}_n' is what each metric was averaged over,
        # which is award_quantile_analysis.py's convention for n_desired / n_undesired
        row = {'author_count': ac, 'n_group': len(grp)}
        for col in ['f', 'e', 'g']:
            vals = grp[col].dropna().values
            row[f'{col}_n'] = len(vals)
            if len(vals) == 0:
                row[f'{col}_mean']    = np.nan
                row[f'{col}_ci_low']  = np.nan
                row[f'{col}_ci_high'] = np.nan
                continue
            row[f'{col}_mean'] = vals.mean()
            boot_means = np.array([
                rng.choice(vals, size=len(vals), replace=True).mean()
                for _ in range(n_bootstrap)
            ])
            row[f'{col}_ci_low']  = np.percentile(boot_means, 2.5)
            row[f'{col}_ci_high'] = np.percentile(boot_means, 97.5)
        result_rows.append(row)

    df_agg = pd.DataFrame(result_rows)
    df_agg.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


# ── 4b. Disruption vs author_count, cached in the SAME df_agg CSV as F/E/G ─────
# Disruption is read from arrange_disruption.py's output disp_full.csv (column disp_y5) and
# aggregated per author_count with the SAME thresholds / author-count window / bootstrap
# settings as F/E/G, then merged into df_agg and written back to the same cache CSV. If
# df_agg was loaded from an older cache that predates the disp columns, it is augmented and
# re-saved here so the disruption is cached too (no separate cache file).
# arrange_disruption.py writes to META_DIRE(+'_with_review')/disruption/disp_full.csv; the
# efg/author data above is the article-only run, so default to the plain (article-only)
# disruption. Flip use_with_review_disp to match a review-inclusive arrange_disruption.py run.
use_with_review_disp = False
disp_main_dire = (META_DIRE + '_with_review') if use_with_review_disp else META_DIRE
disp_full_path = os.path.join(disp_main_dire, 'disruption', f'disp_full{disp_year_tag}.csv')

if USE_QUANTILE:
    pass                      # disp comes from the quantile table's disp_y5 quantile instead
elif 'disp_mean' not in df_agg.columns:
    print(f"Computing disruption-by-author_count from {disp_full_path} ...")
    dfd = pd.read_csv(
        disp_full_path,
        usecols=['pid', 'pub_year', 'up_refer_count', f'citation_count_y{off}', f'disp_y{off}'])
    dfd = dfd[dfd['pub_year']          < select_max_year]
    dfd = dfd[dfd['up_refer_count']    >= reference_threshold]
    dfd = dfd[dfd[f'citation_count_y{off}'] >= citation_threshold]
    dfd['disp'] = dfd[f'disp_y{off}']
    valid_pids_d = set(dfd['pid'])
    print(f"  disruption papers after threshold filter: {len(dfd)}")

    with open(os.path.join(META_DIRE, 'pid2author_list.pkl'), 'rb') as f:
        pid2author_list_d = pickle.load(f)
    rows_d = [{'pid': pid, 'author_count': len(al)}
              for pid, al in tqdm(pid2author_list_d.items(), desc='disp author_count')
              if pid in valid_pids_d]
    dfd = dfd.merge(pd.DataFrame(rows_d), on='pid', how='inner')
    dfd = dfd[(dfd['author_count'] >= 1) & (dfd['author_count'] <= 8)]
    print(f"  disruption papers with author_count 1-8: {len(dfd)}")

    rng_d = np.random.default_rng(42)
    disp_rows = []
    for ac, grp in tqdm(dfd.groupby('author_count'), desc='bootstrap disp'):
        vals = grp['disp'].dropna().values
        row = {'author_count': ac, 'disp_n_group': len(grp), 'disp_n': len(vals)}
        if len(vals) == 0:
            row['disp_mean'] = row['disp_ci_low'] = row['disp_ci_high'] = np.nan
        else:
            row['disp_mean'] = vals.mean()
            boot_means = np.array([
                rng_d.choice(vals, size=len(vals), replace=True).mean()
                for _ in range(n_bootstrap)
            ])
            row['disp_ci_low']  = np.percentile(boot_means, 2.5)
            row['disp_ci_high'] = np.percentile(boot_means, 97.5)
        disp_rows.append(row)

    df_agg = df_agg.merge(pd.DataFrame(disp_rows), on='author_count', how='left')
    df_agg.to_csv(csv_path, index=False)
    print(f"Updated cache with disruption columns: {csv_path}")
else:
    print("disruption columns already present in cache -> reusing")


# ── 4c. disp_f (disruptive foundation) vs author_count, cached in the SAME df_agg CSV ─────
# Only when use_disp_f. All THREE disp_f variants (disruption_f_y5 / disruption_f_down_gt1_y5 /
# disruption_f_down_gt2_y5) are read from get_efg_v2_disp_f.py's merged gen_V3_disp_f_y5.csv and
# aggregated per author_count with the SAME thresholds / author-count window / bootstrap settings
# as F/E/G, then merged into df_agg and written back to the same cache CSV (mirrors the disruption
# block above). Each variant is drawn as its own panel in the second figure row.
dispf_full_path = os.path.join(META_DIRE, 'efg_computation', f'gen_V3_disp_f_y{off}.csv')
dispf_src_cols = [src for _, src, _ in dispf_variants]
if USE_QUANTILE:
    pass                      # the disp_f shares' quantiles come from the quantile table
elif use_disp_f and f'{dispf_variants[0][0]}_mean' not in df_agg.columns:
    print(f"Computing disp_f-by-author_count from {dispf_full_path} ...")
    dff = pd.read_csv(
        dispf_full_path,
        usecols=['pid', 'pub_year', 'up_refer_count', f'citation_count_y{off}'] + dispf_src_cols)
    dff = dff[dff['pub_year']          < select_max_year]
    dff = dff[dff['up_refer_count']    >= reference_threshold]
    dff = dff[dff[f'citation_count_y{off}'] >= citation_threshold]
    valid_pids_f = set(dff['pid'])
    print(f"  disp_f papers after threshold filter: {len(dff)}")

    with open(os.path.join(META_DIRE, 'pid2author_list.pkl'), 'rb') as f:
        pid2author_list_f = pickle.load(f)
    rows_f = [{'pid': pid, 'author_count': len(al)}
              for pid, al in tqdm(pid2author_list_f.items(), desc='disp_f author_count')
              if pid in valid_pids_f]
    dff = dff.merge(pd.DataFrame(rows_f), on='pid', how='inner')
    dff = dff[(dff['author_count'] >= 1) & (dff['author_count'] <= 8)]
    print(f"  disp_f papers with author_count 1-8: {len(dff)}")

    rng_f = np.random.default_rng(42)
    dispf_rows = []
    for ac, grp in tqdm(dff.groupby('author_count'), desc='bootstrap disp_f'):
        row = {'author_count': ac, 'dispf_n_group': len(grp)}
        for key, src, _ in dispf_variants:
            vals = grp[src].dropna().values
            row[f'{key}_n'] = len(vals)
            if len(vals) == 0:
                row[f'{key}_mean'] = row[f'{key}_ci_low'] = row[f'{key}_ci_high'] = np.nan
                continue
            row[f'{key}_mean'] = vals.mean()
            boot_means = np.array([
                rng_f.choice(vals, size=len(vals), replace=True).mean()
                for _ in range(n_bootstrap)
            ])
            row[f'{key}_ci_low']  = np.percentile(boot_means, 2.5)
            row[f'{key}_ci_high'] = np.percentile(boot_means, 97.5)
        dispf_rows.append(row)

    df_agg = df_agg.merge(pd.DataFrame(dispf_rows), on='author_count', how='left')
    df_agg.to_csv(csv_path, index=False)
    print(f"Updated cache with disp_f columns: {csv_path}")
elif use_disp_f:
    print("disp_f columns already present in cache -> reusing")


# ── 4d. The four (i+j+k)-denominator ratios vs author_count, cached in the SAME df_agg CSV ────
# A SECOND, separate 1x4 figure (see step 6): each panel is one ratio vs author_count, with the
# same thresholds / author-count window / bootstrap settings as F/E/G. Definitions are taken
# verbatim from compute_efgd_quantile.py so the two scripts agree:
#   i          = I_count_y5, or the legacy split I_iso_count_y5 + I_start_count_y5 (auto-detected)
#   i+j+k      = i + J_count_y5 + K_count_y5   (k = citations of the focal paper's references
#                that do NOT cite the focal paper itself)
#   f/(i+j+k)      = f_count_y5 / (i+j+k)              <- the raw F COUNT, not the share f_y5
#   disp_f/(i+j+k) = disruption_f_count_y5 / (i+j+k)   <- the raw disp_f_0 COUNT, not the share
#   i/(i+j+k)      = i / (i+j+k)                       <- numerator i ALONE, not i-j
#   (i-j)/(i+j+k)  = disp_y5, read straight from disp_full.csv (it IS the disruption index)
# The f/disp_f COUNT columns live in gen_V3_y5.csv / gen_V3_disp_f_y5.csv; the i/j/k counts and
# disp_y5 in disp_full.csv. disp_full is the filtering base (it carries pub_year / up_refer_count
# / citation_count_y5), f_count is inner-joined, and the disp_f count is LEFT-joined so papers
# without a disp_f value only drop out of the disp_f panel (each panel dropna()s its own column).
plot_ijk_ratio = True
# (short key used in df_agg columns, panel label). C = I+J (the paper's own citations), so C+K
# is the i+j+k denominator -- same notation as the award_quantile_analysis.py 1x4 figure.
ijk_variants = [
    ('f_ijk',     'F / (C+K)'),
    ('dispf_ijk', 'Disp(F) / (C+K)'),
    ('i_ijk',     'I / (C+K)'),
    ('imj_ijk',   '(I-J) / (C+K)'),
]

if USE_QUANTILE:
    pass                      # the (i+j+k) ratios' quantiles come from the quantile table
elif plot_ijk_ratio and f'{ijk_variants[0][0]}_mean' not in df_agg.columns:
    print("Computing the (i+j+k)-ratio-by-author_count aggregates ...")

    # ---- i/j/k counts + disp_y5 from disp_full.csv (the filtering base) ----
    # auto-detect the I-column layout, exactly as compute_efgd_quantile.py does
    disp_header = pd.read_csv(disp_full_path, nrows=0).columns
    if f'I_count_y{off}' in disp_header:
        i_source_cols = [f'I_count_y{off}']
    elif f'I_iso_count_y{off}' in disp_header and f'I_start_count_y{off}' in disp_header:
        i_source_cols = [f'I_iso_count_y{off}', f'I_start_count_y{off}']   # legacy split layout
    else:
        raise AssertionError(
            f"disp_full.csv has neither f'I_count_y{off}' nor f'I_iso_count_y{off}'+f'I_start_count_y{off}': "
            f"{list(disp_header)}")
    print(f"  disp_full I-column layout: {i_source_cols}")
    assert f'K_count_y{off}' in disp_header, (
        f"disp_full.csv is missing f'K_count_y{off}' (needed for the i+j+k denominator): "
        f"{list(disp_header)}")

    dfi = pd.read_csv(
        disp_full_path,
        usecols=(['pid', 'pub_year', 'up_refer_count', f'citation_count_y{off}',
                  f'disp_y{off}', f'J_count_y{off}', f'K_count_y{off}'] + i_source_cols))
    dfi = dfi[dfi['pub_year']          < select_max_year]
    dfi = dfi[dfi['up_refer_count']    >= reference_threshold]
    dfi = dfi[dfi[f'citation_count_y{off}'] >= citation_threshold]
    print(f"  disp_full papers after threshold filter: {len(dfi)}")

    # ---- raw F count (inner join: needed by the f/(i+j+k) panel) ----
    df_fcount = pd.read_csv(os.path.join(META_DIRE, 'efg_computation', f'gen_V3_y{off}.csv'),
                            usecols=['pid', f'f_count_y{off}'])
    dfi = dfi.merge(df_fcount, on='pid', how='inner')
    print(f"  after join with f_count_y5: {len(dfi)}")

    # ---- raw disp_f_0 count (LEFT join: missing -> NaN -> that panel only drops the paper) ----
    df_dfcount = pd.read_csv(dispf_full_path, usecols=['pid', f'disruption_f_count_y{off}'])
    dfi = dfi.merge(df_dfcount, on='pid', how='left')
    print(f"  papers with a disruption_f_count_y5: {int(dfi[f'disruption_f_count_y{off}'].notna().sum())}")

    # ---- author_count, same 1-8 window as F/E/G ----
    with open(os.path.join(META_DIRE, 'pid2author_list.pkl'), 'rb') as f:
        pid2author_list_i = pickle.load(f)
    valid_pids_i = set(dfi['pid'])
    rows_i = [{'pid': pid, 'author_count': len(al)}
              for pid, al in tqdm(pid2author_list_i.items(), desc='ijk author_count')
              if pid in valid_pids_i]
    dfi = dfi.merge(pd.DataFrame(rows_i), on='pid', how='inner')
    dfi = dfi[(dfi['author_count'] >= 1) & (dfi['author_count'] <= 8)]
    print(f"  (i+j+k)-ratio papers with author_count 1-8: {len(dfi)}")

    # ---- the four ratios. denom > 0 always holds under citation_threshold >= 5 (i+j is the
    #      citation count), but guard anyway: a 0 denominator yields NaN, not 0, so a
    #      degenerate paper drops out of the mean instead of dragging it toward 0. ----
    I_total  = dfi[i_source_cols].sum(axis=1)
    denom_ijk = I_total + dfi[f'J_count_y{off}'] + dfi[f'K_count_y{off}']
    dfi['f_ijk']     = (dfi[f'f_count_y{off}'] / denom_ijk).where(denom_ijk > 0)
    dfi['dispf_ijk'] = (dfi[f'disruption_f_count_y{off}'] / denom_ijk).where(denom_ijk > 0)
    dfi['i_ijk']     = (I_total / denom_ijk).where(denom_ijk > 0)
    dfi['imj_ijk']   = dfi[f'disp_y{off}']        # (i-j)/(i+j+k): the disruption index, as-is
    for key, _ in ijk_variants:
        print(f"  {key} stats:", dfi[key].describe().to_dict())

    rng_i = np.random.default_rng(42)
    ijk_rows = []
    for ac, grp in tqdm(dfi.groupby('author_count'), desc='bootstrap ijk ratios'):
        row = {'author_count': ac, 'ijk_n_group': len(grp)}
        for key, _ in ijk_variants:
            vals = grp[key].dropna().values
            row[f'{key}_n'] = len(vals)
            if len(vals) == 0:
                row[f'{key}_mean'] = row[f'{key}_ci_low'] = row[f'{key}_ci_high'] = np.nan
                continue
            row[f'{key}_mean'] = vals.mean()
            boot_means = np.array([
                rng_i.choice(vals, size=len(vals), replace=True).mean()
                for _ in range(n_bootstrap)
            ])
            row[f'{key}_ci_low']  = np.percentile(boot_means, 2.5)
            row[f'{key}_ci_high'] = np.percentile(boot_means, 97.5)
        ijk_rows.append(row)

    df_agg = df_agg.merge(pd.DataFrame(ijk_rows), on='author_count', how='left')
    df_agg.to_csv(csv_path, index=False)
    print(f"Updated cache with (i+j+k)-ratio columns: {csv_path}")
elif plot_ijk_ratio:
    print("(i+j+k)-ratio columns already present in cache -> reusing")


# ── 5. Plot: 3 subplots, x=author_count ──────────────────────────────────────
metrics = [
    ('f', 'Foundation'),
    ('e', 'Extension'),
    ('g', 'Generalization'),
]

mid = len(metrics) // 2   # middle column -> carries the 'Author Count' x-label


def plot_panel(ax, x, mean, low, high, name, color, name_fs):
    """One F/E/G-or-disp_f panel: mean line + bootstrap CI band, cleaned spines, and the
    metric name in the bottom-left corner (in place of a per-panel y-label). `x` is passed in
    (not read from a global) so the raw and quantile aggregates can share this."""
    ax.fill_between(x, low, high, alpha=0.2, color=color)
    ax.plot(x, mean, linewidth=1.5, marker='o', markersize=5, color=color)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.tick_params(axis='x', labelsize=tick_size)
    ax.tick_params(axis='y', labelsize=tick_size)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.03, 0.03, name, transform=ax.transAxes,
            fontsize=name_fs, ha='left', va='bottom')


def draw_feg_figure(df, svg_path, ylabel_row1='Average index', ylabel_row2='Average share'):
    """The F/E/G figure (+ a Disruption inset on the G panel, + a disp_f second row when
    use_disp_f), drawn from `df`'s {key}_mean / _ci_low / _ci_high columns. Saved as SVG."""
    x = df['author_count']
    n_rows = 2 if use_disp_f else 1
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4 * n_rows), squeeze=False)

    # ── Row 1: Foundation / Extension / Generalization ─────────────────────────
    for i, (col, name) in enumerate(metrics):
        ax = axes[0][i]
        plot_panel(ax, x, df[f'{col}_mean'], df[f'{col}_ci_low'], df[f'{col}_ci_high'],
                   name, 'steelblue', label_size + 7)
        if i == 0:
            ax.set_ylabel(ylabel_row1, fontsize=label_size)
        # x-label on the middle panel; if a disp_f row follows, it carries the x-label instead
        if i == mid and not use_disp_f:
            ax.set_xlabel('Author Count', fontsize=label_size)

        # ── inset on the Generalization panel: Disruption vs author_count ──────
        if col == 'g':
            axins = ax.inset_axes([0.55, 0.55, 0.42, 0.40])  # upper-right within the G panel
            axins.fill_between(x, df['disp_ci_low'], df['disp_ci_high'],
                               alpha=0.2, color='indianred')
            axins.plot(x, df['disp_mean'],
                       linewidth=1.2, marker='o', markersize=3, color='indianred')
            axins.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            axins.tick_params(axis='both', labelsize=tick_size - 4)
            axins.spines['top'].set_visible(False)
            axins.spines['right'].set_visible(False)
            # inset label (Disruption) in its bottom-left corner; no x/y-axis labels
            axins.text(0.04, 0.04, 'Disruption', transform=axins.transAxes,
                       fontsize=label_size - 5, ha='left', va='bottom')

    # ── Row 2: disp_f 0 / 1 / 2 (disruptive foundation) vs author_count ────────
    if use_disp_f:
        for i, (key, src, name) in enumerate(dispf_variants):
            ax = axes[1][i]
            plot_panel(ax, x, df[f'{key}_mean'], df[f'{key}_ci_low'], df[f'{key}_ci_high'],
                       name, 'indianred', label_size)
            if i == 0:
                ax.set_ylabel(ylabel_row2, fontsize=label_size)
            if i == mid:
                ax.set_xlabel('Author Count', fontsize=label_size)

    fig.tight_layout()
    fig.savefig(svg_path, format='svg', bbox_inches='tight', transparent=True)
    plt.close(fig)
    print(f"Saved: {svg_path}")


def draw_ijk_figure(df, svg_path, ylabel='Average index'):
    """The 1x4 (i+j+k)-denominator-ratio figure. Same panel style as draw_feg_figure, but with
    ONE shared 'Author Count' x-label under the row rather than a middle-panel one, since a
    4-panel row has no middle. Saved as SVG + PNG."""
    x = df['author_count']
    fig_i, axes_i = plt.subplots(1, 4, figsize=(21, 4))
    for i, (key, name) in enumerate(ijk_variants):
        plot_panel(axes_i[i], x, df[f'{key}_mean'], df[f'{key}_ci_low'], df[f'{key}_ci_high'],
                   name, 'indianred', label_size)
        if i == 0:
            axes_i[i].set_ylabel(ylabel, fontsize=label_size)
    fig_i.tight_layout(rect=[0, 0.04, 1, 1])
    fig_i.supxlabel('Author Count', fontsize=label_size, y=0.02)
    fig_i.savefig(svg_path, format='svg', bbox_inches='tight', transparent=True)
    fig_i.savefig(svg_path[:-4] + '.png', dpi=150, bbox_inches='tight')
    plt.close(fig_i)
    print(f"Saved: {svg_path}")


# ── 5b. QUANTILE aggregates (only when USE_QUANTILE) ──────────────────────────
# Averages each paper's quantile per author_count, exactly as the raw blocks average the raw
# index. Everything comes from the single compute_efgd_quantile.py table -- no joins, no I/J/K
# arithmetic. Cached in its own CSV, keyed by the grouping + thresholds.
if USE_QUANTILE:
    print("quantile table:", quantile_table_path)

    # (df_agg column key, base column in the quantile table, panel label). The keys mirror the
    # raw path's, so draw_feg_figure / draw_ijk_figure work unchanged. 'disp' (the G-panel
    # inset) and 'imj_ijk' (the 1x4's 4th panel) are the SAME column: (i-j)/(i+j+k) IS disp_y5.
    quantile_metrics = (
        [(col, f'{col}_y{off}', name) for col, name in metrics]        # f / e / g
        + [('disp', f'disp_y{off}', 'Disruption')]                         # G-panel inset
        + (list(dispf_variants) if use_disp_f else [])                # disp_f row
        + [('f_ijk', f'f_over_ijk_y{off}', 'F / (C+K)'),
           ('dispf_ijk', f'dispf_over_ijk_y{off}', 'Disp(F) / (C+K)'),
           ('i_ijk', f'i_over_ijk_y{off}', 'I / (C+K)'),
           ('imj_ijk', f'disp_y{off}', '(I-J) / (C+K)')]                   # the 1x4
    )

    q_csv_path = os.path.join(
        SAVE_DIRE,
        f'author_count_feg_agg_quantile_{QUANTILE_GROUP}{year_tag}_rt_{reference_threshold}_'
        f'ct_{citation_threshold}_my_{select_max_year}{quantile_data_suffix}{dispf_tag}.csv')

    if os.path.exists(q_csv_path):
        print(f"Loading cached quantile aggregates from {q_csv_path}")
        df_agg = pd.read_csv(q_csv_path)
    else:
        assert os.path.exists(quantile_table_path), (
            f"{quantile_table_path}\nRun openalex/compute_efgd_quantile.py with "
            f"reference_threshold={reference_threshold}, citation_threshold={citation_threshold} "
            f"and max_year={max_year} (and use_disp_f=True) so its output matches this script's "
            f"thresholds -- the quantile must rank papers among the same peers plotted here.")

        # only the quantile columns are read (deduped: disp/imj_ijk share one). The table is
        # ALREADY filtered by rt / ct / max_year upstream, so THOSE are not redone here -- but
        # select_max_year is this script's OWN extra cut and has to be applied, which is why
        # pub_year is read as well.
        q_cols = sorted({f'{src}{qs}' for _, src, _ in quantile_metrics})
        header = pd.read_csv(quantile_table_path, nrows=0).columns
        missing = [c for c in q_cols if c not in header]
        assert not missing, (
            f"{quantile_table_path} is missing {missing}.\nRe-run compute_efgd_quantile.py with "
            f"use_disp_f=True (the disp_f / dispf_over_ijk quantiles need it).")
        assert 'pub_year' in header, (
            f"{quantile_table_path} has no pub_year column, so select_max_year cannot be "
            f"applied.")
        dfq = pd.read_csv(quantile_table_path, usecols=['pid', 'pub_year'] + q_cols)
        print(f"  quantile table rows (already rt/ct/max_year filtered): {len(dfq)}")

        # the extra pub_year cut, in the table's own strict-less-than convention. The quantile
        # columns are NOT recomputed -- see select_max_year in the settings for why that is
        # sound only for a within-year ranking (asserted there).
        if select_max_year != max_year:
            _n_before = len(dfq)
            dfq = dfq[dfq['pub_year'] < select_max_year]
            print(f"  select_max_year={select_max_year}: {_n_before} -> {len(dfq)} papers "
                  f"(dropped {_n_before - len(dfq)} published in "
                  f"{select_max_year}-{max_year - 1}); quantiles NOT recomputed")
            assert len(dfq), (
                f"no paper is left with pub_year < {select_max_year}; check select_max_year.")

        with open(os.path.join(META_DIRE, 'pid2author_list.pkl'), 'rb') as f:
            pid2author_list_q = pickle.load(f)
        valid_pids_q = set(dfq['pid'])
        rows_q = [{'pid': pid, 'author_count': len(al)}
                  for pid, al in tqdm(pid2author_list_q.items(), desc='quantile author_count')
                  if pid in valid_pids_q]
        dfq = dfq.merge(pd.DataFrame(rows_q), on='pid', how='inner')
        dfq = dfq[(dfq['author_count'] >= 1) & (dfq['author_count'] <= 8)]
        print(f"  quantile papers with author_count 1-8: {len(dfq)}")

        rng_q = np.random.default_rng(42)
        q_rows = []
        for ac, grp in tqdm(dfq.groupby('author_count'), desc='bootstrap quantile'):
            # 'n_group' is the group size; '{key}_n' is what each metric was averaged
            # over -- award_quantile_analysis.py's convention
            row = {'author_count': ac, 'n_group': len(grp)}
            for key, src, _ in quantile_metrics:
                vals = grp[f'{src}{qs}'].dropna().values
                row[f'{key}_n'] = len(vals)   # < n_group when the quantile is NaN
                if len(vals) == 0:
                    row[f'{key}_mean'] = row[f'{key}_ci_low'] = row[f'{key}_ci_high'] = np.nan
                    continue
                row[f'{key}_mean'] = vals.mean()
                boot_means = np.array([
                    rng_q.choice(vals, size=len(vals), replace=True).mean()
                    for _ in range(n_bootstrap)
                ])
                row[f'{key}_ci_low']  = np.percentile(boot_means, 2.5)
                row[f'{key}_ci_high'] = np.percentile(boot_means, 97.5)
            q_rows.append(row)

        df_agg = pd.DataFrame(q_rows)
        df_agg.to_csv(q_csv_path, index=False)
        print(f"Saved quantile aggregates: {q_csv_path}")


# ── 5c. how many papers are behind the aggregate ────────────────────────
# Read off df_agg -- freshly computed or loaded from the cache -- so the counts appear either
# way. The PER-METRIC counts ({key}_n) are the ones to quote: they are the values that survived
# that metric's dropna, which is exactly how award_quantile_analysis.py counts its n_desired /
# n_undesired. A metric's count falls below the group size whenever its value is NaN -- with
# QUANTILE_GROUP='year_domain' the usual cause is a paper with no single top_concept, whose
# (pub_year, top_concept) quantile cannot be defined.
print("\n" + "=" * 78)
print("PAPERS BEHIND THE AGGREGATE  (author_count 1-8, rt=%s, ct=%s, pub_year < %s%s)"
      % (reference_threshold, citation_threshold, select_max_year,
         ", quantile" if USE_QUANTILE else ", raw"))

_metric_n_cols = [c for c in df_agg.columns if c.endswith('_n')]
_group_n_cols = [c for c in df_agg.columns if c.endswith('_n_group')]


def _report(col, label):
    """One line: the total, then the per-author_count breakdown."""
    vals = df_agg[col].fillna(0)
    print(f"  {label:<14} total {int(vals.sum()):>14,}   "
          + "  ".join(f"ac{int(ac)}={int(v):,}"
                      for ac, v in zip(df_agg['author_count'], vals)))


if _metric_n_cols:
    print("  -- per metric, AFTER dropna (same convention as award_quantile_analysis.py) --")
    for _c in _metric_n_cols:
        _report(_c, _c[:-2])
else:
    print("  (this cache predates the per-metric counts -- delete it to rebuild with them)")

if _group_n_cols:
    print("  -- the frames they were drawn from, BEFORE any dropna (not comparable) --")
    for _c in _group_n_cols:
        _report(_c, _c)
print("=" * 78)


# ── 6. draw both figures from whichever aggregate USE_QUANTILE selected ───────
# Same panels either way; only the plotted value, the y-labels and the filenames differ.
if USE_QUANTILE:
    value_tag = f'_quantile_{QUANTILE_GROUP}'
    ylabel_row1 = ylabel_row2 = ijk_ylabel = 'Index quantile'
else:
    value_tag = ''
    ylabel_row1, ylabel_row2, ijk_ylabel = 'Average index', 'Average share', 'Average index'

# the raw figure at the paper's default thresholds IS the Extended Data deliverable; every other
# combination (and every quantile run) carries its params in the filename instead.
if (not USE_QUANTILE and not use_disp_f
        and reference_threshold == 1 and citation_threshold == 5
        and year_offset == 5 and max_year == 2020 and select_max_year == max_year):
    svg_path = os.path.join(SAVE_DIRE, 'Extended Data Fig. 10a.svg')
else:
    svg_path = os.path.join(
        SAVE_DIRE,
        f'author_count_feg_agg{value_tag}{year_tag}_rt_{reference_threshold}_'
        f'ct_{citation_threshold}_my_{select_max_year}{dispf_tag}.svg')
draw_feg_figure(df_agg, svg_path, ylabel_row1=ylabel_row1, ylabel_row2=ylabel_row2)

# second figure: 1x4, the (i+j+k)-denominator ratios vs author_count. Its own file, so the
# figure above is untouched; the filename always carries the thresholds.
if plot_ijk_ratio:
    draw_ijk_figure(
        df_agg,
        os.path.join(SAVE_DIRE,
                     f'author_count_ijk_ratio_agg{value_tag}{year_tag}_rt_{reference_threshold}_'
                     f'ct_{citation_threshold}_my_{select_max_year}.svg'),
        ylabel=ijk_ylabel)

print("Done.")
