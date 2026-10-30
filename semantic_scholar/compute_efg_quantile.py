"""
Semantic Scholar (S2AG) version of the OpenAlex compute_efgd_quantile.py.

Per-paper E/F/G (+ disp_f) quantiles, computed three ways:
    (a) WITHIN its (publication year, top domain) peer group -> {col}_quantile
    (b) WITHIN its publication year only (any/no domain)      -> {col}_quantile_by_year
    (c) GLOBALLY, no grouping (raw value across all papers)   -> {col}_quantile_global
Only (a) is domain-based: it is restricted to papers with exactly ONE top domain
(others get NaN). (b) and (c) rank over ALL papers passing the reference / citation /
max_year filters, regardless of domain.

This S2AG pipeline reads the disruption index D (disp_y{off}) AND the disruption COUNTS (I/J/K)
from arrange_disruption.py's disp_full; the counts derive the disruptive-fraction ratios below,
while D is quantiled as-is (not recomputed). It quantiles:
    E = e_y{off}, F = f_y{off}, G = g_y{off}                 (from get_efg.py output)
    disp_f 0/1/2 = disruption_f_y{off} / disruption_f_down_gt1_y{off} / disruption_f_down_gt2_y{off}
                   (three "disruptive foundation" shares from get_efg_v2_disp_f.py; carried
                    in the same merged gen_V3_disp_f table, so no extra join is needed)
    R = I/(I+J) = ij_ratio_y{off}  -- the disruptive-citation fraction, now taken STRAIGHT from
                   arrange_disruption.py's disp_full counts (I = I_count_y{off}, J = J_count_y{off}),
                   replacing the old g_y{off} + disruption_f_y{off} proxy.
    i/(i+j+k)   = I / (I + J + K)      (disruption numerator i ALONE, not i-j, over i+j+k;
                   K = K_count_y{off} from disp_full)
    f/(i+j+k)   = f_count_y{off} / (I + J + K)   (the raw F COUNT over i+j+k, NOT the share f_y{off})
    disp_f0/(i+j+k) = disruption_f_count_y{off} / (I + J + K)  (the raw disp_f_0 COUNT over i+j+k,
                   NOT the share disruption_f_y{off}; only when use_disp_f; NaN for papers
                   lacking a disp_f value)
    D = (i-j)/(i+j+k) = disp_y{off}  -- the disruption index, read STRAIGHT from disp_full
                   (get_disruption.py already computes it). Unlike every other quantiled column
                   it ranges over [-1, 1], not [0, 1]; its quantile is in (0, 1] all the same.
The disp_full columns are LEFT-joined per pid, so the E/F/G sample and its quantiles are
unchanged; papers without a disruption record simply get NaN for D and the four
disruption-derived ratios above.
plus its citation count (citation_count_y{off}).

"Top domain" comes from one of two concept maps (selected by concept_source below):
  * 'single' (default): cid2single_concept.pkl from get_s2ag_meta_author_domain_s2_simple_domain.py
    -- exactly ONE category per cid (a multi-category paper already had ONE picked at RANDOM
    upstream) or 'Others'. No multi-domain case, so 'Others'/absent papers keep top_concept=NaN.
  * 'list': cid2category_list.pkl from get_s2ag_meta_author_domain.py (cid -> list of categories)
    -- only papers with EXACTLY ONE category get a top domain; no-/multi-domain papers -> NaN.
Papers with top_concept=NaN receive no (year, domain) quantile but still take part in the
by-year and global quantiles.

The quantiles are computed ONLY over the set that passes the reference / citation /
max_year filters, so a paper's quantile reflects its rank among the filtered peers of its
(year, domain) cell.

Note: in the S2AG pipeline a "pid" is a corpusid (cid); the E/F/G scripts keep it in the
'pid' column, so this script uses 'pid' throughout to match get_efg_s2_merge.py output.

Input  : META_DIRE/efg_computation/gen_V3[_disp_f]_y{year_offset}{network_suffix}.csv
             (get_efg_s2_merge.py; E/F/G [+ disp_f shares when use_disp_f]; a superset table)
         META_DIRE/disruption/disp_full[_y{year_offset}]{network_suffix}{window_suffix}.csv
             (the _y tag appears only when year_offset != 5)
             (arrange_disruption.py; I/J/K counts for the disruption-derived ratios; LEFT-joined)
         META_DIRE/cid2single_concept.pkl (get_s2ag_meta_author_domain_s2_simple_domain.py; top
             domain, concept_source='single') OR cid2category_list.pkl
             (get_s2ag_meta_author_domain.py; concept_source='list')
Output : META_DIRE/efg_computation/  (TWO files, one per pub_year cutoff)
           efg_within_year_domain_quantile_y{year_offset}{network_suffix}_rt_{rt}_ct_{ct}_ybefore_{2025 - year_offset}.csv
             -> papers whose full year_offset citation window has elapsed (complete citation_y{off})
           efg_within_year_domain_quantile_y{year_offset}{network_suffix}_rt_{rt}_ct_{ct}_ybefore_2025_full_cover.csv
             -> ALL papers published before 2025; the most recent years lack a complete
                year_offset citation window, but quantiles are computed WITHIN a publication
                year, so each paper is ranked only against same-year peers sharing the same
                (in)complete window.
"""

import os
import sys
import pickle

import pandas as pd

# utils.py is in this same directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from utils import META_DIRE  # noqa: E402

meta_dire = META_DIRE
print("meta_dire", meta_dire)

year_offset = -1    # must match get_efg.py / get_efg_v2_disp_f.py citation window

# When True, additionally quantile the three "disruptive foundation" (disp_f) shares
# (disruption_f_y{off} / disruption_f_down_gt1_y{off} / disruption_f_down_gt2_y{off}) which
# get_efg_v2_disp_f.py wrote into the gen_V3_disp_f table. When False, only E/F/G/citation are
# quantiled and the plain gen_V3 table is read.
use_disp_f = True

# Must match get_efg.py / get_efg_v2_disp_f.py / get_efg_s2_merge.py: when True, read the
# E/F/G(+disp_f) table computed on the OpenAlex-augmented remove_circle network
# (augment_s2_citation_with_openalex.py). Encoded in the filename via network_suffix so the
# augmented and plain variants never collide.
use_openalex_augmented = False
network_suffix = '_openalex_augmented' if use_openalex_augmented else ''

# Must match arrange_disruption.py / get_disruption.py: selects which disp_full variant this
# script reads for the disruption counts (I/J/K). True -> the '_complete_window' run (focal
# papers whose year_offset citation window is not fully observed were dropped upstream), so in
# the '_full_cover' output the recent, incomplete-window papers have no disruption record and
# get NaN disruption-derived ratios. False -> no window filter (every paper may have counts).
skip_incomplete_window = True
# The flag only does anything when there IS a window to be incomplete: at
# year_offset == -1 the citation window is unbounded, get_disruption.py's skip test
# short-circuits, and the run is identical either way -- so the tag is dropped there,
# and one -1 run is not written under two names that hold the same data.
window_suffix = ('_complete_window'
                  if (skip_incomplete_window and year_offset != -1) else '')

print("year_offset", year_offset)
print("use_disp_f", use_disp_f)
print("use_openalex_augmented", use_openalex_augmented, "| network_suffix", repr(network_suffix))
print("skip_incomplete_window", skip_incomplete_window, "| window_suffix", repr(window_suffix))

efg_dire = os.path.join(meta_dire, 'efg_computation')     # E/F/G (+ disp_f) input & output

# filters (same convention as the OpenAlex compute_efgd_quantile.py)
reference_threshold = 1
citation_threshold = 5
# Two pub_year cutoffs, each saved to its own file (distinguished by the ybefore_{max_year}
# suffix). 2025 - year_offset : complete citation window; 2025 (_full_cover) : all papers
# before 2025 (recent years have an incomplete window, but quantiles are computed WITHIN a
# publication year, so every paper is ranked only against same-year peers).
# max_years = [(2025 - year_offset, ''), (2025, '_full_cover')]
max_years = [(2025 - year_offset, '')]

print("reference_threshold", reference_threshold)
print("citation_threshold", citation_threshold)
print("max_years", max_years)

citation_col = f'citation_count_y{year_offset}'
efg_cols = [f'e_y{year_offset}', f'f_y{year_offset}', f'g_y{year_offset}']

# disp_f variants (get_efg_v2_disp_f.py): three "disruptive foundation" shares (a citing paper
# shares NONE of the focal paper's references and co-cites more than T of its other citing
# papers: T = 0 / 1 / 2). Quantiled alongside E/F/G only when use_disp_f.
dispf_cols = ([f'disruption_f_y{year_offset}',
               f'disruption_f_down_gt1_y{year_offset}',
               f'disruption_f_down_gt2_y{year_offset}']
              if use_disp_f else [])

index_cols = efg_cols + dispf_cols        # E/F/G (+ disp_f) indices READ from the input table

# Disruption counts read from arrange_disruption.py's disp_full (I/J/K citations):
#   J = J_count_y{off} (developmental), I = I_count_y{off} (disruptive),
#   K = K_count_y{off} (cite the focal paper's references but NOT the focal paper).
# All four ratios below are DERIVED from these counts (left-joined per pid; papers without a
# disruption record get NaN ratios, so the E/F/G sample and quantiles are unchanged):
#   R = I/(I+J) = ij_ratio_y{off}    (disruptive-citation fraction; now taken straight from the
#                 disp_full counts, NOT the old g + disp_f_0 proxy)
#   i/(i+j+k)   = I/(I+J+K)          (disruption numerator i ALONE, not i-j, over i+j+k)
#   f/(i+j+k)   = f_count_y{off}/(I+J+K)   (the raw F COUNT over i+j+k, NOT the share f_y{off})
#   disp_f0/(i+j+k) = disruption_f_count_y{off}/(I+J+K)  (the raw disp_f_0 COUNT over i+j+k, NOT
#                 the share disruption_f_y{off}; only when use_disp_f; NaN without a disp_f value)
#   D = (I-J)/(I+J+K) = disp_y{off}   -- the disruption index itself, READ STRAIGHT from disp_full
#                 (get_disruption.py computes it as (size_disp - size_dev)/(citation_count + size_K);
#                  it is not re-derived here). Unlike the ratios above it lives in [-1, 1].
j_count_col = f'J_count_y{year_offset}'
k_count_col = f'K_count_y{year_offset}'
i_count_col = f'I_count_y{year_offset}'
disp_col = f'disp_y{year_offset}'        # the disruption index D, read from disp_full
# NUMERATOR count columns (from the gen_V3_disp_f table, get_efg[_v2_disp_f].py) -- raw COUNTS,
# NOT the citation-count-normalised shares f_y{off} / disruption_f_y{off}:
f_count_col = f'f_count_y{year_offset}'                 # raw F count for f/(i+j+k)
dispf0_count_col = f'disruption_f_count_y{year_offset}'  # raw disp_f_0 count for disp_f0/(i+j+k)
ratio_col = f'ij_ratio_y{year_offset}'
i_over_ijk_col = f'i_over_ijk_y{year_offset}'
f_over_ijk_col = f'f_over_ijk_y{year_offset}'
dispf_over_ijk_col = f'dispf_over_ijk_y{year_offset}'

# ==================================================
# 1. load the merged E/F/G (+ disp_f) table (get_efg_s2_merge.py output). The disp_f table is a
#    superset that already carries E/F/G, so a single read covers everything (no extra join).
# ==================================================
file_prefix = 'gen_V3_disp_f' if use_disp_f else 'gen_V3'
efg_path = os.path.join(efg_dire, f'{file_prefix}_y{year_offset}{network_suffix}.csv')
assert os.path.exists(efg_path), (
    "%s\nRun get_efg.py / get_efg_v2_disp_f.py then get_efg_s2_merge.py first." % efg_path)
df_all = pd.read_csv(efg_path)
print("loaded", os.path.basename(efg_path), "shape:", df_all.shape)
# the raw numerator COUNT columns for f/(i+j+k) (+ disp_f0/(i+j+k) when use_disp_f); both live
# in the gen_V3_disp_f superset table alongside E/F/G and the disp_f shares.
numer_count_cols = [f_count_col] + ([dispf0_count_col] if use_disp_f else [])
needed = ['pid', 'pub_year', 'up_refer_count', citation_col] + index_cols + numer_count_cols
missing = [c for c in needed if c not in df_all.columns]
assert not missing, "columns missing from %s: %s\n(did you set use_disp_f to match the merged table?)" % (
    efg_path, missing)
df_all = df_all[needed].copy()

# ---- attach the disruption counts (I/J/K) from arrange_disruption.py's disp_full, then derive
#      R = I/(I+J) and the three (i+j+k)-denominator ratios. LEFT join so the E/F/G sample and
#      quantiles are unchanged; papers without a disruption record get NaN counts -> NaN ratios. ----
# arrange_disruption.py tags the window only when it is not the default 5
disp_year_tag = '' if year_offset == 5 else f'_y{year_offset}'
disp_path = os.path.join(meta_dire, 'disruption',
                         f'disp_full{disp_year_tag}{network_suffix}{window_suffix}.csv')
assert os.path.exists(disp_path), (
    "%s\nRun get_disruption.py then arrange_disruption.py first (match use_openalex_augmented / "
    "skip_incomplete_window)." % disp_path)
df_disp = pd.read_csv(disp_path,
                      usecols=['pid', i_count_col, j_count_col, k_count_col, disp_col])
print("loaded", os.path.basename(disp_path), "rows:", len(df_disp))

n_before_join = len(df_all)
df_all = df_all.merge(df_disp, on='pid', how='left')
n_with_disp = int(df_all[i_count_col].notna().sum())
print(f"join with disruption (left): {n_before_join} papers, {n_with_disp} have a disruption record")

# R = I/(I+J); i/(i+j+k) = I/(I+J+K); f/(i+j+k) = f/(I+J+K); disp_f0/(i+j+k) = disp_f_0/(I+J+K).
# denom is NaN for papers without a disruption record (NaN counts) -> NaN ratio -> NaN quantile;
# .where(denom > 0) also yields NaN if a joined paper somehow has denom == 0 (edge case).
I_total = df_all[i_count_col]
denom_ij = I_total + df_all[j_count_col]
denom_ijk = denom_ij + df_all[k_count_col]
df_all[ratio_col] = (I_total / denom_ij).where(denom_ij > 0)
df_all[i_over_ijk_col] = (I_total / denom_ijk).where(denom_ijk > 0)
df_all[f_over_ijk_col] = (df_all[f_count_col] / denom_ijk).where(denom_ijk > 0)   # raw F COUNT
# D = (I-J)/(I+J+K): NOT recomputed -- taken as-is from disp_full. get_disruption.py stores 0
# (not NaN) when the denominator is 0, so mask those to NaN like the ratios above; a spurious 0
# would otherwise rank mid-distribution instead of dropping out.
df_all[disp_col] = df_all[disp_col].where(denom_ijk > 0)
disp_derived_cols = [ratio_col, i_over_ijk_col, f_over_ijk_col, disp_col]
if use_disp_f:
    # raw disp_f_0 COUNT over i+j+k (NOT the share)
    df_all[dispf_over_ijk_col] = (df_all[dispf0_count_col] / denom_ijk).where(denom_ijk > 0)
    disp_derived_cols.append(dispf_over_ijk_col)
for c in disp_derived_cols:
    print(f"{c} stats:", df_all[c].describe().to_dict())

metric_cols = index_cols + disp_derived_cols   # raw index columns to quantile + output
quantile_cols = metric_cols + [citation_col]

# ==================================================
# 2. attach the single top domain as a column (papers with no top domain -> top_concept = NaN;
#    they are KEPT so they take part in the by-year / global quantiles)
# ==================================================
# Which concept map assigns the (year, domain) grouping domain:
#   'single' -> cid2single_concept.pkl (get_s2ag_meta_author_domain_s2_simple_domain.py):
#               exactly ONE category per cid -- a multi-category paper already had ONE chosen at
#               RANDOM upstream -- or 'Others' when it has no category. There is thus NO
#               multi-domain case here, so many more papers get a (year, domain) quantile.
#   'list'   -> cid2category_list.pkl (get_s2ag_meta_author_domain.py): the full category LIST;
#               only papers with EXACTLY ONE category get a top domain, the rest -> NaN.
concept_source = 'single'   # 'single' | 'list'
assert concept_source in ('single', 'list'), concept_source
concept_file = 'cid2single_concept.pkl' if concept_source == 'single' else 'cid2category_list.pkl'
concept_path = os.path.join(meta_dire, concept_file)
assert os.path.exists(concept_path), (
    f"{concept_path}\nRun get_s2ag_meta_author_domain_s2_simple_domain.py first."
    if concept_source == 'single' else f"{concept_path}\nRun get_s2ag_meta_author_domain.py first.")
with open(concept_path, 'rb') as f:
    pid2concept = pickle.load(f)   # 'single': cid -> one category / 'Others'; 'list': cid -> [categories]
print("concept map:", concept_path, "| source:", concept_source, "| size:", len(pid2concept))

top_concepts = []
n_no_concept = 0   # 'Others' / None / empty / pid missing from the map
n_multiple = 0     # more than one category ('list' source only; 'single' never hits this)
for pid in df_all['pid'].values:
    v = pid2concept.get(pid)
    if concept_source == 'single':
        # v is a single category name, 'Others' (no valid category), or None (cid absent)
        if v is not None and v != 'Others':
            top_concepts.append(v)
        else:
            n_no_concept += 1
            top_concepts.append(None)
    else:
        if isinstance(v, list) and len(v) == 1:
            top_concepts.append(v[0])
        else:
            if isinstance(v, list) and len(v) > 1:
                n_multiple += 1
            else:                   # None, [], or pid not in the map
                n_no_concept += 1
            top_concepts.append(None)
df_all['top_concept'] = top_concepts

n_with_domain = int(df_all['top_concept'].notna().sum())
print(f"top-domain step: {len(df_all)} papers kept; {n_with_domain} have a top domain, "
      f"{n_no_concept} with no domain, {n_multiple} with >1 domain "
      f"(the latter two keep top_concept=NaN)")


def compute_and_save(df_base, max_year, name_tag=''):
    """Filter df_base to pub_year < max_year, compute the three quantile families, save one
    CSV (filename carries ybefore_{max_year}{name_tag})."""
    print("=" * 60)
    print("max_year", max_year)

    # ---- reference / citation / max_year filter ----
    n_before_filter = len(df_base)
    df = df_base[
        (df_base['up_refer_count'] >= reference_threshold) &
        (df_base[citation_col] >= citation_threshold) &
        (df_base['pub_year'] < max_year)
    ].copy()
    print(f"threshold filter: {n_before_filter} -> {len(df)} papers "
          f"(removed {n_before_filter - len(df)})")

    # ---- quantile of each E/F/G (+disp_f) index AND citation count. rank(pct=True) =
    #      percentile rank in (0, 1]. Only (a) is domain-based; (b)/(c) rank over ALL papers.
    # (a) within (pub_year, top_concept) -- domain papers only (NaN top_concept -> NaN quantile)
    group_cols = ['pub_year', 'top_concept']
    df['group_size'] = df.groupby(group_cols)['pid'].transform('size')
    for col in quantile_cols:
        df[f'{col}_quantile'] = df.groupby(group_cols)[col].rank(pct=True, method='average')

    # (b) within pub_year only -- ALL papers (regardless of domain)
    df['year_group_size'] = df.groupby('pub_year')['pid'].transform('size')
    for col in quantile_cols:
        df[f'{col}_quantile_by_year'] = df.groupby('pub_year')[col].rank(pct=True, method='average')

    # (c) globally (no grouping) -- ALL papers: rank each raw value across the whole filtered set
    df['global_size'] = len(df)
    for col in quantile_cols:
        df[f'{col}_quantile_global'] = df[col].rank(pct=True, method='average')

    print("number of (year, domain) groups:", df.groupby(group_cols).ngroups)
    print("group_size (year, domain) distribution:", df['group_size'].describe().to_dict())
    print("number of (year) groups:", df.groupby('pub_year').ngroups)
    print("year_group_size distribution:", df['year_group_size'].describe().to_dict())
    print("global_size (all filtered papers):", len(df))

    # ---- save ----
    os.makedirs(efg_dire, exist_ok=True)
    save_path = os.path.join(
        efg_dire,
        f'efg_within_year_domain_quantile_y{year_offset}{network_suffix}_'
        f'rt_{reference_threshold}_'
        f'ct_{citation_threshold}_'
        f'ybefore_{max_year}{name_tag}.csv'
    )
    out_cols = (['pid', 'pub_year', 'top_concept',
                 'group_size', 'year_group_size', 'global_size',
                 'up_refer_count', citation_col]
                + metric_cols
                + [f'{c}_quantile' for c in quantile_cols]
                + [f'{c}_quantile_by_year' for c in quantile_cols]
                + [f'{c}_quantile_global' for c in quantile_cols])
    df[out_cols].to_csv(save_path, index=False)
    print("output shape:", df[out_cols].shape)
    print("saved to:", save_path)


# ==================================================
# 3. run for each pub_year cutoff (one output file each)
# ==================================================
for max_year, name_tag in max_years:
    compute_and_save(df_all, max_year, name_tag)
print("finished")
