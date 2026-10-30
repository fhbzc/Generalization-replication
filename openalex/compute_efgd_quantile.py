"""
Per-paper E/F/G/D quantiles, computed three ways:
    (a) WITHIN its (publication year, top domain) peer group -> {col}_quantile
    (b) WITHIN its publication year only (any/no domain)      -> {col}_quantile_by_year
    (c) GLOBALLY, no grouping (raw value across all papers)   -> {col}_quantile_global
Only (a) is domain-based: it is restricted to papers with exactly ONE top domain
(others get NaN). (b) and (c) rank over ALL papers passing the reference / citation /
max_year filters, regardless of domain.

For every paper we take its indices:
    E = e_y{off}, F = f_y{off}, G = g_y{off}   (from get_efg_s2_merge.py output)
    D = disp_y{off}  (disruption index, from arrange_disruption.py output)
    R = I / (I + J)  (disruptive-citation fraction, where J = J_count_y{off} and I is the
                      disruptive-citation count from disp_full.csv: the single
                      I_count_y{off} column, or, for legacy outputs, I_iso_count_y{off}
                      + I_start_count_y{off}; auto-detected)
    disp_f 0/1/2 = disruption_f_y{off} / disruption_f_down_gt1_y{off} / disruption_f_down_gt2_y{off}
                      (three "disruptive foundation" shares from gen_V3_disp_f_y{off}.csv,
                      merged in per pid; papers without a value get NaN disp_f quantiles)
    i/(i+j+k)   = I / (I + J + K)      (disruption numerator i ALONE, not i-j, over the full
                      disruption denominator i+j+k; K = K_count_y{off} from disp_full.csv)
    f/(i+j+k)   = f_count_y{off} / (I + J + K)   (the raw F COUNT over i+j+k, NOT the
                      citation-normalised share f_y{off})
    disp_f0/(i+j+k) = disruption_f_count_y{off} / (I + J + K)  (the raw disp_f_0 COUNT over
                      i+j+k, NOT the share disruption_f_y{off}; only when use_disp_f; NaN for
                      papers lacking a disp_f value)
plus its citation count (citation_count_y{off}), and compute, for each of these
separately, the paper's quantile (percentile rank) among all papers published in
the SAME YEAR and belonging to the SAME top domain.

"Top domain" is the single root-level concept from process_top_concepts.py
(pid2root_concepts_level_{root_level}.pkl, pid -> list of root concepts). Papers
with NO top domain or with MORE THAN ONE top domain are KEPT with top_concept=NaN;
they receive no (year, domain) quantile but still take part in the by-year and global
quantiles.

with_review toggle: when True the top domain instead comes from
get_meta_data_preprint_review_s2_simple_concept.py
(META_DIRE_with_review/pid2single_concept.pkl, pid -> ONE top-scored level-0
concept, or 'Others'). There is then no multi-domain case (each paper already has
exactly one concept); papers mapped to 'Others' / absent keep top_concept=NaN (no
(year, domain) quantile, but still in the by-year / global ones). The output
filename gains a '_with_review' suffix so the two variants coexist.

The quantiles are computed ONLY over the set that passes the reference / citation
/ max_year filters, so a paper's quantile reflects its rank among the filtered
peers of its (year, domain) cell.

Input/Output directory (<META> = META_DIRE, or META_DIRE + '_with_review' when with_review):
Input  : <META>/efg_computation/gen_V3_y{year_offset}.csv   (get_efg_s2_merge.py; E/F/G)
         <META>/disruption/disp_full.csv                    (arrange_disruption.py; D = disp_y{off})
         <META>/pid2root_concepts_level_{root_level}.pkl    (process_top_concepts.py; top domain)
             or, when with_review: <META>/pid2single_concept.pkl
             (get_meta_data_preprint_review_s2_simple_concept.py; one top-scored domain)
Output : <META>/efg_computation/  (TWO files, one per pub_year cutoff)
           efgd_within_year_domain_quantile_y{year_offset}{data_suffix}_rt_{rt}_ct_{ct}_ybefore_{2025 - year_offset}.csv
             -> papers whose full year_offset citation window has elapsed (complete citation_y{off})
           efgd_within_year_domain_quantile_y{year_offset}{data_suffix}_rt_{rt}_ct_{ct}_ybefore_2025_full_cover.csv
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

year_offset = -1    # must match get_efg_s2_merge.py / get_efg.py and the disruption window
root_level = 0     # must match process_top_concepts.py

# When True, take the single top domain from get_meta_data_preprint_review_s2_simple_concept.py
# (pid2single_concept.pkl in META_DIRE + '_with_review': pid -> ONE top-scored level-0
# concept, or 'Others'), instead of pid2root_concepts_level_{root_level}.pkl. The E/F/G and
# disruption INPUTS are then also read from the '_with_review' meta tree (its get_efg /
# get_disruption review runs). The output filename carries a '_with_review' suffix.
with_review = False
data_suffix = '_with_review' if with_review else ''

# When with_review, BOTH inputs (E/F/G, disruption, concept) AND the output quantile
# table live in the '_with_review' meta tree; otherwise the base tree. (The output also
# keeps the '_with_review' filename suffix.)
in_meta_dire = META_DIRE + '_with_review' if with_review else META_DIRE
efg_dire = os.path.join(in_meta_dire, 'efg_computation')   # E/F/G + concept inputs
disp_dire = os.path.join(in_meta_dire, 'disruption')       # disruption input
out_efg_dire = os.path.join(in_meta_dire, 'efg_computation')   # output quantile table

# When True, additionally quantile the three "disruptive foundation" (disp_f) shares
# (disruption_f_y{off} / disruption_f_down_gt1_y{off} / disruption_f_down_gt2_y{off}) from
# gen_V3_disp_f_y{off}.csv (get_efg_v2_disp_f.py -> get_efg_s2_merge.py, use_disp_f=True),
# left-joined per pid. The E/F/G/D/R sample and their quantiles are unchanged either way; the
# OUTPUT FILENAME is the same in both cases (disp_f just adds columns), so downstream scripts
# that read the quantile table are unaffected — but a table written with use_disp_f=False will
# not contain the disp_f columns.
use_disp_f = True

print("year_offset", year_offset)
print("root_level", root_level)
print("with_review", with_review)
print("use_disp_f", use_disp_f)
print("in_meta_dire", in_meta_dire)

# filters (same convention as get_efg_s3_merge_with_concept.py)
reference_threshold = 1
citation_threshold = 5
# Two pub_year cutoffs, each saved to its own file (distinguished by the ybefore_{max_year}
# suffix in the filename):
#   * 2025 - year_offset : papers whose full year_offset citation window has elapsed
#     (complete citation_y{off} data).
#   * 2025               : ALL papers published before 2025. Papers from the most recent
#     years do NOT have a complete year_offset citation window, but that is fine here:
#     quantiles are always computed WITHIN a publication year, so every paper is ranked
#     only against same-year peers that share the same (in)complete window.
# each cutoff -> (max_year, extra filename tag). The all-papers-before-2025 variant is
# tagged '_full_cover' to mark that it covers every paper (incl. recent, incomplete-window ones).
# max_years = [(2025 - year_offset, ''), (2025, '_full_cover')]

# For year_offset == -1 there is no elapsed-window cutoff to speak of, so only the
# all-papers-before-2025 '_full_cover' variant is produced.
if year_offset == -1:
    max_years = [(2025, '_full_cover')]
else:
    max_years = [(2025 - year_offset, '')]

print("reference_threshold", reference_threshold)
print("citation_threshold", citation_threshold)
print("max_years", max_years)

citation_col = f'citation_count_y{year_offset}'
disp_col = f'disp_y{year_offset}'
efg_cols = [f'e_y{year_offset}', f'f_y{year_offset}', f'g_y{year_offset}']

# Additional disruptive-fraction metric  R = I / (I + J), where
#   J = J_count_y{off}        (developmental citations)
#   I = disruptive citations
# read from disp_full.csv (get_disruption.py output). Two historical layouts of that file
# are supported (auto-detected from its header):
#   * current : a single   I_count_y{off}                       column
#   * legacy  : two split   I_iso_count_y{off} + I_start_count_y{off} columns (older
#               get_disruption.py); then I = I_iso + I_start.
ratio_col = f'ij_ratio_y{year_offset}'
j_count_col = f'J_count_y{year_offset}'
k_count_col = f'K_count_y{year_offset}'
i_count_col = f'I_count_y{year_offset}'
i_iso_count_col = f'I_iso_count_y{year_offset}'
i_start_count_col = f'I_start_count_y{year_offset}'

# Three additional metrics that all share the FULL disruption denominator (i + j + k),
# where k = K_count_y{off} (citations of the focal paper's references that do NOT cite the
# focal paper itself). Unlike R = I/(I+J) above, the denominator here is i+j+k (the same
# denominator as the disruption index D = (i-j)/(i+j+k)). All three numerators are raw COUNTS
# (NOT the citation-count-normalised shares f_y / disruption_f_y), so every ratio is count/count:
#   * i/(i+j+k)      : the disruption numerator i ON ITS OWN (NOT i-j) over i+j+k
#   * f/(i+j+k)      : the raw F count f_count_y{off} over i+j+k
#   * disp_f0/(i+j+k): the raw disp_f_0 count disruption_f_count_y{off} over i+j+k
#                      (only when use_disp_f; NaN for papers lacking a disp_f value)
i_over_ijk_col = f'i_over_ijk_y{year_offset}'
f_over_ijk_col = f'f_over_ijk_y{year_offset}'
dispf_over_ijk_col = f'dispf_over_ijk_y{year_offset}'
# NUMERATOR count columns (from gen_V3 / gen_V3_disp_f, get_efg[_v2_disp_f].py):
f_count_col = f'f_count_y{year_offset}'                 # raw F count for f/(i+j+k)
dispf0_count_col = f'disruption_f_count_y{year_offset}'  # raw disp_f_0 count for disp_f0/(i+j+k)

# disp_f variants (get_efg_v2_disp_f.py -> get_efg_s2_merge.py with use_disp_f=True): three
# "disruptive foundation" shares (a citing paper shares NONE of the focal paper's references
# and co-cites more than T of its other citing papers: T = 0 / 1 / 2). Quantiled alongside
# E/F/G/D/R only when use_disp_f; read from gen_V3_disp_f_y{off}.csv and merged in per pid.
dispf_cols = ([f'disruption_f_y{year_offset}',
               f'disruption_f_down_gt1_y{year_offset}',
               f'disruption_f_down_gt2_y{year_offset}']
              if use_disp_f else [])

# the three (i+j+k)-denominator ratios; disp_f0/(i+j+k) only when disp_f is loaded
ijk_ratio_cols = [i_over_ijk_col, f_over_ijk_col] + ([dispf_over_ijk_col] if use_disp_f else [])

index_cols = (efg_cols + [disp_col, ratio_col] + dispf_cols + ijk_ratio_cols)
# E/F/G/D + I/(I+J) (+ disp_f) + i/(i+j+k), f/(i+j+k), disp_f0/(i+j+k) indices

# ==================================================
# 1. load the merged E/F/G table (get_efg_s2_merge.py output)
# ==================================================
efg_path = os.path.join(efg_dire, f'gen_V3_y{year_offset}.csv')
assert os.path.exists(efg_path), efg_path
df = pd.read_csv(efg_path)
print("loaded gen_V3 shape:", df.shape)
# keep only the columns we need from the E/F/G side (incl. the raw F count for f/(i+j+k))
df = df[['pid', 'pub_year', 'up_refer_count', citation_col] + efg_cols + [f_count_col]].copy()

# ==================================================
# 2. attach the single top domain as a column (no-/multi-domain papers -> top_concept
#    = NaN; they are KEPT so they take part in the by-year / global quantiles)
# ==================================================
if with_review:
    # one top-scored level-0 concept per pid (or 'Others') from
    # get_meta_data_preprint_review_s2_simple_concept.py (in the '_with_review' tree)
    concept_path = os.path.join(in_meta_dire, 'pid2single_concept.pkl')
else:
    concept_path = os.path.join(meta_dire, f'pid2root_concepts_level_{root_level}.pkl')
assert os.path.exists(concept_path), concept_path
with open(concept_path, 'rb') as f:
    pid2concept = pickle.load(f)
print("concept map:", concept_path)
print("concept map size:", len(pid2concept))

top_concepts = []
n_no_concept = 0   # None / 'Others' / empty / pid missing from the map
n_multiple = 0     # more than one top concept (only possible for the default source)
for pid in df['pid'].values:
    v = pid2concept.get(pid)
    if with_review:
        # v is a single display name, 'Others' (no valid domain), or None (pid absent)
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
df['top_concept'] = top_concepts

# Keep ALL papers: no-/multi-domain papers stay with top_concept = NaN. The
# (year, domain) quantile naturally excludes them (NaN group key -> NaN rank), while
# the by-year and global quantiles rank over ALL papers regardless of domain.
n_with_domain = int(df['top_concept'].notna().sum())
print(f"top-domain step: {len(df)} papers kept; {n_with_domain} have a single top domain, "
      f"{n_no_concept} with no top domain, {n_multiple} with >1 top domain "
      f"(the latter two keep top_concept=NaN)")

# ==================================================
# 3. load the disruption index D (disp_y{off}) and the disruptive/developmental
#    citation counts (J, I) from disp_full{year_tag}.csv (shared across both cutoffs)
# ==================================================
# arrange_disruption.py tags its disp_full output by citation window: year_offset_main == 5
# keeps the untagged 'disp_full.csv', any other window uses 'disp_full_y{off}.csv'. Match that
# naming so year_offset == -1 loads its own disp_full_y-1.csv rather than the y5 file.
disp_year_tag = '' if year_offset == 5 else f'_y{year_offset}'
disp_path = os.path.join(disp_dire, f'disp_full{disp_year_tag}.csv')
assert os.path.exists(disp_path), disp_path

# auto-detect which I-column layout this disp_full.csv uses (current single I_count vs
# legacy split I_iso + I_start) so both historical outputs of get_disruption.py work.
disp_header = pd.read_csv(disp_path, nrows=0).columns
if i_count_col in disp_header:
    i_source_cols = [i_count_col]
    print(f"disp_full I-column layout: single '{i_count_col}'")
elif i_iso_count_col in disp_header and i_start_count_col in disp_header:
    i_source_cols = [i_iso_count_col, i_start_count_col]
    print(f"disp_full I-column layout: legacy '{i_iso_count_col}' + '{i_start_count_col}'")
else:
    raise AssertionError(
        f"disp_full.csv has neither '{i_count_col}' nor "
        f"'{i_iso_count_col}'+'{i_start_count_col}': {list(disp_header)}")

# K_count is required for the (i+j+k) denominator of the three new ratios.
assert k_count_col in disp_header, (
    f"disp_full.csv is missing '{k_count_col}' (needed for the i/(i+j+k), f/(i+j+k), "
    f"disp_f0/(i+j+k) ratios): {list(disp_header)}")

df_disp = pd.read_csv(disp_path,
                      usecols=['pid', disp_col, j_count_col, k_count_col] + i_source_cols)
print("loaded disp_full rows:", len(df_disp))

# ==================================================
# 3b. (optional) load the three disp_f variant shares (gen_V3_disp_f_y{off}.csv); merged LEFT
#     in compute_and_save so the E/F/G/D/R sample and quantiles are unchanged (papers lacking a
#     disp_f value simply get NaN disp_f quantiles). Skipped entirely when use_disp_f is False.
# ==================================================
if use_disp_f:
    dispf_path = os.path.join(efg_dire, f'gen_V3_disp_f_y{year_offset}.csv')
    assert os.path.exists(dispf_path), dispf_path
    # also read the raw disp_f_0 COUNT (disruption_f_count_y{off}) for the disp_f0/(i+j+k) ratio
    df_dispf = pd.read_csv(dispf_path, usecols=['pid'] + dispf_cols + [dispf0_count_col])
    print("loaded disp_f rows:", len(df_dispf))
else:
    df_dispf = None
    print("use_disp_f=False -> disp_f columns are NOT computed")

quantile_cols = index_cols + [citation_col]


def compute_and_save(df_base, max_year, name_tag=''):
    """Filter df_base to pub_year < max_year, attach disruption + R, compute the three
    quantile families, and save one CSV (filename carries ybefore_{max_year}{name_tag})."""
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

    # ---- attach disruption D + I/J counts, then build R = I/(I+J) ----
    n_before_join = len(df)
    df = df.merge(df_disp, on='pid', how='inner')   # keep only papers that also have D
    print(f"join with disruption: {n_before_join} -> {len(df)} papers "
          f"(dropped {n_before_join - len(df)} without a disruption value)")

    # ---- attach the three disp_f variant shares (LEFT: never drops rows; missing -> NaN) ----
    if use_disp_f:
        df = df.merge(df_dispf, on='pid', how='left')
        n_with_dispf = int(df[dispf_cols[0]].notna().sum())
        print(f"join with disp_f (left): {len(df)} papers, {n_with_dispf} have a disp_f value")

    # R = I / (I + J): fraction of disruptive citations among non-K citations.
    # I = I_count (current) or I_iso + I_start (legacy); J = J_count. Denominator (I + J)
    # is the citation count, which is >= citation_threshold here, but guard against 0.
    I_total = df[i_source_cols].sum(axis=1)
    denom_ij = I_total + df[j_count_col]
    df[ratio_col] = (I_total / denom_ij).where(denom_ij > 0, 0.0)
    print(f"{ratio_col} stats:", df[ratio_col].describe().to_dict())

    # ---- three (i+j+k)-denominator ratios (k = K_count); all numerators are raw COUNTS ----
    #   i/(i+j+k)       : disruption numerator i alone (NOT i-j) over the full denominator
    #   f/(i+j+k)       : the raw F count f_count_y{off} over i+j+k
    #   disp_f0/(i+j+k) : the raw disp_f_0 count disruption_f_count_y{off} over i+j+k
    #                     (NaN where disp_f is missing)
    # Guard against denom == 0 (won't happen under citation_threshold>=5, but be safe).
    denom_ijk = I_total + df[j_count_col] + df[k_count_col]
    df[i_over_ijk_col] = (I_total / denom_ijk).where(denom_ijk > 0, 0.0)
    df[f_over_ijk_col] = (df[f_count_col] / denom_ijk).where(denom_ijk > 0, 0.0)
    print(f"{i_over_ijk_col} stats:", df[i_over_ijk_col].describe().to_dict())
    print(f"{f_over_ijk_col} stats:", df[f_over_ijk_col].describe().to_dict())
    if use_disp_f:
        # NaN disp_f count (paper lacked a disp_f value) propagates -> NaN ratio -> NaN quantile
        df[dispf_over_ijk_col] = (df[dispf0_count_col] / denom_ijk).where(denom_ijk > 0, 0.0)
        print(f"{dispf_over_ijk_col} stats:", df[dispf_over_ijk_col].describe().to_dict())

    # ---- quantile of each E/F/G/D index AND citation count. rank(pct=True) = percentile
    #      rank in (0, 1]. Only (a) is domain-based; (b)/(c) rank over ALL papers.
    #        (a) within (pub_year, top_concept) -> {col}_quantile         (DOMAIN papers only;
    #            no-/multi-domain papers have top_concept=NaN -> NaN here)
    #        (b) within pub_year (any/no domain) -> {col}_quantile_by_year (ALL papers)
    #        (c) globally, no grouping           -> {col}_quantile_global  (ALL papers)
    # ----
    # (a) within (pub_year, top_concept) -- domain papers only (NaN top_concept is dropped by
    #     groupby, so those rows get NaN group_size / NaN quantile)
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

    print("number of (year, domain) groups:",
          df.groupby(group_cols).ngroups)
    print("group_size (year, domain) distribution:",
          df['group_size'].describe().to_dict())
    print("number of (year) groups:",
          df.groupby('pub_year').ngroups)
    print("year_group_size distribution:",
          df['year_group_size'].describe().to_dict())
    print("global_size (all filtered papers):", len(df))

    # ---- save ----
    os.makedirs(out_efg_dire, exist_ok=True)
    save_path = os.path.join(
        out_efg_dire,
        f'efgd_within_year_domain_quantile_y{year_offset}{data_suffix}_'
        f'rt_{reference_threshold}_'
        f'ct_{citation_threshold}_'
        f'ybefore_{max_year}{name_tag}.csv'
    )
    out_cols = (['pid', 'pub_year', 'top_concept',
                 'group_size', 'year_group_size', 'global_size',
                 'up_refer_count', citation_col]
                + index_cols
                + [f'{c}_quantile' for c in quantile_cols]
                + [f'{c}_quantile_by_year' for c in quantile_cols]
                + [f'{c}_quantile_global' for c in quantile_cols])
    df[out_cols].to_csv(save_path, index=False)
    print("output shape:", df[out_cols].shape)
    print("saved to:", save_path)


# ==================================================
# 4. run for each pub_year cutoff (one output file each)
# ==================================================
for max_year, name_tag in max_years:
    compute_and_save(df, max_year, name_tag)
print("finished")
