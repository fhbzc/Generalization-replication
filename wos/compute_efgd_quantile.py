"""
WoS version of the OpenAlex compute_efgd_quantile.py.

Per-paper E/F/G/D and I/(I+J) quantiles, computed three ways:
    (a) WITHIN its (publication year, domain) peer group     -> {col}_quantile
    (b) WITHIN its publication year only (any/no domain)      -> {col}_quantile_by_year
    (c) GLOBALLY, no grouping (raw value across all papers)   -> {col}_quantile_global
Only (a) is domain-based: it is restricted to papers with exactly ONE domain (others
get NaN). (b) and (c) rank over ALL papers passing the reference / citation / max_year
filters, regardless of domain.

For every paper we take its indices:
    E = e_y{off}, F = f_y{off}, G = g_y{off}   (from get_efg_s2_merge.py output)
    D = disp_y{off}  (disruption index, from arrange_disruption.py output)
    R = I / (I + J)  (disruptive-citation fraction, where J = J_count_y{off} and
                      I = I_count_y{off}, from get_disruption.py / disp_full.csv)
    disp_f 0/1/2 = disruption_f_y{off} / disruption_f_down_gt1_y{off} / disruption_f_down_gt2_y{off}
                      (three "disruptive foundation" shares from get_efg_v2_disp_f.py ->
                      get_efg_s2_merge.py (use_disp_f=True); only quantiled when use_disp_f is
                      True below, merged in per pid; papers without a value get NaN disp_f quantiles)
    i/(i+j+k)   = I / (I + J + K)      (disruption numerator i ALONE, not i-j, over the full
                      disruption denominator i+j+k; K = K_count_y{off} from disp_full.csv)
    f/(i+j+k)   = f_count_y{off} / (I + J + K)   (the raw F COUNT over i+j+k, NOT the
                      citation-normalised share f_y{off})
    disp_f0/(i+j+k) = disruption_f_count_y{off} / (I + J + K)  (the raw disp_f_0 COUNT over
                      i+j+k, NOT the share disruption_f_y{off}; only when use_disp_f; NaN for
                      papers lacking a disp_f value)
plus its citation count (citation_count_y{off}), and compute, for each of these
separately, the paper's quantile (percentile rank) among all papers published in the
SAME YEAR and belonging to the SAME domain.

"Domain" for WoS is the macro_citation_topic recorded by get_meta_data.py and stored
as concept2pid_list.pkl (topic -> list of pids). We invert it to pid -> topics and,
mirroring the OpenAlex rule, a paper has a domain only if it has EXACTLY ONE topic;
papers with no topic / the 'Others' bucket / more than one topic are KEPT with
top_concept=NaN (no (year, domain) quantile, but still in the by-year / global ones).

The quantiles are computed ONLY over the set that passes the reference / citation /
max_year filters, so a paper's quantile reflects its rank among the filtered peers of
its (year, domain) cell.

Set use_with_review to match get_efg.py / get_disruption.py: when True it reads the
'_with_review' E/F/G and disruption outputs (the doctypes=='review'-inclusive network)
and tags its own output with the same suffix.

Set k_require_after_focal to match get_disruption.py / arrange_disruption.py: it selects
the disruption (D + I/J) variant. False -> the dropped-lower-bound K disruption tagged
'_k_no_lower_bound' (k_suffix); True -> the default K with no tag. The E/F/G table and the
domain assignment are unaffected by this flag; it only changes which disp_full is read and
the output filename (so the two variants never collide).

Set use_disp_f to additionally quantile the three "disruptive foundation" (disp_f) shares:
when True it reads the merged gen_V3_disp_f_y{year_offset}{data_suffix}.csv (get_efg_v2_disp_f.py ->
get_efg_s2_merge.py with use_disp_f=True), left-joins its three disp_f share columns per pid (so the
E/F/G/D/R sample and their quantiles are unchanged; papers lacking a disp_f value get NaN disp_f
quantiles), and tags its output filename with '_disp_f' so the two variants never collide.

Input  : META_DIRE/efg_computation/gen_V3_y{year_offset}{data_suffix}.csv          (get_efg_s2_merge.py; E/F/G)
         META_DIRE/disruption/disp_full[_y{year_offset}]{data_suffix}{k_suffix}.csv (arrange_disruption.py; D + J/I counts;
                                                                    the _y tag appears only when year_offset != 5)
         META_DIRE/concept2pid_list{data_suffix}.pkl                               (get_meta_data[_with_review].py; topic -> [pid])
         META_DIRE/efg_computation/gen_V3_disp_f_y{year_offset}{data_suffix}.csv   (get_efg_s2_merge.py use_disp_f=True; disp_f shares; only when use_disp_f)
Output : META_DIRE/efg_computation/
           efgd_within_year_domain_quantile_y{year_offset}{data_suffix}{k_suffix}{dispf_suffix}_rt_{rt}_ct_{ct}_ybefore_{max_year}{name_tag}.csv
         max_year is 2025 - year_offset with name_tag '', except at year_offset == -1 where
         there is no window to leave room for: then it is 2025 (every paper before the data
         horizon) and the name carries '_full_cover'.
"""

import os
import sys
import pickle
from collections import defaultdict

import pandas as pd

# utils.py is in this same directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from utils import META_DIRE  # noqa: E402

meta_dire = META_DIRE
print("meta_dire", meta_dire)

efg_dire = os.path.join(meta_dire, 'efg_computation')
disp_dire = os.path.join(meta_dire, 'disruption')

year_offset = -1    # must match get_efg.py and the disruption window
print("year_offset", year_offset)

# must match get_efg.py / get_disruption.py: read the '_with_review' variant when True
use_with_review = True
data_suffix = '_with_review' if use_with_review else ''
print("use_with_review", use_with_review)

# must match get_disruption.py / arrange_disruption.py: the K-window disruption variant.
# False -> the dropped-lower-bound K -> '_k_no_lower_bound'; True -> default K -> ''.
k_require_after_focal = True
k_suffix = '' if k_require_after_focal else '_k_no_lower_bound'
print("k_require_after_focal", k_require_after_focal)
print("k_suffix", repr(k_suffix))

# When True, additionally quantile the three "disruptive foundation" (disp_f) shares from
# get_efg_v2_disp_f.py -> get_efg_s2_merge.py (use_disp_f=True), read from the merged
# gen_V3_disp_f_y{year_offset}{data_suffix}.csv and left-joined per pid. The E/F/G/D/R sample and
# their quantiles are unchanged (left join never drops rows); papers lacking a disp_f value get
# NaN disp_f quantiles. The output filename gains a '_disp_f' tag so the two variants never collide.
use_disp_f = True
dispf_suffix = '_disp_f' if use_disp_f else ''
print("use_disp_f", use_disp_f)
print("dispf_suffix", repr(dispf_suffix))

# the macro-topic bucket get_meta_data.py assigns to papers with NO topic; treated as
# "no domain" here and dropped (same spirit as OpenAlex dropping papers with no root).
DUMMY_TOPIC = 'Others'

# filters (same convention as the OpenAlex version)
reference_threshold = 1
citation_threshold = 5
# pub_year cutoff. The usual one leaves room for the citation window
# (pub_year < 2025 - year_offset), but with year_offset == -1 there is no elapsed window to
# leave room for, so the cutoff becomes the data horizon itself and every paper before 2025 is
# kept. That variant is tagged '_full_cover' in the filename -- exactly the convention
# openalex/compute_efgd_quantile.py uses -- so the two never collide.
if year_offset == -1:
    max_year = 2025
    name_tag = '_full_cover'
else:
    max_year = 2025 - year_offset
    name_tag = ''
print("reference_threshold", reference_threshold)
print("citation_threshold", citation_threshold)
print("max_year", max_year, "| name_tag", repr(name_tag))

citation_col = f'citation_count_y{year_offset}'
disp_col = f'disp_y{year_offset}'
efg_cols = [f'e_y{year_offset}', f'f_y{year_offset}', f'g_y{year_offset}']

# Additional disruptive-fraction metric  R = I / (I + J), read from disp_full.csv:
#   J = J_count_y{off}                                  (developmental citations)
#   I = I_count_y{off}                                   (disruptive citations)
ratio_col = f'ij_ratio_y{year_offset}'
j_count_col = f'J_count_y{year_offset}'
k_count_col = f'K_count_y{year_offset}'
i_count_col = f'I_count_y{year_offset}'
disp_count_cols = [j_count_col, k_count_col, i_count_col]

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
# E/F/G/D/R only when use_disp_f; read from gen_V3_disp_f_y{off}{data_suffix}.csv, merged per pid.
dispf_cols = ([f'disruption_f_y{year_offset}',
               f'disruption_f_down_gt1_y{year_offset}',
               f'disruption_f_down_gt2_y{year_offset}']
              if use_disp_f else [])

# the three (i+j+k)-denominator ratios; disp_f0/(i+j+k) only when disp_f is loaded
ijk_ratio_cols = [i_over_ijk_col, f_over_ijk_col] + ([dispf_over_ijk_col] if use_disp_f else [])

index_cols = (efg_cols + [disp_col, ratio_col] + dispf_cols + ijk_ratio_cols)
# E/F/G/D + I/(I+J) (+ disp_f) + i/(i+j+k), f/(i+j+k), disp_f0/(i+j+k) indices we quantile

# ==================================================
# 1. load the merged E/F/G table (get_efg_s2_merge.py output)
#    pid kept as str: WoS ids are digit-like and would otherwise lose leading zeros.
# ==================================================
efg_path = os.path.join(efg_dire, f'gen_V3_y{year_offset}{data_suffix}.csv')
assert os.path.exists(efg_path), efg_path
df = pd.read_csv(efg_path, dtype={'pid': str})
print("loaded gen_V3 shape:", df.shape)
# keep only the columns we need from the E/F/G side (incl. the raw F count for f/(i+j+k))
df = df[['pid', 'pub_year', 'up_refer_count', citation_col] + efg_cols + [f_count_col]].copy()

# ==================================================
# 2. attach the single domain (macro_citation_topic) as a column; no-/multi-domain
#    papers -> top_concept = NaN, but they are KEPT (they take part in the by-year /
#    global quantiles; only the (year, domain) quantile excludes them)
# ==================================================
concept_path = os.path.join(meta_dire, f'concept2pid_list{data_suffix}.pkl')
assert os.path.exists(concept_path), concept_path
with open(concept_path, 'rb') as f:
    topic2pid_list = pickle.load(f)
print("topic2pid_list domains:", len(topic2pid_list))

# invert topic -> [pid] into pid -> set(topics); the no-topic 'Others' bucket is
# ignored, so a paper that is ONLY in 'Others' (or in no topic at all) ends up with
# no domain (top_concept = NaN below).
pid2topics = defaultdict(set)
for topic, pid_list in topic2pid_list.items():
    if topic == DUMMY_TOPIC:
        continue
    for pid in pid_list:
        pid2topics[pid].add(topic)

top_concepts = []
n_no_concept = 0   # no domain: pid not in any real topic (incl. 'Others'-only)
n_multiple = 0     # more than one domain
for pid in df['pid'].values:
    topics = pid2topics.get(pid)
    if topics is not None and len(topics) == 1:
        top_concepts.append(next(iter(topics)))
    else:
        if topics is not None and len(topics) > 1:
            n_multiple += 1
        else:
            n_no_concept += 1
        top_concepts.append(None)
df['top_concept'] = top_concepts

# report the two groups that get top_concept=NaN ('Others'/none counts as no domain)
print(f"papers with NO domain (none or only '{DUMMY_TOPIC}'): {n_no_concept}")
print(f"papers with MORE THAN ONE domain: {n_multiple}")

# Keep ALL papers (no-/multi-domain stay with top_concept=NaN): the (year, domain)
# quantile excludes them via groupby; by-year / global rank over everyone.
n_with_domain = int(df['top_concept'].notna().sum())
print(f"domain step: {len(df)} papers kept; {n_with_domain} have a single domain, "
      f"{n_no_concept} no-domain + {n_multiple} multi-domain keep top_concept=NaN")


# ==================================================
# 3. reference / citation / max_year filter
# ==================================================
n_before_filter = len(df)
df = df[
    (df['up_refer_count'] >= reference_threshold) &
    (df[citation_col] >= citation_threshold) &
    (df['pub_year'] < max_year)
].copy()
print(f"threshold filter: {n_before_filter} -> {len(df)} papers "
      f"(removed {n_before_filter - len(df)})")

# ==================================================
# 4. attach the disruption index D (disp_y{off}) and the disruptive/developmental
#    citation counts (J, I) from disp_full.csv, then build R = I/(I+J)
# ==================================================
# arrange_disruption.py tags the window only when it is not the default 5
disp_year_tag = '' if year_offset == 5 else f'_y{year_offset}'
disp_path = os.path.join(disp_dire, f'disp_full{disp_year_tag}{data_suffix}{k_suffix}.csv')
assert os.path.exists(disp_path), disp_path
df_disp = pd.read_csv(disp_path, usecols=['pid', disp_col] + disp_count_cols,
                      dtype={'pid': str})
print("loaded disp_full rows:", len(df_disp))

n_before_join = len(df)
df = df.merge(df_disp, on='pid', how='inner')   # keep only papers that also have D
print(f"join with disruption: {n_before_join} -> {len(df)} papers "
      f"(dropped {n_before_join - len(df)} without a disruption value)")

# R = I / (I + J): fraction of disruptive citations among non-K citations.
# I = I_count, J = J_count. Denominator (I + J) is the citation count,
# which is >= citation_threshold here, but guard against 0 just in case.
I_total = df[i_count_col]
denom_ij = I_total + df[j_count_col]
df[ratio_col] = (I_total / denom_ij).where(denom_ij > 0, 0.0)
print(f"{ratio_col} stats:", df[ratio_col].describe().to_dict())

# ---- two of the three (i+j+k)-denominator ratios (k = K_count); disp_f0/(i+j+k) is
#      computed in section 4b after the disp_f join. All numerators are raw COUNTS:
#      i/(i+j+k) is the disruption numerator i ALONE (NOT i-j); f/(i+j+k) is the raw F count
#      f_count_y{off} over i+j+k (NOT the share f_y{off}). Guard denom == 0. ----
denom_ijk = I_total + df[j_count_col] + df[k_count_col]
df[i_over_ijk_col] = (I_total / denom_ijk).where(denom_ijk > 0, 0.0)
df[f_over_ijk_col] = (df[f_count_col] / denom_ijk).where(denom_ijk > 0, 0.0)
print(f"{i_over_ijk_col} stats:", df[i_over_ijk_col].describe().to_dict())
print(f"{f_over_ijk_col} stats:", df[f_over_ijk_col].describe().to_dict())

# ==================================================
# 4b. (optional) attach the three disp_f variant shares from the merged
#     gen_V3_disp_f_y{off}{data_suffix}.csv. LEFT join so the E/F/G/D/R sample and quantiles
#     are unchanged; papers lacking a disp_f value simply get NaN disp_f quantiles.
# ==================================================
if use_disp_f:
    dispf_path = os.path.join(efg_dire, f'gen_V3_disp_f_y{year_offset}{data_suffix}.csv')
    assert os.path.exists(dispf_path), dispf_path
    # also read the raw disp_f_0 COUNT (disruption_f_count_y{off}) for the disp_f0/(i+j+k) ratio
    df_dispf = pd.read_csv(dispf_path, usecols=['pid'] + dispf_cols + [dispf0_count_col],
                           dtype={'pid': str})
    print("loaded disp_f rows:", len(df_dispf))

    df = df.merge(df_dispf, on='pid', how='left')
    n_with_dispf = int(df[dispf_cols[0]].notna().sum())
    print(f"join with disp_f (left): {len(df)} papers, {n_with_dispf} have a disp_f value")

    # disp_f0/(i+j+k): the raw disp_f_0 COUNT over the disruption denominator (NOT the share
    # disruption_f_y{off}). NaN disp_f count (paper lacked a disp_f value) propagates -> NaN
    # ratio -> NaN quantile. Recompute the denominator from df columns (post-merge) to stay
    # index-safe.
    denom_ijk_dispf = df[i_count_col] + df[j_count_col] + df[k_count_col]
    df[dispf_over_ijk_col] = (df[dispf0_count_col] / denom_ijk_dispf).where(denom_ijk_dispf > 0, 0.0)
    print(f"{dispf_over_ijk_col} stats:", df[dispf_over_ijk_col].describe().to_dict())

# ==================================================
# 5. quantile of each E/F/G/D index AND citation count. rank(pct=True) = percentile
#    rank in (0, 1]. Only (a) is domain-based; (b)/(c) rank over ALL papers.
#      (a) within (pub_year, top_concept) -> {col}_quantile           (DOMAIN papers only;
#          no-/multi-domain papers have top_concept=NaN -> NaN here)
#      (b) within pub_year (any/no domain) -> {col}_quantile_by_year   (ALL papers)
#      (c) globally, no grouping           -> {col}_quantile_global    (ALL papers)
# ==================================================
quantile_cols = index_cols + [citation_col]

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

# ==================================================
# 6. save
# ==================================================
save_path = os.path.join(
    efg_dire,
    f'efgd_within_year_domain_quantile_y{year_offset}{data_suffix}{k_suffix}{dispf_suffix}_'
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
print("finished")
