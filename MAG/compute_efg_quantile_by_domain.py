"""
Per-paper E/F/G quantiles WITHIN each (publication year, level-0 domain) peer group.

For every level-0 field of study (domain) and every publication year, we take all
papers published that year in that domain that pass the reference / citation filters
(citation_count_y{off} >= citation_threshold and up_refer_count >= reference_threshold)
and compute, for each paper, the percentile rank (quantile in (0, 1]) of its
E = e_y{off}, F = f_y{off}, G = g_y{off} shares among that (year, domain) peer group.

A paper can belong to MORE THAN ONE level-0 domain (pid2field_of_study.pkl maps
pid -> list of level-0 display names), so it is evaluated once per domain: the table
is exploded to one row per (pid, domain), and the quantile of a paper reflects its
rank among the same-year peers of THAT domain.

COMBINE_DOMAINS lets you define MERGED domains: a dict {merged_name: [domain, ...]}. A paper
in ANY listed domain is pooled ONCE (unique pids) into the merged domain, and its E/F/G
quantiles are computed WITHIN that merged domain-year. Merged domains are saved as their own
CSVs (alongside the individual ones), so they can be selected when plotting. Every listed
value domain must exist in the data (asserted before the quantile computation).

DISRUPTION is quantiled the same way: the raw per-paper disruption index (disp_y{off} from
get_disruption.py's disp_full file) is joined by pid and ranked WITHIN each (domain, year) peer
group, producing disp_y{off}_quantile alongside the E/F/G quantiles. Papers without a disruption
value get a NaN quantile and simply don't participate in the disruption ranking.

The SAME metric set as openalex/compute_efgd_quantile.py is quantiled, so the two pipelines are
comparable. Beyond E/F/G and D, each of these is quantiled too -- every one DERIVED from raw
COUNTS, never from the citation-normalised shares f_y{off} / disruption_f_y{off}:
    R = i/(i+j)      = ij_ratio_y{off}        (disruptive-citation fraction; i = I_count_y{off},
                       j = J_count_y{off}, both from disp_full)
    i/(i+j+k)        = i_over_ijk_y{off}      (the numerator i ALONE, not i-j, over i+j+k;
                       k = K_count_y{off} -- citations of the focal paper's references that do
                       NOT cite the focal paper itself)
    f/(i+j+k)        = f_over_ijk_y{off}      (the raw F COUNT f_count_y{off} over i+j+k)
    disp_f0/(i+j+k)  = dispf_over_ijk_y{off}  (the raw disp_f_0 COUNT disruption_f_count_y{off}
                       over i+j+k; use_disp_f only)
    disp_f 0/1/2     = disruption_f_y{off} / disruption_f_down_gt1_y{off} /
                       disruption_f_down_gt2_y{off}  (the three "disruptive foundation" shares
                       from get_efg_v2_disp_f.py; use_disp_f only)
Note (i-j)/(i+j+k) IS the disruption index D, so it is disp_y{off} and is not recomputed.
disp_full and gen_V3_disp_f are LEFT-joined, so a paper missing from either keeps its E/F/G
quantiles and only gets NaN for the metrics it lacks.

Each index ALSO gets a GLOBAL quantile ({col}_quantile_global): the paper is ranked over ALL
threshold-passing papers pooled together -- NEITHER by domain NOR by year (so a paper's global
quantile is identical across every domain it belongs to). Use whichever fits: {col}_quantile
compares within the (domain, year) peer group; {col}_quantile_global compares against the entire
threshold-passing corpus.

Input : <META>/efg_computation/gen_V3_y{year_offset}.csv   (get_efg_s2_merge.py; E/F/G + f_count)
        <META>/disruption/disp_full_k_no_lower_bound.csv   (get_disruption.py; raw disruption +
                                                           the I/J/K counts)
        <META>/efg_computation/gen_V3_disp_f_y{year_offset}.csv  (get_efg_v2_disp_f.py; the three
                                                           disp_f shares + disruption_f_count;
                                                           only when use_disp_f)
        <META>/pid2field_of_study.pkl                      (get_paper_meta.py; pid -> [level-0 names])
Output: <META>/efg_computation/efg_within_year_domain_quantile_y{year_offset}_rt_{rt}_ct_{ct}/*.csv
        (per-domain; columns include the e/f/g/disp/ij_ratio/disp_f/(i+j+k)-ratio values, their
        within-(domain,year) quantiles, and their global quantiles)
"""

import os
import sys
import pickle

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)               # utils.py is in this same directory
from utils import META_DIRE  # noqa: E402

meta_dire = META_DIRE
efg_dire = os.path.join(meta_dire, 'efg_computation')
print("meta_dire", meta_dire)

year_offset = -1     # must match get_efg.py / get_efg_s2_merge.py
print("year_offset", year_offset)

# a paper enters its (year, domain) peer group only if it has enough references and citations
reference_threshold = 1
citation_threshold = 5
print("reference_threshold", reference_threshold)
print("citation_threshold", citation_threshold)

# OPTIONAL combined ("merged") domains. key = new merged-domain name, value = list of existing
# level-0 domains to pool. A paper that belongs to ANY of the listed domains is counted ONCE
# in the merged domain (unique pids), and the E/F/G quantiles are computed WITHIN that merged
# domain-year peer group. The merged domain is saved as its OWN CSV (in addition to the
# individual domains), so a merged domain can be selected when plotting. Every value domain
# must be an existing domain in the data (asserted before the quantile computation).
COMBINE_DOMAINS = {
    'Social science': ['Economics', 'Sociology', 'Psychology', 'Political science'],
    'Geoscience': ['Geography', 'Geology', 'Environmental science'],

    # e.g. 'Physical sciences': ['Physics', 'Chemistry', 'Materials science'],
    # e.g. 'Life & health': ['Biology', 'Medicine'],
}
print("COMBINE_DOMAINS", COMBINE_DOMAINS)

citation_col = f'citation_count_y{year_offset}'
efg_cols = [f'e_y{year_offset}', f'f_y{year_offset}', f'g_y{year_offset}']

# disruption is quantiled the same way as E/F/G (within each (domain, year) peer group).
# raw per-paper disruption comes from get_disruption.py's disp_full file.
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

# ---- the same extra indices openalex/compute_efgd_quantile.py quantiles, so the two pipelines
#      report the same metric set. All are DERIVED here from raw COUNTS (never from the
#      citation-normalised shares f_y{off} / disruption_f_y{off}):
#        i, j, k    = I_count_y{off}, J_count_y{off}, K_count_y{off}  (disp_full; k = citations
#                     of the focal paper's references that do NOT cite the focal paper itself)
#        R = i/(i+j)      = ij_ratio_y{off}       (disruptive-citation fraction)
#        i/(i+j+k)        = i_over_ijk_y{off}     (numerator i ALONE, not i-j)
#        f/(i+j+k)        = f_over_ijk_y{off}     (the raw F COUNT f_count_y{off} over i+j+k)
#        disp_f0/(i+j+k)  = dispf_over_ijk_y{off} (the raw disp_f_0 COUNT over i+j+k; use_disp_f
#                           only; NaN for papers lacking a disp_f value)
#      (i-j)/(i+j+k) is the disruption index D = disp_y{off}, already read from disp_full above.
ratio_col = f'ij_ratio_y{year_offset}'
i_over_ijk_col = f'i_over_ijk_y{year_offset}'
f_over_ijk_col = f'f_over_ijk_y{year_offset}'
dispf_over_ijk_col = f'dispf_over_ijk_y{year_offset}'
j_count_col = f'J_count_y{year_offset}'
k_count_col = f'K_count_y{year_offset}'
i_count_col = f'I_count_y{year_offset}'
i_iso_count_col = f'I_iso_count_y{year_offset}'       # legacy split layout (older get_disruption)
i_start_count_col = f'I_start_count_y{year_offset}'
# NUMERATOR count columns -- raw COUNTS, NOT the shares:
f_count_col = f'f_count_y{year_offset}'                  # gen_V3 (get_efg.py); for f/(i+j+k)
dispf0_count_col = f'disruption_f_count_y{year_offset}'  # gen_V3_disp_f; for disp_f0/(i+j+k)

# When True, also quantile the three "disruptive foundation" SHARES from get_efg_v2_disp_f.py's
# gen_V3_disp_f table (a citing paper shares NONE of the focal paper's references and co-cites
# more than T = 0/1/2 of its other citing papers), and enable disp_f0/(i+j+k). That table is
# LEFT-joined, so papers without a disp_f value simply get NaN disp_f quantiles.
use_disp_f = True
dispf_cols = ([f'disruption_f_y{year_offset}',
               f'disruption_f_down_gt1_y{year_offset}',
               f'disruption_f_down_gt2_y{year_offset}']
              if use_disp_f else [])
dispf_full_path = os.path.join(efg_dire, f'gen_V3_disp_f_y{year_offset}.csv')
print("use_disp_f", use_disp_f)

# the (i+j+k)-denominator ratios; disp_f0/(i+j+k) only when the disp_f counts are loaded
ijk_ratio_cols = ([i_over_ijk_col, f_over_ijk_col]
                  + ([dispf_over_ijk_col] if use_disp_f else []))

# all per-paper indices to quantile: E/F/G + D + I/(I+J) (+ the disp_f shares) + the (i+j+k) ratios
value_cols = efg_cols + [disp_col, ratio_col] + dispf_cols + ijk_ratio_cols
# each index gets TWO quantiles: within (domain, year) [_quantile], and GLOBAL [_quantile_global]
# = ranked over ALL threshold-passing papers pooled (neither by domain nor by year).
global_qcols = [f'{c}_quantile_global' for c in value_cols]
print("value_cols", value_cols)

# ==================================================
# 1. load the merged E/F/G table (get_efg_s2_merge.py output)
# ==================================================
efg_path = os.path.join(efg_dire, f'gen_V3_y{year_offset}.csv')
assert os.path.exists(efg_path), efg_path
df = pd.read_csv(efg_path)
print("loaded gen_V3 shape:", df.shape)
# the raw F COUNT comes along for f/(i+j+k) -- it is NOT the citation-normalised share f_y{off}
df = df[['pid', 'pub_year', 'up_refer_count', citation_col] + efg_cols + [f_count_col]].copy()

# ==================================================
# 2. reference / citation filter
# ==================================================
n_before = len(df)
df = df[
    (df['up_refer_count'] >= reference_threshold) &
    (df[citation_col] >= citation_threshold)
].copy()
print(f"threshold filter: {n_before} -> {len(df)} papers (removed {n_before - len(df)})")

# ---- attach raw disruption AND the I/J/K counts by pid (left join; papers without disruption
#      stay NaN and get NaN disruption / ratio quantiles, so they simply don't participate in
#      those rankings) ----
assert os.path.exists(disp_full_path), disp_full_path
# auto-detect the I-column layout (current single I_count vs the legacy split I_iso + I_start),
# exactly as openalex/compute_efgd_quantile.py does, so both get_disruption.py vintages work.
disp_header = pd.read_csv(disp_full_path, nrows=0).columns
if i_count_col in disp_header:
    i_source_cols = [i_count_col]
elif i_iso_count_col in disp_header and i_start_count_col in disp_header:
    i_source_cols = [i_iso_count_col, i_start_count_col]      # I = I_iso + I_start
else:
    raise AssertionError(
        f"{disp_full_path} has neither '{i_count_col}' nor "
        f"'{i_iso_count_col}'+'{i_start_count_col}': {list(disp_header)}")
print("disp_full I-column layout:", i_source_cols)
assert k_count_col in disp_header, (
    f"{disp_full_path} is missing '{k_count_col}' (needed for the i+j+k denominator): "
    f"{list(disp_header)}")

disp_df = pd.read_csv(
    disp_full_path,
    usecols=['pid', disp_col, j_count_col, k_count_col] + i_source_cols).drop_duplicates('pid')
df = df.merge(disp_df, on='pid', how='left')
print(f"attached disruption: {df[disp_col].notna().sum()} / {len(df)} papers have {disp_col}")

# ---- attach the raw disp_f_0 COUNT + the three disp_f SHARES (left join; missing -> NaN) ----
if use_disp_f:
    assert os.path.exists(dispf_full_path), (
        f"{dispf_full_path}\nRun get_efg_v2_disp_f.py then get_efg_s2_merge.py first "
        f"(or set use_disp_f = False).")
    dispf_df = pd.read_csv(
        dispf_full_path, usecols=['pid'] + dispf_cols + [dispf0_count_col]).drop_duplicates('pid')
    df = df.merge(dispf_df, on='pid', how='left')
    print(f"attached disp_f: {df[dispf0_count_col].notna().sum()} / {len(df)} papers have a value")

# ---- derive the count-based ratios (same definitions as openalex/compute_efgd_quantile.py).
#      Guard against a 0 denominator (it cannot happen while citation_threshold >= 1, since
#      i+j IS the citation count, but be safe). A NaN count -- i.e. a paper missing from
#      disp_full / gen_V3_disp_f -- propagates to a NaN ratio -> NaN quantile. ----
I_total = df[i_source_cols].sum(axis=1)
denom_ij = I_total + df[j_count_col]
denom_ijk = denom_ij + df[k_count_col]
df[ratio_col] = (I_total / denom_ij).where(denom_ij > 0, 0.0)
df[i_over_ijk_col] = (I_total / denom_ijk).where(denom_ijk > 0, 0.0)
df[f_over_ijk_col] = (df[f_count_col] / denom_ijk).where(denom_ijk > 0, 0.0)   # raw F COUNT
if use_disp_f:
    # raw disp_f_0 COUNT over i+j+k (NOT the share disruption_f_y{off})
    df[dispf_over_ijk_col] = (df[dispf0_count_col] / denom_ijk).where(denom_ijk > 0, 0.0)
for c in [ratio_col] + ijk_ratio_cols:
    print(f"{c} stats:", df[c].describe().to_dict())

# ---- GLOBAL quantile: rank each index over ALL threshold-passing papers pooled together --
#      NEITHER by domain NOR by year (df here is one row per pid). This value is a property of the
#      paper, so it is the SAME across every domain the paper is later exploded into. ----
for col in value_cols:
    df[f'{col}_quantile_global'] = df[col].rank(pct=True, method='average')

# ==================================================
# 3. attach level-0 domains and explode to one row per (pid, domain);
#    then add merged-domain rows (pooled, unique pids) for any COMBINE_DOMAINS
# ==================================================
with open(os.path.join(meta_dire, 'pid2field_of_study.pkl'), 'rb') as f:
    pid2field_of_study = pickle.load(f)
print("pid2field_of_study load finished, size", len(pid2field_of_study))

# de-duplicate domains per paper (a paper may map to the same level-0 field via
# several sub-fields) -> a set per pid
pid2domains = {pid: set(names) for pid, names in pid2field_of_study.items()}

# one row per (pid, domain): the individual (unmerged) level-0 domains
df['domain'] = df['pid'].map(lambda p: list(pid2domains.get(p, ())))
df = df.explode('domain')
df = df[df['domain'].notna()].copy()
print("after explode by domain, rows:", len(df))
print("number of distinct level-0 domains:", df['domain'].nunique())

# ---- assert every merged value-domain exists in the data (before any computation) ----
available_domains = set(df['domain'].unique())
for merged_name, value_domains in COMBINE_DOMAINS.items():
    for d in value_domains:
        assert d in available_domains, (
            f"COMBINE_DOMAINS[{merged_name!r}]: domain {d!r} not found in the data. "
            f"available domains: {sorted(available_domains)}")

# ---- build merged-domain rows: a paper in ANY constituent domain is counted ONCE ----
if COMBINE_DOMAINS:
    base_cols = (['pid', 'pub_year', 'up_refer_count', citation_col]
                 + value_cols + global_qcols)                       # one row per pid
    df_papers = df.drop_duplicates('pid')[base_cols]
    merged_frames = []
    for merged_name, value_domains in COMBINE_DOMAINS.items():
        vset = set(value_domains)
        keep_pids = {p for p, ds in pid2domains.items() if vset & ds}
        sub = df_papers[df_papers['pid'].isin(keep_pids)].copy()  # already unique per pid
        sub['domain'] = merged_name
        merged_frames.append(sub)
        print(f"merged domain {merged_name!r} <- {value_domains}: {len(sub)} unique papers")
    df = pd.concat([df] + merged_frames, ignore_index=True)
    print("after adding merged domains, rows:", len(df))

# ==================================================
# 4. quantile of each E/F/G share within (pub_year, domain)
#    rank(pct=True) = percentile rank in (0, 1]
# ==================================================
group_cols = ['domain', 'pub_year']
df['group_size'] = df.groupby(group_cols)['pid'].transform('size')
for col in value_cols:   # E/F/G and disruption, each ranked within its (domain, year) peers
    df[f'{col}_quantile'] = df.groupby(group_cols)[col].rank(pct=True, method='average')

print("number of (domain, year) groups:", df.groupby(group_cols).ngroups)
print("group_size distribution:", df['group_size'].describe().to_dict())

# ==================================================
# 5. save one CSV per level-0 domain
# ==================================================
out_dire = os.path.join(
    efg_dire,
    f'efg_within_year_domain_quantile_y{year_offset}_'
    f'rt_{reference_threshold}_ct_{citation_threshold}'
)
os.makedirs(out_dire, exist_ok=True)

out_cols = (['pid', 'pub_year', 'domain', 'group_size', 'up_refer_count', citation_col]
            + value_cols
            + [f'{c}_quantile' for c in value_cols]        # within (domain, year)
            + global_qcols)                                 # within year, all domains pooled


def safe_filename(name):
    # turn a domain display name into a filesystem-safe filename
    return "".join(c if (c.isalnum() or c in (' ', '-', '_')) else '_' for c in str(name)).strip().replace(' ', '_')


param_suffix = f'y{year_offset}_rt_{reference_threshold}_ct_{citation_threshold}'
for domain, df_domain in df.groupby('domain'):
    save_path = os.path.join(out_dire, f'{safe_filename(domain)}_{param_suffix}.csv')
    df_domain[out_cols].to_csv(save_path, index=False)
    print(f"saved domain {domain!r}: {df_domain.shape[0]} rows -> {save_path}")

print("output dir:", out_dire)
print("finished")
