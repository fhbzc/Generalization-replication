import os
import sys
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)             # utils.py is in the parent directory
from utils import META_DIRE

meta_dire = META_DIRE
print("meta_dire:", meta_dire)
# with open(os.path.join(meta_dire, 'concept2pid_list.pkl'), 'rb') as f:
#     concept2pid_list = pickle.load(f)
# print(f"  concept2pid_list size: {len(concept2pid_list)}")
# print(concept2pid_list.keys())

# assert False
# Nobel prize data is in the local project folder (two levels up from replication/openalex/)
nobel_prize_dire = os.path.join(meta_dire, 'nobel_prize')
print("nobel_prize_dire:", os.path.abspath(nobel_prize_dire))

save_dire = os.path.join(meta_dire, 'nobel_analysis')
os.makedirs(save_dire, exist_ok=True)
print("save_dire:", save_dire)



# ── Thresholds (same as existing pipeline) ───────────────────────────────────
target_domains = {'Chemistry', 'Medicine', 'Physics'}
domain_str = '_'.join(sorted(target_domains))
reference_threshold = 1
citation_threshold  = 5
# Year range is DERIVED from the matched Nobel papers' pub_year (same method as
# nobel_logistic_regression_sep.py), then clipped to [floor, cap]:
manual_min_year_floor = 1951   # derived min_year will be max(derived, floor)
manual_max_year_cap   = 2020   # derived max_year (exclusive) will be min(derived, cap+1)
print("target_domains:", target_domains)
print("domain_str:", domain_str)
print("reference_threshold:", reference_threshold)
print("citation_threshold:", citation_threshold)
print("manual_min_year_floor:", manual_min_year_floor)
print("manual_max_year_cap:", manual_max_year_cap)

# ── 1. Build doi → pid mapping ────────────────────────────────────────────────
print("\nLoading pid2doi ...")
with open(os.path.join(meta_dire, 'pid2doi.pkl'), 'rb') as f:
    pid2doi = pickle.load(f)
print(f"  pid2doi size: {len(pid2doi)}")

doi2pid_list = defaultdict(list)
for pid, doi in tqdm(pid2doi.items(), desc='building doi2pid_list'):
    doi2pid_list[doi].append(pid)

# ── 2. Nobel prize DOIs → PIDs ────────────────────────────────────────────────
print("\nLoading Nobel prize records ...")
doi_set_nobel = set()
for fname in os.listdir(nobel_prize_dire):
    if 'Prize-winning' in fname:
        continue
    fpath = os.path.join(nobel_prize_dire, fname)
    df_n = pd.read_csv(fpath, usecols=['Is prize-winning paper', 'DOI'], encoding='latin1')
    prize_rows = df_n[df_n['Is prize-winning paper'] == 'YES']
    doi_set_nobel |= set(prize_rows['DOI'].dropna())

print(f"  Nobel DOIs found: {len(doi_set_nobel)}")

nobel_pids = set()
for doi in doi_set_nobel:
    if doi in doi2pid_list and len(doi2pid_list[doi]) == 1:
        nobel_pids.add(doi2pid_list[doi][0])
print(f"  Nobel PIDs (unique mapping): {len(nobel_pids)}")

# ── 3. Load generalization data (base frame: has pub_year, up_refer_count, citation_count_y5) ──
# Same source as nobel_logistic_regression_sep.py: the merged E/F/G table gen_V3_y5.csv
# (get_efg_s2_merge.py), using the f_y5/e_y5/g_y5 shares (renamed to f/e/g).
print("\nLoading generalization data ...")
df_gen = pd.read_csv(
    os.path.join(meta_dire, 'efg_computation', 'gen_V3_y5.csv'),
    usecols=['pid', 'pub_year', 'up_refer_count', 'citation_count_y5',
             'f_y5', 'e_y5', 'g_y5']
)
df_gen.rename(columns={'f_y5': 'f', 'e_y5': 'e', 'g_y5': 'g'}, inplace=True)
print(f"  generalization rows: {len(df_gen)}")

# ── 3b. Filter to Chemistry / Medicine / Physics ──────────────────────────────
print("\nFiltering to Chemistry / Medicine / Physics ...")
with open(os.path.join(meta_dire, 'concept2pid_list.pkl'), 'rb') as f:
    concept2pid_list = pickle.load(f)
print(f"  concept2pid_list size: {len(concept2pid_list)}")

valid_pids_domain = set()
for domain in target_domains:
    valid_pids_domain.update(concept2pid_list.get(domain, []))
print(f"  PIDs in target domains: {len(valid_pids_domain)}")

df_gen = df_gen[df_gen['pid'].isin(valid_pids_domain)]
print(f"  generalization rows after domain filter: {len(df_gen)}")

# missing_pids = set(nobel_pids) - set(df_gen['pid'])
# print("missing_pids", missing_pids)

# assert False
# Report Nobel coverage before any threshold filtering
# df_nobel_before = df_gen['pid'].isin(nobel_pids)
# print(max(df_gen[df_gen['pid'].isin(nobel_pids)]['pub_year']))
# assert False
n_nobel_before = df_gen['pid'].isin(nobel_pids).sum()
print(f"  Nobel papers before threshold filter: {n_nobel_before}")

# Reference / citation filters first (keep papers with sufficient references and citations),
# then derive the year range from the surviving Nobel papers — mirrors nobel_logistic_regression_sep.py,
# which derives from its rt/ct-filtered frame before applying the year window.
df_gen = df_gen[df_gen['up_refer_count']    >= reference_threshold]
df_gen = df_gen[df_gen['citation_count_y5'] >= citation_threshold]

# ── Derive year range from matched Nobel papers' pub_year (same method as _sep) ──
prize_pub_years = df_gen.loc[df_gen['pid'].isin(nobel_pids), 'pub_year'].dropna()
assert len(prize_pub_years) > 0, "no Nobel PIDs found in df_gen after domain + ref/citation filter"
raw_min_year = int(prize_pub_years.min())
raw_max_year = int(prize_pub_years.max())
print(f"  Nobel paper pub_year range (raw derived): {raw_min_year} – {raw_max_year}")

min_year = raw_min_year
max_year = raw_max_year + 1   # exclusive upper bound
if manual_min_year_floor is not None:
    min_year = max(min_year, manual_min_year_floor)
if manual_max_year_cap is not None:
    max_year = min(max_year, manual_max_year_cap + 1)
print(f"  Final derived year window (after manual clip): min_year={min_year}, "
      f"max_year={max_year} (inclusive range {min_year} – {max_year - 1})")

# Apply the derived pub_year window
df_gen = df_gen[df_gen['pub_year'] >= min_year]
df_gen = df_gen[df_gen['pub_year'] <  max_year]
print(f"  after threshold filter: {len(df_gen)}")

# F/E/G already renamed from f_y5/e_y5/g_y5 at load (same as nobel_logistic_regression_sep.py)

# ── 4. Load disruption data ───────────────────────────────────────────────────
print("\nLoading disruption data ...")
df_disp = pd.read_csv(
    os.path.join(meta_dire, 'disruption', 'disp_full.csv'),
    usecols=['pid', 'disp_y5']
)
print(f"  disruption rows: {len(df_disp)}")

# ── 4b. Load disp_f data (3 variants: down>0 / down>1 / down>2) ────────────────
# From get_efg_v2_disp_f.py -> get_efg_s2_merge.py (use_disp_f=True). The three
# "disruptive foundation" shares each get their own logistic regression below (mirrors
# nobel_logistic_regression_sep.py). Merged LEFT so it never shrinks the other models'
# sample; the disp_f models drop their own NaN rows via patsy.
DISPF_COLS = ['disruption_f_y5', 'disruption_f_down_gt1_y5', 'disruption_f_down_gt2_y5']
print("\nLoading disp_f data ...")
df_dispf = pd.read_csv(
    os.path.join(meta_dire, 'efg_computation', 'gen_V3_disp_f_y5.csv'),
    usecols=['pid'] + DISPF_COLS
)
print(f"  disp_f rows: {len(df_dispf)}")

# ── 5. Load top concept data ──────────────────────────────────────────────────
print("\nLoading top concept data ...")
with open(os.path.join(meta_dire, 'pid2root_concepts_level_0.pkl'), 'rb') as f:
    pid2root_concepts = pickle.load(f)

pid2top_concept = {
    pid: concepts[0]
    for pid, concepts in pid2root_concepts.items()
    if isinstance(concepts, list) and len(concepts) == 1
}
print(f"  papers with unique top concept: {len(pid2top_concept)}")

df_concept = pd.DataFrame(
    list(pid2top_concept.items()), columns=['pid', 'top_concept']
)

# ── 6. Load semantic distance data ───────────────────────────────────────────
print("\nLoading semantic distance data ...")
df_dist = pd.read_csv(
    os.path.join(meta_dire, 'merged_stats',
                 'merged_dist_min_y_1951_max_y_2025_p_l_5_r_t_1_c_t_5_no_abs.csv'),
    usecols=['pid', 'title_dist_mean', 'refer_dist_mean']
)
print(f"  distance rows: {len(df_dist)}")

# ── 7. Merge ──────────────────────────────────────────────────────────────────
print("\nMerging ...")
df = df_gen.merge(df_disp, on='pid', how='inner')
df = df.merge(df_dist, on='pid', how='left')
df = df.merge(df_concept, on='pid', how='inner')
# disp_f merged LEFT so it never shrinks the existing models' sample; the disp_f models
# drop their own NaN rows via patsy (disp_f is NOT added to the reg_vars dropna below).
df = df.merge(df_dispf, on='pid', how='left')
print(f"  merged rows: {len(df)}")

# ── 8. Add Nobel prize indicator ──────────────────────────────────────────────
df['is_nobel'] = df['pid'].isin(nobel_pids).astype(int)
print(f"  Nobel papers in sample: {df['is_nobel'].sum()}")
print(f"  Non-Nobel papers in sample: {(df['is_nobel'] == 0).sum()}")

# Log-transform skewed control variables
df['log_up_refer_count']  = np.log1p(df['up_refer_count'])
df['log_citation_count']  = np.log1p(df['citation_count_y5'])

# Standardise all metric variables (mean=0, std=1); disp_f cols standardised over their
# own non-NaN rows (pandas mean/std skip NaN), matching nobel_logistic_regression_sep.py.
for col in ['disp_y5', 'f', 'e', 'g', 'refer_dist_mean', 'title_dist_mean'] + DISPF_COLS:
    df[col] = (df[col] - df[col].mean()) / df[col].std()

# Drop rows with NaN in any regression variable (can arise from division by zero)
reg_vars = ['is_nobel', 'log_up_refer_count', 'log_citation_count', 'pub_year',
            'disp_y5', 'f', 'e', 'g', 'title_dist_mean', 'refer_dist_mean',
            'top_concept']
df = df.dropna(subset=reg_vars)
print(f"  rows after dropping NaN: {len(df)}")

# pub_year must be treated as categorical for fixed effects
df['pub_year'] = df['pub_year'].astype(int)

# Fixed effect = non-overlapping 5-year bins anchored at min_year (bin 0 = [min_year, min_year+4],
# bin 1 = [min_year+5, min_year+9], ...), then DROP every 5-year bin with NO positive (is_nobel==1)
# paper. Coarser bins mean far fewer FE levels than per-year dummies, and dropping the all-control
# bins removes the quasi-complete separation that breaks logit convergence / drives pseudo-R2
# negative. Every Nobel paper is kept (a bin holding a positive is retained by construction).
bin_start = min_year + 5 * ((df['pub_year'] - min_year) // 5)
df['year_bin'] = (bin_start.astype(int).astype(str) + '_' +
                  (bin_start + 4).astype(int).astype(str))
bins_with_prize = set(df.loc[df['is_nobel'] == 1, 'year_bin'])
n_bins_before = int(df['year_bin'].nunique())
df = df[df['year_bin'].isin(bins_with_prize)].copy()
print(f"  5y-bins kept {int(df['year_bin'].nunique())}/{n_bins_before} "
      f"(dropped {n_bins_before - int(df['year_bin'].nunique())} bins with no Nobel paper)")
print(f"  rows after dropping no-prize bins: {len(df)}")

# ── 9. Logistic regressions ───────────────────────────────────────────────────
models_spec = [
    ('disp',       'is_nobel ~ log_up_refer_count + log_citation_count + C(year_bin) + C(top_concept) + disp_y5'),
    ('disp_f_0',   'is_nobel ~ log_up_refer_count + log_citation_count + C(year_bin) + C(top_concept) + disruption_f_y5'),
    ('disp_f_1',   'is_nobel ~ log_up_refer_count + log_citation_count + C(year_bin) + C(top_concept) + disruption_f_down_gt1_y5'),
    ('disp_f_2',   'is_nobel ~ log_up_refer_count + log_citation_count + C(year_bin) + C(top_concept) + disruption_f_down_gt2_y5'),
    ('f_index',    'is_nobel ~ log_up_refer_count + log_citation_count + C(year_bin) + C(top_concept) + f'),
    ('e_index',    'is_nobel ~ log_up_refer_count + log_citation_count + C(year_bin) + C(top_concept) + e'),
    ('g_index',    'is_nobel ~ log_up_refer_count + log_citation_count + C(year_bin) + C(top_concept) + g'),
    ('title_dist', 'is_nobel ~ log_up_refer_count + log_citation_count + C(year_bin) + C(top_concept) + title_dist_mean'),
    ('refer_dist', 'is_nobel ~ log_up_refer_count + log_citation_count + C(year_bin) + C(top_concept) + refer_dist_mean'),
    # ('f_and_refer_dist', 'is_nobel ~ log_up_refer_count + log_citation_count + C(year_bin) + C(top_concept) + f + refer_dist_mean'),
]

for model_name, formula in tqdm(models_spec, desc='fitting logistic regressions'):
    output_path = os.path.join(save_dire, f'nobel_logistic_regression_{model_name}_rt_{reference_threshold}_ct_{citation_threshold}_miny_{min_year}_my_{max_year}_dom_{domain_str}.xlsx')
    if os.path.exists(output_path):
        print(f"\nSkipping model [{model_name}]: output already exists at {output_path}")
        continue

    print(f"\nFitting model [{model_name}]: {formula}")
    try:
        result = smf.logit(formula, data=df).fit(
            disp=False,
            method='newton',
            maxiter=200,
            warn_convergence=True
        )
    except Exception as e:
        print(f"  newton failed ({e}), retrying with bfgs ...")
        result = smf.logit(formula, data=df).fit(
            disp=False,
            method='bfgs',
            maxiter=500,
            warn_convergence=True
        )

    params = result.params
    bse    = result.bse
    tvals  = result.tvalues
    pvals  = result.pvalues

    # Compute 95% CI from normal approximation (robust even when conf_int() struggles)
    ci_low  = params - 1.96 * bse
    ci_high = params + 1.96 * bse

    converged = result.mle_retvals.get('converged', None)
    print(f"  converged: {converged}")
    print(f"  n_obs={int(result.nobs)}, pseudo_R2={result.prsquared:.4f}, AIC={result.aic:.1f}")

    # Only report non-fixed-effect terms (exclude year and top_concept FEs)
    keep_vars = [v for v in params.index
                 if not v.startswith('C(year_bin)') and not v.startswith('C(top_concept)')]

    results_rows = []
    for var in keep_vars:
        results_rows.append({
            'model':        model_name,
            'variable':     var,
            'coef':         params[var],
            'std_err':      bse[var],
            'z_stat':       tvals[var],
            'p_value':      pvals[var],
            'ci_2_5':       ci_low[var],
            'ci_97_5':      ci_high[var],
            'n_obs':        int(result.nobs),
            'n_nobel':      int(df['is_nobel'].sum()),
            'pseudo_r2':    result.prsquared,
            'aic':          result.aic,
            'bic':          result.bic,
            'converged':    converged,
        })

    # ── 10. Save results per model ────────────────────────────────────────────
    pd.DataFrame(results_rows).to_excel(output_path, index=False)
    print(f"  Saved: {output_path}")

print("Done.")
