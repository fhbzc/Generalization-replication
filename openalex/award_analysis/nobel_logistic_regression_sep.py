import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)             # utils.py is in the parent directory
from utils import META_DIRE, NOBEL_DATA_DIRE

meta_dire = META_DIRE
print("meta_dire:", meta_dire)

nobel_prize_dire = NOBEL_DATA_DIRE          # external dataset (see utils.py)
save_dire        = os.path.join(meta_dire, 'nobel_analysis_sep')
os.makedirs(save_dire, exist_ok=True)
print("nobel_prize_dire:", os.path.abspath(nobel_prize_dire))
print("save_dire:", save_dire)

# ── Hyperparameters ───────────────────────────────────────────────────────────
reference_threshold    = 1
citation_threshold     = 5
manual_min_year_floor  = 1951   # derived min_year will be max(derived, floor)
manual_max_year_cap    = 2020   # derived max_year (exclusive) will be min(derived, cap+1)
print("reference_threshold:", reference_threshold)
print("citation_threshold:", citation_threshold)
print("manual_min_year_floor:", manual_min_year_floor)
print("manual_max_year_cap:", manual_max_year_cap)

# ── Per-domain config ─────────────────────────────────────────────────────────
NOBEL_FILES = [
    {'file': 'Medicine publication record.csv',  'domain': 'Medicine'},
    {'file': 'Chemistry publication record.csv', 'domain': 'Chemistry'},
    {'file': 'Physics publication record.csv',   'domain': 'Physics'},
]

# ── 1. Build doi → pid mapping ────────────────────────────────────────────────
print("\nLoading pid2doi ...")
with open(os.path.join(meta_dire, 'pid2doi.pkl'), 'rb') as f:
    pid2doi = pickle.load(f)
print(f"  pid2doi size: {len(pid2doi)}")

doi2pid_list = defaultdict(list)
for pid, doi in tqdm(pid2doi.items(), desc='building doi2pid_list'):
    doi2pid_list[doi].append(pid)

# ── 2. Load concept → pid list ────────────────────────────────────────────────
print("\nLoading concept2pid_list ...")
with open(os.path.join(meta_dire, 'concept2pid_list.pkl'), 'rb') as f:
    concept2pid_list = pickle.load(f)
print(f"  concept2pid_list size: {len(concept2pid_list)}")

# ── 3. Load generalization data (full, pre-domain-split) ─────────────────────
print("\nLoading generalization data ...")
df_gen_full = pd.read_csv(
    os.path.join(meta_dire, 'efg_computation', 'gen_V3_y5.csv'),
    usecols=['pid', 'pub_year', 'up_refer_count', 'citation_count_y5',
             'f_y5', 'e_y5', 'g_y5']
)
df_gen_full.rename(columns={
    'f_y5': 'f',
    'e_y5': 'e',
    'g_y5': 'g',
}, inplace=True)
print(f"  generalization rows: {len(df_gen_full)}")
df_gen_full = df_gen_full[df_gen_full['up_refer_count']    >= reference_threshold]
df_gen_full = df_gen_full[df_gen_full['citation_count_y5'] >= citation_threshold]
print(f"  generalization rows after threshold filter: {len(df_gen_full)}")

# ── 4. Load disruption data ───────────────────────────────────────────────────
print("\nLoading disruption data ...")
df_disp = pd.read_csv(
    os.path.join(meta_dire, 'disruption', 'disp_full.csv'),
    usecols=['pid', 'disp_y5']
)
print(f"  disruption rows: {len(df_disp)}")

# ── 4b. Load disp_f data (3 variants: down>0 / down>1 / down>2) ────────────────
# From get_efg_v2_disp_f.py -> get_efg_s2_merge.py (use_disp_f=True). The three
# "disruptive foundation" shares each get their own logistic regression below.
DISPF_COLS = ['disruption_f_y5', 'disruption_f_down_gt1_y5', 'disruption_f_down_gt2_y5']
print("\nLoading disp_f data ...")
df_dispf = pd.read_csv(
    os.path.join(meta_dire, 'efg_computation', 'gen_V3_disp_f_y5.csv'),
    usecols=['pid'] + DISPF_COLS
)
print(f"  disp_f rows: {len(df_dispf)}")

# ── 5. Load semantic distance data ───────────────────────────────────────────
print("\nLoading semantic distance data ...")
df_dist = pd.read_csv(
    os.path.join(meta_dire, 'merged_stats',
                 'merged_dist_min_y_1951_max_y_2025_p_l_5_r_t_1_c_t_5_no_abs.csv'),
    usecols=['pid', 'title_dist_mean', 'refer_dist_mean']
)
print(f"  distance rows: {len(df_dist)}")

# ── Regression formulas (no top_concept FE: only one domain per run) ─────────
models_spec = [
    ('disp',       'is_nobel ~ log_up_refer_count + log_citation_count + C(year_bin) + disp_y5'),
    ('disp_f_0',   'is_nobel ~ log_up_refer_count + log_citation_count + C(year_bin) + disruption_f_y5'),
    ('disp_f_1',   'is_nobel ~ log_up_refer_count + log_citation_count + C(year_bin) + disruption_f_down_gt1_y5'),
    ('disp_f_2',   'is_nobel ~ log_up_refer_count + log_citation_count + C(year_bin) + disruption_f_down_gt2_y5'),
    ('f_index',    'is_nobel ~ log_up_refer_count + log_citation_count + C(year_bin) + f'),
    ('e_index',    'is_nobel ~ log_up_refer_count + log_citation_count + C(year_bin) + e'),
    ('g_index',    'is_nobel ~ log_up_refer_count + log_citation_count + C(year_bin) + g'),
    ('title_dist', 'is_nobel ~ log_up_refer_count + log_citation_count + C(year_bin) + title_dist_mean'),
    ('refer_dist', 'is_nobel ~ log_up_refer_count + log_citation_count + C(year_bin) + refer_dist_mean'),
]

# ═════════════════════════════════════════════════════════════════════════════
# Main loop: one domain per Nobel CSV file
# ═════════════════════════════════════════════════════════════════════════════
for cfg in tqdm(NOBEL_FILES, desc='domains', position=0):
    domain = cfg['domain']
    fpath  = os.path.join(nobel_prize_dire, cfg['file'])

    print(f"\n{'='*60}")
    print(f"Domain: {domain}  |  file: {cfg['file']}")

    # ── Load Nobel prize file and extract prize-winning DOIs ─────────────────
    df_n = pd.read_csv(fpath, usecols=['Is prize-winning paper', 'DOI'], encoding='latin1')
    prize_rows = df_n[df_n['Is prize-winning paper'] == 'YES']
    doi_set = set(prize_rows['DOI'].dropna())
    print(f"  Nobel DOIs found: {len(doi_set)}")

    # Map DOI → PID (unique mapping only)
    nobel_pids = set()
    for doi in doi_set:
        if doi in doi2pid_list and len(doi2pid_list[doi]) == 1:
            nobel_pids.add(doi2pid_list[doi][0])
    print(f"  Nobel PIDs (unique DOI mapping): {len(nobel_pids)}")

    # ── Derive year range from matched Nobel papers' pub_year ────────────────
    prize_pub_years = df_gen_full.loc[df_gen_full['pid'].isin(nobel_pids), 'pub_year'].dropna()
    if len(prize_pub_years) == 0:
        print("  WARNING: no Nobel PIDs found in df_gen_full, skipping.")
        continue
    min_year = int(prize_pub_years.min())
    max_year = int(prize_pub_years.max()) + 1   # exclusive upper bound
    if manual_min_year_floor is not None:
        min_year = max(min_year, manual_min_year_floor)
    if manual_max_year_cap is not None:
        max_year = min(max_year, manual_max_year_cap + 1)
    print(f"  Nobel paper pub_year range (after manual clip): {min_year} – {max_year - 1}")

    # ── Filter df_gen to this domain and year range ───────────────────────────
    domain_pids = set(concept2pid_list.get(domain, []))
    if not domain_pids:
        print(f"  WARNING: domain '{domain}' not found in concept2pid_list, skipping.")
        continue
    print(f"  PIDs in domain '{domain}': {len(domain_pids)}")

    df_gen = df_gen_full[df_gen_full['pid'].isin(domain_pids) &
                         (df_gen_full['pub_year'] >= min_year) &
                         (df_gen_full['pub_year'] <  max_year)].copy()
    print(f"  df_gen after domain + year filter: {len(df_gen)}")

    # ── Merge ─────────────────────────────────────────────────────────────────
    df = df_gen.merge(df_disp, on='pid', how='inner')
    df = df.merge(df_dist, on='pid', how='left')
    # disp_f merged LEFT so it never shrinks the existing models' sample; the disp_f models
    # drop their own NaN rows via patsy (disp_f is NOT added to the global reg_vars dropna).
    df = df.merge(df_dispf, on='pid', how='left')
    print(f"  merged rows: {len(df)}")

    # ── Nobel indicator ───────────────────────────────────────────────────────
    df['is_nobel'] = df['pid'].isin(nobel_pids).astype(int)
    print(f"  Nobel papers in sample: {df['is_nobel'].sum()}")
    print(f"  non-Nobel papers in sample: {(df['is_nobel'] == 0).sum()}")

    if df['is_nobel'].sum() == 0:
        print("  WARNING: no Nobel papers in sample after filters, skipping.")
        continue

    # Log-transform controls
    df['log_up_refer_count'] = np.log1p(df['up_refer_count'])
    df['log_citation_count'] = np.log1p(df['citation_count_y5'])

    # Standardise metrics (disp_f cols standardised over their own non-NaN rows)
    for col in ['disp_y5', 'f', 'e', 'g', 'refer_dist_mean', 'title_dist_mean'] + DISPF_COLS:
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    reg_vars = ['is_nobel', 'log_up_refer_count', 'log_citation_count', 'pub_year',
                'disp_y5', 'f', 'e', 'g', 'title_dist_mean', 'refer_dist_mean']
    df = df.dropna(subset=reg_vars)
    df['pub_year'] = df['pub_year'].astype(int)
    print(f"  rows after dropping NaN: {len(df)}")

    # Fixed effect = non-overlapping 5-year bins anchored at min_year (bin 0 = [min_year, min_year+4],
    # bin 1 = [min_year+5, min_year+9], ...), then DROP every 5-year bin with NO positive
    # (is_nobel==1) paper. Coarser bins mean far fewer FE levels than per-year dummies, and dropping
    # the all-control bins removes the quasi-complete separation that breaks logit convergence.
    # Every Nobel paper is kept (a bin holding a positive is retained by construction).
    bin_start = min_year + 5 * ((df['pub_year'] - min_year) // 5)
    df['year_bin'] = (bin_start.astype(int).astype(str) + '_' +
                      (bin_start + 4).astype(int).astype(str))
    bins_with_prize = set(df.loc[df['is_nobel'] == 1, 'year_bin'])
    n_bins_before = int(df['year_bin'].nunique())
    df = df[df['year_bin'].isin(bins_with_prize)].copy()
    print(f"  5y-bins kept {int(df['year_bin'].nunique())}/{n_bins_before} "
          f"(dropped {n_bins_before - int(df['year_bin'].nunique())} bins with no Nobel paper); "
          f"rows now {len(df)}")

    stem = (f'dom_{domain}_rt_{reference_threshold}_ct_{citation_threshold}'
            f'_miny_{min_year}_my_{max_year - 1}')

    # ── Regressions ───────────────────────────────────────────────────────────
    for model_name, formula in tqdm(models_spec, desc=f'  [{domain}] regressions', position=1, leave=False):
        output_path = os.path.join(save_dire, f'nobel_logistic_sep_{model_name}_{stem}.xlsx')
        if os.path.exists(output_path):
            print(f"  Skipping [{model_name}]: already exists")
            continue

        print(f"  Fitting [{model_name}]: {formula}")
        try:
            result = smf.logit(formula, data=df).fit(
                disp=False, method='newton', maxiter=1000, warn_convergence=True
            )
        except Exception as e:
            print(f"  newton failed ({e}), retrying with bfgs ...")
            result = smf.logit(formula, data=df).fit(
                disp=False, method='bfgs', maxiter=1000, warn_convergence=True
            )

        params    = result.params
        bse       = result.bse
        tvals     = result.tvalues
        pvals     = result.pvalues
        ci_low    = params - 1.96 * bse
        ci_high   = params + 1.96 * bse
        converged = result.mle_retvals.get('converged', None)
        print(f"  [{model_name}] converged={converged}  n_obs={int(result.nobs)}"
              f"  pseudo_R2={result.prsquared:.4f}  AIC={result.aic:.1f}")

        keep_vars = [v for v in params.index if not v.startswith('C(year_bin)')]
        results_rows = []
        for var in keep_vars:
            results_rows.append({
                'domain':    domain,
                'model':     model_name,
                'variable':  var,
                'coef':      params[var],
                'std_err':   bse[var],
                'z_stat':    tvals[var],
                'p_value':   pvals[var],
                'ci_2_5':    ci_low[var],
                'ci_97_5':   ci_high[var],
                'n_obs':     int(result.nobs),
                'n_nobel':   int(df['is_nobel'].sum()),
                'pseudo_r2': result.prsquared,
                'aic':       result.aic,
                'bic':       result.bic,
                'converged': converged,
            })

        pd.DataFrame(results_rows).to_excel(output_path, index=False)
        print(f"  Saved: {output_path}")

print("\nDone.")
