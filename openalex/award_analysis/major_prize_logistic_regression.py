import os
import re
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
from utils import META_DIRE, PRIZE_DATA_DIRE

meta_dire = META_DIRE
print("meta_dire:", meta_dire)

prize_data_dire = PRIZE_DATA_DIRE          # external dataset (see utils.py)
save_dire = os.path.join(meta_dire, 'major_prize_analysis')
os.makedirs(save_dire, exist_ok=True)
print("prize_data_dire:", prize_data_dire)
print("save_dire:", save_dire)

# ── Hyperparameters ───────────────────────────────────────────────────────────
reference_threshold    = 1
citation_threshold     = 5
# Manual year bounds: derived min/max_year will be clipped to these limits.
# Set to None to apply no limit.
manual_min_year_floor  = 1951   # e.g. 1951 — derived min_year will be max(derived, floor)
manual_max_year_cap    = 2020   # e.g. 2008 — derived max_year (exclusive) will be min(derived, cap+1)
print("reference_threshold:", reference_threshold)
print("citation_threshold:", citation_threshold)
print("manual_min_year_floor:", manual_min_year_floor)
print("manual_max_year_cap:", manual_max_year_cap)

# ── Per-file config ───────────────────────────────────────────────────────────
# prize_col: column whose string we check for "Nobel" to exclude
PRIZE_FILES = [
    {
        'file':      'Medicine Prize Winners Cleaned 2.xls',
        'domain':    'Medicine',
        'prize_col': 'Prize',
    },
    {
        'file':      'Biology Prize Winners (12.30).xlsx',
        'domain':    'Biology',
        'prize_col': 'Prize',
    },
    {
        'file':      'Chemistry Prize Winners Cleaned 2.xlsx',
        'domain':    'Chemistry',
        'prize_col': 'Award',
    },
]

# ── Load concept → pid list ───────────────────────────────────────────────────
print("\nLoading concept2pid_list ...")
with open(os.path.join(meta_dire, 'concept2pid_list.pkl'), 'rb') as f:
    concept2pid_list = pickle.load(f)
print(f"  concept2pid_list size: {len(concept2pid_list)}")
print(f"  available domains: {list(concept2pid_list.keys())}")

# ── Load pid → PMID, build pmid → pid ────────────────────────────────────────
print("\nLoading pid2pmid ...")
with open(os.path.join(meta_dire, 'pid2pmid.pkl'), 'rb') as f:
    pid2pmid = pickle.load(f)
pmid2pid = {v: k for k, v in pid2pmid.items()}
print(f"  pid2pmid size: {len(pid2pmid)}")

# ── Load generalization data ──────────────────────────────────────────────────
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

# ── Load disruption data ──────────────────────────────────────────────────────
print("\nLoading disruption data ...")
df_disp = pd.read_csv(
    os.path.join(meta_dire, 'disruption', 'disp_full.csv'),
    usecols=['pid', 'disp_y5']
)
print(f"  disruption rows: {len(df_disp)}")

# ── Load disp_f data (3 variants: down>0 / down>1 / down>2) ────────────────────
# From get_efg_v2_disp_f.py -> get_efg_s2_merge.py (use_disp_f=True). The three
# "disruptive foundation" shares each get their own logistic regression below.
DISPF_COLS = ['disruption_f_y5', 'disruption_f_down_gt1_y5', 'disruption_f_down_gt2_y5']
print("\nLoading disp_f data ...")
df_dispf = pd.read_csv(
    os.path.join(meta_dire, 'efg_computation', 'gen_V3_disp_f_y5.csv'),
    usecols=['pid'] + DISPF_COLS
)
print(f"  disp_f rows: {len(df_dispf)}")

# ── Load semantic distance data ───────────────────────────────────────────────
print("\nLoading semantic distance data ...")
df_dist = pd.read_csv(
    os.path.join(meta_dire, 'merged_stats',
                 'merged_dist_min_y_1951_max_y_2025_p_l_5_r_t_1_c_t_5_no_abs.csv'),
    usecols=['pid', 'title_dist_mean', 'refer_dist_mean']
)
print(f"  distance rows: {len(df_dist)}")

# ── Regression formulas (no top_concept FE: only one domain per run) ─────────
models_spec = [
    ('disp',       'is_prize ~ log_up_refer_count + log_citation_count + C(year_bin) + disp_y5'),
    ('disp_f_0',   'is_prize ~ log_up_refer_count + log_citation_count + C(year_bin) + disruption_f_y5'),
    ('disp_f_1',   'is_prize ~ log_up_refer_count + log_citation_count + C(year_bin) + disruption_f_down_gt1_y5'),
    ('disp_f_2',   'is_prize ~ log_up_refer_count + log_citation_count + C(year_bin) + disruption_f_down_gt2_y5'),
    ('f_index',    'is_prize ~ log_up_refer_count + log_citation_count + C(year_bin) + f'),
    ('e_index',    'is_prize ~ log_up_refer_count + log_citation_count + C(year_bin) + e'),
    ('g_index',    'is_prize ~ log_up_refer_count + log_citation_count + C(year_bin) + g'),
    ('title_dist', 'is_prize ~ log_up_refer_count + log_citation_count + C(year_bin) + title_dist_mean'),
    ('refer_dist', 'is_prize ~ log_up_refer_count + log_citation_count + C(year_bin) + refer_dist_mean'),
    # ('f_and_refer_dist', 'is_prize ~ log_up_refer_count + log_citation_count + C(year_bin) + f + refer_dist_mean'),
]

# ═════════════════════════════════════════════════════════════════════════════
# Main loop: one domain per file
# ═════════════════════════════════════════════════════════════════════════════
for cfg in tqdm(PRIZE_FILES, desc='domains', position=0):
    domain    = cfg['domain']
    fpath     = os.path.join(prize_data_dire, cfg['file'])
    prize_col = cfg['prize_col']

    print(f"\n{'='*60}")
    print(f"Domain: {domain}  |  file: {cfg['file']}")

    # ── Load prize file ───────────────────────────────────────────────────────
    engine = 'xlrd' if cfg['file'].endswith('.xls') else 'openpyxl'
    df_prize = pd.read_excel(fpath, engine=engine)

    # Auto-detect prize column if not specified
    # Exclude Nobel prize rows
    n_before = len(df_prize)
    df_prize = df_prize[~df_prize[prize_col].str.lower().str.contains('nobel', na=False)]
    print(f"  rows before/after Nobel filter: {n_before} → {len(df_prize)}")

    # Gather PMIDs from ALL "PMID*" columns (PMID, PMID 1, PMID 2, ...), not just the first.
    # Empty / non-numeric cells are simply skipped.
    pmid_cols = [c for c in df_prize.columns
                 if re.match(r'^\s*PMID\s*\d*\s*$', str(c), re.IGNORECASE)]
    print(f"  PMID columns used: {pmid_cols}")

    # Map every listed PMID (across all PMID columns) → PID
    prize_pids = set()
    for col in pmid_cols:
        vals = pd.to_numeric(df_prize[col], errors='coerce').dropna()
        for pmid in vals.astype(int).astype(str):
            pid = pmid2pid.get(pmid)
            if pid is not None:
                prize_pids.add(pid)
    print(f"  prize PIDs matched (all PMID columns): {len(prize_pids)}")

    # Derive year range from matched prize papers' pub_year in df_gen_full
    prize_pub_years = df_gen_full.loc[df_gen_full['pid'].isin(prize_pids), 'pub_year'].dropna()
    if len(prize_pub_years) == 0:
        print("  WARNING: no prize PIDs found in df_gen_full, skipping.")
        continue
    min_year = int(prize_pub_years.min())
    max_year = int(prize_pub_years.max()) + 1   # exclusive upper bound
    if manual_min_year_floor is not None:
        min_year = max(min_year, manual_min_year_floor)
    if manual_max_year_cap is not None:
        max_year = min(max_year, manual_max_year_cap + 1)
    print(f"  prize paper pub_year range (after manual clip): {min_year} – {max_year - 1}")

    # ── Filter df_gen to this domain ──────────────────────────────────────────
    domain_pids = set(concept2pid_list.get(domain, []))
    if not domain_pids:
        print(f"  WARNING: domain '{domain}' not found in concept2pid_list, skipping.")
        continue
    print(f"  PIDs in domain '{domain}': {len(domain_pids)}")

    df_gen = df_gen_full[df_gen_full['pid'].isin(domain_pids) &
                         (df_gen_full['pub_year'] >= min_year) &
                         (df_gen_full['pub_year'] <  max_year)].copy()
    print(f"  df_gen after filters: {len(df_gen)}")

    # ── Merge ─────────────────────────────────────────────────────────────────
    df = df_gen.merge(df_disp, on='pid', how='inner')
    df = df.merge(df_dist, on='pid', how='left')
    # disp_f merged LEFT so it never shrinks the existing models' sample; the disp_f models
    # drop their own NaN rows via patsy (disp_f is NOT added to the global reg_vars dropna).
    df = df.merge(df_dispf, on='pid', how='left')
    print(f"  merged rows: {len(df)}")

    # ── Prize indicator ───────────────────────────────────────────────────────
    df['is_prize'] = df['pid'].isin(prize_pids).astype(int)
    print(f"  prize papers in sample: {df['is_prize'].sum()}")
    print(f"  non-prize papers in sample: {(df['is_prize'] == 0).sum()}")

    if df['is_prize'].sum() == 0:
        print("  WARNING: no prize papers in sample after filters, skipping.")
        continue

    # Log-transform controls
    df['log_up_refer_count'] = np.log1p(df['up_refer_count'])
    df['log_citation_count'] = np.log1p(df['citation_count_y5'])

    # Standardise metrics (disp_f cols standardised over their own non-NaN rows)
    for col in ['disp_y5', 'f', 'e', 'g', 'refer_dist_mean', 'title_dist_mean'] + DISPF_COLS:
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    reg_vars = ['is_prize', 'log_up_refer_count', 'log_citation_count', 'pub_year',
                'disp_y5', 'f', 'e', 'g', 'title_dist_mean', 'refer_dist_mean']
    df = df.dropna(subset=reg_vars)
    df['pub_year'] = df['pub_year'].astype(int)
    print(f"  rows after dropping NaN: {len(df)}")

    # Fixed effect = non-overlapping 5-year bins anchored at min_year (bin 0 = [min_year, min_year+4],
    # bin 1 = [min_year+5, min_year+9], ...), then DROP every 5-year bin with NO positive
    # (is_prize==1) paper. Coarser bins mean far fewer FE levels than per-year dummies, and dropping
    # the all-control bins removes the quasi-complete separation that breaks logit convergence.
    # Every prize paper is kept (a bin holding a positive is retained by construction).
    bin_start = min_year + 5 * ((df['pub_year'] - min_year) // 5)
    df['year_bin'] = (bin_start.astype(int).astype(str) + '_' +
                      (bin_start + 4).astype(int).astype(str))
    bins_with_prize = set(df.loc[df['is_prize'] == 1, 'year_bin'])
    n_bins_before = int(df['year_bin'].nunique())
    df = df[df['year_bin'].isin(bins_with_prize)].copy()
    print(f"  5y-bins kept {int(df['year_bin'].nunique())}/{n_bins_before} "
          f"(dropped {n_bins_before - int(df['year_bin'].nunique())} bins with no prize paper); "
          f"rows now {len(df)}")

    stem = (f'dom_{domain}_rt_{reference_threshold}_ct_{citation_threshold}'
            f'_miny_{min_year}_my_{max_year - 1}')

    # ── Regressions ───────────────────────────────────────────────────────────
    for model_name, formula in tqdm(models_spec, desc=f'  [{domain}] regressions', position=1, leave=False):
        output_path = os.path.join(save_dire, f'major_prize_logistic_{model_name}_{stem}.xlsx')
        if os.path.exists(output_path):
            print(f"  Skipping [{model_name}]: already exists")
            continue

        try:
            result = smf.logit(formula, data=df).fit(
                disp=False, method='newton', maxiter=1000, warn_convergence=True
            )
        except Exception as e:
            print(f"  newton failed ({e}), retrying with bfgs ...")
            result = smf.logit(formula, data=df).fit(
                disp=False, method='bfgs', maxiter=1000, warn_convergence=True
            )

        params   = result.params
        bse      = result.bse
        tvals    = result.tvalues
        pvals    = result.pvalues
        ci_low   = params - 1.96 * bse
        ci_high  = params + 1.96 * bse
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
                'n_prize':   int(df['is_prize'].sum()),
                'pseudo_r2': result.prsquared,
                'aic':       result.aic,
                'bic':       result.bic,
                'converged': converged,
            })

        pd.DataFrame(results_rows).to_excel(output_path, index=False)
        print(f"  Saved: {output_path}")

print("\nDone.")
