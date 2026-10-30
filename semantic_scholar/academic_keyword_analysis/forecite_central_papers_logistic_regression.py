"""
Logistic regression: does a paper's index predict being a CONCEPT-INTRODUCING (central) paper?

Companion to forecite_central_papers_efg_s2_plot.py -- same inputs, same central-paper
definitions, but instead of drawing the central-vs-rest distributions it regresses the central
indicator on ONE index at a time, controlling for reference count, citation count and
publication-year fixed effects. The exact specifications are the `models_spec` list below
(mirroring openalex/award_analysis/major_prize_logistic_regression.py); every one of them is

    is_central ~ scaled_<index> + log_up_refer_count + log_citation_count + C(pub_year)

with
    scaled_<index>     = the index, z-scored -> scaled(index)
    log_up_refer_count = log(up_refer_count + 1)
    log_citation_count = log(citation_count_y{off} + 1)
    C(pub_year)        = publication-year fixed effects

Each index is standardised WITHIN its own model's estimation sample (after the NaN and year
drops below), so a focal coefficient reads as "the log-odds change per one standard deviation of
this index among the papers this model is fitted on". The mean/sd actually used are written to
the output CSV (scale_mean / scale_sd) so the raw-unit effect can always be recovered.

The NINE focal indices (one model each; C = I + J is the citation count, so C + K = I+J+K):

    foundation        f_y{off}              F share
    extension         e_y{off}              E share
    generalization    g_y{off}              G share
    disruption        disp_y{off}           D = (I-J)/(I+J+K)
    disruption_ratio  ij_ratio_y{off}       I/(I+J)
    disp_ij           disp_ij_y{off}        (I-J)/(I+J)
    f_over_ck         f_over_ijk_y{off}     f_count/(I+J+K)
    e_over_ck         e_over_ijk_y{off}     e_count/(I+J+K)
    g_over_ck         g_over_ijk_y{off}     g_count/(I+J+K)

The last three are all derived HERE from the SAME pair of files -- the raw f/e/g COUNTS out of the
gen_V3[_disp_f] table over the I+J+K counts out of disp_full -- rather than f/(C+K) being read
back from compute_efg_quantile.py's precomputed f_over_ijk column (which takes its f_count from
the gen_V3_disp_f superset while e/g would have come from whichever table use_disp_f selected).
The formula is identical to compute_efg_quantile.py's, so the values are unchanged; what changes
is that one setting now feeds all three numerators.

NOTE (I-J)/(I+J) = 2*I/(I+J) - 1 is an exact affine transform of I/(I+J), so after standardising
`disp_ij` and `disruption_ratio` are the SAME regressor: both models are fitted (they were both
asked for) but their focal coefficients will agree to numerical precision.

The outcome comes from forecite_central_papers_efg_s1_split.py's cached central sets: a paper is
central (is_central = 1) if it is in that definition's set, and 0 otherwise. Step 1 builds four
definitions and each gets the whole models_spec group; trim DEFINITIONS below to just the one
that matters (e.g. max_score) to skip the others entirely.

The "0" class (non-central papers) is REST_UNIVERSE, exactly as in step 2:
  'all_efg' -> every other paper in the efg quantile table (central vs the general population)
  'scored'  -> only papers scoring some above-threshold keyword, minus the central set
               (central vs non-central WITHIN the ForeCite-scored papers)

Publication years holding NO central paper are dropped before fitting: their year dummy predicts
the outcome perfectly (quasi-complete separation, which breaks logit convergence) and they carry
no information about the focal index anyway. Every central paper is kept by construction.

Input  : ./figures/forecite_gt_<threshold>/central_sets_<out_tag>.pkl   [step 1]
         META_DIRE/efg_computation/efg_within_year_domain_quantile_y{off}{network_suffix}_
             rt_{rt}_ct_{ct}_ybefore_{max_year}{name_tag}.csv            [compute_efg_quantile.py]
         -- only when f_over_ck / e_over_ck / g_over_ck are in models_spec:
         META_DIRE/efg_computation/gen_V3[_disp_f]_y{off}{network_suffix}.csv
             (get_efg_s2_merge.py; the raw f_count/e_count/g_count numerators -- ALL THREE come
              from this ONE file, so f/(C+K) is built exactly like e/(C+K) and g/(C+K))
         META_DIRE/disruption/disp_full[_y{off}]{network_suffix}{window_suffix}.csv
             (arrange_disruption.py; the I/J/K counts making up the C+K denominator)
Output : ./data/central_paper_logistic_regression/ -- ONE CSV PER FOCAL VARIABLE, with the
         variable (and the definition) named in the file, plus one concatenation of them all:
             central_logit_<variable>_def_<definition>_<stem>.csv
             central_logit_coefs_ALL_<stem>.csv
         <stem> = gt_<threshold>_<universe>_y{off}{network_suffix}_rt_{rt}_ct_{ct}_
                  ybefore_{max_year}{name_tag}_<out_tag>
         An existing per-variable CSV is SKIPPED rather than refitted unless OVERWRITE, so a
         killed run resumes where it stopped. Each file holds
             one row per (definition, model, variable): coef / std_err / z / p / 95% CI / odds
             ratio, plus the model's formula, the scale_mean/scale_sd used, the sample sizes
             (n_obs = total / n_central = positives / n_non_central / central_rate, and the
             pre-drop n_frame_rows / n_frame_central), pseudo-R2, AIC/BIC and convergence flag.
             The C(pub_year) fixed-effect dummies are not written out.

Usage  : python forecite_central_papers_logistic_regression.py [FORECITE_THRESHOLD]
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

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
FORECITE_THRESHOLD = 0.5       # keep keywords whose ForeCite score is > this; CLI override below
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
year_offset = 5
use_openalex_augmented = False
network_suffix = '_openalex_augmented' if use_openalex_augmented else ''
reference_threshold = 1       # rt
citation_threshold = 5        # ct
max_year = 2025 - year_offset               # (2025 - year_offset, '') complete window | (2025, '_full_cover') all
name_tag = ''

# ---- only used when a model needs the f_count/e_count/g_count numerators and the I/J/K counts,
#      i.e. when any of f_over_ck / e_over_ck / g_over_ck is in models_spec. ONE table supplies
#      all three numerators, so the three C+K ratios cannot drift apart. Must match
#      get_efg_s2_merge.py / arrange_disruption.py, exactly as in compute_efg_quantile.py --
#      get_efg_s2_merge.py and compute_efg_quantile.py both run with use_disp_f=True, and the
#      _disp_f table is a superset carrying the same f/e/g counts, so True is the matching value.
use_disp_f = True                     # True -> read the gen_V3_disp_f superset table
skip_incomplete_window = True        # True -> disp_full '_complete_window' variant
# The flag only does anything when there IS a window to be incomplete: at
# year_offset == -1 the citation window is unbounded, get_disruption.py's skip test
# short-circuits, and the run is identical either way -- so the tag is dropped there,
# and one -1 run is not written under two names that hold the same data.
window_suffix = ('_complete_window'
                  if (skip_incomplete_window and year_offset != -1) else '')

# "0"-class universe, as in forecite_central_papers_efg_s2_plot.py:
#   'all_efg' = every other paper in the efg table; 'scored' = only papers scoring some
#   above-threshold keyword, minus the central set.
REST_UNIVERSE = 'all_efg'

# Optional case-control subsampling of the non-central papers: keep ALL central papers and at
# most NEG_PER_POS non-central papers per central paper (None -> keep every non-central paper).
# Under case-control sampling the slope coefficients stay consistent; only the intercept shifts.
# Set it if the full table is too large to fit with year dummies.
NEG_PER_POS = None
RANDOM_SEED = 0

# Optional publication-year window on top of the efg table's own filters (None -> no limit).
MIN_PUB_YEAR = None
MAX_PUB_YEAR = None

OVERWRITE = False             # False -> skip the run entirely if the coefficient CSV exists

off = year_offset
citation_col = f'citation_count_y{off}'

# ===========================================================================
# Regression specifications
# ===========================================================================
# One model per focal index; controls and the pub_year fixed effect are identical throughout.
# `scaled_<name>` is the index standardised within this model's own estimation sample (built in
# the fitting loop below); log_up_refer_count = log(up_refer_count + 1) and log_citation_count =
# log(citation_count_y{off} + 1) are the two log controls; C(pub_year) is the year fixed effect.
models_spec = [
    ('foundation',       'is_central ~ scaled_foundation       + log_up_refer_count + log_citation_count + C(pub_year)'),
    ('extension',        'is_central ~ scaled_extension        + log_up_refer_count + log_citation_count + C(pub_year)'),
    ('generalization',   'is_central ~ scaled_generalization   + log_up_refer_count + log_citation_count + C(pub_year)'),
    ('disruption',       'is_central ~ scaled_disruption       + log_up_refer_count + log_citation_count + C(pub_year)'),
    ('disruption_ratio', 'is_central ~ scaled_disruption_ratio + log_up_refer_count + log_citation_count + C(pub_year)'),
    ('disp_ij',          'is_central ~ scaled_disp_ij          + log_up_refer_count + log_citation_count + C(pub_year)'),
    ('f_over_ck',        'is_central ~ scaled_f_over_ck        + log_up_refer_count + log_citation_count + C(pub_year)'),
    ('e_over_ck',        'is_central ~ scaled_e_over_ck        + log_up_refer_count + log_citation_count + C(pub_year)'),
    ('g_over_ck',        'is_central ~ scaled_g_over_ck        + log_up_refer_count + log_citation_count + C(pub_year)'),
]

# Where each focal index comes from. 'source':
#   'table'   -> read straight out of the efg quantile table (compute_efg_quantile.py)
#   'ij_lin'  -> (I-J)/(I+J) = 2 * ij_ratio - 1, derived from the table's I/(I+J) column
#   'over_ck' -> <num>_count_y{off} / (I + J + K); needs the gen_V3 numerator counts and the
#                disp_full I/J/K counts. f/e/g_over_ck ALL use this route (same numerator table,
#                same denominator) -- f is deliberately NOT read from the quantile table's
#                precomputed f_over_ijk column, so the three share one source.
METRIC_SPEC = {
    'foundation':       {'source': 'table',   'col': f'f_y{off}',          'desc': 'F share'},
    'extension':        {'source': 'table',   'col': f'e_y{off}',          'desc': 'E share'},
    'generalization':   {'source': 'table',   'col': f'g_y{off}',          'desc': 'G share'},
    'disruption':       {'source': 'table',   'col': f'disp_y{off}',       'desc': 'D = (I-J)/(I+J+K)'},
    'disruption_ratio': {'source': 'table',   'col': f'ij_ratio_y{off}',   'desc': 'I/(I+J)'},
    'disp_ij':          {'source': 'ij_lin',  'col': f'disp_ij_y{off}',    'desc': '(I-J)/(I+J)'},
    'f_over_ck':        {'source': 'over_ck', 'col': f'f_over_ijk_y{off}', 'desc': 'f_count/(I+J+K)',
                         'num': f'f_count_y{off}'},
    'e_over_ck':        {'source': 'over_ck', 'col': f'e_over_ijk_y{off}', 'desc': 'e_count/(I+J+K)',
                         'num': f'e_count_y{off}'},
    'g_over_ck':        {'source': 'over_ck', 'col': f'g_over_ijk_y{off}', 'desc': 'g_count/(I+J+K)',
                         'num': f'g_count_y{off}'},
}
MODEL_NAMES = [m for m, _ in models_spec]
assert len(MODEL_NAMES) == len(set(MODEL_NAMES)), "duplicate model name in models_spec"
for m, formula in models_spec:
    assert m in METRIC_SPEC, "models_spec model '%s' has no METRIC_SPEC entry" % m
    assert f'scaled_{m} ' in formula, (
        "models_spec formula for '%s' does not use its focal regressor scaled_%s: %s"
        % (m, m, formula))

# the column each model is standardised from, and which of them have to be derived here
focal_col = {m: METRIC_SPEC[m]['col'] for m in MODEL_NAMES}
table_models = [m for m in MODEL_NAMES if METRIC_SPEC[m]['source'] == 'table']
over_ck_models = [m for m in MODEL_NAMES if METRIC_SPEC[m]['source'] == 'over_ck']
need_ijk_join = bool(over_ck_models)

print("FORECITE_THRESHOLD", FORECITE_THRESHOLD, "| out_tag", out_tag)
print("REST_UNIVERSE", REST_UNIVERSE, "| NEG_PER_POS", NEG_PER_POS)
print("models (%d):" % len(models_spec))
for m, formula in models_spec:
    print("   %-17s [%-22s %-18s] %s"
          % (m, METRIC_SPEC[m]['desc'], focal_col[m], formula))

# INPUT dir: step 1's central-sets cache sits with the figures, in a per-threshold folder
# (./figures/forecite_gt_<threshold>/, shared with steps 1 and 2). Never written to here.
central_dir = os.path.join(BASE_DIR, 'figures', f'forecite_gt_{FORECITE_THRESHOLD}')
# OUTPUT dir: ./data/<RESULT_SUBDIR>/. The threshold is no longer in the path, so it moves into
# the filename, together with EVERY other setting that changes the numbers -- otherwise a rerun
# with a different citation window / filter would silently overwrite this one. Two independent
# year bounds are encoded: 'b_/e_' (inside out_tag) is the KEYWORD corpus range that decides
# which papers can be central, while 'ybefore_' is the efg table's own pub_year cutoff
# (2025 - year_offset), which is what actually bounds the regression sample.
RESULT_SUBDIR = 'central_paper_logistic_regression'
save_dir = os.path.join(BASE_DIR, 'data', RESULT_SUBDIR)
os.makedirs(save_dir, exist_ok=True)
# every run parameter that changes the numbers, shared by all output filenames
stem = (f'gt_{FORECITE_THRESHOLD}_{REST_UNIVERSE}_y{off}{network_suffix}_'
        f'rt_{reference_threshold}_ct_{citation_threshold}_ybefore_{max_year}{name_tag}_{out_tag}')


def model_csv_path(def_name, model_name):
    """One CSV per (central-paper definition, focal variable) -- the variable is in the NAME, as
    in major_prize_logistic_regression.py. With OVERWRITE=False an existing file makes that model
    be skipped, so a killed run resumes where it stopped."""
    return os.path.join(save_dir, f'central_logit_{model_name}_def_{def_name}_{stem}.csv')


# plus one concatenation of them all, for convenience when reading the results back
all_csv = os.path.join(save_dir, f'central_logit_coefs_ALL_{stem}.csv')
print("central_dir (step-1 cache, input)", central_dir)
print("save_dir", save_dir)
print("per-model CSV", model_csv_path('<definition>', '<variable>'))
print("combined CSV ", all_csv)

# the per-paper index table most values come from (compute_efg_quantile.py)
efg_csv = os.path.join(
    meta_dire, 'efg_computation',
    f'efg_within_year_domain_quantile_y{off}{network_suffix}_'
    f'rt_{reference_threshold}_ct_{citation_threshold}_ybefore_{max_year}{name_tag}.csv')
# the two extra sources for e/(C+K) and g/(C+K) (only read when those models are requested)
gen_csv = os.path.join(meta_dire, 'efg_computation',
                       f"{'gen_V3_disp_f' if use_disp_f else 'gen_V3'}_y{off}{network_suffix}.csv")
# arrange_disruption.py tags the window only when it is not the default 5
disp_year_tag = '' if year_offset == 5 else f'_y{year_offset}'
disp_csv = os.path.join(meta_dire, 'disruption',
                        f'disp_full{disp_year_tag}{network_suffix}{window_suffix}.csv')

# each definition -> readable description. Must match the keys step 1 stores in central_sets.
# Trim this list to run only the outcome you need (e.g. just max_score).
DEFINITIONS = [
    ('max_score', 'MAX-score paper per keyword'),
    # ('top5_score', f'TOP-{TOP_K_PAPERS} score papers per keyword'),
    # ('earliest', 'EARLIEST (>0 score) paper per keyword'),
    # ('earliest_above_mean', 'EARLIEST paper among those scoring > keyword mean'),
]


# ===========================================================================
# The central sets come from step 1 -- this script never builds them.
# ===========================================================================
central_cache_path = os.path.join(central_dir, f'central_sets_{out_tag}.pkl')
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


central_sets, scored_pids, n_keywords = load_central_sets()
for name, _ in DEFINITIONS:
    assert name in central_sets, "definition '%s' missing from the step-1 cache" % name
    print("  central set %-19s: %d unique papers" % (name, len(central_sets[name])))

# ===========================================================================
# Analysis frame: the efg table (pid + every outcome-independent regressor)
# ===========================================================================
assert os.path.exists(efg_csv), "%s\nRun compute_efg_quantile.py first." % efg_csv
need_cols = {'pid', 'pub_year', 'up_refer_count', citation_col}
need_cols |= {focal_col[m] for m in table_models}
need_cols |= {f'ij_ratio_y{off}'}                      # source of disp_ij
need_cols = sorted(need_cols)
df = pd.read_csv(efg_csv, usecols=need_cols)
print("loaded efg table:", os.path.basename(efg_csv), "shape", df.shape, "| columns read:", need_cols)

# ---- (I-J)/(I+J) = 2 * I/(I+J) - 1: an exact affine transform of the table's ij_ratio, so no
#      count join is needed for it. ----
df[f'disp_ij_y{off}'] = 2.0 * df[f'ij_ratio_y{off}'] - 1.0

# ---- f/(C+K), e/(C+K) and g/(C+K): the raw f_count/e_count/g_count numerators (gen_V3[_disp_f]
#      table) and the I/J/K counts (disp_full) are joined in here and all three ratios derived
#      from that ONE numerator table, exactly as compute_efg_quantile.py derives f_over_ijk:
#      <num>/(I+J+K), NaN when a paper has no disruption record or a zero denominator. Both joins
#      are LEFT joins, so the sample of the other six models is untouched. ----
if need_ijk_join:
    num_cols = sorted({METRIC_SPEC[m]['num'] for m in over_ck_models})
    assert os.path.exists(gen_csv), (
        "%s\nRun get_efg.py / get_efg_v2_disp_f.py then get_efg_s2_merge.py first "
        "(match use_disp_f)." % gen_csv)
    df_num = pd.read_csv(gen_csv, usecols=['pid'] + num_cols)
    print("loaded numerator counts:", os.path.basename(gen_csv), "rows:", len(df_num),
          "| columns:", num_cols)

    assert os.path.exists(disp_csv), (
        "%s\nRun get_disruption.py then arrange_disruption.py first (match "
        "use_openalex_augmented / skip_incomplete_window)." % disp_csv)
    ijk_cols = [f'I_count_y{off}', f'J_count_y{off}', f'K_count_y{off}']
    df_ijk = pd.read_csv(disp_csv, usecols=['pid'] + ijk_cols)
    print("loaded I/J/K counts:", os.path.basename(disp_csv), "rows:", len(df_ijk))

    df = df.merge(df_num, on='pid', how='left').merge(df_ijk, on='pid', how='left')
    del df_num, df_ijk
    denom_ck = df[ijk_cols[0]] + df[ijk_cols[1]] + df[ijk_cols[2]]     # C + K = I + J + K
    print("papers with a disruption record: %d / %d" % (int(denom_ck.notna().sum()), len(df)))
    for m in over_ck_models:
        df[focal_col[m]] = (df[METRIC_SPEC[m]['num']] / denom_ck).where(denom_ck > 0)
        print("  %-12s = %s / (I+J+K): non-NaN %d | %s"
              % (focal_col[m], METRIC_SPEC[m]['num'], int(df[focal_col[m]].notna().sum()),
                 df[focal_col[m]].describe().to_dict()))
    df.drop(columns=ijk_cols + num_cols, inplace=True)

if REST_UNIVERSE == 'scored':
    # central vs non-central WITHIN the ForeCite-scored papers: every central paper scores some
    # above-threshold keyword by construction, so this only trims the 0 class.
    n_before = len(df)
    df = df[df['pid'].isin(scored_pids)].copy()
    print("REST_UNIVERSE='scored': %d -> %d papers" % (n_before, len(df)))
else:
    assert REST_UNIVERSE == 'all_efg', REST_UNIVERSE

df = df.dropna(subset=['pub_year', 'up_refer_count', citation_col])
df['pub_year'] = df['pub_year'].astype(int)
if MIN_PUB_YEAR is not None:
    df = df[df['pub_year'] >= MIN_PUB_YEAR]
if MAX_PUB_YEAR is not None:
    df = df[df['pub_year'] <= MAX_PUB_YEAR]
print("analysis frame after year / control filters:", len(df), "papers")

# ---- the two log controls ----
df['log_up_refer_count'] = np.log1p(df['up_refer_count'])
df['log_citation_count'] = np.log1p(df[citation_col])

rng = np.random.default_rng(RANDOM_SEED)

# ===========================================================================
# Regressions: the whole models_spec group, once per central-paper definition
# ===========================================================================
results_rows = []
for def_name, def_desc in tqdm(DEFINITIONS, desc='definitions', position=0):
    print("\n" + "=" * 70)
    print("definition: %s  (%s)" % (def_name, def_desc))

    d = df.copy()
    d['is_central'] = d['pid'].isin(central_sets[def_name]).astype(int)
    n_central_all = int(d['is_central'].sum())
    print("  central papers present in the efg frame: %d / %d"
          % (n_central_all, len(central_sets[def_name])))
    if n_central_all == 0:
        print("  WARNING: no central paper of this definition survives the filters, skipping.")
        continue

    # Drop pub_years with NO central paper: their year dummy predicts the outcome perfectly
    # (quasi-complete separation), which breaks logit convergence. Every central paper is kept.
    years_with_central = set(d.loc[d['is_central'] == 1, 'pub_year'])
    n_years_before = d['pub_year'].nunique()
    d = d[d['pub_year'].isin(years_with_central)].copy()
    print("  pub_years kept %d/%d (dropped %d with no central paper); rows now %d"
          % (d['pub_year'].nunique(), n_years_before,
             n_years_before - d['pub_year'].nunique(), len(d)))

    # Optional case-control subsampling of the 0 class (all central papers always kept).
    if NEG_PER_POS is not None:
        pos_idx = d.index[d['is_central'] == 1].to_numpy()
        neg_idx = d.index[d['is_central'] == 0].to_numpy()
        keep_n = min(len(neg_idx), NEG_PER_POS * n_central_all)
        if keep_n < len(neg_idx):
            keep = rng.choice(neg_idx, size=keep_n, replace=False)
            d = d.loc[np.concatenate([pos_idx, keep])].copy()
            print("  case-control subsample: kept %d non-central + %d central = %d rows"
                  % (keep_n, len(pos_idx), len(d)))

    for model_name, formula in tqdm(models_spec, desc='  [%s] regressions' % def_name,
                                    position=1, leave=False):
        fcol = focal_col[model_name]
        scaled = f'scaled_{model_name}'
        m_csv = model_csv_path(def_name, model_name)
        if not OVERWRITE and os.path.exists(m_csv):
            # already fitted in an earlier run -> read it back so the combined CSV stays complete
            results_rows.extend(pd.read_csv(m_csv).to_dict('records'))
            print("  [%-17s] already on disk -> skipped (OVERWRITE=True to refit): %s"
                  % (model_name, os.path.basename(m_csv)))
            continue
        dm = d.dropna(subset=[fcol, 'log_up_refer_count', 'log_citation_count', 'pub_year'])
        if int(dm['is_central'].sum()) == 0:
            print("  [%s] no central paper left after dropping NaN %s, skipping."
                  % (model_name, fcol))
            continue
        # NaN in the focal index can empty a year of its central papers again -> re-drop those.
        years_ok = set(dm.loc[dm['is_central'] == 1, 'pub_year'])
        dm = dm[dm['pub_year'].isin(years_ok)].copy()
        n_central_model = int(dm['is_central'].sum())

        # scaled(index): standardise over THIS model's estimation sample -- the rows above are
        # exactly the ones the logit will use -- so the coefficient is per one SD of the index
        # among those papers. mean/sd are recorded in the output CSV.
        mu, sd = dm[fcol].mean(), dm[fcol].std()
        assert sd > 0, "%s is constant in the %s sample" % (fcol, model_name)
        dm[scaled] = (dm[fcol] - mu) / sd

        try:
            result = smf.logit(formula, data=dm).fit(
                disp=False, method='newton', maxiter=1000, warn_convergence=True)
        except Exception as e:
            print("  [%s] newton failed (%s), retrying with bfgs ..." % (model_name, e))
            result = smf.logit(formula, data=dm).fit(
                disp=False, method='bfgs', maxiter=1000, warn_convergence=True)

        params, bse = result.params, result.bse
        converged = result.mle_retvals.get('converged', None)
        print("  [%-17s] coef=%+.4f (se %.4f, p=%.3g)  scale mean=%.6f sd=%.6f  n_obs=%d  "
              "n_central=%d  n_non_central=%d  converged=%s  pseudo_R2=%.4f"
              % (model_name, params[scaled], bse[scaled], result.pvalues[scaled], mu, sd,
                 int(result.nobs), n_central_model, int(result.nobs) - n_central_model,
                 converged, result.prsquared))

        # the C(pub_year) dummies are the fixed effect -- not written out
        keep_vars = [v for v in params.index if not v.startswith('C(pub_year)')]
        model_rows = []
        for var in keep_vars:
            model_rows.append({
                'definition':   def_name,
                'description':  def_desc,
                'model':        model_name,
                'index_meaning': METRIC_SPEC[model_name]['desc'],
                'focal_column': fcol,
                'formula':      formula,
                'variable':     var,
                'is_focal':     int(var == scaled),
                'coef':         params[var],
                'std_err':      bse[var],
                'z_stat':       result.tvalues[var],
                'p_value':      result.pvalues[var],
                'ci_2_5':       params[var] - 1.96 * bse[var],
                'ci_97_5':      params[var] + 1.96 * bse[var],
                'odds_ratio':   float(np.exp(params[var])),
                'scale_mean':   mu,
                'scale_sd':     sd,
                # sample sizes of THIS fit: n_obs = total, n_central = positives (is_central==1),
                # n_non_central = the 0 class. n_frame_* are the counts BEFORE this definition's
                # year drop and this model's NaN drop, so the shrinkage is visible per row.
                'n_obs':          int(result.nobs),
                'n_central':      n_central_model,
                'n_non_central':  int(result.nobs) - n_central_model,
                'central_rate':   n_central_model / float(result.nobs),
                'n_frame_rows':   len(df),
                'n_frame_central': n_central_all,
                'pseudo_r2':    result.prsquared,
                'aic':          result.aic,
                'bic':          result.bic,
                'converged':    converged,
                'rest_universe': REST_UNIVERSE,
                'forecite_threshold': FORECITE_THRESHOLD,
                'n_keywords':   n_keywords,
            })

        # Save after EVERY model: each fit over the full table is slow, so a run that is killed
        # (or crashes on a later model) keeps everything finished so far -- and with
        # OVERWRITE=False the next run skips exactly those models. The combined CSV is rewritten
        # alongside the per-model one; both are tiny next to the fit itself.
        pd.DataFrame(model_rows).to_csv(m_csv, index=False)
        results_rows.extend(model_rows)
        pd.DataFrame(results_rows).to_csv(all_csv, index=False)
        print("       saved %d rows -> %s" % (len(model_rows), os.path.basename(m_csv)))

assert results_rows, "no model was fitted -- check the central sets and the efg table filters"
pd.DataFrame(results_rows).to_csv(all_csv, index=False)
n_models = len({(r['definition'], r['model']) for r in results_rows})
print("\n%d per-model CSVs in %s" % (n_models, save_dir))
print("combined coefficients (%d rows): %s" % (len(results_rows), all_csv))
print("finished")
