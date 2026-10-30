"""
Project every word onto the disruption dimension axes and relate those projections to
three outcome quantiles: Foundation (F), Disruptive-foundation (disp_f) and I/(I+J).

Dimensions (SAME word groups as plot_word_cloud_umap_disruption.py)
------------------------------------------------------------------
    dimension 1 = A - B : every (a, b) in GROUP_A x GROUP_B  -> a direction unit(vec(a)-vec(b))
    dimension 2 = C - D : every (c, d) in GROUP_C x GROUP_D
Each product pair is one "direction". A word's coordinate on a direction is the cosine of
its unit vector with the (unit) direction. The joint-model figures iterate every
(dimension-1 direction, dimension-2 direction) combination.

Figures produced (nothing else)
-------------------------------
1. average_r_by_dimension:            grouped bar, x = the 3 outcome indices, 2 bars each =
   the MEAN Pearson r (over that dimension's directions) for dimension 1 and dimension 2.
2. average_scaled_effect_by_dimension: grouped bar, x = the 3 outcomes, 2 bars each = the
   MEAN standardized OLS coefficient (dim1 and dim2 z-scored, jointly regressed on the
   outcome), averaged over every (dimension-1 direction, dimension-2 direction) combination.
3. per_dimension_r/ (subdir):         one figure per direction (both dimensions) -- its r and
   95% CI against each of the 3 outcomes.
4. per_combination_scaled_effect/ (subdir): one figure per (dim1 direction, dim2 direction)
   combination -- the two standardized coefficients (dim1, dim2) and their 95% CI from the
   joint model, for each of the 3 outcomes.

The per-word projections + outcome values are cached (keyed by the groups / filters / model),
so a re-run redraws the figures without reloading the model or re-projecting the vocabulary.

Inputs (mirror train_title_word2vec_coappearance.py / compute_term_freq_disp_quantile.py):
    META_DIRE/text_embedding_coappearance/word_embedding_coappearance_before_{year_cutoff}_dim_{dim}_
        min_count_{mc}_epoch_{ep}_seed_{seed}{efgd_tag}.model
    META_DIRE/term_freq_disp_quantile/term_freq_disp_quantile_before_{year_cutoff}_{kind}_rt_{rt}_ct_{ct}.pkl
Output:
    ./figures/axis_projection_disruption/<param_subdir>/  (figures 1-2 + the two subdirs + CSVs + cache)
"""

import os
import sys
import pickle
import itertools

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec

# utils.py lives three levels up (openalex/); add it to the path regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import META_DIRE

meta_dire = META_DIRE
print("meta_dire", meta_dire)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
year_cutoff = 2020                # papers/titles before this year; CLI override below
if len(sys.argv) > 1:
    year_cutoff = int(sys.argv[1])

# ---- word groups -> two dimensions (SAME as plot_word_cloud_umap_disruption.py) ----
# dimension 1 = A - B (every a in A vs every b in B), dimension 2 = C - D.
GROUP_A = ('software', 'device', 'tool', 'technique')            # dimension 1 positive end
GROUP_B = ('theory', 'hypothesis', 'model')                     # dimension 1 negative end
GROUP_C = ('introduction', 'creation', 'create', 'introduce')   # dimension 2 positive end
GROUP_D = ('refinement', 'extension', 'refine', 'extend')       # dimension 2 negative end

# the three outcome quantiles: (key, per-term-table column, display label)
OUTCOMES = [
    ('foundation', 'avg_f_quantile',        'Foundation'),
    ('disp_f',     'avg_disp_f_quantile',   'Disruptive\nfoundation'),
    ('ij_ratio',   'avg_ij_ratio_quantile', 'I/(I+J)'),
]

# words to include: in the embedding vocab, non-stopword, df >= MIN_DF, tf > MIN_FREQ,
# and a non-NaN value for ALL three outcomes (so every figure shares one sample).
MIN_DF = 20
MIN_FREQ = 800

# colours: dimension 1 red, dimension 2 blue (matches the word-cloud figure)
DIM1_COLOR = '#d62728'
DIM2_COLOR = '#1f5fd6'

FORCE_RECOMPUTE = False           # ignore the projection cache and rebuild it

# ---- locate the word2vec model (must match train_title_word2vec_coappearance.py) ----
embed_size = 128
min_count = 0
epochs = 3
seed = 100
restrict_to_efgd = True
quantile_kind = ''            # '' (year,domain) | '_by_year' | '_global'
reference_threshold = 1       # rt
citation_threshold = 5        # ct
with_review = False
data_suffix = '_with_review' if with_review else ''
efgd_tag = ('_efgd%s_%s_rt%s_ct%s'
            % (data_suffix, quantile_kind.lstrip('_') or 'year_domain',
               reference_threshold, citation_threshold)
            if restrict_to_efgd else '')

model_path = os.path.join(
    meta_dire, 'text_embedding_coappearance',
    'word_embedding_coappearance_before_%s_dim_%s_min_count_%s_epoch_%s_seed_%s%s.model'
    % (year_cutoff, embed_size, min_count, epochs, seed, efgd_tag))

# ---- locate the per-term table (must match compute_term_freq_disp_quantile.py) ----
kind_tag = quantile_kind if quantile_kind else '_year_domain'
term_pkl = os.path.join(
    meta_dire, 'term_freq_disp_quantile',
    'term_freq_disp_quantile_before_%s_%s_rt_%s_ct_%s.pkl'
    % (year_cutoff, kind_tag.lstrip('_'), reference_threshold, citation_threshold))


def _as_word_tuple(g):
    return (g,) if isinstance(g, str) else tuple(g)


# each dimension's directions = the pairwise product (pos, neg) of its two groups
axis1_pairs = list(dict.fromkeys(itertools.product(_as_word_tuple(GROUP_A), _as_word_tuple(GROUP_B))))
axis2_pairs = list(dict.fromkeys(itertools.product(_as_word_tuple(GROUP_C), _as_word_tuple(GROUP_D))))

print("year_cutoff", year_cutoff)
print("MIN_DF", MIN_DF, "| MIN_FREQ", MIN_FREQ)
print("dimension 1 directions (A x B, %d):" % len(axis1_pairs), axis1_pairs)
print("dimension 2 directions (C x D, %d):" % len(axis2_pairs), axis2_pairs)
print("outcomes:", [o[0] for o in OUTCOMES])

# ---------------------------------------------------------------------------
# t critical value for the 95% CI band (scipy if present, else the normal approx)
# ---------------------------------------------------------------------------
try:
    from scipy import stats as _scipy_stats

    def _t_ppf(p, dof):
        return float(_scipy_stats.t.ppf(p, dof))
except Exception:
    def _t_ppf(p, dof):
        return 1.96


def _zscore(a):
    """Standardise to mean 0, sd 1 (population sd); constant input -> centred zeros."""
    a = np.asarray(a, dtype=float)
    sd = a.std(ddof=0)
    return (a - a.mean()) / sd if sd > 0 else (a - a.mean())


def ols_fit(X, y):
    """OLS of y on X (n x k, WITHOUT intercept -- one is added). Returns (coef, se, r2, n)
    with coef/se length k+1 (intercept first). Standard errors use sigma^2 (X'X)^-1."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, dtype=float)
    n = len(y)
    A = np.column_stack([np.ones(n), X])
    k = A.shape[1]
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    sse = float(resid @ resid)
    dof = n - k
    XtX_inv = np.linalg.pinv(A.T @ A)
    sigma2 = sse / dof if dof > 0 else float('nan')
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - sse / ss_tot if ss_tot > 0 else float('nan')
    return beta, se, r2, n


def pearson_r_ci(x, y):
    """Pearson r between x and y with a 95% CI via the Fisher z-transform. Returns (r, lo, hi, n)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float('nan'), float('nan'), float('nan'), n
    r = float(np.corrcoef(x, y)[0, 1])
    if n < 4 or abs(r) >= 1.0:
        return r, float('nan'), float('nan'), n
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    lo = float(np.tanh(z - 1.96 * se))
    hi = float(np.tanh(z + 1.96 * se))
    return r, lo, hi, n


def ols_joint_ci(d1z, d2z, y):
    """Joint OLS: y ~ z(dim1) + z(dim2). Returns (dim1_stats, dim2_stats, r2, n) where each
    *_stats is a dict {coef, se, lo, hi} (95% CI = coef +/- t*se)."""
    beta, se, r2, n = ols_fit(np.column_stack([d1z, d2z]), y)
    dof = n - 3
    tcrit = _t_ppf(0.975, dof) if dof > 0 else 1.96
    d1 = {'coef': float(beta[1]), 'se': float(se[1]),
          'lo': float(beta[1] - tcrit * se[1]), 'hi': float(beta[1] + tcrit * se[1])}
    d2 = {'coef': float(beta[2]), 'se': float(se[2]),
          'lo': float(beta[2] - tcrit * se[2]), 'hi': float(beta[2] + tcrit * se[2])}
    return d1, d2, float(r2), n


def _two_hist(ax, v1, v2, l1, l2, c1, c2, xlabel, title):
    """Overlaid histograms of v1 and v2 with dashed mean lines and a reference line at 0."""
    v1 = np.asarray([v for v in v1 if v == v], dtype=float)
    v2 = np.asarray([v for v in v2 if v == v], dtype=float)
    allv = np.concatenate([v1, v2]) if (len(v1) + len(v2)) else np.array([0.0])
    lo, hi = float(np.min(allv)), float(np.max(allv))
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    bins = np.linspace(lo, hi, 20)
    if len(v1):
        ax.hist(v1, bins=bins, color=c1, alpha=0.5, label='%s (mean=%.3f, n=%d)' % (l1, v1.mean(), len(v1)))
        ax.axvline(v1.mean(), color=c1, lw=2, ls='--')
    if len(v2):
        ax.hist(v2, bins=bins, color=c2, alpha=0.5, label='%s (mean=%.3f, n=%d)' % (l2, v2.mean(), len(v2)))
        ax.axvline(v2.mean(), color=c2, lw=2, ls='--')
    ax.axvline(0, color='grey', lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('count')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Output layout
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
param_subdir = ('before_%s_%s_rt_%s_ct_%s_mindf_%s_minfreq_%s'
                % (year_cutoff, kind_tag.lstrip('_'), reference_threshold,
                   citation_threshold, MIN_DF, MIN_FREQ))
save_dir = os.path.join(BASE_DIR, 'figures', 'axis_projection_disruption', param_subdir)
os.makedirs(save_dir, exist_ok=True)
per_dim_dir = os.path.join(save_dir, 'per_dimension_r')
per_combo_dir = os.path.join(save_dir, 'per_combination_scaled_effect')
os.makedirs(per_dim_dir, exist_ok=True)
os.makedirs(per_combo_dir, exist_ok=True)
cache_path = os.path.join(save_dir, 'projection_cache.pkl')
print("save_dir", save_dir)

# config the cache must match to be reusable
cache_key = {'A': tuple(GROUP_A), 'B': tuple(GROUP_B), 'C': tuple(GROUP_C), 'D': tuple(GROUP_D),
             'outcomes': tuple(c for _, c, _ in OUTCOMES), 'MIN_DF': MIN_DF, 'MIN_FREQ': MIN_FREQ,
             'year_cutoff': year_cutoff, 'model_path': model_path, 'term_pkl': term_pkl}


def build_projection_cache():
    """Load the model + per-term table, build the eligible-word sample, and project it onto
    every dimension direction. Returns the dict that gets pickled to cache_path."""
    assert os.path.exists(model_path), (
        "%s\nRun train_title_word2vec_coappearance.py first." % model_path)
    model = Word2Vec.load(model_path)
    wv = model.wv
    print("loaded model, vocab size", len(wv))

    assert os.path.exists(term_pkl), (
        "%s\nRun compute_term_freq_disp_quantile.py first." % term_pkl)
    with open(term_pkl, 'rb') as f:
        term2stats = pickle.load(f)
    print("loaded per-term table, terms", len(term2stats))
    sample_stats = next(iter(term2stats.values()))
    for _, col, _ in OUTCOMES:
        assert col in sample_stats, "outcome column %r not in per-term table %s" % (col, list(sample_stats))

    try:
        stop_words_set = set(stopwords.words('english'))
    except LookupError:
        print("NLTK stopwords corpus not found, downloading ...")
        nltk.download('stopwords')
        stop_words_set = set(stopwords.words('english'))

    outcome_cols = [c for _, c, _ in OUTCOMES]
    words = []
    outcome_vals = {c: [] for c in outcome_cols}
    for word, stats in term2stats.items():
        if word not in wv or word in stop_words_set:
            continue
        if stats.get('df', 0) < MIN_DF or stats.get('tf', 0) <= MIN_FREQ:
            continue
        vals = [stats.get(c) for c in outcome_cols]
        if any(v is None or v != v for v in vals):        # require all outcomes present
            continue
        words.append(word)
        for c, v in zip(outcome_cols, vals):
            outcome_vals[c].append(float(v))
    assert words, "no eligible words (loosen MIN_DF / MIN_FREQ or check outcome columns)"
    print("eligible words:", len(words))

    outcomes = {c: np.asarray(outcome_vals[c], dtype=float) for c in outcome_cols}

    # unit-vector matrix for the eligible words, then project onto each direction
    U = np.vstack([wv.get_vector(w, norm=True) for w in words]).astype(np.float64)

    def build_dirs(pairs, group_name):
        out = []
        for pos, neg in pairs:
            if pos not in wv or neg not in wv:
                print("  [skip %s direction] %s-%s (missing word)" % (group_name, pos, neg))
                continue
            d = wv.get_vector(pos, norm=True) - wv.get_vector(neg, norm=True)
            nrm = np.linalg.norm(d)
            if nrm == 0:
                print("  [skip %s direction] %s-%s (identical vectors)" % (group_name, pos, neg))
                continue
            proj = (U @ (d / nrm)).astype(float)
            out.append({'label': '%s-%s' % (pos, neg), 'pair': (pos, neg), 'proj': proj})
        return out

    dim1 = build_dirs(axis1_pairs, 'dim1')
    dim2 = build_dirs(axis2_pairs, 'dim2')
    print("built directions: dim1=%d, dim2=%d" % (len(dim1), len(dim2)))

    return {'key': cache_key, 'words': words, 'outcomes': outcomes, 'dim1': dim1, 'dim2': dim2}


# ---- load cache or (re)build it ----
cache = None
if not FORCE_RECOMPUTE and os.path.exists(cache_path):
    with open(cache_path, 'rb') as f:
        loaded = pickle.load(f)
    if loaded.get('key') == cache_key:
        cache = loaded
        print("loaded projection cache:", cache_path, "(%d words)" % len(cache['words']))
    else:
        print("cache config changed -> rebuilding")
if cache is None:
    cache = build_projection_cache()
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f, pickle.HIGHEST_PROTOCOL)
    print("saved projection cache:", cache_path)

words = cache['words']
outcomes = cache['outcomes']
dim1_dirs = cache['dim1']
dim2_dirs = cache['dim2']
outcome_keys = [k for k, _, _ in OUTCOMES]
outcome_cols = {k: c for k, c, _ in OUTCOMES}
outcome_labels = {k: lab for k, _, lab in OUTCOMES}

# ===========================================================================
# Per-direction Pearson r (+95% CI) against each outcome  -> subdir figures + CSV
# ===========================================================================
per_dim_rows = []


def per_direction_figure(direction, dim_name, color, out_dir):
    """One figure: this direction's r (+95% CI) against each of the 3 outcomes."""
    proj = direction['proj']
    xs = np.arange(len(OUTCOMES))
    rs, los, his = [], [], []
    for k in outcome_keys:
        r, lo, hi, n = pearson_r_ci(proj, outcomes[outcome_cols[k]])
        rs.append(r)
        los.append(r - lo if lo == lo else 0.0)
        his.append(hi - r if hi == hi else 0.0)
        per_dim_rows.append({'dimension': dim_name, 'direction': direction['label'],
                             'outcome': k, 'r': r, 'ci_lo': lo, 'ci_hi': hi, 'n': n})
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(xs, rs, yerr=[los, his], fmt='o', color=color, capsize=5,
                markersize=8, lw=1.5)
    ax.axhline(0, color='grey', lw=0.8, ls='--')
    ax.set_xticks(xs)
    ax.set_xticklabels([outcome_labels[k] for k in outcome_keys])
    ax.set_ylabel('Pearson r (95% CI)')
    ax.set_title('%s  |  %s' % (dim_name, direction['label']))
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fname = '%s__%s.svg' % (dim_name, direction['label'].replace('/', '_'))
    fig.savefig(os.path.join(out_dir, fname), format='svg', bbox_inches='tight')
    plt.close(fig)


print("drawing per-direction r figures ...")
for d in dim1_dirs:
    per_direction_figure(d, 'dimension1', DIM1_COLOR, per_dim_dir)
for d in dim2_dirs:
    per_direction_figure(d, 'dimension2', DIM2_COLOR, per_dim_dir)
pd.DataFrame(per_dim_rows).to_csv(os.path.join(save_dir, 'per_dimension_r.csv'), index=False)
print("  saved %d per-direction figures + per_dimension_r.csv" % (len(dim1_dirs) + len(dim2_dirs)))

# ===========================================================================
# FIGURE 1: mean r per dimension (average over that dimension's directions)
# ===========================================================================
def mean_r_for_dim(dirs):
    """mean over directions of Pearson r vs each outcome -> {outcome_key: mean_r}."""
    out = {}
    for k in outcome_keys:
        y = outcomes[outcome_cols[k]]
        rs = [pearson_r_ci(d['proj'], y)[0] for d in dirs]
        rs = [r for r in rs if r == r]
        out[k] = float(np.mean(rs)) if rs else float('nan')
    return out


mean_r_dim1 = mean_r_for_dim(dim1_dirs)
mean_r_dim2 = mean_r_for_dim(dim2_dirs)

xs = np.arange(len(OUTCOMES))
w = 0.38
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.bar(xs - w / 2, [mean_r_dim1[k] for k in outcome_keys], w, color=DIM1_COLOR, label='dimension 1 (A-B)')
ax.bar(xs + w / 2, [mean_r_dim2[k] for k in outcome_keys], w, color=DIM2_COLOR, label='dimension 2 (C-D)')
ax.axhline(0, color='grey', lw=0.8)
ax.set_xticks(xs)
ax.set_xticklabels([outcome_labels[k] for k in outcome_keys])
ax.set_ylabel('mean Pearson r (over directions)')
ax.set_title('Average correlation of each dimension with the outcome quantiles (before %s)' % year_cutoff)
ax.legend()
ax.grid(True, axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(save_dir, 'average_r_by_dimension.svg'), format='svg', bbox_inches='tight')
fig.savefig(os.path.join(save_dir, 'average_r_by_dimension.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
pd.DataFrame([{'outcome': k, 'mean_r_dim1': mean_r_dim1[k], 'mean_r_dim2': mean_r_dim2[k],
               'n_dim1_directions': len(dim1_dirs), 'n_dim2_directions': len(dim2_dirs)}
              for k in outcome_keys]).to_csv(
    os.path.join(save_dir, 'average_r_by_dimension.csv'), index=False)
print("saved: average_r_by_dimension.{svg,png,csv}")

# ===========================================================================
# Per-(dim1, dim2) joint model: standardized effect (+95% CI)  -> subdir + CSV
# FIGURE 2: mean standardized effect per dimension (average over all combinations)
# ===========================================================================
per_combo_rows = []
# accumulate the coefficients to average for figure 2
coef_accum = {k: {'dim1': [], 'dim2': []} for k in outcome_keys}

combos = list(itertools.product(dim1_dirs, dim2_dirs))
print("drawing per-combination scaled-effect figures (%d combinations) ..." % len(combos))
for d1, d2 in combos:
    d1z = _zscore(d1['proj'])
    d2z = _zscore(d2['proj'])
    xs = np.arange(len(OUTCOMES))
    c1s, c1lo, c1hi, c2s, c2lo, c2hi = [], [], [], [], [], []
    for k in outcome_keys:
        y = outcomes[outcome_cols[k]]
        s1, s2, r2, n = ols_joint_ci(d1z, d2z, y)
        c1s.append(s1['coef']); c1lo.append(s1['coef'] - s1['lo']); c1hi.append(s1['hi'] - s1['coef'])
        c2s.append(s2['coef']); c2lo.append(s2['coef'] - s2['lo']); c2hi.append(s2['hi'] - s2['coef'])
        coef_accum[k]['dim1'].append(s1['coef'])
        coef_accum[k]['dim2'].append(s2['coef'])
        per_combo_rows.append({'dim1_direction': d1['label'], 'dim2_direction': d2['label'],
                               'outcome': k, 'coef_dim1': s1['coef'], 'ci_lo_dim1': s1['lo'],
                               'ci_hi_dim1': s1['hi'], 'coef_dim2': s2['coef'], 'ci_lo_dim2': s2['lo'],
                               'ci_hi_dim2': s2['hi'], 'r2': r2, 'n': n})

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    off = 0.12
    ax.errorbar(xs - off, c1s, yerr=[c1lo, c1hi], fmt='o', color=DIM1_COLOR, capsize=5,
                markersize=7, lw=1.5, label='dimension 1: %s' % d1['label'])
    ax.errorbar(xs + off, c2s, yerr=[c2lo, c2hi], fmt='s', color=DIM2_COLOR, capsize=5,
                markersize=7, lw=1.5, label='dimension 2: %s' % d2['label'])
    ax.axhline(0, color='grey', lw=0.8, ls='--')
    ax.set_xticks(xs)
    ax.set_xticklabels([outcome_labels[k] for k in outcome_keys])
    ax.set_ylabel('standardized coefficient (95% CI)')
    ax.set_title('joint model  |  %s  +  %s' % (d1['label'], d2['label']), fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fname = 'd1_%s__d2_%s.svg' % (d1['label'].replace('/', '_'), d2['label'].replace('/', '_'))
    fig.savefig(os.path.join(per_combo_dir, fname), format='svg', bbox_inches='tight')
    plt.close(fig)

pd.DataFrame(per_combo_rows).to_csv(
    os.path.join(save_dir, 'per_combination_scaled_effect.csv'), index=False)
print("  saved %d per-combination figures + per_combination_scaled_effect.csv" % len(combos))

# figure 2: mean standardized effect per dimension (over all combinations)
mean_effect_dim1 = {k: float(np.mean(coef_accum[k]['dim1'])) if coef_accum[k]['dim1'] else float('nan')
                    for k in outcome_keys}
mean_effect_dim2 = {k: float(np.mean(coef_accum[k]['dim2'])) if coef_accum[k]['dim2'] else float('nan')
                    for k in outcome_keys}

xs = np.arange(len(OUTCOMES))
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.bar(xs - w / 2, [mean_effect_dim1[k] for k in outcome_keys], w, color=DIM1_COLOR, label='dimension 1 (A-B)')
ax.bar(xs + w / 2, [mean_effect_dim2[k] for k in outcome_keys], w, color=DIM2_COLOR, label='dimension 2 (C-D)')
ax.axhline(0, color='grey', lw=0.8)
ax.set_xticks(xs)
ax.set_xticklabels([outcome_labels[k] for k in outcome_keys])
ax.set_ylabel('mean standardized coefficient (joint model)')
ax.set_title('Average scaled effect of each dimension on the outcome quantiles (before %s)' % year_cutoff)
ax.legend()
ax.grid(True, axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(save_dir, 'average_scaled_effect_by_dimension.svg'), format='svg', bbox_inches='tight')
fig.savefig(os.path.join(save_dir, 'average_scaled_effect_by_dimension.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
pd.DataFrame([{'outcome': k, 'mean_scaled_effect_dim1': mean_effect_dim1[k],
               'mean_scaled_effect_dim2': mean_effect_dim2[k], 'n_combinations': len(combos)}
              for k in outcome_keys]).to_csv(
    os.path.join(save_dir, 'average_scaled_effect_by_dimension.csv'), index=False)
print("saved: average_scaled_effect_by_dimension.{svg,png,csv}")

# ===========================================================================
# DISTRIBUTION figures: one per outcome, each with (left) the distribution of dimension 1
# vs dimension 2 Pearson r over their directions, and (right) the distribution of the d1
# vs d2 standardized joint-model coefficients over all (dim1, dim2) combinations.
# ===========================================================================
print("drawing distribution figures ...")
for k in outcome_keys:
    label = outcome_labels[k].replace('\n', ' ')
    r_dim1 = [row['r'] for row in per_dim_rows if row['dimension'] == 'dimension1' and row['outcome'] == k]
    r_dim2 = [row['r'] for row in per_dim_rows if row['dimension'] == 'dimension2' and row['outcome'] == k]
    c_dim1 = coef_accum[k]['dim1']
    c_dim2 = coef_accum[k]['dim2']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    _two_hist(axes[0], r_dim1, r_dim2, 'dimension 1 (A-B)', 'dimension 2 (C-D)',
              DIM1_COLOR, DIM2_COLOR, 'Pearson r',
              'r distribution over directions')
    _two_hist(axes[1], c_dim1, c_dim2, 'dimension 1 (d1)', 'dimension 2 (d2)',
              DIM1_COLOR, DIM2_COLOR, 'standardized coefficient',
              'scaled coefficient distribution over (d1, d2) combinations')
    fig.suptitle('%s  (before %s)' % (label, year_cutoff), fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'distribution_%s.svg' % k), format='svg', bbox_inches='tight')
    fig.savefig(os.path.join(save_dir, 'distribution_%s.png' % k), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  saved: distribution_%s.{svg,png}" % k)

print("finished")
