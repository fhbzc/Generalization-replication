"""
Per-term frequency + average E/F/G/D-style quantiles of the papers that use the term.

Over the SAME corpus the co-appearance word2vec is trained on (train_title_word2vec_coappearance.py):
title tokens of papers that (a) pass compute_efgd_quantile.py's rt/ct filters, i.e. appear in its
output CSV, and (b) are published before `year_cutoff`. For every title token t ("term") we compute:

    tf(t)   = total occurrences of t across the corpus titles
    df(t)   = number of papers (titles) that contain t

and, averaged over the papers that CONTAIN t (skipping papers whose value is NaN):
    avg_e_quantile      : mean of  e_y{off}_quantile{kind}                (E index, extension)
    avg_f_quantile      : mean of  f_y{off}_quantile{kind}
    avg_g_quantile      : mean of  g_y{off}_quantile{kind}
    avg_disp_quantile     : mean of  disp_y{off}_quantile{kind}          (disruption index D)
    avg_disp_f_quantile   : mean of  disruption_f_y{off}_quantile{kind}  (disruptive-foundation share)
    avg_ij_ratio_quantile : mean of  ij_ratio_y{off}_quantile{kind}      (R = I/(I+J), disruptive-citation fraction)

`quantile_kind` picks which quantile family from compute_efgd_quantile.py:
    ''          -> within (pub_year, top_domain)   (NaN for no-/multi-domain papers)
    '_by_year'  -> within pub_year only            (defined for ALL papers)   [default]
    '_global'   -> global, no grouping             (defined for ALL papers)

Input  : META_DIRE/pid2eng_token_list_{parse_method}_before_{year_before}.pkl  (parse_title_to_token.py)
         META_DIRE/pid2pub_year.pkl
         META_DIRE/efg_computation/efgd_within_year_domain_quantile_y{off}{data_suffix}_rt_{rt}_ct_{ct}_ybefore_{max_year}{name_tag}.csv
             (compute_efgd_quantile.py; needs use_disp_f=True for the disp_f quantile column)
Output : META_DIRE/term_freq_disp_quantile/
             term_freq_disp_quantile_before_{year_cutoff}_{quantile_kind}_rt_{rt}_ct_{ct}.csv
             term_freq_disp_quantile_before_{year_cutoff}_{quantile_kind}_rt_{rt}_ct_{ct}.pkl  (term -> stats dict)
"""

import os
import sys
import csv
import pickle
from collections import defaultdict

from tqdm import tqdm
import pandas as pd
import nltk
from nltk.corpus import stopwords

# utils.py lives three levels up (openalex/); add it to the path regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import META_DIRE


def get_english_stopwords():
    try:
        return set(stopwords.words('english'))
    except LookupError:
        print("NLTK stopwords corpus not found, downloading ...")
        nltk.download('stopwords')
        return set(stopwords.words('english'))


stop_words_set = get_english_stopwords()

meta_dire = META_DIRE
print("meta_dire", meta_dire)

# ---------------------------------------------------------------------------
# Configuration (keep aligned with train_title_word2vec_coappearance.py)
# ---------------------------------------------------------------------------
year_cutoff = 2020            # only papers published strictly before this year; CLI override below
if len(sys.argv) > 1:
    year_cutoff = int(sys.argv[1])

year_before = 2025            # selects the parse_title_to_token.py token dump file
parse_method = 'lemma'        # must match parse_title_to_token.py's method_string

MIN_DF = 1                    # drop terms occurring in fewer than this many papers (1 = keep all)

# which quantile family from compute_efgd_quantile.py: '' (year,domain) | '_by_year' | '_global'
quantile_kind = ''

# ---- efgd quantile CSV selection (must match compute_efgd_quantile.py) ----
efgd_year_offset = 5
efgd_reference_threshold = 1   # rt
efgd_citation_threshold = 5    # ct
efgd_max_year = 2025
efgd_name_tag = '_full_cover'  # '' complete-window table, '_full_cover' all-before-2025
efgd_data_suffix = ''          # '_with_review' if compute_efgd_quantile.py ran with with_review=True

off = efgd_year_offset
# base index column -> readable metric name; quantile column = f'{base}_quantile{quantile_kind}'
metrics = {
    'e':        f'e_y{off}',           # E index (extension)
    'f':        f'f_y{off}',
    'g':        f'g_y{off}',
    'disp':     f'disp_y{off}',
    'disp_f':   f'disruption_f_y{off}',
    'ij_ratio': f'ij_ratio_y{off}',   # R = I/(I+J), disruptive-citation fraction
}
qcol = {m: f'{base}_quantile{quantile_kind}' for m, base in metrics.items()}

print("year_cutoff", year_cutoff, "| year_before", year_before, "| parse_method", parse_method)
print("quantile_kind", quantile_kind or '(year,domain)', "| MIN_DF", MIN_DF)
print("quantile columns:", qcol)

# ---------------------------------------------------------------------------
# Load inputs
# ---------------------------------------------------------------------------
with open(os.path.join(meta_dire, 'pid2eng_token_list_%s_before_%s.pkl' % (parse_method, year_before)), 'rb') as f:
    pid2token_list = pickle.load(f)
print("load pid2token_list finished, size", len(pid2token_list))

with open(os.path.join(meta_dire, 'pid2pub_year.pkl'), 'rb') as f:
    pid2pub_year = pickle.load(f)
print("load pid2pub_year finished, size", len(pid2pub_year))

efgd_csv = os.path.join(
    meta_dire, 'efg_computation',
    'efgd_within_year_domain_quantile_y%s%s_rt_%s_ct_%s_ybefore_%s%s.csv'
    % (off, efgd_data_suffix, efgd_reference_threshold,
       efgd_citation_threshold, efgd_max_year, efgd_name_tag))
assert os.path.exists(efgd_csv), "%s\nRun compute_efgd_quantile.py first." % efgd_csv
qdf = pd.read_csv(efgd_csv, usecols=['pid'] + list(qcol.values()))
qdf = qdf.drop_duplicates('pid').set_index('pid')
# pid -> quantile value (float, may be NaN) for each metric
qval = {m: qdf[qcol[m]].to_dict() for m in metrics}
efgd_pids = set(qdf.index)
print("load efgd quantiles finished, papers with quantile:", len(efgd_pids))

# ---------------------------------------------------------------------------
# One pass over the corpus: accumulate per-term tf, df, and per-metric quantile sums.
# A paper contributes to a term's df / quantile sums at most once (distinct tokens).
# ---------------------------------------------------------------------------
term_tf = defaultdict(int)                                   # term -> total occurrences
term_df = defaultdict(int)                                   # term -> #papers containing it
term_qsum = {m: defaultdict(float) for m in metrics}        # metric -> term -> sum of quantiles
term_qcnt = {m: defaultdict(int) for m in metrics}          # metric -> term -> #papers with a non-NaN value

n_papers = 0
for pid, token_list in tqdm(pid2token_list.items(), desc='count terms'):
    pub_year = pid2pub_year.get(pid)
    if pub_year is None or pub_year >= year_cutoff:
        continue
    if pid not in efgd_pids:                                 # keep only rt/ct-passing (has-quantile) papers
        continue

    valid_tokens = [tok for tok in token_list
                    if tok not in stop_words_set and len(tok) > 1]
    if len(valid_tokens) < 2:                                # match the training corpus (>=2 co-appearing tokens)
        continue

    n_papers += 1
    for tok in valid_tokens:                                 # tf counts every occurrence
        term_tf[tok] += 1

    this_vals = {m: qval[m].get(pid) for m in metrics}
    for tok in set(valid_tokens):                            # df / quantile: once per paper per term
        term_df[tok] += 1
        for m in metrics:
            v = this_vals[m]
            if v is not None and v == v:                     # v == v is False only for NaN
                term_qsum[m][tok] += v
                term_qcnt[m][tok] += 1

print("corpus papers used:", n_papers, "| distinct terms:", len(term_df))
assert n_papers > 0, "no papers matched; check year_cutoff / efgd CSV / pid types"

# ---------------------------------------------------------------------------
# Assemble per-term rows: frequency + average quantiles
# ---------------------------------------------------------------------------
rows = []
for term, df_t in term_df.items():
    if df_t < MIN_DF:
        continue
    row = {
        'term': term,
        'tf': term_tf[term],   # total occurrences
        'df': df_t,            # #papers containing the term
    }
    for m in metrics:
        c = term_qcnt[m][term]
        row['avg_%s_quantile' % m] = (term_qsum[m][term] / c) if c > 0 else float('nan')
        row['n_%s_quantile' % m] = c
    rows.append(row)

rows.sort(key=lambda r: r['df'], reverse=True)   # most frequent terms first
print("terms kept (df >= %d): %d" % (MIN_DF, len(rows)))

# ---------------------------------------------------------------------------
# Save CSV + pickle
# ---------------------------------------------------------------------------
save_dir = os.path.join(meta_dire, 'term_freq_disp_quantile')
os.makedirs(save_dir, exist_ok=True)
kind_tag = quantile_kind if quantile_kind else '_year_domain'
base_name = ('term_freq_disp_quantile_before_%s_%s_rt_%s_ct_%s'
             % (year_cutoff, kind_tag.lstrip('_'), efgd_reference_threshold, efgd_citation_threshold))

fieldnames = (['term', 'tf', 'df']
              + ['avg_%s_quantile' % m for m in metrics]
              + ['n_%s_quantile' % m for m in metrics])
csv_path = os.path.join(save_dir, base_name + '.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow(r)
print("saved:", csv_path)

pkl_path = os.path.join(save_dir, base_name + '.pkl')
with open(pkl_path, 'wb') as f:
    pickle.dump({r['term']: r for r in rows}, f, pickle.HIGHEST_PROTOCOL)
print("saved:", pkl_path)
print("finished")
