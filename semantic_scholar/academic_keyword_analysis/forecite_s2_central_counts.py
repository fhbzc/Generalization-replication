"""
ForeCite academic-keyword extraction, STAGE 2 of 3.  (see forecite_s1_paper_terms.py header)

Aggregates stage-1 per-year outputs and walks the de-circularized citation network to compute
the two ingredients the ForeCite score needs per term t:
  * term2year2count[t][year]  -- how many papers mentioning t were published each year
        (lets stage 3 compute f_t = #papers mentioning t published from a central paper's year on)
  * term_central2count[t][p]  -- f^p_t : how many papers that mention t cite candidate central
        paper p. Computed edge-wise: for every citation edge q -> p (q cites p), each term shared
        by BOTH q and p (so p is in G_t and q is a mentioning citer of p) increments (t, p).
Also records cid2indegree[p] = #papers citing p (over the same de-circularized network), used by
stage 3 for ForeCite's ">= MIN_CITATIONS citations to be a scorable central paper" rule.

Candidate terms are restricted to those with total document frequency >= MIN_TERM_DF (drops the
long tail of hapax tokens / typos; set MIN_TERM_DF = 1 to keep everything). The citation network
is the temporally-consistent one from get_s2ag_citation_s2_remove_circle.py, so every citer q of p
is published on/after p and thus a genuine "future" mentioning paper.

Input  : META_DIRE/academic_keyword_forecite/ (stage 1)
             term2df_y_<year>_<pmi_tag>.pkl, cid2term_set_y_<year>_<pmi_tag>.pkl
         META_DIRE/citing2cited_cid_list_remove_circle.pkl
             (from get_s2ag_citation_s2_remove_circle.py; q -> [cited p])
Output : META_DIRE/academic_keyword_forecite/  (tag = <pmi_tag>_df_<MIN_TERM_DF>_b_<begin>_e_<finish>)
             term2year2count_<tag>.pkl, term_central2count_<tag>.pkl,
             cid2indegree_<tag>.pkl, candidate_terms_<tag>.pkl
"""

import os
import sys
import pickle
from collections import defaultdict

from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
from utils import META_DIRE

# ---- parameters (must match forecite_s1_paper_terms.py) ----
PMI_bigram = 0
begin_year = 1951
finish_year = 2025          # inclusive
MIN_TERM_DF = 10            # keep only terms mentioned by >= this many papers overall (1 = all)
pmi_tag = f'pmib_{PMI_bigram}'
tag = f'{pmi_tag}_df_{MIN_TERM_DF}_b_{begin_year}_e_{finish_year}'

meta_dire = META_DIRE
kw_dire = os.path.join(meta_dire, 'academic_keyword_forecite')

print('tag', tag)
print('MIN_TERM_DF', MIN_TERM_DF)
print('kw_dire', kw_dire)

# ==================================================
# 1. aggregate per-year term document frequencies -> term2year2count + total df, then pick the
#    candidate terms (total df >= MIN_TERM_DF)
# ==================================================
term2year2count = defaultdict(dict)
term2df_total = defaultdict(int)
for year in tqdm(range(begin_year, finish_year + 1), desc='aggregate df'):
    path = os.path.join(kw_dire, f'term2df_y_{year}_{pmi_tag}.pkl')
    if not os.path.exists(path):
        print(f'  [y{year}] term2df not found, skipping.')
        continue
    with open(path, 'rb') as f:
        term2df = pickle.load(f)
    for t, df in term2df.items():
        term2year2count[t][year] = term2year2count[t].get(year, 0) + df
        term2df_total[t] += df

candidate_terms = {t for t, df in term2df_total.items() if df >= MIN_TERM_DF}
print(f'terms total: {len(term2df_total)}; candidates (df >= {MIN_TERM_DF}): {len(candidate_terms)}')

# keep term2year2count only for candidates
term2year2count = {t: term2year2count[t] for t in candidate_terms}
with open(os.path.join(kw_dire, f'term2year2count_{tag}.pkl'), 'wb') as f:
    pickle.dump(term2year2count, f, pickle.HIGHEST_PROTOCOL)
with open(os.path.join(kw_dire, f'candidate_terms_{tag}.pkl'), 'wb') as f:
    pickle.dump(candidate_terms, f, pickle.HIGHEST_PROTOCOL)

# ==================================================
# 2. load every paper's candidate-term set (restricted to candidate terms)
# ==================================================
cid2terms = {}
for year in tqdm(range(begin_year, finish_year + 1), desc='load term sets'):
    path = os.path.join(kw_dire, f'cid2term_set_y_{year}_{pmi_tag}.pkl')
    if not os.path.exists(path):
        continue
    with open(path, 'rb') as f:
        cid2term_set = pickle.load(f)
    for cid, terms in cid2term_set.items():
        kept = [t for t in terms if t in candidate_terms]
        if kept:
            cid2terms[cid] = set(kept)
print('papers with >=1 candidate term:', len(cid2terms))

# ==================================================
# 3. walk citation edges q -> p; for each term shared by q and p, count it toward central paper p
# ==================================================
with open(os.path.join(meta_dire, 'citing2cited_cid_list_remove_circle.pkl'), 'rb') as f:
    citing2cited = pickle.load(f)
print('citing papers in network:', len(citing2cited))

term_central2count = defaultdict(lambda: defaultdict(int))
cid2indegree = defaultdict(int)
for q, cited_list in tqdm(citing2cited.items(), desc='citation edges'):
    tq = cid2terms.get(q)
    for p in cited_list:
        if p == q:
            continue
        cid2indegree[p] += 1
        tp = cid2terms.get(p)
        if not tp or not tq:
            continue
        shared = tq & tp
        if not shared:
            continue
        for t in shared:
            term_central2count[t][p] += 1

# convert nested defaultdicts to plain dicts for pickling
term_central2count = {t: dict(pc) for t, pc in term_central2count.items()}
with open(os.path.join(kw_dire, f'term_central2count_{tag}.pkl'), 'wb') as f:
    pickle.dump(term_central2count, f, pickle.HIGHEST_PROTOCOL)
with open(os.path.join(kw_dire, f'cid2indegree_{tag}.pkl'), 'wb') as f:
    pickle.dump(dict(cid2indegree), f, pickle.HIGHEST_PROTOCOL)

print('terms with >=1 central-paper citation:', len(term_central2count))
print('finished')
