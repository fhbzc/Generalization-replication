"""
ForeCite academic-keyword extraction, STAGE 1 of 3.

Method: ForeCite (King, Downey & Weld, "High-Precision Extraction of Emerging Concepts from
Scientific Literature", SIGIR 2020; 2006.06877). A candidate phrase t is scored by its TERM
CITATION GRAPH G_t (the papers mentioning t plus their citation edges):
    ForeCite(t) = max_{p in G_t}  log(f^p_t + 1) * (f^p_t / f_t)
where p ranges over papers mentioning t, f^p_t = #papers mentioning t that cite p, and
f_t = #papers mentioning t published from p's year onward. Concepts tend to be introduced by a
single central paper that later papers mentioning the concept disproportionately cite.

Candidate phrases here = the SINGLE grams from get_global_trigam_bigram_s1_eng.py's per-paper
lemmatized token stream + the VALID bigrams from get_global_trigam_bigram_s2.py (PMI-filtered).
A term is one of:
    single  : a lemma string                w
    bigram  : a tuple                        (w1, w2)   -- kept iff in valid_bigram2pmi
(trigrams are intentionally not used.)

THIS STAGE builds, per publication year, (a) the set of candidate terms each paper mentions and
(b) each term's document frequency that year. Stage 2 aggregates these + the citation network to
count central-paper citations; stage 3 computes the ForeCite score and emits the keyword list.

Input  : META_DIRE/global_tribigram_eng/
             cid2parsed_token_list_<year>.pkl           (from get_global_trigam_bigram_s1_eng.py)
             valid_bigram2pmi_t_<PMI_bigram>.pkl         (from get_global_trigam_bigram_s2.py)
Output : META_DIRE/academic_keyword_forecite/  (pmi_tag = pmib_<PMI_bigram>)
             cid2term_set_y_<year>_<pmi_tag>.pkl         (cid -> list of candidate terms)
             term2df_y_<year>_<pmi_tag>.pkl              (term -> #papers that year containing it)
             finished_s1_y_<year>_<pmi_tag>.pkl          (resume marker)
"""

import os
import sys
import pickle
from collections import defaultdict

from tqdm import tqdm

# utils.py lives in the parent (semantic_scholar) directory; make it importable regardless of CWD
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
from utils import META_DIRE

# ---- parameters ----
# PMI_bigram + (VOCAB_START_YEAR, VOCAB_END_YEAR) must match the valid_bigram2pmi file written by
# get_global_trigam_bigram_s2.py (its PMI_bigram and its start_year / end_year).
PMI_bigram = 0
VOCAB_START_YEAR = 1951
VOCAB_END_YEAR = 2026       # get_global_trigam_bigram_s2.py's end_year (range end, exclusive)
begin_year = 1951           # years to process here
finish_year = 2025          # inclusive
pmi_tag = f'pmib_{PMI_bigram}'

meta_dire = META_DIRE
load_dire = os.path.join(meta_dire, 'global_tribigram_eng')
save_dire = os.path.join(meta_dire, 'academic_keyword_forecite')
os.makedirs(save_dire, exist_ok=True)

print('PMI_bigram', PMI_bigram)
print('years', begin_year, '..', finish_year)
print('load_dire', load_dire)
print('save_dire', save_dire)

# ---- the valid bigram vocabulary (membership tests only) ----
valid_bigram_name = f'valid_bigram2pmi_t_{PMI_bigram}_b_{VOCAB_START_YEAR}_e_{VOCAB_END_YEAR}.pkl'
with open(os.path.join(load_dire, valid_bigram_name), 'rb') as f:
    valid_bigram_set = set(pickle.load(f).keys())
print('valid bigrams:', len(valid_bigram_set))

bigram_contains = valid_bigram_set.__contains__


def paper_terms(list_of_token_list):
    """Candidate terms mentioned by one paper: every single lemma, plus adjacent pairs that are
    valid bigrams. (Trigrams are not used.)"""
    terms = set()
    for token_list in list_of_token_list:
        l = len(token_list)
        for i in range(l):
            w = token_list[i]
            terms.add(w)                                  # single
            if i <= l - 2:
                bg = (w, token_list[i + 1])
                if bigram_contains(bg):
                    terms.add(bg)                         # bigram
    return terms


for year in tqdm(range(begin_year, finish_year + 1), desc='years'):
    finish_path = os.path.join(save_dire, f'finished_s1_y_{year}_{pmi_tag}.pkl')
    if os.path.exists(finish_path):
        continue
    parsed_path = os.path.join(load_dire, f'cid2parsed_token_list_{year}.pkl')
    if not os.path.exists(parsed_path):
        print(f'  [y{year}] {parsed_path} not found, skipping.')
        continue

    with open(parsed_path, 'rb') as f:
        cid2list_of_token_list = pickle.load(f)

    cid2term_set = {}
    term2df = defaultdict(int)
    for cid, list_of_token_list in tqdm(cid2list_of_token_list.items(),
                                        position=1, leave=False, desc=f'y{year}'):
        terms = paper_terms(list_of_token_list)
        if not terms:
            continue
        cid2term_set[cid] = list(terms)
        for t in terms:
            term2df[t] += 1

    with open(os.path.join(save_dire, f'cid2term_set_y_{year}_{pmi_tag}.pkl'), 'wb') as f:
        pickle.dump(cid2term_set, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(save_dire, f'term2df_y_{year}_{pmi_tag}.pkl'), 'wb') as f:
        pickle.dump(dict(term2df), f, pickle.HIGHEST_PROTOCOL)
    with open(finish_path, 'wb') as f:
        pickle.dump([], f, pickle.HIGHEST_PROTOCOL)
    print(f'  [y{year}] papers with terms: {len(cid2term_set)}; distinct terms: {len(term2df)}')

print('finished')
