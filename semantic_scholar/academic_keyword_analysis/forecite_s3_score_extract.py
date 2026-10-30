"""
ForeCite academic-keyword extraction, STAGE 3 of 4.  (see forecite_s1_paper_terms.py header)

Computes the per-term ForeCite score in a CHECKPOINTED, RESUMABLE way, then leaves the final
table assembly to forecite_s4_combine.py.

For term t (a "word": single lemma or valid bigram), over each candidate central paper p that has
>= MIN_CITATIONS citations:
    f^p_t   = term_central2count[t][p]              (papers mentioning t that cite p)
    f_t     = sum of term2year2count[t][y] for y >= pub_year(p)  (papers mentioning t from p's year on)
    score_p = log(f^p_t + 1) * (f^p_t / f_t)        (> 0 for every scorable central paper)
    ForeCite(t) = max_p score_p
We use the exact f_t from the per-year counts rather than the paper's 500-sample estimate.

Checkpointing:
  * Only terms with >= MIN_CENTRAL_PAPERS candidate central papers (raw len(term_central2count[t]))
    are kept; the surviving count is printed before scoring.
  * The surviving terms are split, in a STABLE order (term_central2count iteration order), into
    N_LISTS contiguous lists. List index i always maps to the same terms across runs.
  * Each list is scored independently and its result saved to  <ckpt_dir>/list_<i>.pkl , followed
    by an empty marker  <ckpt_dir>/list_<i>_finished . The marker is written only AFTER the .pkl is
    fully flushed, so its presence guarantees a complete list.
  * On restart, any list whose marker exists is skipped, so a run resumes from the unfinished lists.
Each  list_<i>.pkl  holds {'fc': word->score, 'pid': word->{pid:score_p}, 'det': word->detail}
for that list's terms. forecite_s4_combine.py unions all N_LISTS files into the final deliverables.

The per-list scoring is CPU-bound and independent, so lists are fanned out over PROCESS_COUNT
workers; the two large read-only lookups (cid2indegree, cid2pub_year) are loaded once in the parent
and inherited by the workers copy-on-write via fork (Linux).

Input  : META_DIRE/academic_keyword_forecite/ (stage 2, same tag)
             term2year2count_<tag>.pkl, term_central2count_<tag>.pkl, cid2indegree_<tag>.pkl
         META_DIRE/cid2pub_year.pkl              (from get_s2ag_meta.py)
Output : META_DIRE/academic_keyword_forecite/forecite_s3_partial/<out_tag>/
             list_<i>.pkl, list_<i>_finished      (i = 0 .. N_LISTS-1;  out_tag = <tag>_mcp_<MIN_CENTRAL_PAPERS>)
         (batch files stay in this dedicated subdirectory; only forecite_s4_combine.py writes the
          final table back into academic_keyword_forecite/.)
Then run forecite_s4_combine.py to build the final word2forecite_score / CSV.
"""

import os
import sys
import math
import pickle
from bisect import bisect_left
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
from utils import META_DIRE

# ---- parameters (must match forecite_s2_central_counts.py / forecite_s4_combine.py) ----
PMI_bigram = 0
begin_year = 1951
finish_year = 2025
MIN_TERM_DF = 10
MIN_CITATIONS = 3           # ForeCite: only papers with >= this many citations are scorable centres
MIN_CENTRAL_PAPERS = 10     # keep a term only if it has >= this many candidate central papers (raw len(term_central2count[t]))

N_LISTS = 128               # split qualifying terms into this many lists; each scored + checkpointed separately (resumable)
PROCESS_COUNT = 16          # parallel worker processes for scoring (set to 1 to run sequentially)

pmi_tag = f'pmib_{PMI_bigram}'
tag = f'{pmi_tag}_df_{MIN_TERM_DF}_b_{begin_year}_e_{finish_year}'   # stage-2 inputs use this tag
out_tag = f'{tag}_mcp_{MIN_CENTRAL_PAPERS}'                          # outputs also carry the central-paper floor

meta_dire = META_DIRE
kw_dire = os.path.join(meta_dire, 'academic_keyword_forecite')       # final combined table lands here (via s4)
# per-list batch files live in their OWN subdirectory tree, kept apart from the final outputs; only
# forecite_s4_combine.py writes the combined table back into kw_dire.
partial_root = os.path.join(kw_dire, 'forecite_s3_partial')          # dedicated parent for all batch files
ckpt_dir = os.path.join(partial_root, out_tag)                       # this run's per-list partials + markers (shared with s4)


def _partial_path(i):
    return os.path.join(ckpt_dir, 'list_%d.pkl' % i)


def _marker_path(i):
    return os.path.join(ckpt_dir, 'list_%d_finished' % i)


def term_type_text(t):
    """(type, human-readable text) for a single / bigram term key."""
    if isinstance(t, str):
        return 'single', t
    return 'bigram', f'{t[0]} {t[1]}'                    # (w1, w2)


# ---- the two large read-only lookups live as module globals: loaded once in the parent and
#      inherited by the workers copy-on-write via fork, so they are never re-shipped as task
#      arguments nor reloaded per worker ----
_cid2indegree = None
_cid2pub_year = None


def _score_list(items):
    """Score one list of terms. `items` is [(t, p2count, years, suffix), ...] where `years`/`suffix`
    are the term's sorted years and suffix sums so f_t = suffix[bisect_left(years, pub_year)].
    Returns the three per-list dicts (fc, pid, det) keyed by the readable word."""
    w2fc = {}      # word -> ForeCite score (max over central papers)
    w2pid = {}     # word -> {pid: score_p}  (every scorable central paper, score_p > 0)
    w2det = {}     # word -> (type, best_pid, f_p_t, f_t, df_total)  (for the readable CSV in s4)
    for t, p2count, years, suffix in items:
        pid2score = {}
        best = None                  # (score, pid, f_p_t, f_t)
        for p, fpt in p2count.items():
            if _cid2indegree.get(p, 0) < MIN_CITATIONS:   # only papers with >= MIN_CITATIONS are centres
                continue
            py = _cid2pub_year.get(p)
            if py is None:
                continue
            ft = suffix[bisect_left(years, py)]
            if ft <= 0:
                continue
            score = math.log(fpt + 1) * (fpt / ft)
            if score <= 0:
                continue
            pid2score[p] = score
            if best is None or score > best[0]:
                best = (score, p, fpt, ft)
        if pid2score:
            typ, text = term_type_text(t)
            w2fc[text] = best[0]
            w2pid[text] = pid2score
            w2det[text] = (typ, best[1], best[2], best[3], suffix[0])   # suffix[0] = total df of t
    return w2fc, w2pid, w2det


def _split(seq, n):
    """Split `seq` into n contiguous lists (stable: index i -> same slice given the same seq).
    Trailing lists may be empty when len(seq) < n."""
    k, m = divmod(len(seq), n)
    out = []
    start = 0
    for i in range(n):
        size = k + (1 if i < m else 0)
        out.append(seq[start:start + size])
        start += size
    return out


def main():
    global _cid2indegree, _cid2pub_year
    print('tag', tag, '| MIN_CITATIONS', MIN_CITATIONS,
          '| MIN_CENTRAL_PAPERS', MIN_CENTRAL_PAPERS, '| N_LISTS', N_LISTS, '| PROCESS_COUNT', PROCESS_COUNT)

    # ---- load term_central2count (all the prefilter needs) ----
    with open(os.path.join(kw_dire, f'term_central2count_{tag}.pkl'), 'rb') as f:
        term_central2count = pickle.load(f)
    print('terms to score:', len(term_central2count))

    # ---- pre-filter: keep only terms with >= MIN_CENTRAL_PAPERS candidate central papers (raw
    #      len(term_central2count[t])), in term_central2count iteration order so the split below is
    #      stable. Needs no citation/year lookups, so it runs before those load. ----
    qualifying_terms = [t for t, p2count in tqdm(term_central2count.items(),
                                                 total=len(term_central2count), desc='prefilter')
                        if len(p2count) >= MIN_CENTRAL_PAPERS]
    print('terms with >= %d central papers: %d (of %d)'
          % (MIN_CENTRAL_PAPERS, len(qualifying_terms), len(term_central2count)))

    # ---- split into N_LISTS stable lists, and figure out which still need scoring ----
    os.makedirs(ckpt_dir, exist_ok=True)
    lists = _split(qualifying_terms, N_LISTS)
    todo = [i for i in range(N_LISTS) if not os.path.exists(_marker_path(i))]
    print('checkpoint dir:', ckpt_dir)
    print('lists: %d total | %d finished | %d to compute'
          % (N_LISTS, N_LISTS - len(todo), len(todo)))
    if not todo:
        print('all lists finished; nothing to do. run forecite_s4_combine.py to build the final table.')
        return

    # ---- load the two large read-only lookups into module globals so forked workers inherit them
    #      (copy-on-write) instead of receiving them as task arguments ----
    with open(os.path.join(kw_dire, f'cid2indegree_{tag}.pkl'), 'rb') as f:
        _cid2indegree = pickle.load(f)
    with open(os.path.join(meta_dire, 'cid2pub_year.pkl'), 'rb') as f:
        _cid2pub_year = pickle.load(f)

    # ---- load term2year2count (only needed to build f_t) and precompute, for the terms in the
    #      unfinished lists only, sorted years + suffix sums so f_t is an O(log n) lookup ----
    with open(os.path.join(kw_dire, f'term2year2count_{tag}.pkl'), 'rb') as f:
        term2year2count = pickle.load(f)
    todo_terms = [t for i in todo for t in lists[i]]
    term2year_suffix = {}          # term -> (sorted_years, suffix_sum_from_that_year_onward)
    for t in todo_terms:
        y2c = term2year2count[t]
        years = sorted(y2c)
        suffix = [0] * (len(years) + 1)
        for i in range(len(years) - 1, -1, -1):
            suffix[i] = suffix[i + 1] + y2c[years[i]]
        term2year_suffix[t] = (years, suffix)
    del term2year2count

    def make_payload(terms):
        return [(t, term_central2count[t], *term2year_suffix[t]) for t in terms]

    def save_list(i, result):
        w2fc, w2pid, w2det = result
        # write the .pkl first, then the marker, so a marker always implies a complete .pkl
        with open(_partial_path(i), 'wb') as f:
            pickle.dump({'fc': w2fc, 'pid': w2pid, 'det': w2det}, f, pickle.HIGHEST_PROTOCOL)
        open(_marker_path(i), 'w').close()

    # ---- score each unfinished list, checkpointing as soon as it completes (as_completed, so an
    #      interrupt keeps every list that had already finished, not just a contiguous prefix) ----
    if PROCESS_COUNT > 1:
        with ProcessPoolExecutor(max_workers=PROCESS_COUNT) as executor:
            fut2i = {executor.submit(_score_list, make_payload(lists[i])): i for i in todo}
            for fut in tqdm(as_completed(fut2i), total=len(fut2i), desc='score lists'):
                save_list(fut2i[fut], fut.result())
    else:
        for i in tqdm(todo, desc='score lists'):
            save_list(i, _score_list(make_payload(lists[i])))

    print('finished scoring %d lists into %s' % (len(todo), ckpt_dir))
    print('run forecite_s4_combine.py to build the final word2forecite_score / CSV.')


if __name__ == '__main__':
    main()
