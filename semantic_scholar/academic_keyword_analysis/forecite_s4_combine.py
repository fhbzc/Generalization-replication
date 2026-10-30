"""
ForeCite academic-keyword extraction, STAGE 4 of 4.  (see forecite_s1_paper_terms.py header)

Combines the per-list partial results written by forecite_s3_score_extract.py into the TWO
deliverable dicts + a readable ranked CSV. Stage 3 splits the surviving terms into N_LISTS lists and
saves each list's scores to <ckpt_dir>/list_<i>.pkl (+ a list_<i>_finished marker). This stage
unions all N_LISTS files, in list-index order (so the ranking's tie-breaking matches a single-pass
run), optionally applies MIN_SCORE / TOP_K, and writes the final outputs.

The two output dicts (keyed by the readable word):
    word2forecite_score      : word -> ForeCite score (the max over its central papers)
    word2pid2forecite_score  : word -> {pid: score_p} for EVERY pid with score_p > 0
word2forecite_score[w] == max(word2pid2forecite_score[w].values()).

Input  : META_DIRE/academic_keyword_forecite/forecite_s3_partial/<out_tag>/
             list_<i>.pkl, list_<i>_finished   (i = 0 .. N_LISTS-1)
Output : META_DIRE/academic_keyword_forecite/  (out_tag = <tag>_mcp_<MIN_CENTRAL_PAPERS>)
             word2forecite_score_<out_tag>.pkl       (word -> ForeCite score)
             word2pid2forecite_score_<out_tag>.pkl   (word -> {pid: score_p}, all score_p > 0)
             academic_keyword_list_<out_tag>.csv     (readable ranking: rank, keyword, type, score, ...)
"""

import os
import sys
import csv
import pickle

from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
from utils import META_DIRE

# ---- parameters (must match forecite_s3_score_extract.py) ----
PMI_bigram = 0
begin_year = 1951
finish_year = 2025
MIN_TERM_DF = 10
MIN_CENTRAL_PAPERS = 10     # must match stage 3 (selects the checkpoint dir / out_tag)
N_LISTS = 128               # must match stage 3

TOP_K = None                # if set, also write the top-K keywords to a *_top<K>.csv (None = skip)
MIN_SCORE = None            # if set, only keep terms with ForeCite score >= this (None = keep all)

pmi_tag = f'pmib_{PMI_bigram}'
tag = f'{pmi_tag}_df_{MIN_TERM_DF}_b_{begin_year}_e_{finish_year}'
out_tag = f'{tag}_mcp_{MIN_CENTRAL_PAPERS}'

meta_dire = META_DIRE
kw_dire = os.path.join(meta_dire, 'academic_keyword_forecite')       # final combined table lands here
partial_root = os.path.join(kw_dire, 'forecite_s3_partial')          # dedicated parent for all batch files
ckpt_dir = os.path.join(partial_root, out_tag)                       # this run's per-list partials + markers (from s3)


def _partial_path(i):
    return os.path.join(ckpt_dir, 'list_%d.pkl' % i)


def _marker_path(i):
    return os.path.join(ckpt_dir, 'list_%d_finished' % i)


def main():
    print('tag', tag, '| MIN_CENTRAL_PAPERS', MIN_CENTRAL_PAPERS, '| N_LISTS', N_LISTS)
    print('checkpoint dir:', ckpt_dir)

    # ---- require every list to be finished, else the combined table would be incomplete ----
    missing = [i for i in range(N_LISTS) if not os.path.exists(_marker_path(i))]
    if missing:
        preview = ', '.join(map(str, missing[:20])) + (' ...' if len(missing) > 20 else '')
        print('ERROR: %d of %d lists are not finished yet: [%s]' % (len(missing), N_LISTS, preview))
        print('run forecite_s3_score_extract.py again to finish them, then re-run this stage.')
        sys.exit(1)

    # ---- union all per-list partials, in list-index order (== stage-3 term order) ----
    word2forecite_score = {}
    word2pid2forecite_score = {}
    word2detail = {}                 # word -> (type, best_pid, f_p_t, f_t, df)
    for i in tqdm(range(N_LISTS), desc='combine lists'):
        with open(_partial_path(i), 'rb') as f:
            d = pickle.load(f)
        word2forecite_score.update(d['fc'])
        word2pid2forecite_score.update(d['pid'])
        word2detail.update(d['det'])
    print('combined academic keywords:', len(word2forecite_score))

    # optional score floor (applied to the word's max score; drops the word from both dicts)
    if MIN_SCORE is not None:
        keep = {w for w, s in word2forecite_score.items() if s >= MIN_SCORE}
        word2forecite_score = {w: word2forecite_score[w] for w in keep}
        word2pid2forecite_score = {w: word2pid2forecite_score[w] for w in keep}
        word2detail = {w: word2detail[w] for w in keep}
        print('after MIN_SCORE >= %s:' % MIN_SCORE, len(word2forecite_score))

    ranked = sorted(word2forecite_score.items(), key=lambda kv: kv[1], reverse=True)

    # ---- save the TWO deliverable dicts + a readable ranked CSV ----
    with open(os.path.join(kw_dire, f'word2forecite_score_{out_tag}.pkl'), 'wb') as f:
        pickle.dump(word2forecite_score, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(kw_dire, f'word2pid2forecite_score_{out_tag}.pkl'), 'wb') as f:
        pickle.dump(word2pid2forecite_score, f, pickle.HIGHEST_PROTOCOL)

    def write_csv(path, rows):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['rank', 'keyword', 'type', 'forecite_score',
                        'best_central_cid', 'f_p_t', 'f_t', 'df', 'n_pids_scored'])
            for rank, (word, score) in enumerate(rows, 1):
                typ, best_pid, fpt, ft, df_total = word2detail[word]
                w.writerow([rank, word, typ, f'{score:.6f}',
                            best_pid, fpt, ft, df_total, len(word2pid2forecite_score[word])])

    csv_path = os.path.join(kw_dire, f'academic_keyword_list_{out_tag}.csv')
    write_csv(csv_path, ranked)
    print('saved:', csv_path)

    if TOP_K is not None:
        topk_path = os.path.join(kw_dire, f'academic_keyword_list_{out_tag}_top{TOP_K}.csv')
        write_csv(topk_path, ranked[:TOP_K])
        print('saved:', topk_path)

    print('finished')


if __name__ == '__main__':
    main()
