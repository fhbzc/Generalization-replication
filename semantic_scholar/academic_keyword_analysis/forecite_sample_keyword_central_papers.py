"""
Sanity/inspection script over the ForeCite outputs (forecite_s3_score_extract.py).

Produces TWO keyword tables (all keywords have a max ForeCite score > 0 by construction):
  (1) a RANDOM sample of SAMPLE_N keywords, and
  (2) the TOP_N keywords by max ForeCite score.
For each keyword both tables report the same columns:
  * the keyword's ForeCite score (= max over its scoring papers, from word2forecite_score)
  * the MAX-SCORE paper: the pid with the highest ForeCite score for this keyword (the ForeCite
    "central" paper)
  * the EARLIEST paper: among ALL papers that have a score for this keyword, the one published
    earliest (by cid2pub_date; falls back to cid2pub_year when no date)
  * the EARLIEST-ABOVE-MEAN paper: same, but restricted to papers whose score is strictly greater
    than the mean score over all scoring papers (empty when none are above the mean, e.g. a single
    scoring paper)
each paper is reported with its ForeCite score and its publication date. Lets you eyeball whether
the top-scoring / above-average paper for a keyword is indeed an early, introducing paper (the
intuition ForeCite is built on).

Note: in the S2AG pipeline a "pid" is a corpusid (cid); dates come from get_s2ag_meta.py's
cid2pub_date.pkl / cid2pub_year.pkl.

Input  : META_DIRE/academic_keyword_forecite/ (from forecite_s4_combine.py, out_tag = <tag>_mcp_<MIN_CENTRAL_PAPERS>)
             word2forecite_score_<out_tag>.pkl, word2pid2forecite_score_<out_tag>.pkl
         META_DIRE/{cid2pub_date.pkl, cid2pub_year.pkl, cid2title.pkl}   (from get_s2ag_meta.py)
Output : ./ (next to this script)
             sampled_keyword_central_papers_<out_tag>_n<SAMPLE_N>_seed<SEED>.csv   (random sample)
             top_keyword_central_papers_<out_tag>_top<TOP_N>.csv                   (top by max score)
"""

import os
import sys
import csv
import pickle
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
from utils import META_DIRE

# ---- parameters (tag + MIN_CENTRAL_PAPERS must match forecite_s3/s4) ----
PMI_bigram = 0
begin_year = 1951
finish_year = 2025
MIN_TERM_DF = 10
MIN_CENTRAL_PAPERS = 10      # must match forecite_s3_score_extract.py / forecite_s4_combine.py
tag = f'pmib_{PMI_bigram}_df_{MIN_TERM_DF}_b_{begin_year}_e_{finish_year}'
out_tag = f'{tag}_mcp_{MIN_CENTRAL_PAPERS}'   # forecite_s4_combine.py writes the deliverables under out_tag

SAMPLE_N = 1000              # size of the random keyword sample
TOP_N = 1000                 # also emit the top-N keywords by max ForeCite score
SEED = 0                     # fixed seed so the random sample is reproducible
TOP_K_PAPERS = 5             # also list the top-K scoring papers (pid + title) per keyword

# keep only keywords whose max ForeCite score is > this (strict >, same rule as
# forecite_central_papers_efg_s1_split.py). Both the random sample and the top-N are drawn from
# the surviving terms. CLI override: python forecite_sample_keyword_central_papers.py [THRESHOLD]
FORECITE_THRESHOLD = 0.5
if len(sys.argv) > 1:
    FORECITE_THRESHOLD = float(sys.argv[1])

meta_dire = META_DIRE
kw_dire = os.path.join(meta_dire, 'academic_keyword_forecite')
print('tag', tag, '| MIN_CENTRAL_PAPERS', MIN_CENTRAL_PAPERS, '| SAMPLE_N', SAMPLE_N, '| SEED', SEED,
      '| FORECITE_THRESHOLD', FORECITE_THRESHOLD)

# ---- load the ForeCite dicts + publication dates/years ----
with open(os.path.join(kw_dire, f'word2forecite_score_{out_tag}.pkl'), 'rb') as f:
    word2forecite_score = pickle.load(f)
with open(os.path.join(kw_dire, f'word2pid2forecite_score_{out_tag}.pkl'), 'rb') as f:
    word2pid2forecite_score = pickle.load(f)
with open(os.path.join(meta_dire, 'cid2pub_date.pkl'), 'rb') as f:
    cid2pub_date = pickle.load(f)
with open(os.path.join(meta_dire, 'cid2pub_year.pkl'), 'rb') as f:
    cid2pub_year = pickle.load(f)
with open(os.path.join(meta_dire, 'cid2title.pkl'), 'rb') as f:
    cid2title = pickle.load(f)
print('keywords available:', len(word2forecite_score), '| titles:', len(cid2title))


def title_str(cid):
    """Paper title for display (empty string when unknown)."""
    t = cid2title.get(cid)
    return str(t) if t is not None else ''


def pub_key(cid):
    """Sortable (year, month, day) for a paper; date preferred, else year, else +inf (unknown)."""
    d = cid2pub_date.get(cid)
    if d:
        parts = str(d)[:10].split('-')
        try:
            y = int(parts[0])
            m = int(parts[1]) if len(parts) > 1 and parts[1] else 0
            day = int(parts[2]) if len(parts) > 2 and parts[2] else 0
            return (y, m, day)
        except ValueError:
            pass
    y = cid2pub_year.get(cid)
    if y is not None:
        try:
            return (int(y), 0, 0)
        except (ValueError, TypeError):
            pass
    return (10 ** 9, 0, 0)


def pub_str(cid):
    """Human-readable publication date/year for display."""
    d = cid2pub_date.get(cid)
    if d:
        return str(d)
    y = cid2pub_year.get(cid)
    return str(y) if y is not None else ''


FIELDS = ['keyword', 'forecite_score', 'n_scoring_pids', 'mean_score',
          'max_score_pid', 'max_score_pid_title', 'max_score_pid_score', 'max_score_pid_pub',
          'earliest_pid', 'earliest_pid_title', 'earliest_pid_score', 'earliest_pid_pub',
          'earliest_above_mean_pid', 'earliest_above_mean_pid_title',
          'earliest_above_mean_pid_score', 'earliest_above_mean_pid_pub']
# per-keyword top-K scoring papers (pid + score + title), one set of columns per rank
TOP_K_FIELDS = [f'top{k}_{suffix}'
                for k in range(1, TOP_K_PAPERS + 1)
                for suffix in ('pid', 'score', 'title')]
FIELDS += TOP_K_FIELDS
# score columns to format with 6 decimals when written
SCORE_FIELDS = (['max_score_pid_score', 'earliest_pid_score', 'earliest_above_mean_pid_score']
                + [f'top{k}_score' for k in range(1, TOP_K_PAPERS + 1)])


def build_rows(keyword_list):
    """One row per keyword: its ForeCite score + the max-score / earliest / earliest-above-mean
    papers (each with score + publication date)."""
    rows = []
    for kw in keyword_list:
        pid2score = word2pid2forecite_score.get(kw, {})
        if not pid2score:
            continue
        kw_score = word2forecite_score[kw]          # == max(pid2score.values())
        n = len(pid2score)
        mean_score = sum(pid2score.values()) / n

        # MAX-SCORE paper (tie-break: earliest published, then smallest cid, for determinism)
        max_pid = max(pid2score, key=lambda p: (pid2score[p], tuple(-x for x in pub_key(p)), -p))
        # EARLIEST paper among ALL papers scoring this keyword
        earliest_pid = min(pid2score, key=lambda p: (pub_key(p), p))
        # EARLIEST paper among those strictly above the mean score
        above = [p for p, s in pid2score.items() if s > mean_score]
        early_above_pid = min(above, key=lambda p: (pub_key(p), p)) if above else None

        # TOP-K scoring papers: highest score first (tie-break earliest published, then smallest cid)
        top_pids = sorted(pid2score, key=lambda p: (-pid2score[p], pub_key(p), p))[:TOP_K_PAPERS]

        row = {
            'keyword': kw,
            'forecite_score': kw_score,
            'n_scoring_pids': n,
            'mean_score': mean_score,
            'max_score_pid': max_pid,
            'max_score_pid_title': title_str(max_pid),
            'max_score_pid_score': pid2score[max_pid],
            'max_score_pid_pub': pub_str(max_pid),
            'earliest_pid': earliest_pid,
            'earliest_pid_title': title_str(earliest_pid),
            'earliest_pid_score': pid2score[earliest_pid],
            'earliest_pid_pub': pub_str(earliest_pid),
            'earliest_above_mean_pid': early_above_pid if early_above_pid is not None else '',
            'earliest_above_mean_pid_title': title_str(early_above_pid) if early_above_pid is not None else '',
            'earliest_above_mean_pid_score': pid2score[early_above_pid] if early_above_pid is not None else '',
            'earliest_above_mean_pid_pub': pub_str(early_above_pid) if early_above_pid is not None else '',
        }
        # top-K scoring papers (pid + score + title); blank columns when fewer than K papers
        for rank in range(1, TOP_K_PAPERS + 1):
            if rank <= len(top_pids):
                p = top_pids[rank - 1]
                row[f'top{rank}_pid'] = p
                row[f'top{rank}_score'] = pid2score[p]
                row[f'top{rank}_title'] = title_str(p)
            else:
                row[f'top{rank}_pid'] = ''
                row[f'top{rank}_score'] = ''
                row[f'top{rank}_title'] = ''
        rows.append(row)
    return rows


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            r = dict(r)
            r['forecite_score'] = f"{r['forecite_score']:.6f}"
            r['mean_score'] = f"{r['mean_score']:.6f}"
            for k in SCORE_FIELDS:
                if r[k] != '':
                    r[k] = f"{r[k]:.6f}"
            w.writerow(r)
    print('saved:', path, '| rows:', len(rows))


# outputs go next to this script (./)
OUT_DIR = BASE_DIR
# only keywords whose max ForeCite score is > FORECITE_THRESHOLD (strict >, matching step 1)
keywords = [w for w, s in word2forecite_score.items() if s > FORECITE_THRESHOLD]
print('keywords with ForeCite score > %g: %d (of %d)'
      % (FORECITE_THRESHOLD, len(keywords), len(word2forecite_score)))
assert keywords, 'no keywords above the threshold; lower FORECITE_THRESHOLD'
# threshold tag so filtered outputs never overwrite an unfiltered (or differently-thresholded) run
thr_tag = f'_gt{FORECITE_THRESHOLD}'

# ---- 1. random sample of SAMPLE_N keywords ----
rng = random.Random(SEED)
sample = keywords if len(keywords) <= SAMPLE_N else rng.sample(keywords, SAMPLE_N)
rows_sample = build_rows(sample)
rows_sample.sort(key=lambda r: r['forecite_score'], reverse=True)
# show each sampled term with its ForeCite score (high -> low), for a quick eyeball
print('sampled %d keywords (term : ForeCite score), high to low:' % len(rows_sample))
for r in rows_sample:
    print('  %s\t%.6f' % (r['keyword'], r['forecite_score']))
write_csv(os.path.join(OUT_DIR,
          f'sampled_keyword_central_papers_{out_tag}{thr_tag}_n{SAMPLE_N}_seed{SEED}.csv'),
          rows_sample)

# ---- 2. top TOP_N keywords by max ForeCite score ----
top = sorted(keywords, key=lambda w: word2forecite_score[w], reverse=True)[:TOP_N]
rows_top = build_rows(top)                           # already in descending-score order
write_csv(os.path.join(OUT_DIR, f'top_keyword_central_papers_{out_tag}{thr_tag}_top{TOP_N}.csv'),
          rows_top)

print('finished')
