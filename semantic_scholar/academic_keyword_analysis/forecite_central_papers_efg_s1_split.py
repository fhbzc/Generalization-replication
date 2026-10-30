"""
STEP 1 of 2: split the papers into CENTRAL vs NON-CENTRAL, and cache the result.

(Step 2 is forecite_central_papers_efg_s2_plot.py, which draws the distributions. This script
does no plotting and never touches the efg table -- it only decides which papers are central.)

Given a ForeCite score threshold, take every keyword with ForeCite score > threshold
(forecite_s4_combine.py output). For each keyword a set of "central" papers is picked in FOUR
ways, so each way yields its own central class (and, implicitly, its non-central rest = every
other paper):

  max_score           : the single MAX-ForeCite-score paper of each keyword
  top5_score          : the TOP-{TOP_K_PAPERS} ForeCite-score papers of each keyword
  earliest            : the EARLIEST-published paper among those scoring the keyword (score > 0)
  earliest_above_mean : the EARLIEST-published paper among those scoring ABOVE the mean score
                        over that keyword's papers -- i.e. one paper per keyword, not the whole
                        above-mean set. 'earliest' picks the earliest of ALL scoring papers;
                        this one first restricts to the above-mean papers, so it is the earliest
                        paper that is also a strong one for the keyword.

So three of the four definitions contribute at most ONE paper per keyword; only top5_score
contributes several.

Each central set is a UNIQUE union over keywords (a paper central for several keywords is
counted once). Also saved is `scored_pids` -- every paper scoring ANY above-threshold keyword --
which step 2 uses when its REST_UNIVERSE='scored' compares central vs non-central WITHIN the
ForeCite-scored papers rather than against the whole corpus.

The build is the expensive part of the pipeline (a pure-Python loop over every keyword), so its
output is cached and keyed ONLY by what the central sets actually depend on (threshold, out_tag,
TOP_K_PAPERS). Step 2's plotting parameters are deliberately NOT part of that key: re-running
step 2 with different metrics / quantiles never invalidates this.

Input  : META_DIRE/academic_keyword_forecite/{word2forecite_score,word2pid2forecite_score}_{out_tag}.pkl
             (forecite_s4_combine.py; out_tag = <tag>_mcp_<MIN_CENTRAL_PAPERS>)
         META_DIRE/{cid2pub_date.pkl, cid2pub_year.pkl}   (get_s2ag_meta.py; for "earliest")
Output : ./figures/forecite_gt_<threshold>/central_sets_<out_tag>.pkl
             {'key': ..., 'central_sets': {name -> set(pid)}, 'scored_pids': set(pid),
              'n_keywords': int}
         ./figures/forecite_gt_<threshold>/central_set_sizes_<out_tag>.csv
             (one row per definition: how many papers ended up central)

Usage  : python forecite_central_papers_efg_s1_split.py [FORECITE_THRESHOLD]
"""

import os
import sys
import pickle
import heapq

import pandas as pd
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
from utils import META_DIRE  # noqa: E402

meta_dire = META_DIRE
print("meta_dire", meta_dire)

# ---------------------------------------------------------------------------
# Parameters. FORECITE_THRESHOLD / TOP_K_PAPERS / out_tag together key the output cache, so
# forecite_central_papers_efg_s2_plot.py MUST be run with the same three values to find it.
# ---------------------------------------------------------------------------
FORECITE_THRESHOLD = 2     # keep keywords whose ForeCite score is > this; CLI override below
if len(sys.argv) > 1:
    FORECITE_THRESHOLD = float(sys.argv[1])

TOP_K_PAPERS = 5             # 'top5_score': top-K scoring papers per keyword

# ---- ForeCite dicts (tag + MIN_CENTRAL_PAPERS must match forecite_s3/s4) ----
PMI_bigram = 0
begin_year = 1951
finish_year = 2025
MIN_TERM_DF = 10
MIN_CENTRAL_PAPERS = 10
tag = f'pmib_{PMI_bigram}_df_{MIN_TERM_DF}_b_{begin_year}_e_{finish_year}'
out_tag = f'{tag}_mcp_{MIN_CENTRAL_PAPERS}'

FORCE_RECOMPUTE = False       # ignore any existing central-sets cache and rebuild

# Parallelism for building the central-paper sets (a CPU-bound pure-Python loop). This uses
# multiprocessing (fork), NOT threads -- Python threads would not help here because the GIL
# serialises pure-Python work. Fork lets workers share the big ForeCite dict copy-on-write
# (no pickling of the input). 1 -> run serially; None/0 -> use all CPUs.
PARALLEL_WORKERS = None

print("FORECITE_THRESHOLD", FORECITE_THRESHOLD, "| out_tag", out_tag,
      "| TOP_K_PAPERS", TOP_K_PAPERS)

# output dir: ./figures/forecite_gt_<threshold>/  (shared with step 2)
save_dir = os.path.join(BASE_DIR, 'figures', f'forecite_gt_{FORECITE_THRESHOLD}')
os.makedirs(save_dir, exist_ok=True)
print("save_dir", save_dir)

# each definition -> readable description (kept for logging / the sizes CSV)
DEFINITIONS = [
    ('max_score', 'MAX-score paper per keyword'),
    ('top5_score', f'TOP-{TOP_K_PAPERS} score papers per keyword'),
    ('earliest', 'EARLIEST (>0 score) paper per keyword'),
    ('earliest_above_mean', 'EARLIEST paper among those scoring > keyword mean'),
]


# ===========================================================================
# Central-set worker (module-level so multiprocessing 'fork' workers can call it; the big
# input dicts are shared copy-on-write via the globals below, set before the Pool is created).
# ===========================================================================
_W2P = None          # word -> {pid: score}
_CID2DATE = None     # cid -> pub date string
_CID2YEAR = None     # cid -> pub year


def _process_kw_chunk(args):
    """Build the four central-paper sets for a chunk of keywords. Returns
    (set_max, set_top5, set_earliest, set_above, scored_pids). Uses the module globals so no
    big data is pickled to the worker. `args` = (kw_chunk, show_progress, position)."""
    kw_chunk, show_progress, position = args
    w2p = _W2P
    pk_cache = {}   # cid -> (y, m, d); parse each date at most once (per worker)

    def pub_key(cid):
        c = pk_cache.get(cid)
        if c is not None:
            return c
        result = (10 ** 9, 0, 0)
        d = _CID2DATE.get(cid)
        if d:
            parts = str(d)[:10].split('-')
            try:
                y = int(parts[0])
                m = int(parts[1]) if len(parts) > 1 and parts[1] else 0
                day = int(parts[2]) if len(parts) > 2 and parts[2] else 0
                result = (y, m, day)
            except ValueError:
                pass
        if result == (10 ** 9, 0, 0):
            yv = _CID2YEAR.get(cid)
            if yv is not None:
                try:
                    result = (int(yv), 0, 0)
                except (ValueError, TypeError):
                    pass
        pk_cache[cid] = result
        return result

    set_max, set_top5, set_earliest, set_above, scored_pids = set(), set(), set(), set(), set()
    iterator = (tqdm(kw_chunk, desc='build central sets', position=position)
                if show_progress else kw_chunk)
    for kw in iterator:
        pid2s = w2p.get(kw, {})
        if not pid2s:
            continue
        scored_pids.update(pid2s.keys())
        topk = heapq.nlargest(TOP_K_PAPERS, pid2s,
                              key=lambda p: (pid2s[p], tuple(-x for x in pub_key(p)), -p))
        set_max.add(topk[0])
        set_top5.update(topk)
        scored = [p for p, s in pid2s.items() if s > 0]
        if scored:
            set_earliest.add(min(scored, key=lambda p: (pub_key(p), p)))
        # earliest_above_mean: restrict to the above-mean papers, then take the EARLIEST one --
        # ONE paper per keyword, not the whole above-mean set. Same date ordering (and pid
        # tie-break) as 'earliest' above; papers with no known date sort last.
        mean_s = sum(pid2s.values()) / len(pid2s)
        above = [p for p, s in pid2s.items() if s > mean_s]
        if above:
            set_above.add(min(above, key=lambda p: (pub_key(p), p)))
    return set_max, set_top5, set_earliest, set_above, scored_pids


def _build_central_sets(kw_list, word2pid2forecite_score, cid2pub_date, cid2pub_year):
    """Build the four central-paper sets over kw_list, in parallel (multiprocessing fork) when
    PARALLEL_WORKERS != 1 and fork is available; otherwise serially."""
    global _W2P, _CID2DATE, _CID2YEAR
    _W2P, _CID2DATE, _CID2YEAR = word2pid2forecite_score, cid2pub_date, cid2pub_year

    n_workers = (os.cpu_count() or 1) if not PARALLEL_WORKERS else PARALLEL_WORKERS
    n_workers = max(1, min(n_workers, len(kw_list)))

    sets = [set(), set(), set(), set(), set()]   # max, top5, earliest, above, scored
    if n_workers > 1:
        import multiprocessing as mp
        try:
            ctx = mp.get_context('fork')          # fork -> workers inherit the big dicts (no pickling)
        except ValueError:
            ctx = None
        if ctx is not None:
            # round-robin split keeps chunks balanced regardless of keyword ordering
            chunks = [kw_list[i::n_workers] for i in range(n_workers)]
            args = [(chunk, False, 0) for chunk in chunks]
            print("build central sets: %d workers (fork), %d keywords" % (n_workers, len(kw_list)))
            with ctx.Pool(n_workers) as pool:
                for partial in tqdm(pool.imap_unordered(_process_kw_chunk, args),
                                    total=len(args), desc='build central sets (workers)'):
                    for acc, part in zip(sets, partial):
                        acc |= part
            return sets
        print("fork unavailable -> serial")

    # serial fallback (single call, with the per-keyword progress bar)
    partial = _process_kw_chunk((kw_list, True, 0))
    return [set(p) for p in partial]


# ===========================================================================
# Build (or reuse) the central sets and save them.
# ===========================================================================
# Keyed ONLY by what the central sets depend on, so step 2's plotting params never invalidate it.
# 'defs' versions the DEFINITIONS themselves: the threshold/out_tag/TOP_K_PAPERS triple cannot
# tell that a definition's RULE changed, so any change to how a set is picked must bump it --
# otherwise an existing cache still looks valid and a stale set is silently reused.
#   v1 -> above_mean = ALL papers scoring above the keyword mean
#   v2 -> earliest_above_mean = the EARLIEST of those papers (one per keyword)
central_cache_path = os.path.join(save_dir, f'central_sets_{out_tag}.pkl')
central_key = {'threshold': FORECITE_THRESHOLD, 'out_tag': out_tag,
               'TOP_K_PAPERS': TOP_K_PAPERS, 'defs': 'v2_earliest_above_mean'}
print("central_cache_path", central_cache_path)

if not FORCE_RECOMPUTE and os.path.exists(central_cache_path):
    with open(central_cache_path, 'rb') as f:
        cc = pickle.load(f)
    if cc.get('key') == central_key:
        central_sets = cc['central_sets']
        scored_pids = cc['scored_pids']
        n_keywords = cc['n_keywords']
        print("central-sets cache already up to date (%d keywords) -> nothing to do:"
              % n_keywords, central_cache_path)
        print("   (set FORCE_RECOMPUTE=True to rebuild it)")
    else:
        print("central-sets config changed -> rebuilding:", central_cache_path)
        cc = None
else:
    cc = None

if cc is None or cc.get('key') != central_key:
    # ---- ForeCite dicts + publication dates/years (only needed to BUILD the sets) ----
    kw_dire = os.path.join(meta_dire, 'academic_keyword_forecite')
    with open(os.path.join(kw_dire, f'word2forecite_score_{out_tag}.pkl'), 'rb') as f:
        word2forecite_score = pickle.load(f)
    with open(os.path.join(kw_dire, f'word2pid2forecite_score_{out_tag}.pkl'), 'rb') as f:
        word2pid2forecite_score = pickle.load(f)
    with open(os.path.join(meta_dire, 'cid2pub_date.pkl'), 'rb') as f:
        cid2pub_date = pickle.load(f)
    with open(os.path.join(meta_dire, 'cid2pub_year.pkl'), 'rb') as f:
        cid2pub_year = pickle.load(f)
    print("keywords available:", len(word2forecite_score))

    kw_list = [kw for kw, s in word2forecite_score.items() if s > FORECITE_THRESHOLD]
    print("keywords with ForeCite score > %g: %d" % (FORECITE_THRESHOLD, len(kw_list)))
    assert kw_list, "no keywords above the threshold; lower FORECITE_THRESHOLD"

    set_max, set_top5, set_earliest, set_above, scored_pids = _build_central_sets(
        kw_list, word2pid2forecite_score, cid2pub_date, cid2pub_year)
    central_sets = {'max_score': set_max, 'top5_score': set_top5,
                    'earliest': set_earliest, 'earliest_above_mean': set_above}
    n_keywords = len(kw_list)
    with open(central_cache_path, 'wb') as f:
        pickle.dump({'key': central_key, 'central_sets': central_sets,
                     'scored_pids': scored_pids, 'n_keywords': n_keywords},
                    f, pickle.HIGHEST_PROTOCOL)
    print("saved central-sets cache:", central_cache_path)

# ---- report: how many papers each definition marks central ----
print("keywords used:", n_keywords, "| papers scoring any of them:", len(scored_pids))
size_rows = []
for name, desc in DEFINITIONS:
    n_central = len(central_sets[name])
    print("  central set %-11s: %8d unique papers  (%s)" % (name, n_central, desc))
    size_rows.append({'definition': name, 'description': desc,
                      'n_central': n_central, 'n_scored_pids': len(scored_pids),
                      'n_keywords': n_keywords})
sizes_csv = os.path.join(save_dir, f'central_set_sizes_{out_tag}.csv')
pd.DataFrame(size_rows).to_csv(sizes_csv, index=False)
print("saved central-set sizes:", sizes_csv)

print("finished -- now run forecite_central_papers_efg_s2_plot.py to draw the figures")
