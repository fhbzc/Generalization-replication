"""
Select, for each keyword, all pids whose TITLE contains that keyword.

Matching PREFERS whole-token equality against the lemmatized title tokens: if
META_DIRE/pid2eng_token_list_lemma_before_2025.pkl exists (from parse_title_to_token.py),
a keyword is selected for a pid only when it equals one of that title's tokens EXACTLY.
So keyword "review" matches the token "review" but NOT "reviews" -- a plain substring scan
would wrongly match the latter. Keywords are lower-cased before comparing.

If the token file is absent, it falls back to case-insensitive SUBSTRING matching on the
raw title (a keyword "rna" then also matches a title "mRNA vaccine"). Note: whole-token
matching only makes sense for single-token keywords; a multi-word phrase can match only in
the substring fallback.

Output: one pickle file PER keyword under META_DIRE/keyword_pids/, each holding
    [keyword, [pid, pid, ...]]
i.e. the first element is the (lower-cased) keyword name and the second is the
list of all pids whose title contains it.

Per-keyword files let the script be incremental: a keyword that already has a
saved file is skipped, and if EVERY keyword is already saved the script exits
immediately without even loading the (large) token/title file.

Input  : META_DIRE/pid2eng_token_list_lemma_before_2025.pkl   (preferred; whole-token match)
         falls back to META_DIRE/pid2title_no_html_before_2025.pkl (HTML-stripped titles)
         or           META_DIRE/pid2title.pkl                      (raw titles)
Output : META_DIRE/keyword_pids/<safe_keyword>.pkl      (one per keyword)
"""

import os
import re
import pickle
import sys
from tqdm import tqdm

# utils.py lives one level up (openalex/), not in this text_analysis/ directory;
# add the parent directory to the path so the import resolves regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import META_DIRE

meta_dire = META_DIRE
print("meta_dire", meta_dire)

# ==================================================
# keywords to search for (fill these in; case does not matter, they get lowercased)
# ==================================================
keyword_list = ['device', 'tool', 'software', 'review','guideline', 'tutorial', 'new', 'novel', 'innovative',
                'theory', 'metric', 'hypothesis', 'discovery', 'dataset', 'observation', 'overview', 'method']

# normalize: lower-case + drop blanks/duplicates while preserving order
seen = set()
keywords = []
for kw in keyword_list:
    kw = kw.lower().strip()
    if kw and kw not in seen:
        seen.add(kw)
        keywords.append(kw)
print("keywords:", keywords)

# ==================================================
# per-keyword output paths; a keyword whose file already exists is skipped
# ==================================================
out_dir = os.path.join(meta_dire, 'keyword_pids')
os.makedirs(out_dir, exist_ok=True)

def keyword_path(kw):
    # map a keyword to a stable, filesystem-safe filename (spaces/punct -> '_')
    safe = re.sub(r'[^0-9a-z]+', '_', kw).strip('_') or 'kw'
    return os.path.join(out_dir, f'{safe}.pkl')

todo = [kw for kw in keywords if not os.path.exists(keyword_path(kw))]
done = [kw for kw in keywords if kw not in todo]
if done:
    print(f"already saved ({len(done)}): {done}")

# if every keyword is already saved, exit before loading the large title file
if not todo:
    print("all keywords already saved; nothing to do.")
    print("finished")
    sys.exit(0)

print(f"to process ({len(todo)}): {todo}")

# ==================================================
# choose the matching source: PREFER the lemmatized whole-token list (exact-token match);
# fall back to substring matching on the raw title if that file is missing.
# ==================================================
# year_threshold must match parse_title_to_token.py (which wrote before_2025)
year_threshold = 2025
token_path = os.path.join(meta_dire, f'pid2eng_token_list_lemma_before_{year_threshold}.pkl')

keyword2pids = {kw: [] for kw in todo}

if os.path.exists(token_path):
    # ---- whole-token match: keyword selected iff it is EXACTLY one of the title's tokens ----
    print("token_path (whole-token match):", token_path)
    with open(token_path, 'rb') as f:
        pid2token_list = pickle.load(f)
    print("pid2token_list load finished | size:", len(pid2token_list))

    for pid, token_list in tqdm(pid2token_list.items(), desc='scan tokens'):
        if not token_list:
            continue
        token_set = {t.lower() for t in token_list}  # lower-cased; whole-token equality
        for kw in todo:
            if kw in token_set:              # "review" matches token "review", not "reviews"
                keyword2pids[kw].append(pid)
else:
    # ---- fallback: case-insensitive SUBSTRING match on titles (HTML-stripped preferred) ----
    title_no_html_path = os.path.join(meta_dire, 'pid2title_no_html_before_2025.pkl')
    title_raw_path = os.path.join(meta_dire, 'pid2title.pkl')
    title_path = title_no_html_path if os.path.exists(title_no_html_path) else title_raw_path
    print("token file not found -> substring title match. title_path:", title_path)
    assert os.path.exists(title_path), title_path

    with open(title_path, 'rb') as f:
        pid2title = pickle.load(f)
    print("pid2title load finished | size:", len(pid2title))

    for pid, title in tqdm(pid2title.items(), desc='scan titles'):
        if title is None:
            continue
        title_l = title.lower()
        for kw in todo:
            if kw in title_l:
                keyword2pids[kw].append(pid)

# ==================================================
# save one file per keyword -> [keyword, [pids...]]
# ==================================================
for kw in todo:
    pids = keyword2pids[kw]
    path = keyword_path(kw)
    with open(path, 'wb') as f:
        pickle.dump([kw, pids], f, pickle.HIGHEST_PROTOCOL)
    print(f"  '{kw}': {len(pids)} pids -> {path}")

print("finished")
