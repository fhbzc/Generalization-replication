"""
get_issn_top_domain.py

For each ISSN (venue), determines its top domain by:
  1. Loading pid2root_concepts_level_0 — keeps only pids with exactly one concept.
  2. Loading pid2primary_location_issn — maps pid -> issn_l.
  3. For each issn, counts how many of its papers belong to each domain,
     and assigns the plurality domain as the top_domain.

Output: META_DIRE/issn2top_domain.pkl  (dict: issn -> top_domain str)
"""

import os
import sys
import pickle
from collections import Counter
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)             # utils.py is in the parent directory
from utils import META_DIRE

print("meta_dire:", META_DIRE)

# ── 1. Load pid2root_concepts → keep only single-concept pids ────────────────
print("\nLoading pid2root_concepts_level_0 ...")
with open(os.path.join(META_DIRE, 'pid2root_concepts_level_0.pkl'), 'rb') as f:
    pid2root_concepts = pickle.load(f)
print(f"  total pids: {len(pid2root_concepts)}")

pid2domain = {
    pid: concepts[0]
    for pid, concepts in pid2root_concepts.items()
    if isinstance(concepts, list) and len(concepts) == 1
}
print(f"  pids with exactly one concept: {len(pid2domain)}")

# ── 2. Load pid2primary_location_issn ────────────────────────────────────────
print("\nLoading pid2primary_location_issn ...")
with open(os.path.join(META_DIRE, 'pid2primary_location_issn.pkl'), 'rb') as f:
    pid2issn = pickle.load(f)
print(f"  pid2issn size: {len(pid2issn)}")

# ── 3. Aggregate domain counts per issn ──────────────────────────────────────
print("\nAggregating domain counts per issn ...")
issn2domain_counter = {}

for pid, issn in tqdm(pid2issn.items(), desc='building issn domain counts'):
    if pid not in pid2domain:
        continue
    domain = pid2domain[pid]
    if issn not in issn2domain_counter:
        issn2domain_counter[issn] = Counter()
    issn2domain_counter[issn][domain] += 1

print(f"  issns with domain counts: {len(issn2domain_counter)}")

# ── 4. Pick plurality domain per issn ────────────────────────────────────────
issn2top_domain = {
    issn: counter.most_common(1)[0][0]
    for issn, counter in issn2domain_counter.items()
}
print(f"  issns with top domain: {len(issn2top_domain)}")

# ── 5. Save ───────────────────────────────────────────────────────────────────
out_path = os.path.join(META_DIRE, 'issn2top_domain.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(issn2top_domain, f, pickle.HIGHEST_PROTOCOL)
print(f"\nSaved to {out_path}")
print("Done.")
