"""
Extract a SINGLE field-of-study category per paper from the Semantic Scholar Academic Graph
(S2AG) "papers" dump -- the S2AG analogue of
openalex/get_meta_data_preprint_review_s2_simple_concept.py.

Same parsing as get_s2ag_meta_author_domain.py (gzipped JSONL shards, keyed by corpusid, a
paper's distinct s2fieldsofstudy categories), but instead of keeping the full category list per
paper this keeps exactly ONE category per cid, with the SAME LOGIC as the OpenAlex version:

  * a paper's "valid" categories are its distinct s2fieldsofstudy categories;
  * ONE of them is kept -- chosen at RANDOM. (Unlike OpenAlex concepts, S2AG s2fieldsofstudy
    categories carry no score, so there is no "highest score" rule: every category is equally
    likely. A paper with a single category keeps it directly.)
  * if a paper has no valid category, it is assigned 'Others'.

The RNG (used for the random pick) is seeded once at startup; reproducibility across runs is
NOT guaranteed (and not needed). Shards are split into batches and processed in parallel, and
the per-shard results merged.

Input  : S2AG_DIRE/papers/*.gz   (S2AG papers dump; S2AG_DIRE set in utils.py)
Outputs (to META_DIRE):
  cid2single_concept.pkl        cid -> chosen category display name (or 'Others')
  category2cid_list_single.pkl  category -> [cids assigned to it] (each cid under exactly ONE)
"""

import json
import os
import sys
import gzip
import pickle
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

# utils.py is in this same directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from utils import S2AG_DIRE, META_DIRE  # noqa: E402

file_dire = os.path.join(S2AG_DIRE, 'papers')
meta_dire = META_DIRE

# seed once; the per-paper category pick is then a plain (fast) random.choice. We do NOT care
# about reproducibility, so no per-cid re-seeding. (Each worker process re-imports this module
# and re-seeds, which is fine -- the picks need not be reproducible.)
random.seed(42)


def get_single_concept(file_path):
    """One shard -> ({cid: single category or 'Others'}, {category: [cids]}). For each record
    with a corpusid, collect its distinct s2fieldsofstudy categories and keep ONE (random when
    there is more than one); a record with no category is assigned 'Others'."""
    cid2single_concept = {}
    category2cid_list = defaultdict(list)
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get('corpusid') is None:
                continue
            cid = obj['corpusid']

            fields = obj.get('s2fieldsofstudy')
            if fields:
                categories = sorted({field['category'] for field in fields
                                     if field.get('category') is not None})
            else:
                categories = []

            if not categories:
                chosen = 'Others'
            elif len(categories) == 1:
                chosen = categories[0]
            else:
                chosen = random.choice(categories)

            cid2single_concept[cid] = chosen
            category2cid_list[chosen].append(cid)
    return cid2single_concept, category2cid_list


# main
if __name__ == "__main__":
    save_dire = meta_dire
    os.makedirs(save_dire, exist_ok=True)

    if os.path.exists(os.path.join(save_dire, 'cid2single_concept.pkl')):
        print("cid2single_concept.pkl already exists, nothing to do.")
        sys.exit(0)

    file_list = sorted(os.listdir(file_dire))
    batch_count = 32
    print("batch_count:", batch_count)

    batch2file = defaultdict(list)
    for i, file in enumerate(file_list):
        batch2file[i % batch_count].append(file)

    max_workers = 4
    print("Using workers:", max_workers)

    cid2single_concept_M = {}
    category2cid_list_M = defaultdict(list)

    if max_workers > 1:
        for batch, files in tqdm(batch2file.items(), desc="Batches"):
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(get_single_concept, os.path.join(file_dire, file))
                    for file in files
                ]
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc=f"Batch {batch}", leave=False):
                    cid2single_concept, category2cid_list = future.result()
                    cid2single_concept_M.update(cid2single_concept)
                    for category, cid_list in category2cid_list.items():
                        category2cid_list_M[category].extend(cid_list)
    else:
        for batch, files in tqdm(batch2file.items(), desc="Batches"):
            for file in files:
                cid2single_concept, category2cid_list = get_single_concept(
                    os.path.join(file_dire, file))
                cid2single_concept_M.update(cid2single_concept)
                for category, cid_list in category2cid_list.items():
                    category2cid_list_M[category].extend(cid_list)

    with open(os.path.join(save_dire, 'cid2single_concept.pkl'), 'wb') as f:
        pickle.dump(cid2single_concept_M, f, pickle.HIGHEST_PROTOCOL)

    # each cid appears under exactly one category, but dedupe defensively before saving
    category2cid_list_M = {category: list(set(cid_list))
                           for category, cid_list in tqdm(category2cid_list_M.items())}
    with open(os.path.join(save_dire, 'category2cid_list_single.pkl'), 'wb') as f:
        pickle.dump(category2cid_list_M, f, pickle.HIGHEST_PROTOCOL)

    print("size of cid2single_concept", len(cid2single_concept_M))
    print("number of categories", len(category2cid_list_M))
    print("finished")
