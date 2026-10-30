"""
Extract per-paper field-of-study categories and author counts from the Semantic
Scholar Academic Graph (S2AG) "papers" dump (gzipped JSONL shards, one paper per
line). For each record (keyed by corpusid) collect its distinct s2fieldsofstudy
categories and its number of authors, and build the inverse category -> [corpusid]
index. Shards are split into batches and processed in parallel, and the per-shard
results merged.

Input  : S2AG_DIRE/papers/*.gz   (S2AG papers dump; S2AG_DIRE set in utils.py)
Outputs: META_DIRE/{cid2category_list, cid2author_count, category2cid_list}.pkl
"""

import json
import os
import sys
import gzip
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

# utils.py is in this same directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from utils import S2AG_DIRE, META_DIRE  # noqa: E402

file_dire = os.path.join(S2AG_DIRE, 'papers')
meta_dire = META_DIRE


def get_meta(file_path):
    cid2category_list = {}
    cid2author_count = {}
    category2cid_list = defaultdict(list)


    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            if 'corpusid' in obj and obj['corpusid'] is not None:
                cid = obj['corpusid']
            else:
                continue

            if obj['s2fieldsofstudy'] is not None:
                category_set = set([field['category'] for field in obj['s2fieldsofstudy']])


                for category in category_set:
                    assert category is not None

                    category2cid_list[category].append(cid)
            else:
                category_set = set()

            if len(category_set) > 0:
                cid2category_list[cid] = list(category_set)
            if obj['authors'] is not None:

                author_count = len(obj['authors'])

                cid2author_count[cid] = author_count
    return cid2category_list, cid2author_count, category2cid_list


# main
if __name__ == "__main__":

    file_list = sorted(os.listdir(file_dire))
    batch_count = 32
    print("batch_count:", batch_count)

    batch2file = defaultdict(list)
    for i, file in enumerate(file_list):
        batch2file[i % batch_count].append(file)

    save_dire = meta_dire
    os.makedirs(save_dire, exist_ok=True)

    max_workers = 4
    print("Using workers:", max_workers)


    cid2category_list_M = {}
    cid2author_count_M = {}
    category2cid_list_M = defaultdict(list)
    if max_workers > 1:
        for batch, files in tqdm(batch2file.items(), desc="Batches"):


            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        get_meta,
                        os.path.join(file_dire, file)
                    )
                    for file in files
                ]

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Batch {batch}",
                    leave=False
                ):
                    cid2category_list, cid2author_count, category2cid_list = future.result()
                    cid2category_list_M.update(cid2category_list)
                    cid2author_count_M.update(cid2author_count)

                    for category, cid_list in category2cid_list.items():
                        category2cid_list_M[category].extend(cid_list)
    else:

        for batch, files in tqdm(batch2file.items(), desc="Batches"):


            for file in files:

                cid2category_list, cid2author_count, category2cid_list = get_meta(os.path.join(file_dire, file))
                cid2category_list_M.update(cid2category_list)
                cid2author_count_M.update(cid2author_count)

                for category, cid_list in category2cid_list.items():
                    category2cid_list_M[category].extend(cid_list)



    with open(os.path.join(save_dire,'cid2category_list.pkl'),'wb') as f:
        pickle.dump(cid2category_list_M, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_dire,'cid2author_count.pkl'),'wb') as f:
        pickle.dump(cid2author_count_M, f, pickle.HIGHEST_PROTOCOL)

    category2cid_list_M = {category: list(set(cid_list)) for category, cid_list in tqdm(category2cid_list_M.items())}
    with open(os.path.join(save_dire,'category2cid_list.pkl'),'wb') as f:
        pickle.dump(category2cid_list_M, f, pickle.HIGHEST_PROTOCOL)

    print("size of cid2category_list_M", len(cid2category_list_M))
    print("size of cid2author_count_M", len(cid2author_count_M))
    print("size of category2cid_list_M", len(category2cid_list_M))

    print("finished")
