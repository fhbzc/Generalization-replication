"""
Extract each paper's venue from the Semantic Scholar Academic Graph (S2AG) "papers"
dump (gzipped JSONL shards, one paper per line). For every record keyed by corpusid,
record its 'venue' string (papers with a null/missing venue are skipped). Shards are
split into batches and processed in parallel, and the per-shard dicts merged.

Input  : S2AG_DIRE/papers/*.gz   (S2AG papers dump; S2AG_DIRE set in utils.py)
Output : META_DIRE/cid2venue.pkl  (corpusid -> venue string)
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


# single-file worker (runs in a subprocess)
def get_meta(file_path):
    cid2venue = {}
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            if 'corpusid' in obj and obj['corpusid'] is not None:
                cid = obj['corpusid']
            else:
                continue

            venue = obj.get('venue')
            if venue is not None:
                cid2venue[cid] = venue
    return cid2venue


if __name__ == "__main__":

    save_dire = meta_dire
    os.makedirs(save_dire, exist_ok=True)

    file_list = sorted(os.listdir(file_dire))
    batch_count = 32
    print("batch_count:", batch_count)

    batch2file = defaultdict(list)
    for i, file in enumerate(file_list):
        batch2file[i % batch_count].append(file)

    max_workers = 4
    print("Using workers:", max_workers)

    cid2venue_M = {}
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
                    cid2venue = future.result()
                    cid2venue_M.update(cid2venue)

    else:

        for batch, files in tqdm(batch2file.items(), desc="Batches"):

            for file in files:

                cid2venue = get_meta(os.path.join(file_dire, file))
                cid2venue_M.update(cid2venue)

    with open(os.path.join(meta_dire, 'cid2venue.pkl'), 'wb') as f:
        pickle.dump(cid2venue_M, f, pickle.HIGHEST_PROTOCOL)

    print("size of cid2venue_M", len(cid2venue_M))

    print("finished")
