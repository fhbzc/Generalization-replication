"""
Extract per-paper metadata from the Semantic Scholar Academic Graph (S2AG)
"papers" dump (gzipped JSONL shards, one paper per line). For each record (keyed
by corpusid) collect publication year, publication date, title, and DOI (from
externalids). Shards are split into batches and processed in parallel, and the
per-shard dicts merged.

Input  : S2AG_DIRE/papers/*.gz   (S2AG papers dump; S2AG_DIRE set in utils.py)
Outputs: META_DIRE/{cid2pub_year, cid2pub_date, cid2title, cid2DOI}.pkl
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
    corpus_id2pub_year = {}
    corpus_id2pub_date = {}
    corpus_id2title = {}
    corpus_id2DOI = {}
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            if 'corpusid' in obj and obj['corpusid'] is not None:
                cid = obj['corpusid']
            else:
                continue

            if 'externalids' in obj and 'DOI' in obj['externalids'] and obj['externalids']['DOI'] is not None:
                doi = obj['externalids']['DOI']
                corpus_id2DOI[cid] = doi

            if 'title' in obj:
                corpus_id2title[cid] = obj['title']

            if 'year' in obj and obj['year'] is not None:
                corpus_id2pub_year[cid] = obj['year']

            if 'publicationdate' in obj and obj['publicationdate'] is not None:
                corpus_id2pub_date[cid] = obj['publicationdate']

    return corpus_id2pub_year, corpus_id2pub_date, corpus_id2title, corpus_id2DOI

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


    corpus_id2pub_year_M = {}
    corpus_id2pub_date_M = {}
    corpus_id2title_M = {}
    corpus_id2DOI_M = {}
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
                    corpus_id2pub_year, corpus_id2pub_date, corpus_id2title, corpus_id2DOI = future.result()
                    corpus_id2pub_year_M.update(corpus_id2pub_year)
                    corpus_id2pub_date_M.update(corpus_id2pub_date)
                    corpus_id2title_M.update(corpus_id2title)
                    corpus_id2DOI_M.update(corpus_id2DOI)

    else:

        for batch, files in tqdm(batch2file.items(), desc="Batches"):


            for file in files:

                corpus_id2pub_year, corpus_id2pub_date, corpus_id2title, corpus_id2DOI = get_meta(os.path.join(file_dire, file))
                corpus_id2pub_year_M.update(corpus_id2pub_year)
                corpus_id2pub_date_M.update(corpus_id2pub_date)
                corpus_id2title_M.update(corpus_id2title)
                corpus_id2DOI_M.update(corpus_id2DOI)





    with open(os.path.join(save_dire,'cid2pub_year.pkl'),'wb') as f:
        pickle.dump(corpus_id2pub_year_M, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_dire,'cid2pub_date.pkl'),'wb') as f:
        pickle.dump(corpus_id2pub_date_M, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_dire,'cid2title.pkl'),'wb') as f:
        pickle.dump(corpus_id2title_M, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_dire,'cid2DOI.pkl'),'wb') as f:
        pickle.dump(corpus_id2DOI_M, f, pickle.HIGHEST_PROTOCOL)

            
    print("finished")
