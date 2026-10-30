import json
import os
import sys
import gzip
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import re

# utils.py lives in the parent (semantic_scholar) directory; make it importable regardless of CWD
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
from utils import S2AG_DIRE, META_DIRE

file_dire = os.path.join(S2AG_DIRE, 'abstracts')
meta_dire = META_DIRE
save_dire = os.path.join(meta_dire, 'cid2abs_year')
os.makedirs(save_dire, exist_ok = True)
print("V1")

# =====================
# 正则
# =====================


with open(os.path.join(meta_dire,'cid2pub_year.pkl'),'rb') as f:
    cid2pub_year = pickle.load(f)
print("load cid2pub_year", len(cid2pub_year))
# =====================
# 单文件处理函数（子进程）
# =====================
def get_meta(file_path):
    
    pub_year2cid2abstract = defaultdict(dict)
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            if 'corpusid' in obj and obj['corpusid'] is not None:
                cid = obj['corpusid']
            else:
                continue

            if cid not in cid2pub_year:
                continue

            pub_year = cid2pub_year[cid]
            if (pub_year < 1951) or (pub_year > 2025):
                continue
            if 'abstract' in obj and obj['abstract'] is not None and len(obj['abstract']) > 0:
                pub_year2cid2abstract[pub_year][cid] = obj['abstract']

    return pub_year2cid2abstract


# =====================
# 主程序
# =====================
if __name__ == "__main__":
    # file_list = os.listdir(file_dire)
    # for file in file_list:
    #     get_meta(os.path.join(file_dire, file))
    
    # assert False
    file_list = sorted(os.listdir(file_dire))
    batch_count = 1
    print("batch_count:", batch_count)

    batch2file = defaultdict(list)
    for i, file in enumerate(file_list):
        batch2file[i % batch_count].append(file)

    # 建议：CPU 核数 - 1
    max_workers = 4
    print("Using workers:", max_workers)


    pub_year2cid2abstract_M = defaultdict(dict)
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
                    pub_year2cid2abstract = future.result()
                    for pub_year, cid2abs in pub_year2cid2abstract.items():
                        pub_year2cid2abstract_M[pub_year].update(cid2abs)


    else:

        for batch, files in tqdm(batch2file.items(), desc="Batches"):


            for file in files:


                pub_year2cid2abstract = get_meta(os.path.join(file_dire, file))
                for pub_year, cid2abs in pub_year2cid2abstract.items():
                    pub_year2cid2abstract_M[pub_year].update(cid2abs)


    for pub_year, cid2abs in tqdm(pub_year2cid2abstract_M.items()):
        with open(os.path.join(save_dire,f'cid2pub_date_y_{str(pub_year)}.pkl'),'wb') as f:
            pickle.dump(pub_year2cid2abstract_M[pub_year], f, pickle.HIGHEST_PROTOCOL)

            
    print("finished")
