import json
import os
import gzip
from tqdm import tqdm
import pickle
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import pandas as pd
import sys
# utils.py lives one level up (openalex/), not in this text_analysis/ directory;
# add the parent directory to the path so the import resolves regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import META_DIRE, OPENALEX_DIRE


path_openalex = OPENALEX_DIRE
meta_dire = META_DIRE
os.makedirs(meta_dire, exist_ok = True)
process_count = 1

print("meta_dire",meta_dire)
parse_method = 'lemma'
year_before = 2025
year_after = 2020
print("parse_method",parse_method)
print("process_count",process_count)

print("year_before",year_before)
print("year_after",year_after)


with open(os.path.join(meta_dire, 'pid2pub_year.pkl'), 'rb') as f:
    pid2pub_year = pickle.load(f)
print("load pid2pub_year finished")

valid_pid_set = set([pid for pid, pub_year in tqdm(pid2pub_year.items(), desc = 'process year', leave = False) if pub_year >= year_after])
del pid2pub_year
print("size of valid_pid_set s1", len(valid_pid_set))

with open(os.path.join(meta_dire, 'pid2eng_token_list_%s_before_%s.pkl'%(parse_method, year_before)), 'rb') as f:
    pid2token_list = pickle.load(f)
print("load pid2token_list finished")

valid_pid_set &= set(pid2token_list)
del pid2token_list


print("size of valid_pid_set s2", len(valid_pid_set))
def get_abs(dir_):
    pid2abstract = {}
    
    # for dir_ in tqdm(dirs, desc = 'process works'):
    if '_updated_date' not in dir_ and 'updated_date' in dir_:
        files = sorted(os.listdir(os.path.join(path_work, dir_)))
        for file in files:
            if '.gz' in file and '_part' not in file:
                # Open the .gz file in read mode
                main_dire = os.path.join(path_work, dir_, file)
                with gzip.open(main_dire, 'rb') as f:
                    for line in tqdm(f, position = 1, desc = 'process works in file %s'%file, leave = False):
                        line = json.loads(line)

                        id_ = line['id'][21:]


                        if id_ in valid_pid_set:

                            if 'abstract_inverted_index' in line and line['abstract_inverted_index'] is not None:
                                inverted_index = line['abstract_inverted_index']
                                if len(inverted_index) == 0:
                                    continue
                                total_length = max(pos for positions in inverted_index.values() for pos in positions) + 1
                                words_list = [""] * total_length

                                for word, positions in inverted_index.items():
                                    for pos in positions:
                                        words_list[pos] = word

                                # 3. Join into a single string
                                abstract = " ".join(words_list)
                                pid2abstract[id_] = abstract
    return pid2abstract

path_work = os.path.join(path_openalex, 'openalex-snapshot/data/works/')
dirs = sorted(os.listdir(path_work))
pid2abstract_M = {}

if process_count > 1:

    with ProcessPoolExecutor(max_workers=process_count) as executor:
        with tqdm(total=len(dirs)) as pbar:
            for pid2abstract in executor.map(get_abs, dirs):
                pbar.update(1)
                
                pid2abstract_M.update(pid2abstract)
else:
    for dire in tqdm(dirs):
        pid2abstract_M.update(get_abs(dire))

with open(os.path.join(meta_dire, 'pid2abstract_method_%s_before_%s_after_%s.pkl'%(parse_method, year_before, year_after)), 'wb') as f:
    pickle.dump(pid2abstract_M, f, pickle.HIGHEST_PROTOCOL)
print("size of pid2abstract_M", len(pid2abstract_M))
print("finished")