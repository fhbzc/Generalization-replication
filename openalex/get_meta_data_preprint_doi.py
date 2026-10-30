import json
import os
import gzip
from tqdm import tqdm
import pickle
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import pandas as pd
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)               # utils.py is in this same directory
from utils import OPENALEX_DIRE, META_DIRE

path_openalex = OPENALEX_DIRE
save_dire = META_DIRE
os.makedirs(save_dire, exist_ok = True)
process_count = 1

print("save_dire",save_dire)
print("process_count",process_count)


def get_meta(dir_):
    pid2doi = {}
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

                            if 'doi' in line and line['doi'] is not None:
                                doi = line['doi']

                                doi = doi[16:]

                                pid2doi[id_] = doi
    return pid2doi



if os.path.exists(os.path.join(save_dire, 'pid2doi.pkl')) == False:
    with open(os.path.join(save_dire, 'pid2pub_year.pkl'), 'rb') as f:
        pid2pub_year = pickle.load(f)

    valid_pid_set = set(pid2pub_year.keys())


    path_work = os.path.join(path_openalex, 'openalex-snapshot/data/works/')
    dirs = sorted(os.listdir(path_work))

    pid2doi_M = {}

    for dire in tqdm(dirs, desc = 'iterate over files'):
        pid2doi_sig = get_meta(dire)
        pid2doi_M.update(pid2doi_sig)
    # assert False



    with open(os.path.join(save_dire, 'pid2doi.pkl'), 'wb') as f:
        pickle.dump(pid2doi_M, f, pickle.HIGHEST_PROTOCOL)


    print("size of pid2doi_M", len(pid2doi_M))


print("finished")