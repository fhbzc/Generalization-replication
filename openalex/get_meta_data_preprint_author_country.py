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
    pid2author_list = {}
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

                        # if True:                        
                        if id_ in valid_pid_set:
                            author_list = []
                            if 'authorships' in line and line['authorships'] is not None:

                                for author in line['authorships']:
                                    if 'author' in author and 'id' in author['author'] and author['author']['id'] is not None:
                                        
                                        institution_country_list = []
                                        if 'institutions' in author and author['institutions'] is not None:
                                            for institution in author['institutions']:
                                                if 'country_code' in institution and institution['country_code'] is not None:

                                                    institution_country_list.append(institution['country_code'])

                                        institution_country_list = list(set(institution_country_list))
                                        author_list.append((author['author']['id'][21:], institution_country_list))

                                if len(author_list) > 0:
                                    pid2author_list[id_] = author_list
    return pid2author_list



if os.path.exists(os.path.join(save_dire, 'pid2author_country_list.pkl')) == False:
    with open(os.path.join(save_dire, 'pid2pub_year.pkl'), 'rb') as f:
        pid2pub_year = pickle.load(f)

    valid_pid_set = set(pid2pub_year.keys())


    path_work = os.path.join(path_openalex, 'openalex-snapshot/data/works/')
    dirs = sorted(os.listdir(path_work))

    pid2author_list_M = {}

    for dire in tqdm(dirs, desc = 'iterate over files'):
        pid2author_list_sig = get_meta(dire)
        pid2author_list_M.update(pid2author_list_sig)


    with open(os.path.join(save_dire, 'pid2author_country_list.pkl'), 'wb') as f:
        pickle.dump(pid2author_list_M, f, pickle.HIGHEST_PROTOCOL)

    print("size of pid2author_list_M", len(pid2author_list_M))


print("finished")