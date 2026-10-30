import json
import os
import gzip
from tqdm import tqdm
import pickle
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)               # utils.py is in this same directory
from utils import OPENALEX_DIRE, META_DIRE



path_openalex = OPENALEX_DIRE
# save to a separate directory so the article/preprint-only meta files are not overwritten
save_dire = META_DIRE + '_with_review'
os.makedirs(save_dire, exist_ok = True)
process_count = 1

# work types kept in the meta data (now also includes 'review')
KEEP_TYPES = {'article', 'preprint', 'review'}

print("save_dire",save_dire)
print("process_count",process_count)
print("KEEP_TYPES",KEEP_TYPES)

def get_valid_pid(file_path):
    pid2year_single = {}
    pid2pub_date_single = {}
    pid2title_single = {}
    pid2primary_location_single = {}
    id2displayname_single = {}
    # file-level worker: process a single .gz file (parallelism is over files)
    with gzip.open(file_path, 'rb') as f:
        for line in f:
            line = json.loads(line)

            id_ = line['id'][21:]


            if line and line['publication_year'] and line['authorships'] and line['type'] and (line['type'] in KEEP_TYPES):
                # try:
                pid2year_single[id_] = line['publication_year']
                title = line['title']
                pid2pub_date_single[id_] = line['publication_date']
                pid2title_single[id_] = title
                # venue (primary location source), same as get_meta_data_preprint_primary_location.py
                if (
                    line.get('primary_location')
                    and line['primary_location'].get('source')
                    and line['primary_location']['source'].get('id')
                ):
                    source = line['primary_location']['source']
                    source_id = source['id'][21:]
                    pid2primary_location_single[id_] = source_id
                    # source id -> human-readable venue name
                    id2displayname_single[source_id] = source.get('display_name')

    return pid2year_single, pid2pub_date_single, pid2title_single, pid2primary_location_single, id2displayname_single



def get_reference(file_path):
    pid2up_refer_single = {}
    pid2down_refer_single = defaultdict(list)
    # file-level worker: process a single .gz file (parallelism is over files)
    with gzip.open(file_path, 'rb') as f:
        for line in f:
            line = json.loads(line)

            id_ = line['id'][21:]

            if id_ in valid_pid_set:
                source_work_date = pid2pub_date[id_]
                if line['referenced_works']:
                    refs = set([r[21:] for r in line['referenced_works'] if r[21:] in valid_pid_set and pid2pub_date[r[21:]] < source_work_date])
                    # this should remove items where the reference has a size of 0
                    refs = list(refs)
                    pid2up_refer_single[id_] = refs
                    for ref in refs:
                        pid2down_refer_single[ref].append(id_)
    return pid2up_refer_single, pid2down_refer_single


path_work = os.path.join(path_openalex, 'openalex-snapshot/data/works/')
dirs = sorted(os.listdir(path_work))

# ===============================
# Collect all valid gz files (file-level parallelism, like get_meta_data_preprint_arxiv.py)
# ===============================
gz_files = []
for d in dirs:
    if '_updated_date' not in d and 'updated_date' in d:
        dpath = os.path.join(path_work, d)
        for fname in sorted(os.listdir(dpath)):
            if fname.endswith('.gz') and '_part' not in fname:
                gz_files.append(os.path.join(dpath, fname))
print("Total gz files:", len(gz_files))



if os.path.exists(os.path.join(save_dire, 'source_id2display_name.pkl')) ==  False:
    JournalArticle_Year = {}
    pid2title = {}
    pid2pub_date = {}
    pid2primary_location = {}
    id2displayname = {}
    with ProcessPoolExecutor(max_workers=process_count) as executor:
        with tqdm(total=len(gz_files)) as pbar:
            for pid2year_single, pid2pub_date_single, pid2title_single, pid2primary_location_single, id2displayname_single in executor.map(get_valid_pid, gz_files):
                pbar.update(1)
                JournalArticle_Year.update(pid2year_single)
                pid2pub_date.update(pid2pub_date_single)
                pid2title.update(pid2title_single)
                pid2primary_location.update(pid2primary_location_single)
                id2displayname.update(id2displayname_single)
    with open(os.path.join(save_dire, 'pid2pub_year.pkl'), 'wb') as f:
        pickle.dump(JournalArticle_Year, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_dire, 'pid2pub_date.pkl'), 'wb') as f:
        pickle.dump(pid2pub_date, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_dire, 'pid2title.pkl'), 'wb') as f:
        pickle.dump(pid2title, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_dire, 'pid2primay_location.pkl'), 'wb') as f:
        pickle.dump(pid2primary_location, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_dire, 'source_id2display_name.pkl'), 'wb') as f:
        pickle.dump(id2displayname, f, pickle.HIGHEST_PROTOCOL)




else:

    with open(os.path.join(save_dire, 'pid2pub_date.pkl'), 'rb') as f:
        pid2pub_date = pickle.load(f)

valid_pid_set = set(pid2pub_date)
if os.path.exists(os.path.join(save_dire, 'cited_pid2citing_pids_list.pkl')) ==  False:
    pid2up_refer = {}
    pid2down_citation = {}

    if process_count > 1:
        with ProcessPoolExecutor(max_workers=process_count) as executor:
            with tqdm(total=len(gz_files), desc = 'process reference') as pbar:
                for pid2up_refer_single, pid2down_refer_single in executor.map(get_reference, gz_files):
                    pbar.update(1)
                    pid2up_refer.update(pid2up_refer_single)
                    for pid in pid2down_refer_single:
                        if pid not in pid2down_citation:
                            pid2down_citation[pid] = []

                        pid2down_citation[pid].extend(pid2down_refer_single[pid])
    else:
        for file_path in tqdm(gz_files, desc = 'process reference'):
            pid2up_refer_single, pid2down_refer_single = get_reference(file_path)
            pid2up_refer.update(pid2up_refer_single)
            for pid in pid2down_refer_single:
                if pid not in pid2down_citation:
                    pid2down_citation[pid] = []

                pid2down_citation[pid].extend(pid2down_refer_single[pid])
    with open(os.path.join(save_dire, 'citing_pid2cited_pids_list.pkl'), 'wb') as f:
        pickle.dump(pid2up_refer, f, pickle.HIGHEST_PROTOCOL)

    pid2down_citation = {pid: list(set(pid2down_citation[pid])) for pid in tqdm(pid2down_citation, desc = 'process pid2down_citation')}
    with open(os.path.join(save_dire, 'cited_pid2citing_pids_list.pkl'), 'wb') as f:
        pickle.dump(pid2down_citation, f, pickle.HIGHEST_PROTOCOL)

    print("size of pid2up_refer", len(pid2up_refer))
    print("size of pid2down_citation", len(pid2down_citation))


print("finished")
