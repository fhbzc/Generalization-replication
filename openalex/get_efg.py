import os
import pickle
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import random
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)               # utils.py is in this same directory
from  utils import META_DIRE


process_count = 3
print("process_count",process_count)

# Choose the meta directory. The review-inclusive meta (get_meta_data_preprint_review.py,
# KEEP_TYPES includes 'review') writes to META_DIRE + '_with_review' using the SAME
# filenames as the article/preprint-only run. Switching the base directory therefore makes
# get_efg.py read that variant's pid2pub_year / citation-network pkls and write its E/F/G
# output under the same separate tree (efg_computation/ there), so the two runs never
# collide. Mirrors USE_REVIEW_META in venue_validation/get_annual_review_papers_and_refs.py.
use_with_review = False
meta_dire = META_DIRE + '_with_review' if use_with_review else META_DIRE
print("use_with_review", use_with_review)

print("meta_dire",meta_dire)

division_count = 20
print("division_count",division_count)


max_year = 2025
print("max_year", max_year)



year_offset = -1 # -1 for no offset

print("year_offset", year_offset)


# Which citation network to compute E/F/G on:
#   'normal'           -> citing/cited_pid2cited_pids_list.pkl (full network, get_meta_data_preprint.py)
#   'no_self_citation' -> ..._no_self_citation.pkl  (get_citation_network_no_self_citation.py;
#                         citing<->cited pairs that share an author are dropped)
#   'no_year_filter'   -> ..._no_year_filter.pkl    (get_meta_data_preprint_no_year_filter.py;
#                         references kept regardless of publication-year ordering)
# The choice is encoded in both the input network filenames and the output filenames
# (network_suffix) so the variants never overwrite each other.
network_mode = 'normal'
NETWORK_SUFFIX_MAP = {
    'normal': '',
    'no_self_citation': '_no_self_citation',
    'no_year_filter': '_no_year_filter',
}
assert network_mode in NETWORK_SUFFIX_MAP, network_mode
network_suffix = NETWORK_SUFFIX_MAP[network_mode]
print("network_mode", network_mode)
print("network_suffix", repr(network_suffix))


# normal mode only: loads the real citation network. The reshuffled-network
# variant lives in atypicality/get_efg_reshuffle.py.
with open(os.path.join(meta_dire, 'pid2pub_year.pkl'), 'rb') as f:
    pid2year = pickle.load(f)
print("pid2year load finished")





def disp_analysis(pid_list, 
                  batch_index):



    pid2e = {}
    pid2f = {}
    pid2g = {}
    pid2total_citation = {}
    pid2up_refer = {}

    for focal_pid in tqdm(pid_list, desc = 'batch %s'%batch_index, position = 3, leave = False):

        e_count = 0
        f_count = 0
        g_count = 0
        focal_pub_year = pid2year[focal_pid]
        # year_offset == -1: keep focal papers up to AND including max_year (<= 2025 all count);
        # a finite offset still drops focal_pub_year >= max_year (incomplete citation window).
        if focal_pub_year > (max_year if year_offset == -1 else max_year - 1) or focal_pub_year < 1945:
            continue
        focal_reference_set = citing_pid2cited_works.get(focal_pid, set())

        max_year_threshold = min(focal_pub_year + year_offset, max_year - 1)

        citing_focal_papers = set()

        if year_offset == -1:
            # all future citations up to AND including max_year (<= 2025 all count)
            citing_focal_papers = set([citing_work for citing_work in cited_pid2citing_works.get(focal_pid, []) if pid2year[citing_work] <= max_year])

        else:

            if focal_pid in cited_pid2citing_works:
                citing_focal_papers = set([citing_work for citing_work in cited_pid2citing_works[focal_pid] if pid2year[citing_work] <= max_year_threshold])


        for citing_paper in citing_focal_papers:
            up_of_down = citing_pid2cited_works.get(citing_paper, set())

            up_count = len(focal_reference_set & up_of_down)
            down_count = len(citing_focal_papers & up_of_down)

            if up_count == 0 and down_count == 0:
                g_count += 1
            else:
                if up_count == down_count:
                    e_count += 0.5
                    f_count += 0.5
                else:
                    if up_count > down_count:
                        e_count += 1
                    else:
                        f_count += 1

        pid2total_citation[focal_pid] = len(citing_focal_papers)
        pid2up_refer[focal_pid] = len(focal_reference_set)
        pid2e[focal_pid] = e_count
        pid2f[focal_pid] = f_count
        pid2g[focal_pid] = g_count

    return pid2e, pid2f, pid2g, pid2up_refer, pid2total_citation



save_dire = os.path.join(meta_dire, 'efg_computation')
os.makedirs(save_dire, exist_ok = True)

citing_pid2cited_pids_path = os.path.join(meta_dire, f'citing_pid2cited_pids_list{network_suffix}.pkl')
cited_pid2citing_pids_path = os.path.join(meta_dire, f'cited_pid2citing_pids_list{network_suffix}.pkl')

print("citing_pid2cited_pids_path", citing_pid2cited_pids_path)
print("cited_pid2citing_pids_path", cited_pid2citing_pids_path)

with open(citing_pid2cited_pids_path, 'rb') as f:
    citing_pid2cited_works = pickle.load(f)

print("citing_pid2cited_works load finished")

citing_pid2cited_works = {pid: set(pid_list) for pid, pid_list in tqdm(citing_pid2cited_works.items(), desc = 'process citing_pid2cited_works')}
with open(cited_pid2citing_pids_path, 'rb') as f:
    cited_pid2citing_works = pickle.load(f)
print("cited_pid2citing_works load finished")


cid_list_M = sorted(list(pid2year.keys()))

division2cid_list = defaultdict(list)

for i, cid in enumerate(cid_list_M):
    division_index = i % division_count
    division2cid_list[division_index].append(cid)



for division_index in tqdm(range(division_count), position = 1):

    finish_dire = os.path.join(save_dire, f'gen_V3_y{year_offset}{network_suffix}_finished_d_{division_index}_max_{division_count}.pkl')
    if os.path.exists(finish_dire) == True:
        continue

    pid_list = division2cid_list[division_index]

    pid2e_M = {}
    pid2f_M = {}
    pid2g_M = {} 
    pid2up_refer_M = {}
    pid2total_citation_M = {}
    if process_count > 1:

        batches = [[] for _ in range(process_count)]
        batch_indexs = [i for i in range(process_count)]
        pid_index = 0
        for pid in tqdm(pid_list, desc = 'process batch data s1', position = 2, leave = False):
            batch_index = pid_index % process_count
            batches[batch_index].append(pid)

            pid_index += 1



        with ProcessPoolExecutor(max_workers=process_count) as executor:

            with tqdm(total=process_count, desc = 'disp_analysis', position = 2, leave = False) as pbar:
                for res in executor.map(disp_analysis, batches, batch_indexs):
                    pbar.update(1)


                    pid2e, \
                    pid2f, \
                    pid2g, \
                    pid2up_refer, \
                    pid2total_citation, \
                    = res

                    pid2e_M.update(pid2e)
                    pid2f_M.update(pid2f)
                    pid2g_M.update(pid2g)
                    pid2total_citation_M.update(pid2total_citation)
                    pid2up_refer_M.update(pid2up_refer)

    else:
        pid2e_M, \
        pid2f_M, \
        pid2g_M, \
        pid2up_refer_M, \
        pid2total_citation_M, \
        = disp_analysis(pid_list, 0)



    df_res = []
    traverse_list = list(pid2e_M.keys())
    for pid in tqdm(traverse_list, desc = 'process pid2disp', position = 2, leave = False):

        orderdict = OrderedDict()
        orderdict['pid'] = pid


        orderdict['pub_year'] = pid2year[pid]

        citation_count = pid2total_citation_M[pid]

        orderdict['up_refer_count'] = pid2up_refer_M[pid]
        orderdict[f'citation_count_y{year_offset}'] = citation_count
        orderdict[f'e_count_y{year_offset}'] = pid2e_M[pid]
        orderdict[f'f_count_y{year_offset}'] = pid2f_M[pid]
        orderdict[f'g_count_y{year_offset}'] = pid2g_M[pid]

        # normalized shares; set to 0 when there are no citations
        if citation_count == 0:
            orderdict[f'e_y{year_offset}'] = 0
            orderdict[f'f_y{year_offset}'] = 0
            orderdict[f'g_y{year_offset}'] = 0
        else:
            orderdict[f'e_y{year_offset}'] = pid2e_M[pid] / citation_count
            orderdict[f'f_y{year_offset}'] = pid2f_M[pid] / citation_count
            orderdict[f'g_y{year_offset}'] = pid2g_M[pid] / citation_count

        df_res.append(orderdict)

    df_res = pd.DataFrame(df_res)
    df_res.to_csv(os.path.join(save_dire, f'gen_V3_y{year_offset}{network_suffix}_d_{division_index}_max_{division_count}.csv'), index = False)
    
    with open(finish_dire, 'wb') as f:
        pickle.dump(tuple(), f, pickle.HIGHEST_PROTOCOL)

    


print("finished")