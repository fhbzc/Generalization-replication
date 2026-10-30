import os
import pickle
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import random
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import META_DIRE
process_count = 4
print("process_count",process_count)

meta_dire = META_DIRE

save_dire = os.path.join(meta_dire, 'disruption_pair_division')
os.makedirs(save_dire, exist_ok = True)

print("meta_dire",meta_dire)

year_threshold = 5


print("year_threshold",year_threshold)


division_count = 20
print("division_count",division_count)
with open(os.path.join(meta_dire, 'pid2pub_year.pkl'), 'rb') as f:
    pid2year = pickle.load(f)
print("pid2year load finished")





def disp_analysis(pid_list, 
                  batch_index):


    pid2up_refer2i = {}
    pid2up_refer2j = {}
    pid2up_refer2i_k_j = {}


    for focal_pid in tqdm(pid_list, desc = 'batch %s, division %s'%(batch_index, division_index), position = 2, leave = False):
        focal_pub_year = pid2year[focal_pid]
        if focal_pub_year >= 2025 - year_threshold or focal_pub_year < 1945:
            continue




        reference_all_set = set(citing_pid2cited_works[focal_pid]) if focal_pid in citing_pid2cited_works else set()

        max_year_threshold = focal_pub_year + year_threshold



        # citing lists are unordered, so scan all (no break); comprehensions keep it fast
        citing_focal_papers = {
            c for c in cited_pid2citing_works.get(focal_pid, ())
            if pid2year[c] <= max_year_threshold
        }

        up_refer2total_citation_count = {
            up_refer: sum(
                1 for c in cited_pid2citing_works.get(up_refer, ())
                if focal_pub_year <= pid2year[c] <= max_year_threshold
            )
            for up_refer in reference_all_set
        }

        up_refer2dev_citation_count = {up_refer:0 for up_refer in reference_all_set}
        up_refer2disp_citation_count = {up_refer:0 for up_refer in reference_all_set}
        for citing_pid in citing_focal_papers:
            up_reference_citing_pid = set(citing_pid2cited_works[citing_pid]) if citing_pid in citing_pid2cited_works else set()
            
            overlap_reference = up_reference_citing_pid & reference_all_set
            non_overlap_reference = reference_all_set - overlap_reference

            for up_refer in overlap_reference:
                up_refer2dev_citation_count[up_refer] += 1

            for up_refer in non_overlap_reference:
                up_refer2total_citation_count[up_refer] += 1 # add to K+j+i
                up_refer2disp_citation_count[up_refer] += 1


        pid2up_refer2i[focal_pid] = up_refer2disp_citation_count
        pid2up_refer2j[focal_pid] = up_refer2dev_citation_count
        pid2up_refer2i_k_j[focal_pid] = up_refer2total_citation_count

    return pid2up_refer2i, \
            pid2up_refer2j, \
            pid2up_refer2i_k_j

with open(os.path.join(meta_dire, 'citing_pid2cited_pids_list.pkl'), 'rb') as f:
    citing_pid2cited_works = pickle.load(f)
print("citing_pid2cited_works load finished")

with open(os.path.join(meta_dire, 'cited_pid2citing_pids_list.pkl'), 'rb') as f:
    cited_pid2citing_works = pickle.load(f)
print("cited_pid2citing_works load finished")


# Per-division focal pid lists are computed on the fly (same chunking as
# get_disruption.py); there is no precomputed pid_list_d_* file. A fixed seed keeps
# the split stable across restarts so the disp_d_*_finished.pkl resume markers stay valid.
main_pid_list = sorted(list(pid2year))
random.seed(42)
random.shuffle(main_pid_list)  # even out per-division compute time
division_size = int(len(main_pid_list) / division_count) + 1


for division_index in tqdm(range(division_count), position = 0):


    finish_dire = os.path.join(save_dire, 'disp_d_%s_finished.pkl'%division_index)
    if os.path.exists(finish_dire) == True:
        continue
    pid_list = main_pid_list[division_index * division_size: (division_index + 1) * division_size]

    if process_count > 1:

        batches = [[] for _ in range(process_count)]
        batch_indexs = [i for i in range(process_count)]
        pid_index = 0
        for pid in tqdm(pid_list, desc = 'process batch data s1', position = 1, leave = False):
            batch_index = pid_index % process_count
            batches[batch_index].append(pid)

            pid_index += 1


        pid2up_refer2i_M = {}
        pid2up_refer2j_M = {}
        pid2up_refer2i_k_j_M = {}
        with ProcessPoolExecutor(max_workers=process_count) as executor:

            with tqdm(total=process_count, desc = 'disp_analysis', position = 1, leave = False) as pbar:
                for res in executor.map(disp_analysis, batches, batch_indexs):
                    pbar.update(1)

                    pid2up_refer2i, \
                    pid2up_refer2j, \
                    pid2up_refer2i_k_j, \
                    = res

                    pid2up_refer2i_M.update(pid2up_refer2i)
                    pid2up_refer2j_M.update(pid2up_refer2j)
                    pid2up_refer2i_k_j_M.update(pid2up_refer2i_k_j)

    else:
        pid2up_refer2i_M, \
        pid2up_refer2j_M, \
        pid2up_refer2i_k_j_M, \
        = disp_analysis(pid_list, 0)

    
    with open(os.path.join(save_dire, 'pid2up2i_count_d_%s_max_%s.pkl'%(division_index, division_count)), 'wb') as f:
        pickle.dump(pid2up_refer2i_M, f, pickle.HIGHEST_PROTOCOL)
    del pid2up_refer2i_M
    
    with open(os.path.join(save_dire, 'pid2up2j_count_d_%s_max_%s.pkl'%(division_index, division_count)), 'wb') as f:
        pickle.dump(pid2up_refer2j_M, f, pickle.HIGHEST_PROTOCOL)
    del pid2up_refer2j_M
    
    with open(os.path.join(save_dire, 'pid2up2i_j_k_count_d_%s_max_%s.pkl'%(division_index, division_count)), 'wb') as f:
        pickle.dump(pid2up_refer2i_k_j_M, f, pickle.HIGHEST_PROTOCOL)
    del pid2up_refer2i_k_j_M
    with open(finish_dire, 'wb') as f:
        pickle.dump(tuple(), f, pickle.HIGHEST_PROTOCOL)

print("finished")