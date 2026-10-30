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
from utils import META_DIRE

process_count = 3
print("process_count",process_count)



# Must match get_efg.py: the review-inclusive meta (get_meta_data_preprint_review.py)
# lives in META_DIRE + '_with_review' with the SAME filenames, so switching the base
# directory makes this step read that variant's pid2pub_year / citation-network pkls and
# write its disruption output under the same separate tree (no collision with the
# article-only run).
use_with_review = False
meta_dire = META_DIRE + '_with_review' if use_with_review else META_DIRE

print("use_with_review", use_with_review)
print("meta_dire",meta_dire)

print("process_count",process_count)

year_offset_list = [10]
assert len(year_offset_list) == 1
year_offset_main = year_offset_list[0]
print("year_offset_main",year_offset_main)

# year_offset_main = -1 is a SENTINEL meaning "no citation window": instead of counting only
# citations up to focal_pub_year + year_offset_main, count everything published up to
# NO_WINDOW_MAX_YEAR. (A literal -1 offset -- citations from BEFORE the focal paper -- is not a
# meaningful window, so the value is free to carry this meaning.)
NO_WINDOW = (year_offset_main == -1)
NO_WINDOW_MAX_YEAR = 2025
print("NO_WINDOW", NO_WINDOW, "| max_year_threshold =",
      NO_WINDOW_MAX_YEAR if NO_WINDOW else "focal_pub_year + %s" % year_offset_main)

# Every file this script writes carries year_tag, so runs with different citation windows never
# overwrite each other. BACKWARD COMPATIBILITY: the original code had no such tag and was run
# with year_offset_main = 5, so 5 keeps the untagged names -- the already-saved y5 output stays
# valid and is still found by arrange_disruption.py.
year_tag = '' if year_offset_main == 5 else f'_y{year_offset_main}'
print("year_tag", repr(year_tag))

division_count = 20
print("division_count",division_count)
# Optionally restrict which divisions to compute (each run is resumable via the
# disp_finished flags, so the work can be split across machines / invocations).
# Divisions processed are [start_division, end_division) -- end is exclusive.
# None means no limit (start at 0 / run through division_count).
start_division = 15  # inclusive
end_division = 20    # exclusive
division_start = 0 if start_division is None else start_division
division_end = division_count if end_division is None else end_division
print("division range [%s, %s)" % (division_start, division_end))




with open(os.path.join(meta_dire, 'pid2pub_year.pkl'), 'rb') as f:
    pid2year = pickle.load(f)
print("pid2year load finished")







def get_disp_metrics(J_set, I_set, K_set):

    '''
    Compute disruption metrics
    '''
    size_disp = len(I_set)
    size_dev = len(J_set)
    citation_count = size_dev + size_disp
    size_K = len(K_set) - size_dev # the raw k - J

    denominator_disp = (citation_count + size_K)

    disp_index = (size_disp - size_dev) / denominator_disp if denominator_disp > 0 else 0

    return  disp_index,\
            size_dev, \
            size_disp, \
            size_K


def disp_analysis(pid_list,
                  batch_index):


    pid2up_reference_id_list = {}

    # year_offset_list is asserted to have length 1, so each paper maps to a single
    # scalar value (not a per-offset list) -- mirrors the faster WoS get_disruption.py.
    pid2year_disp_index = {}

    pid2year_I_count = {}
    pid2year_J_count = {}
    pid2year_K_count = {}


    for focal_pid in tqdm(pid_list, desc = 'batch %s'%batch_index, position = 3, leave = False):
        focal_pub_year = pid2year[focal_pid]
        if focal_pub_year > 2025 or focal_pub_year < 1945:
            continue
        focal_reference_set = set(citing_pid2cited_works.get(focal_pid, ()))
        pid2up_reference_id_list[focal_pid] = list(focal_reference_set)

        # NO_WINDOW (year_offset_main = -1) -> no citation window: count everything up to
        # NO_WINDOW_MAX_YEAR rather than a window relative to the focal paper.
        max_year_threshold = NO_WINDOW_MAX_YEAR if NO_WINDOW else focal_pub_year + year_offset_main

        # citing list is not time-ordered: a single linear scan with a year filter.
        citing_focal_papers = {
            citing_works
            for citing_works in cited_pid2citing_works.get(focal_pid, ())
            if pid2year[citing_works] <= max_year_threshold
        }

        # Union the citers of ALL the focal paper's references first (de-dups papers that
        # cite several of P's references), THEN apply the year-window filter once per
        # distinct citer -- so the pid2year lookup + comparison run |union| times instead
        # of once per (reference, citer) incidence. Same K_set, fewer repeated lookups.
        citers_of_refs = set()
        for up_refer in focal_reference_set:
            citers_of_refs.update(cited_pid2citing_works.get(up_refer, ()))
        K_set = {
            up_citing_works
            for up_citing_works in citers_of_refs
            if focal_pub_year <= pid2year[up_citing_works] <= max_year_threshold
        }

        developmental_citation_set = set()
        disruption_citation_set = set()

        # every paper in citing_focal_papers is already within the year_offset_main window
        for citing_pid in citing_focal_papers:
            # isdisjoint() takes any iterable, so compare the focal set against the raw
            # cited list directly -- avoids building a set per citing paper and short-
            # circuits as soon as a shared reference is found.
            up_reference_citing_works = citing_pid2cited_works.get(citing_pid, ())

            # check whether the citing paper also cites one of the focal paper's references
            if not focal_reference_set.isdisjoint(up_reference_citing_works):
                # developmental citations
                developmental_citation_set.add(citing_pid)
            else:
                # disruption citations (I = I_iso + I_start)
                disruption_citation_set.add(citing_pid)

        disp_index, dev_size, disp_size, K_size = get_disp_metrics(developmental_citation_set,
                                                                   disruption_citation_set,
                                                                   K_set)
        # Note K_size is not the size of K_set

        pid2year_disp_index[focal_pid] = disp_index
        pid2year_I_count[focal_pid] = disp_size
        pid2year_J_count[focal_pid] = dev_size
        pid2year_K_count[focal_pid] = K_size


    return  pid2up_reference_id_list,\
            pid2year_disp_index,\
            pid2year_I_count,\
            pid2year_J_count,\
            pid2year_K_count



random.seed(42)
main_pid_list = sorted(list(pid2year))
random.shuffle(main_pid_list) # even the distribution so that the program will be more efficient
division_size = int(len(main_pid_list) / division_count) + 1





computed_pid_count = 0
save_dire = os.path.join(meta_dire, 'disruption_division')
os.makedirs(save_dire, exist_ok = True)

with open(os.path.join(meta_dire, 'citing_pid2cited_pids_list.pkl'), 'rb') as f:
    citing_pid2cited_works = pickle.load(f)

print("citing_pid2cited_works load finished")

with open(os.path.join(meta_dire, 'pid2title.pkl'), 'rb') as f:
    pid2title_M = pickle.load(f)
print("pid2title_M load finished")

with open(os.path.join(meta_dire, 'cited_pid2citing_pids_list.pkl'), 'rb') as f:
    cited_pid2citing_works = pickle.load(f)
print("cited_pid2citing_works load finished")





for division_index in tqdm(range(division_start, division_end), position = 1):

    pid_list = main_pid_list[division_index * division_size: (division_index + 1) * division_size]
    computed_pid_count += len(pid_list)
    if os.path.exists(os.path.join(save_dire, 'disp_finished%s_d_%s_max_%s.pkl'%(year_tag, division_index, division_count))) == True:
        continue



    if process_count > 1:

        batches = [[] for _ in range(process_count)]
        batch_indexs = [i for i in range(process_count)]
        pid_index = 0
        for pid in tqdm(pid_list, desc = 'process batch data s1', position = 2, leave = False):
            batch_index = pid_index % process_count
            batches[batch_index].append(pid)

            pid_index += 1


        pid2up_reference_id_list_M = {}
        pid2year_disp_index_M = {}
        pid2year_I_count_M = {}
        pid2year_J_count_M = {}
        pid2year_K_count_M = {}

        with ProcessPoolExecutor(max_workers=process_count) as executor:

            with tqdm(total=process_count, desc = 'disp_analysis', position = 2, leave = False) as pbar:
                for res in executor.map(disp_analysis, batches, batch_indexs):
                    pbar.update(1)

                    pid2up_reference_id_list,\
                    pid2year_disp_index,\
                    pid2year_I_count,\
                    pid2year_J_count,\
                    pid2year_K_count \
                    = res

                    pid2up_reference_id_list_M.update(pid2up_reference_id_list)
                    pid2year_disp_index_M.update(pid2year_disp_index)
                    pid2year_I_count_M.update(pid2year_I_count)
                    pid2year_J_count_M.update(pid2year_J_count)
                    pid2year_K_count_M.update(pid2year_K_count)
    else:
        pid2up_reference_id_list_M,\
        pid2year_disp_index_M,\
        pid2year_I_count_M,\
        pid2year_J_count_M,\
        pid2year_K_count_M \
        = disp_analysis(pid_list, 0)


    if os.path.exists(os.path.join(save_dire, 'pid2up_reference_id_list%s_d_%s_max_%s.pkl'%(year_tag, division_index, division_count))) == False:
        with open(os.path.join(save_dire, 'pid2up_reference_id_list%s_d_%s_max_%s.pkl'%(year_tag, division_index, division_count)), 'wb') as f:
            pickle.dump(pid2up_reference_id_list_M, f, pickle.HIGHEST_PROTOCOL)
    df_res = []
    traverse_list = list(pid2up_reference_id_list_M)
    for pid in tqdm(traverse_list, desc = 'process pid2disp', position = 2, leave = False):

        orderdict = OrderedDict()
        orderdict['pid'] = pid
        orderdict['title'] = pid2title_M[pid]
        orderdict['pub_year'] = pid2year[pid]

        up_refer_count = len(pid2up_reference_id_list_M[pid])
        orderdict['up_refer_count'] = up_refer_count

        # year_offset_list has length 1, so each per-pid map holds a single scalar.
        year_offset = year_offset_main

        J = pid2year_J_count_M[pid]
        I = pid2year_I_count_M[pid]
        K = pid2year_K_count_M[pid]

        orderdict['J_count_y%s'%year_offset] = J
        orderdict['I_count_y%s'%year_offset] = I
        orderdict['K_count_y%s'%year_offset] = K

        citation = I + J

        orderdict['citation_count_y%s'%year_offset] = citation


        orderdict['disp_y%s'%year_offset] = pid2year_disp_index_M[pid]

        if I == 0:
            orderdict['disp_no_J_numer_y%s'%year_offset] = 0
        else:
            orderdict['disp_no_J_numer_y%s'%year_offset] = I / (I +  J + K)


        df_res.append(orderdict)

    df_res = pd.DataFrame(df_res)
    df_res.to_csv(os.path.join(save_dire, 'disp_paper%s_d_%s_max_%s.csv'%(year_tag, division_index, division_count)), index = False)

    with open(os.path.join(save_dire, 'disp_finished%s_d_%s_max_%s.pkl'%(year_tag, division_index, division_count)), 'wb') as f:
        pickle.dump(tuple(), f, pickle.HIGHEST_PROTOCOL)




print("computed_pid_count",computed_pid_count)
print("len(pid2year)",len(pid2year))
# Only a full run (all divisions) is expected to cover every pid exactly once.
if division_start == 0 and division_end == division_count:
    assert len(main_pid_list) == computed_pid_count, 'Computation error'

print("finished")