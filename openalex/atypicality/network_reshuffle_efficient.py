"""
Build a degree-preserving reshuffled (null-model) citation network, year-stratified,
following Uzzi et al. (Science, 2013). Loads the real citation network (citing pid ->
cited pids) and per-paper publication years, restricts to citing papers published in
[lower_year, upper_year], and within each (source_year, target_year) block reshuffles
the edges ~2*E*ln(E) times (reshuffle_network_approx). The reshuffled edges are saved
per source year so construct_reshuffled_network.py can merge them.

Input  : META_DIRE/pid2pub_year.pkl and the citation network
         META_DIRE/citing_pid2cited_pids_list.pkl   (from get_meta_data_preprint.py)
Output : META_DIRE/random_shuffle_citation/{target_iteration}/
           citing_pid2cited_works_list_shuffled_source_{source_year}.pkl
           reshuffle_source_{source_year}_finished.pkl   (per-source-year marker)
"""

import os
import sys
import random
import pickle
from collections import defaultdict

import numpy as np
from tqdm import tqdm

# utils.py is in the parent directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
from utils import META_DIRE  # noqa: E402

meta_dire = META_DIRE
print("meta_dire",meta_dire)


target_iteration = 4
print("target_iteration",target_iteration)

save_dire = os.path.join(meta_dire, 'random_shuffle_citation', str(target_iteration))
os.makedirs(save_dire, exist_ok = True)
lower_year = 2020
upper_year = 2024

print("lower_year", lower_year)
print("upper_year", upper_year)

def reshuffle_network_approx(
            source_target_list, 
            N):

    '''
    Compute random reshuffle of existing networks as used in

    Atypical combinations and scientific impact, Uzzi, et.al, Science, 2013


    Input:
        source2target_set: key is the id of link source, value is a set of target ids
        N: The number of random shuffle to be conducted


    Output:
        source2target_set after network reshuffle
    '''

    # if the value in input dictionary is not a set of ids, convert it to set of ids

    index_list = [i for i in range(len(source_target_list))]

    choice1_select = random.choices(index_list, k = N)
    choice2_select = random.choices(index_list, k = N)

    for chose_index in tqdm(range(N), desc = 'conduct network reshuffle', leave = False, position = 1):
        choice_index1 = choice1_select[chose_index]
        choice_index2 = choice2_select[chose_index]
        if choice_index1 == choice_index2:
            continue

        source_target_1 = source_target_list[choice_index1]
        source_target_2 = source_target_list[choice_index2]

        source1, target1 = source_target_1
        source2, target2 = source_target_2


        if source1 != target2 and source2 != target1 and source1 != source2:

            source_target_1[1] = target2
            source_target_2[1] = target1

    return source_target_list


def reshuffle_network(
            source_target_list, 
            N):

    '''
    Compute random reshuffle of existing networks as used in

    Atypical combinations and scientific impact, Uzzi, et.al, Science, 2013


    Input:
        source2target_set: key is the id of link source, value is a set of target ids
        N: The number of random shuffle to be conducted


    Output:
        source2target_set after network reshuffle
    '''

    # if the value in input dictionary is not a set of ids, convert it to set of ids

    index_list = [i for i in range(len(source_target_list))]

    choice1_select = random.choices(index_list, k = N)
    choice2_select = random.choices(index_list, k = N)

    link2fre = defaultdict(int)
    for link in tqdm(source_target_list, desc = 'process link fre', leave = False, position = 1):
        source_target_tuple = tuple(link)
        link2fre[source_target_tuple] = 1

    for chose_index in tqdm(range(N), desc = 'conduct network reshuffle', leave = False, position = 1):
        choice_index1 = choice1_select[chose_index]
        choice_index2 = choice2_select[chose_index]
        if choice_index1 == choice_index2:
            continue

        source_target_1 = source_target_list[choice_index1]
        source_target_2 = source_target_list[choice_index2]

        source1, target1 = source_target_1
        source2, target2 = source_target_2


        if source1 != target2 and source2 != target1 and source1 != source2:
            source_target_1_tuple = tuple(source_target_1)
            source_target_2_tuple = tuple(source_target_2)

            post_tuple1 = (source1, target2)
            post_tuple2 = (source2, target1)



            if link2fre[post_tuple1] == 0 and \
                link2fre[post_tuple2] == 0:

                link2fre[post_tuple1] += 1
                link2fre[post_tuple2] += 1

                link2fre[source_target_1_tuple] -= 1
                link2fre[source_target_2_tuple] -= 1


                source_target_1[1] = target2
                source_target_2[1] = target1

    return source_target_list

def process_yearly_network(
            pid2cited_pid_list,
            pid2year,
            lower_bound,
            upper_bound):

    target_year2source2target_set = defaultdict(dict)

    source_year2target_year2edge_count = defaultdict(lambda: defaultdict(int))


    for pid, cited_pid_list in tqdm(pid2cited_pid_list.items(), desc = 'process citation network'):
        if len(cited_pid_list) == 0:
            continue

        source_year = pid2year[pid]

        if upper_bound >= 0 and source_year > upper_bound:
            continue

        if lower_bound >= 0 and source_year < lower_bound:
            continue

        target_year2target_set = defaultdict(set)
        for cited_pid in cited_pid_list:
            target_year = pid2year[cited_pid]
            target_year2target_set[target_year].add(cited_pid)

        for target_year, target_set in target_year2target_set.items():

                
            target_year2source2target_set[target_year][pid] = target_set
            source_year2target_year2edge_count[source_year][target_year] += len(target_set)


    source_year2target_year2shuffle_count = defaultdict(dict)
    for source_year in source_year2target_year2edge_count:
        for target_year in source_year2target_year2edge_count[source_year]:
            E = source_year2target_year2edge_count[source_year][target_year]
            source_year2target_year2shuffle_count[source_year][target_year] = \
                int(2 * np.log(E) * E)



    return target_year2source2target_set, source_year2target_year2shuffle_count

def get_source2target_list(source_target_list):
    citing_pid2cited_pid_set = defaultdict(set)

    for source_target_item in tqdm(source_target_list, desc = 'compact citing_pid2cited_pid_set', leave = False):
        source, target = source_target_item
        citing_pid2cited_pid_set[source].add(target)

    citing_pid2cited_pid_set = {citing_pid: list(cited_pid_set) for citing_pid, cited_pid_set in tqdm(citing_pid2cited_pid_set.items())}
    return citing_pid2cited_pid_set

def reshuffle_network_in_groups(
    target_year2source2target_set,
    source_year2target_year2shuffle_count,
    pid2year):



    year2pid_list = defaultdict(list)
    for pid,year in tqdm(pid2year.items(), desc = 'process pid2year'):
        year2pid_list[year].append(pid)



    for source_year, target_year2shuffle_count in tqdm(source_year2target_year2shuffle_count.items(), desc = 'process reshuffle all years'):
        source_set = set(year2pid_list[source_year])

        source_year_link_list_M = []

        if os.path.exists(os.path.join(save_dire, 'reshuffle_source_%s_finished.pkl'%(source_year))):
            continue
        for target_year, shuffle_count in target_year2shuffle_count.items():



            if shuffle_count > 0:

                target_year2source2target_set_in_target_year = target_year2source2target_set[target_year]

                has_target_source_set = target_year2source2target_set_in_target_year.keys() & source_set

                input_source_target_list = [
                    [source, target]
                    for source in has_target_source_set
                    for target in target_year2source2target_set_in_target_year[source]
                ]

                output_source_target_list = reshuffle_network_approx(input_source_target_list, shuffle_count)
                source_year_link_list_M.extend(output_source_target_list)


        with open(os.path.join(save_dire, 'citing_pid2cited_works_list_shuffled_source_%s.pkl'%(source_year)), 'wb') as f:
            pickle.dump(get_source2target_list(source_year_link_list_M), f, pickle.HIGHEST_PROTOCOL)


        with open(os.path.join(save_dire, 'reshuffle_source_%s_finished.pkl'%(source_year)), 'wb') as f:
            pickle.dump([], f, pickle.HIGHEST_PROTOCOL)




with open(os.path.join(meta_dire, 'pid2pub_year.pkl'), 'rb') as f:
    pid2year = pickle.load(f)
print("pid2year load finished")

with open(os.path.join(meta_dire, 'citing_pid2cited_pids_list.pkl'), 'rb') as f:
    citing_pid2cited_works = pickle.load(f)
print("citing_pid2cited_pids_list load finished")

for lower_bound, upper_bound in [(lower_year, upper_year)]:

    target_year2source2target_set, source_year2target_year2shuffle_count = \
            process_yearly_network(citing_pid2cited_works, pid2year, lower_bound, upper_bound)

    reshuffle_network_in_groups(target_year2source2target_set,
                                source_year2target_year2shuffle_count,
                                pid2year)
    del target_year2source2target_set
    del source_year2target_year2shuffle_count

print("finished")