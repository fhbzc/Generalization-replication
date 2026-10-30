"""
MAG version of the WoS get_disruption.py: compute the disruption index (and the
J / I / K citation counts) for every MAG journal paper from the MAG citation
network built by get_paper_meta.py.

For a focal paper P we look at the papers that cite P within `year_offset` years of
P's publication and classify each citing paper C as:
    developmental (J): C also cites at least one of P's references
    disruption  (I): C cites none of P's references
K is the set of papers that cite P's references within the same window (minus J).
The disruption index is then (I - J) / denominator.

There is NO '_with_review' variant here: the MAG citation network built by
get_paper_meta.py keeps journal papers only (DocType == 'Journal'), so unlike the WoS
pipeline there is a single network and no data_suffix.

MAG year bounds mirror get_efg.py: the Dec-2021 snapshot's last complete publication
year is 2021, so focal papers must have 1945 <= pub_year < max_year (=2021) and the
citation window is capped at max_year - 1 (= 2020):
    max_year_threshold = min(focal_pub_year + year_offset, max_year - 1)

Papers are split into `division_count` divisions; each division's per-paper table is
written to disruption_division/ with a finished-flag so the job is resumable.
arrange_disruption.py then concatenates the per-division CSVs into disp_full.csv.

K has two definitions, selected by k_require_after_focal: True counts only references'
citers published on/after P (focal_pub_year <= year <= max_year_threshold); False
(default) drops the lower bound so K also counts references' citers published before P
(year <= max_year_threshold). The False variant tags its outputs with '_k_no_lower_bound'
(k_suffix) so it never clashes with the default run. (pid2up_reference_id_list is the
same in both variants and is NOT k_suffix-tagged.)

Input  : META_DIRE/{pid2pub_year, pid2title, citing_pid2cited_pids_list,
         cited_pid2citing_pids_list}.pkl   (from get_paper_meta.py)
Output : META_DIRE/disruption_division/disp_paper[_k_no_lower_bound]_d_{division}_max_{division_count}.csv
         (the names carry year_tag = '_y{year_offset_main}', empty at the default 5)
         (+ disp_finished{year_tag}[_k_no_lower_bound]_d_{division}_max_{division_count}.pkl markers,
          + pid2up_reference_id_list_d_{division}_max_{division_count}.pkl)
"""

import os
import pickle
import random
import sys
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from tqdm import tqdm

# utils.py is in this same directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from utils import META_DIRE  # noqa: E402

process_count = 4
print("process_count", process_count)

meta_dire = META_DIRE
print("meta_dire", meta_dire)

# MAG Dec 2021 snapshot: the last complete publication year in the data is 2021, so
# focal papers must satisfy 1945 <= pub_year < max_year and citations are bounded by
# max_year - 1 (same convention as get_efg.py).
max_year = 2021
print("max_year", max_year)

# K year-window lower bound. When True a paper counts toward K only if it cites one of
# P's references AND was published on/after P (focal_pub_year <= year <= max_year_threshold).
# When False (default), drop the lower bound so K ALSO counts references' citers published
# BEFORE P (year <= max_year_threshold only). Encoded in the output filenames via k_suffix so
# the two variants never overwrite each other; arrange_disruption.py must use the same value.
k_require_after_focal = True
k_suffix = '' if k_require_after_focal else '_k_no_lower_bound'
print("k_require_after_focal", k_require_after_focal)
print("k_suffix", repr(k_suffix))


year_offset_main = -1
print("year_offset_main", year_offset_main)

# year_offset_main = -1 is a SENTINEL meaning "no citation window": instead of counting only
# the citations up to focal_pub_year + year_offset_main, count everything published up to the
# data horizon. (A literal -1 offset -- citations from BEFORE the focal paper -- is not a
# meaningful window, so the value is free to carry this meaning. Without this branch the
# min(focal_pub_year + (-1), ...) below would silently ask for citations from the year BEFORE
# publication and return almost nothing.)
# NO_WINDOW_MAX_YEAR is MAG's horizon: max_year = 2021 is the first EXCLUDED year, so the last
# year with data is max_year - 1. This is the same bound MAG's get_efg.py and
# get_efg_v2_disp_f.py already use for their own year_offset == -1 branch.
NO_WINDOW = (year_offset_main == -1)
NO_WINDOW_MAX_YEAR = max_year - 1
print("NO_WINDOW", NO_WINDOW, "| max_year_threshold =",
      NO_WINDOW_MAX_YEAR if NO_WINDOW else
      "min(focal_pub_year + %s, %s)" % (year_offset_main, max_year - 1))

# Every file this script writes carries year_tag, so runs with different citation windows never
# overwrite each other. BACKWARD COMPATIBILITY: the original code had no such tag and was run
# with year_offset_main = 5, so 5 keeps the untagged names -- the already-saved output stays
# valid and is still found by arrange_disruption.py.
year_tag = '' if year_offset_main == 5 else f'_y{year_offset_main}'
print("year_tag", repr(year_tag))
division_count = 20
print("division_count", division_count)
# Optionally restrict which divisions to compute (each run is resumable via the
# disp_finished flags, so the work can be split across machines / invocations).
# Divisions processed are [start_division, end_division) -- end is exclusive.
# None means no limit (start at 0 / run through division_count).
start_division = 0  # inclusive
end_division = None    # exclusive
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
    size_K = len(K_set) - size_dev  # the raw k - J

    denominator_disp = (citation_count + size_K)

    disp_index = (size_disp - size_dev) / denominator_disp if denominator_disp > 0 else 0

    return  disp_index,\
            size_dev, \
            size_disp, \
            size_K


def disp_analysis(pid_list, batch_index):

    pid2up_reference_id_list = {}

    pid2disp_index = {}
    pid2I_count = {}
    pid2J_count = {}
    pid2K_count = {}

    for focal_pid in tqdm(pid_list, desc='batch %s' % batch_index, position=3, leave=False):
        focal_pub_year = pid2year[focal_pid]
        if focal_pub_year >= max_year or focal_pub_year < 1945:
            continue
        focal_reference_set = set(citing_pid2cited_works[focal_pid]) if focal_pid in citing_pid2cited_works else set()
        pid2up_reference_id_list[focal_pid] = list(focal_reference_set)

        # NO_WINDOW (year_offset_main = -1) -> no citation window: count everything up to
        # the data horizon rather than a window relative to the focal paper. Otherwise cap the
        # window at that same horizon, like get_efg.py.
        max_year_threshold = (NO_WINDOW_MAX_YEAR if NO_WINDOW
                              else min(focal_pub_year + year_offset_main, max_year - 1))

        # citing list is not time-ordered: a single linear scan with a year filter.
        citing_focal_papers = {
            citing_works
            for citing_works in cited_pid2citing_works.get(focal_pid, ())
            if pid2year[citing_works] <= max_year_threshold
        }

        # Union the citers of ALL the focal paper's references first (de-dups papers
        # that cite several of P's references), THEN apply the year-window filter once
        # per distinct citer -- so the pid2year lookup + comparisons run |union| times
        # instead of once per (reference, citer) incidence. Same K_set as the
        # per-incidence version, just fewer repeated lookups when citers overlap.
        citers_of_refs = set()
        for up_refer in focal_reference_set:
            citers_of_refs.update(cited_pid2citing_works.get(up_refer, ()))
        if k_require_after_focal:
            K_set = {
                up_citing_works
                for up_citing_works in citers_of_refs
                if focal_pub_year <= pid2year[up_citing_works] <= max_year_threshold
            }
        else:
            # drop the lower bound: K also counts references' citers published before P
            K_set = {
                up_citing_works
                for up_citing_works in citers_of_refs
                if pid2year[up_citing_works] <= max_year_threshold
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

        pid2disp_index[focal_pid] = disp_index
        pid2I_count[focal_pid] = disp_size
        pid2J_count[focal_pid] = dev_size
        pid2K_count[focal_pid] = K_size

    return  pid2up_reference_id_list,\
            pid2disp_index,\
            pid2I_count,\
            pid2J_count,\
            pid2K_count


random.seed(42)
main_pid_list = sorted(list(pid2year))
random.shuffle(main_pid_list)  # even the distribution so that the program will be more efficient

division_size = int(len(main_pid_list) / division_count) + 1


computed_pid_count = 0
save_dire = os.path.join(meta_dire, 'disruption_division')
os.makedirs(save_dire, exist_ok=True)

with open(os.path.join(meta_dire, 'citing_pid2cited_pids_list.pkl'), 'rb') as f:
    citing_pid2cited_works = pickle.load(f)
print("citing_pid2cited_works load finished")

with open(os.path.join(meta_dire, 'pid2title.pkl'), 'rb') as f:
    pid2title_M = pickle.load(f)
print("pid2title_M load finished")

with open(os.path.join(meta_dire, 'cited_pid2citing_pids_list.pkl'), 'rb') as f:
    cited_pid2citing_works = pickle.load(f)
print("cited_pid2citing_works load finished")


for division_index in tqdm(range(division_start, division_end), position=1):

    pid_list = main_pid_list[division_index * division_size: (division_index + 1) * division_size]
    computed_pid_count += len(pid_list)
    if os.path.exists(os.path.join(save_dire, 'disp_finished%s%s_d_%s_max_%s.pkl' % (year_tag, k_suffix, division_index, division_count))) == True:
        continue

    if process_count > 1:

        batches = [[] for _ in range(process_count)]
        batch_indexs = [i for i in range(process_count)]
        pid_index = 0
        for pid in tqdm(pid_list, desc='process batch data s1', position=2, leave=False):
            batch_index = pid_index % process_count
            batches[batch_index].append(pid)

            pid_index += 1

        pid2up_reference_id_list_M = {}
        pid2disp_index_M = {}
        pid2I_count_M = {}
        pid2J_count_M = {}
        pid2K_count_M = {}

        with ProcessPoolExecutor(max_workers=process_count) as executor:

            with tqdm(total=process_count, desc='disp_analysis', position=2, leave=False) as pbar:
                for res in executor.map(disp_analysis, batches, batch_indexs):
                    pbar.update(1)

                    pid2up_reference_id_list,\
                    pid2disp_index,\
                    pid2I_count,\
                    pid2J_count,\
                    pid2K_count \
                    = res

                    pid2up_reference_id_list_M.update(pid2up_reference_id_list)
                    pid2disp_index_M.update(pid2disp_index)
                    pid2I_count_M.update(pid2I_count)
                    pid2J_count_M.update(pid2J_count)
                    pid2K_count_M.update(pid2K_count)
    else:
        pid2up_reference_id_list_M,\
        pid2disp_index_M,\
        pid2I_count_M,\
        pid2J_count_M,\
        pid2K_count_M \
        = disp_analysis(pid_list, 0)


    if os.path.exists(os.path.join(save_dire, 'pid2up_reference_id_list%s_d_%s_max_%s.pkl' % (year_tag, division_index, division_count))) == False:
        with open(os.path.join(save_dire, 'pid2up_reference_id_list%s_d_%s_max_%s.pkl' % (year_tag, division_index, division_count)), 'wb') as f:
            pickle.dump(pid2up_reference_id_list_M, f, pickle.HIGHEST_PROTOCOL)


    df_res = []
    traverse_list = list(pid2up_reference_id_list_M)
    for pid in tqdm(traverse_list, desc='process pid2disp', position=2, leave=False):

        orderdict = OrderedDict()
        orderdict['pid'] = pid
        orderdict['title'] = pid2title_M[pid]
        orderdict['pub_year'] = pid2year[pid]

        up_refer_count = len(pid2up_reference_id_list_M[pid])
        orderdict['up_refer_count'] = up_refer_count

        year_offset = year_offset_main

        J = pid2J_count_M[pid]
        I = pid2I_count_M[pid]
        K = pid2K_count_M[pid]

        orderdict['J_count_y%s' % year_offset] = J
        orderdict['I_count_y%s' % year_offset] = I
        orderdict['K_count_y%s' % year_offset] = K

        citation = I + J

        orderdict['citation_count_y%s' % year_offset] = citation

        orderdict['disp_y%s' % year_offset] = pid2disp_index_M[pid]

        if I == 0:
            orderdict['disp_no_J_numer_y%s' % year_offset] = 0
        else:
            orderdict['disp_no_J_numer_y%s' % year_offset] = I / (I + J + K)


        df_res.append(orderdict)

    df_res = pd.DataFrame(df_res)
    df_res.to_csv(os.path.join(save_dire, 'disp_paper%s%s_d_%s_max_%s.csv' % (year_tag, k_suffix, division_index, division_count)), index=False)

    with open(os.path.join(save_dire, 'disp_finished%s%s_d_%s_max_%s.pkl' % (year_tag, k_suffix, division_index, division_count)), 'wb') as f:
        pickle.dump(tuple(), f, pickle.HIGHEST_PROTOCOL)


print("computed_pid_count", computed_pid_count)
print("len(pid2year)", len(pid2year))
# Only a full run (all divisions) is expected to cover every pid exactly once.
if division_start == 0 and division_end == division_count:
    assert len(main_pid_list) == computed_pid_count, 'Computation error'

print("finished")
