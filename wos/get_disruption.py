"""
WoS version of the OpenAlex get_disruption.py: compute the disruption index (and the
J / I / K citation counts) for every WoS paper from the WoS citation
network built by get_meta_data.py.

For a focal paper P (published in [1945, 2025]) we look at the papers that cite P
within `year_offset_main` years of P's publication and classify each citing paper C as:
    developmental (J): C also cites at least one of P's references
    disruption  (I): C cites none of P's references
K is the set of papers that cite P's references within the same window (minus J).
The disruption index is then (I - J) / denominator.

Papers are split into `division_count` divisions; each division's per-paper table is
written to disruption_division/ with a finished-flag so the job is resumable.
arrange_disruption.py then concatenates the per-division CSVs into disp_full.csv.

Set use_with_review=True to compute from the get_meta_data_with_review.py outputs
(the '_with_review' citation network that also keeps doctypes == 'review'); the
division outputs then carry the same suffix so they never clash with the article-only run.

year_offset_main == -1 means NO OFFSET, exactly as in get_efg.py: the citation window has
no upper bound at all, so every citation of P that the network holds counts, however late it
was published (and likewise for K). Any other value bounds the window at
focal_pub_year + year_offset_main. Note the output columns are named after the setting, so a
-1 run writes disp_y-1 / I_count_y-1 / ... , matching get_efg.py's e_count_y-1 convention;
downstream scripts have to be pointed at the same year_offset.

K has two definitions, selected by k_require_after_focal: True (default) counts only
references' citers published on/after P (focal_pub_year <= year <= max_year_threshold);
False drops the lower bound so K also counts references' citers published before P
(year <= max_year_threshold). Under year_offset_main == -1 the upper bound falls away from
both variants, so True leaves only focal_pub_year <= year and False leaves no year test. The False variant tags its outputs with '_k_no_lower_bound'
(k_suffix) so it never clashes with the default run. (pid2up_reference_id_list is the
same in both variants and is NOT k_suffix-tagged.)

Input  : META_DIRE/{pid2pub_year, pid2title, citing_pid2cited_pids_list,
         cited_pid2citing_pids_list}[_with_review].pkl   (from get_meta_data[_with_review].py)
Output : META_DIRE/disruption_division/disp_paper[_y{year_offset_main}][_with_review][_k_no_lower_bound]_d_{division}_max_{division_count}.csv
         (+ disp_finished[_y{year_offset_main}][_with_review][_k_no_lower_bound]_d_{division}_max_{division_count}.pkl markers,
          + pid2up_reference_id_list[_with_review]_d_{division}_max_{division_count}.pkl)
         The window is in the filename only when it is NOT the default 5, so the 5-year run keeps
         the names it already has on disk and any other window (e.g. -1, the no-offset variant)
         lands beside it instead of overwriting it.
         Two windows therefore never mix, and a resumed run never picks up another window's
         finished-flags. The tag is NOT put on pid2up_reference_id_list: that map is just each
         paper's reference list, which no citation window touches. arrange_disruption.py must
         use the same year_offset_main.
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

# If True, use the get_meta_data_with_review.py outputs ('_with_review' network that
# also includes doctypes == 'review'); the division outputs get the same suffix.
use_with_review = False
data_suffix = '_with_review' if use_with_review else ''
print("use_with_review", use_with_review)
print("data_suffix", repr(data_suffix))

# K year-window lower bound. When True (original behaviour) a paper counts toward K only
# if it cites one of P's references AND was published on/after P
# (focal_pub_year <= year <= max_year_threshold). When False, drop the lower bound so K
# ALSO counts references' citers published BEFORE P (year <= max_year_threshold only).
# Encoded in the output filenames via k_suffix so the two variants never overwrite each
# other; arrange_disruption.py must use the same k_require_after_focal.
k_require_after_focal = True
k_suffix = '' if k_require_after_focal else '_k_no_lower_bound'
print("k_require_after_focal", k_require_after_focal)
print("k_suffix", repr(k_suffix))


# Citation window, in years after the focal paper's publication. -1 means NO OFFSET (every
# citation counts, with no upper year bound) -- the same convention get_efg.py uses.
year_offset_main = -1
NO_OFFSET = (year_offset_main == -1)
# The window goes into every per-division filename UNLESS it is the default 5, whose files
# already exist on disk under the untagged names.
year_tag = '' if year_offset_main == 5 else '_y%s' % year_offset_main
print("year_offset_main", year_offset_main, "| NO_OFFSET", NO_OFFSET,
      "| year_tag", repr(year_tag))
division_count = 20
print("division_count", division_count)
# Optionally restrict which divisions to compute (each run is resumable via the
# disp_finished flags, so the work can be split across machines / invocations).
# Divisions processed are [start_division, end_division) -- end is exclusive.
# None means no limit (start at 0 / run through division_count).
start_division = 0 # inclusive
end_division = None # exclusive
division_start = 0 if start_division is None else start_division
division_end = division_count if end_division is None else end_division
print("division range [%s, %s)" % (division_start, division_end))


with open(os.path.join(meta_dire, f'pid2pub_year{data_suffix}.pkl'), 'rb') as f:
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
        if focal_pub_year > 2025 or focal_pub_year < 1945:
            continue
        focal_reference_set = set(citing_pid2cited_works[focal_pid]) if focal_pid in citing_pid2cited_works else set()
        pid2up_reference_id_list[focal_pid] = list(focal_reference_set)

        max_year_threshold = focal_pub_year + year_offset_main   # unused when NO_OFFSET

        # citing list is not time-ordered: a single linear scan with a year filter.
        # NO_OFFSET (year_offset_main == -1) takes every citer with no upper bound, exactly as
        # get_efg.py does; a real offset keeps only those inside the window.
        if NO_OFFSET:
            citing_focal_papers = set(cited_pid2citing_works.get(focal_pid, ()))
        else:
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
        # Under NO_OFFSET the upper bound disappears from both variants, mirroring the
        # citing-side window above; k_require_after_focal still decides the lower bound.
        if k_require_after_focal:
            if NO_OFFSET:
                K_set = {
                    up_citing_works
                    for up_citing_works in citers_of_refs
                    if focal_pub_year <= pid2year[up_citing_works]
                }
            else:
                K_set = {
                    up_citing_works
                    for up_citing_works in citers_of_refs
                    if focal_pub_year <= pid2year[up_citing_works] <= max_year_threshold
                }
        else:
            # drop the lower bound: K also counts references' citers published before P
            if NO_OFFSET:
                K_set = set(citers_of_refs)
            else:
                K_set = {
                    up_citing_works
                    for up_citing_works in citers_of_refs
                    if pid2year[up_citing_works] <= max_year_threshold
                }

        developmental_citation_set = set()
        disruption_citation_set = set()

        # every paper in citing_focal_papers already satisfies the year_offset_main window
        # (which is no constraint at all when NO_OFFSET)
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

with open(os.path.join(meta_dire, f'citing_pid2cited_pids_list{data_suffix}.pkl'), 'rb') as f:
    citing_pid2cited_works = pickle.load(f)
print("citing_pid2cited_works load finished")

with open(os.path.join(meta_dire, f'pid2title{data_suffix}.pkl'), 'rb') as f:
    pid2title_M = pickle.load(f)
print("pid2title_M load finished")

with open(os.path.join(meta_dire, f'cited_pid2citing_pids_list{data_suffix}.pkl'), 'rb') as f:
    cited_pid2citing_works = pickle.load(f)
print("cited_pid2citing_works load finished")


for division_index in tqdm(range(division_start, division_end), position=1):

    pid_list = main_pid_list[division_index * division_size: (division_index + 1) * division_size]
    computed_pid_count += len(pid_list)
    if os.path.exists(os.path.join(save_dire, 'disp_finished%s%s%s_d_%s_max_%s.pkl' % (year_tag, data_suffix, k_suffix, division_index, division_count))) == True:
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


    if os.path.exists(os.path.join(save_dire, 'pid2up_reference_id_list%s_d_%s_max_%s.pkl' % (data_suffix, division_index, division_count))) == False:
        with open(os.path.join(save_dire, 'pid2up_reference_id_list%s_d_%s_max_%s.pkl' % (data_suffix, division_index, division_count)), 'wb') as f:
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
    df_res.to_csv(os.path.join(save_dire, 'disp_paper%s%s%s_d_%s_max_%s.csv' % (year_tag, data_suffix, k_suffix, division_index, division_count)), index=False)

    with open(os.path.join(save_dire, 'disp_finished%s%s%s_d_%s_max_%s.pkl' % (year_tag, data_suffix, k_suffix, division_index, division_count)), 'wb') as f:
        pickle.dump(tuple(), f, pickle.HIGHEST_PROTOCOL)


print("computed_pid_count", computed_pid_count)
print("len(pid2year)", len(pid2year))
# Only a full run (all divisions) is expected to cover every pid exactly once.
if division_start == 0 and division_end == division_count:
    assert len(main_pid_list) == computed_pid_count, 'Computation error'

print("finished")
