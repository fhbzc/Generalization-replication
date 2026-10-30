"""
Semantic Scholar (S2AG) version of the OpenAlex get_efg_reshuffle_control_venue.py.

Compute the E/F/G index on a venue-controlled RESHUFFLED (null-model) citation network.
It loads the reshuffled network from the random_shuffle_citation_control_venue/ tree
(built by construct_reshuffled_network_control_venue.py) and writes its outputs to a
separate efg_computation_control_venue/ directory.

The node id is the S2AG corpusid; as in get_efg.py it is loaded into a `pid2year`
variable and written out under the 'pid' column, so the downstream merge / z-score /
visualize scripts read it the same way.

Input  : META_DIRE/cid2pub_year.pkl (from get_s2ag_meta.py)
         META_DIRE/random_shuffle_citation_control_venue/{target_iteration}/constructed_network/
           shuffled_{citing_pid2cited,cited_pid2citing}_pids_list_before_{upper_year}.pkl
           (from construct_reshuffled_network_control_venue.py)
Output : META_DIRE/efg_computation_control_venue/gen_V3_y{year_offset}_iter{target_iteration}_before_{upper_year}_d_{division}_max_{division_count}.csv
         (+ ..._finished_... markers)
"""

import os
import sys
import pickle
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from tqdm import tqdm

# utils.py is in the parent (semantic_scholar/) directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
from utils import META_DIRE  # noqa: E402


process_count = 2
print("process_count", process_count)

meta_dire = META_DIRE

print("meta_dire", meta_dire)

division_count = 20
print("division_count", division_count)


max_year = 2025
print("max_year", max_year)



year_offset = 5  # -1 for no offset

print("year_offset", year_offset)


# Which network to run the E/F/G computation on:
#   target_iteration = <int>  : a venue-controlled reshuffled-network iteration saved by
#                               construct_reshuffled_network_control_venue.py (null model).
#   target_iteration = None   : the OBSERVED (non-reshuffled) venue-filtered network from
#                               filter_network_valid_venue.py -- i.e. the real network,
#                               restricted to the same upper_year edge subset as the
#                               reshuffles so the result is a comparable baseline.
target_iteration = None
print("target_iteration", target_iteration)

# must match the upper_year used in construct_reshuffled_network_control_venue.py (selects
# which reshuffled-network files to load: shuffled_*_pids_list_before_{upper_year}.pkl).
# In the observed (target_iteration is None) case it is the cutoff applied to the venue-
# filtered network so it matches the reshuffles' edge subset (source/citing year < upper_year).
upper_year = 2025
print("upper_year", upper_year)

if target_iteration is None:
    # observed run on the venue-filtered (valid_venue) network; the suffix spells that out
    # so the output is not mistaken for an uncontrolled observed run.
    file_suffix = f'_obs_valid_venue_before_{upper_year}'
else:
    file_suffix = f'_iter{target_iteration}_before_{upper_year}'
print("file_suffix", file_suffix)

with open(os.path.join(meta_dire, 'cid2pub_year.pkl'), 'rb') as f:
    pid2year = pickle.load(f)
print("cid2pub_year load finished")





def disp_analysis(pid_list,
                  batch_index):



    pid2e = {}
    pid2f = {}
    pid2g = {}
    pid2total_citation = {}
    pid2up_refer = {}

    for focal_pid in tqdm(pid_list, desc='batch %s' % batch_index, position=3, leave=False):

        e_count = 0
        f_count = 0
        g_count = 0
        focal_pub_year = pid2year[focal_pid]
        # defensive guard, consistent with the cid_list_M pre-filter below: focal
        # papers must be in [1945, min(max_year, upper_year)).
        if focal_pub_year >= min(max_year, upper_year) or focal_pub_year < 1945:
            continue
        focal_reference_set = citing_pid2cited_works.get(focal_pid, set())

        max_year_threshold = min(focal_pub_year + year_offset, max_year - 1)

        citing_focal_papers = set()

        if year_offset == -1:
            # all future citations, but still bounded by the data horizon (max_year - 1)
            citing_focal_papers = set([citing_work for citing_work in cited_pid2citing_works.get(focal_pid, []) if pid2year[citing_work] <= max_year - 1])

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



save_dire = os.path.join(meta_dire, 'efg_computation_control_venue')
os.makedirs(save_dire, exist_ok=True)

if target_iteration is None:
    # OBSERVED (non-reshuffled) venue-filtered network from
    # filter_network_valid_venue.py. Both directions are available there.
    citing_pid2cited_pids_path = os.path.join(meta_dire, 'citing2cited_cid_list_remove_circle_valid_venue.pkl')
    cited_pid2citing_pids_path = os.path.join(meta_dire, 'cited2citing_cid_list_remove_circle_valid_venue.pkl')
else:
    # a venue-controlled reshuffled-network iteration
    network_dire = os.path.join(meta_dire, 'random_shuffle_citation_control_venue', str(target_iteration), 'constructed_network')
    citing_pid2cited_pids_path = os.path.join(network_dire, f'shuffled_citing_pid2cited_pids_list_before_{upper_year}.pkl')
    cited_pid2citing_pids_path = os.path.join(network_dire, f'shuffled_cited_pid2citing_pids_list_before_{upper_year}.pkl')

print("citing_pid2cited_pids_path", citing_pid2cited_pids_path)
print("cited_pid2citing_pids_path", cited_pid2citing_pids_path)

with open(citing_pid2cited_pids_path, 'rb') as f:
    citing_pid2cited_works = pickle.load(f)

print("citing_pid2cited_works load finished")

citing_pid2cited_works = {pid: set(pid_list) for pid, pid_list in tqdm(citing_pid2cited_works.items(), desc='process citing_pid2cited_works')}
with open(cited_pid2citing_pids_path, 'rb') as f:
    cited_pid2citing_works = pickle.load(f)
print("cited_pid2citing_works load finished")

if target_iteration is None:
    # The reshuffled networks only contain citation edges whose source/citing paper's
    # year is STRICTLY BEFORE upper_year (construct_reshuffled_network_control_venue.py's
    # cut). The observed venue-filtered network is NOT year-bounded, so restrict it to the
    # SAME subset here; otherwise the observed e/f/g would be computed over more citing
    # papers than the null model and the two would not be comparable.
    n_citing_before = len(citing_pid2cited_works)
    citing_pid2cited_works = {
        pid: cited for pid, cited in tqdm(citing_pid2cited_works.items(), desc='restrict citing to year < upper_year')
        if pid2year.get(pid) is not None and pid2year[pid] < upper_year
    }
    cited_pid2citing_works = {
        cited: [c for c in citing if pid2year.get(c) is not None and pid2year[c] < upper_year]
        for cited, citing in tqdm(cited_pid2citing_works.items(), desc='restrict cited->citing to year < upper_year')
    }
    # drop cited papers left with no in-range citing source
    cited_pid2citing_works = {cited: citing for cited, citing in cited_pid2citing_works.items() if citing}
    print(f"observed network restricted to source year < {upper_year}: "
          f"citing keys {n_citing_before} -> {len(citing_pid2cited_works)}")


# Filter the focal-paper list up front by max_year AND upper_year, rather than only
# skipping inside disp_analysis. A focal paper is analyzable in this reshuffled
# control-venue network only if its pub_year is in [1945, max_year) AND strictly
# before upper_year: the reshuffled network keeps only citation edges whose source
# year < upper_year, so papers published >= upper_year have no references/citations
# in it and would all be skipped anyway. Filtering here avoids iterating over them.
focal_year_upper = min(max_year, upper_year)
print("focal_year_upper", focal_year_upper)
cid_list_M = sorted([pid for pid in pid2year.keys() if 1945 <= pid2year[pid] < focal_year_upper])

division2cid_list = defaultdict(list)

for i, cid in enumerate(cid_list_M):
    division_index = i % division_count
    division2cid_list[division_index].append(cid)



for division_index in tqdm(range(division_count), position=1):

    finish_dire = os.path.join(save_dire, f'gen_V3_y{year_offset}{file_suffix}_finished_d_{division_index}_max_{division_count}.pkl')
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
        for pid in tqdm(pid_list, desc='process batch data s1', position=2, leave=False):
            batch_index = pid_index % process_count
            batches[batch_index].append(pid)

            pid_index += 1



        with ProcessPoolExecutor(max_workers=process_count) as executor:

            with tqdm(total=process_count, desc='disp_analysis', position=2, leave=False) as pbar:
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
    for pid in tqdm(traverse_list, desc='process pid2disp', position=2, leave=False):

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
    df_res.to_csv(os.path.join(save_dire, f'gen_V3_y{year_offset}{file_suffix}_d_{division_index}_max_{division_count}.csv'), index=False)

    with open(finish_dire, 'wb') as f:
        pickle.dump(tuple(), f, pickle.HIGHEST_PROTOCOL)




print("finished")
