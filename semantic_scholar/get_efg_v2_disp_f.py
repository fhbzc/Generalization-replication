"""
Semantic Scholar (S2AG) version of the OpenAlex get_efg_v2_disp_f.py.

Same as get_efg.py (E / F / G generalization index over the S2AG citation network, in
per-division shards), but ALSO computes the three "disruptive foundation" (disp_f) variants
in the same pass. A citing paper C of a focal paper P is counted as a disruptive-foundation
citation when it shares NONE of P's references (up_count == 0) yet co-cites more than T of P's
other citing papers (down_count > T). Three thresholds are tallied at once:
    T = 0  -> down_count > 0   (the original disruption_f; kept under the legacy column names)
    T = 1  -> down_count > 1
    T = 2  -> down_count > 2

Paper ids are S2AG corpus ids (cid), kept in the 'pid' column so the downstream merge /
quantile steps stay identical to the OpenAlex and WoS pipelines. get_efg_s2_merge.py
(use_disp_f=True) then concatenates the per-division CSVs.

The citation network is the temporally-consistent ("remove_circle") network from
get_s2ag_citation_s2_remove_circle.py. Set use_openalex_augmented=True to instead run on the
OpenAlex-augmented remove_circle network from augment_s2_citation_with_openalex.py; that choice
is tagged onto both the input filenames and the output file_base (network_suffix) so the two
never collide.

Input  : META_DIRE/cid2pub_year.pkl                          (get_s2ag_meta.py)
         META_DIRE/citing2cited_cid_list_remove_circle[_openalex_augmented].pkl   (citing -> cited)
         META_DIRE/cited2citing_cid_list_remove_circle[_openalex_augmented].pkl   (cited -> citing)
         (from get_s2ag_citation_s2_remove_circle.py, or augment_s2_citation_with_openalex.py)
Output : META_DIRE/efg_computation/gen_V3_disp_f_y{year_offset}[_openalex_augmented]_d_{division}_max_{division_count}.csv
         (+ ..._finished_d_{division}_max_{division_count}.pkl markers)
"""

import os
import sys
import pickle
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from tqdm import tqdm

# utils.py is in this same directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from utils import META_DIRE  # noqa: E402

process_count = 3
print("process_count", process_count)

meta_dire = META_DIRE
print("meta_dire", meta_dire)

division_count = 20
print("division_count", division_count)

max_year = 2025
print("max_year", max_year)

year_offset = -1  # -1 for no offset
print("year_offset", year_offset)

# disp_f variants: a citing paper is a "disruptive foundation" citation when it shares NONE
# of the focal paper's references (up_count == 0) yet co-cites more than T of the focal paper's
# other citing papers (down_count > T). Three thresholds are computed in one pass:
#   T = 0  -> down_count > 0   (the original disruption_f; kept under the legacy column names)
#   T = 1  -> down_count > 1
#   T = 2  -> down_count > 2
DISP_F_DOWN_THRESHOLDS = [0, 1, 2]
print("DISP_F_DOWN_THRESHOLDS", DISP_F_DOWN_THRESHOLDS)


def dispf_colname(down_threshold, kind):
    """Column name for a disp_f variant. `kind` is 'count' or 'share'. The T=0 variant keeps
    the original names (disruption_f_count_y{off} / disruption_f_y{off}) so existing downstream
    scripts keep working; T>0 variants get a _down_gt{T} tag."""
    base = 'disruption_f' if down_threshold == 0 else f'disruption_f_down_gt{down_threshold}'
    suffix = '_count' if kind == 'count' else ''
    return f'{base}{suffix}_y{year_offset}'


# When True, run on the OpenAlex-augmented '_remove_circle' network produced by
# augment_s2_citation_with_openalex.py (the S2AG remove_circle network plus the citation
# edges OpenAlex has but S2AG was missing) instead of the plain S2AG remove_circle
# network. The choice is encoded in BOTH the input network filenames AND the output
# file_base via network_suffix, so the augmented and non-augmented results never overwrite
# each other. get_efg_s2_merge.py must use the same use_openalex_augmented.
use_openalex_augmented = False
network_suffix = '_openalex_augmented' if use_openalex_augmented else ''
print("use_openalex_augmented", use_openalex_augmented)
print("network_suffix", repr(network_suffix))

# base name shared by the per-division CSVs and finished-flags
file_base = f'gen_V3_disp_f_y{year_offset}{network_suffix}'


with open(os.path.join(meta_dire, 'cid2pub_year.pkl'), 'rb') as f:
    pid2year = pickle.load(f)
print("pid2year load finished")


def disp_analysis(pid_list, batch_index):
    pid2e = {}
    pid2f = {}
    pid2g = {}
    pid2disruption_f = {}
    pid2total_citation = {}
    pid2up_refer = {}

    for focal_pid in tqdm(pid_list, desc='batch %s' % batch_index, position=3, leave=False):

        e_count = 0
        f_count = 0
        g_count = 0
        # one disruptive-foundation counter per down_count threshold
        disruption_f_counts = {t: 0 for t in DISP_F_DOWN_THRESHOLDS}
        focal_pub_year = pid2year[focal_pid]
        if focal_pub_year >= max_year or focal_pub_year < 1945:
            continue
        focal_reference_set = citing_pid2cited_works.get(focal_pid, set())

        max_year_threshold = min(focal_pub_year + year_offset, max_year - 1)

        citing_focal_papers = set()

        if year_offset == -1:
            # all future citations, but still bounded by the data horizon (max_year - 1)
            citing_focal_papers = set([
                citing_work for citing_work in cited_pid2citing_works.get(focal_pid, [])
                if pid2year[citing_work] <= max_year - 1
            ])
        else:
            if focal_pid in cited_pid2citing_works:
                citing_focal_papers = set([
                    citing_work for citing_work in cited_pid2citing_works[focal_pid]
                    if pid2year[citing_work] <= max_year_threshold
                ])

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

            # disruption_f: citing papers that are BOTH foundation-contributing AND
            # disruptive, i.e. they share none of the focal paper's references (up_count == 0)
            # yet co-cite more than T of the focal paper's other citing papers (down_count > T).
            # Count all thresholds in DISP_F_DOWN_THRESHOLDS in this single pass.
            if up_count == 0:
                for t in DISP_F_DOWN_THRESHOLDS:
                    if down_count > t:
                        disruption_f_counts[t] += 1

        pid2total_citation[focal_pid] = len(citing_focal_papers)
        pid2up_refer[focal_pid] = len(focal_reference_set)
        pid2e[focal_pid] = e_count
        pid2f[focal_pid] = f_count
        pid2g[focal_pid] = g_count
        pid2disruption_f[focal_pid] = disruption_f_counts

    return pid2e, pid2f, pid2g, pid2disruption_f, pid2up_refer, pid2total_citation


save_dire = os.path.join(meta_dire, 'efg_computation')
os.makedirs(save_dire, exist_ok=True)

citing_pid2cited_pids_path = os.path.join(meta_dire, f'citing2cited_cid_list_remove_circle{network_suffix}.pkl')
cited_pid2citing_pids_path = os.path.join(meta_dire, f'cited2citing_cid_list_remove_circle{network_suffix}.pkl')

print("citing_pid2cited_pids_path", citing_pid2cited_pids_path)
print("cited_pid2citing_pids_path", cited_pid2citing_pids_path)

with open(citing_pid2cited_pids_path, 'rb') as f:
    citing_pid2cited_works = pickle.load(f)
print("citing_pid2cited_works load finished")

citing_pid2cited_works = {pid: set(pid_list) for pid, pid_list in
                          tqdm(citing_pid2cited_works.items(), desc='process citing_pid2cited_works')}
with open(cited_pid2citing_pids_path, 'rb') as f:
    cited_pid2citing_works = pickle.load(f)
print("cited_pid2citing_works load finished")


cid_list_M = sorted(list(pid2year.keys()))

division2cid_list = defaultdict(list)
for i, cid in enumerate(cid_list_M):
    division_index = i % division_count
    division2cid_list[division_index].append(cid)


for division_index in tqdm(range(division_count), position=1):

    finish_dire = os.path.join(save_dire, f'{file_base}_finished_d_{division_index}_max_{division_count}.pkl')
    if os.path.exists(finish_dire) == True:
        continue

    pid_list = division2cid_list[division_index]

    pid2e_M = {}
    pid2f_M = {}
    pid2g_M = {}
    pid2disruption_f_M = {}
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

                    pid2e, pid2f, pid2g, pid2disruption_f, pid2up_refer, pid2total_citation = res

                    pid2e_M.update(pid2e)
                    pid2f_M.update(pid2f)
                    pid2g_M.update(pid2g)
                    pid2disruption_f_M.update(pid2disruption_f)
                    pid2total_citation_M.update(pid2total_citation)
                    pid2up_refer_M.update(pid2up_refer)
    else:
        pid2e_M, pid2f_M, pid2g_M, pid2disruption_f_M, pid2up_refer_M, pid2total_citation_M = \
            disp_analysis(pid_list, 0)

    df_res = []
    traverse_list = list(pid2e_M.keys())
    for pid in tqdm(traverse_list, desc='process pid2disp', position=2, leave=False):

        orderdict = OrderedDict()
        orderdict['pid'] = pid
        orderdict['pub_year'] = pid2year[pid]

        citation_count = pid2total_citation_M[pid]
        dispf_counts = pid2disruption_f_M[pid]   # {down_threshold: count}

        orderdict['up_refer_count'] = pid2up_refer_M[pid]
        orderdict[f'citation_count_y{year_offset}'] = citation_count
        orderdict[f'e_count_y{year_offset}'] = pid2e_M[pid]
        orderdict[f'f_count_y{year_offset}'] = pid2f_M[pid]
        orderdict[f'g_count_y{year_offset}'] = pid2g_M[pid]
        for t in DISP_F_DOWN_THRESHOLDS:
            orderdict[dispf_colname(t, 'count')] = dispf_counts[t]

        # normalized shares; set to 0 when there are no citations
        if citation_count == 0:
            orderdict[f'e_y{year_offset}'] = 0
            orderdict[f'f_y{year_offset}'] = 0
            orderdict[f'g_y{year_offset}'] = 0
            for t in DISP_F_DOWN_THRESHOLDS:
                orderdict[dispf_colname(t, 'share')] = 0
        else:
            orderdict[f'e_y{year_offset}'] = pid2e_M[pid] / citation_count
            orderdict[f'f_y{year_offset}'] = pid2f_M[pid] / citation_count
            orderdict[f'g_y{year_offset}'] = pid2g_M[pid] / citation_count
            for t in DISP_F_DOWN_THRESHOLDS:
                orderdict[dispf_colname(t, 'share')] = dispf_counts[t] / citation_count

        df_res.append(orderdict)

    df_res = pd.DataFrame(df_res)
    df_res.to_csv(os.path.join(save_dire, f'{file_base}_d_{division_index}_max_{division_count}.csv'), index=False)

    with open(finish_dire, 'wb') as f:
        pickle.dump(tuple(), f, pickle.HIGHEST_PROTOCOL)


print("finished")
