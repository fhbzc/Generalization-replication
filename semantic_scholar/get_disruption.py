"""
Semantic Scholar (S2AG) version of the OpenAlex get_disruption.py.

Compute the disruption (CD-style) index for every S2AG paper from the S2AG
citation network, in per-division shards. Paper ids are S2AG corpus ids (cid); they
are kept in the 'pid' column on output so the downstream merge / quantile steps stay
identical to the OpenAlex and WoS pipelines.

For a focal paper P (published in [1945, max_year)) we look at the papers that cite
P within `year_offset` years of P's publication (year_offset == -1 disables the
window, but the horizon max_year - 1 always applies). Over P's citing papers C we
partition:
    J (developmental) : C also cites at least one of P's references
    I (disruptive)    : C cites none of P's references
and, over the citers of P's references (within the same year window):
    K (raw)           : papers citing P's references; size_K = |K_raw| - |J|
The disruption index is
    disp = (|I| - |J|) / (|I| + |J| + size_K)
mirroring the OpenAlex/WoS get_disruption.py exactly.

Papers are split into `division_count` divisions (round-robin, matching get_efg.py);
each division is computed (optionally over `process_count` worker processes), written
to its own CSV, and marked with a finished-flag so the job is resumable.
arrange_disruption.py then concatenates the per-division CSVs.

The citation network is the temporally-consistent ("remove_circle") network from
get_s2ag_citation_s2_remove_circle.py, the same one every other S2 downstream step uses.
Set use_openalex_augmented=True to instead run on the OpenAlex-augmented remove_circle
network from augment_s2_citation_with_openalex.py; that choice is tagged onto both the
input filenames and the output filenames (network_suffix) so the two never collide.

Input  : META_DIRE/cid2pub_year.pkl                                            (get_s2ag_meta.py)
         META_DIRE/cid2title.pkl                                               (get_s2ag_meta.py)
         META_DIRE/citing2cited_cid_list_remove_circle[_openalex_augmented].pkl  (citing -> cited)
         META_DIRE/cited2citing_cid_list_remove_circle[_openalex_augmented].pkl  (cited -> citing)
         (from get_s2ag_citation_s2_remove_circle.py, or augment_s2_citation_with_openalex.py)
Output : META_DIRE/disruption_division/disp_paper[_y{year_offset}][_openalex_augmented][_complete_window]_d_{division}_max_{division_count}.csv
         (+ disp_finished[_y{year_offset}][_openalex_augmented][_complete_window]_d_{division}_max_{division_count}.pkl markers)
         The window appears in the names only when it is NOT the default 5, so the 5-year run
         keeps the names it already has on disk and any other window (e.g. -1, the no-offset
         variant) lands beside it instead of overwriting it.
         Two windows therefore never mix, and a resumed run never picks up another window's
         finished-flags. arrange_disruption.py must use the same year_offset.
         (the _complete_window tag is present when skip_incomplete_window is True AND
          year_offset != -1 -- at -1 the flag changes nothing, so it is left out of the name)
"""

import os
import sys
import pickle
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

division_count = 20
print("division_count", division_count)

max_year = 2025
print("max_year", max_year)

year_offset = -1  # -1 for no offset
# Every per-division output carries the window UNLESS it is the default 5, whose files already
# exist on disk under the untagged names.
year_tag = '' if year_offset == 5 else '_y%s' % year_offset
print("year_offset", year_offset, "| year_tag", repr(year_tag))

# When True (and year_offset != -1), skip focal papers whose citation window would run
# past the data horizon -- i.e. focal_pub_year >= max_year - year_offset -- because their
# year_offset-year window is incomplete (only partially observed) and would bias the
# disruption index. No effect when year_offset == -1.
skip_incomplete_window = True
print("skip_incomplete_window", skip_incomplete_window)
# tag the output filenames so the skip / no-skip variants never overwrite each other.
# The flag only does anything when there IS a window to be incomplete: at
# year_offset == -1 the citation window is unbounded, get_disruption.py's skip test
# short-circuits, and the run is identical either way -- so the tag is dropped there,
# and one -1 run is not written under two names that hold the same data.
window_suffix = ('_complete_window'
                  if (skip_incomplete_window and year_offset != -1) else '')
print("window_suffix", repr(window_suffix))

# Optionally restrict which divisions to compute (each run is resumable via the
# disp_finished flags, so the work can be split across machines / invocations).
# Divisions processed are [start_division, end_division) -- end is exclusive.
# None means no limit (start at 0 / run through division_count).
start_division = 10  # inclusive
end_division = 15     # exclusive
division_start = 0 if start_division is None else start_division
division_end = division_count if end_division is None else end_division
print("division range [%s, %s)" % (division_start, division_end))

# When True, run on the OpenAlex-augmented '_remove_circle' network produced by
# augment_s2_citation_with_openalex.py (the S2AG remove_circle network plus the citation
# edges OpenAlex has but S2AG was missing) instead of the plain S2AG remove_circle
# network. The choice is encoded in BOTH the input network filenames AND the output
# filenames via network_suffix, so the augmented and non-augmented disruption results
# never overwrite each other. arrange_disruption.py must use the same use_openalex_augmented.
use_openalex_augmented = False
network_suffix = '_openalex_augmented' if use_openalex_augmented else ''
print("use_openalex_augmented", use_openalex_augmented)
print("network_suffix", repr(network_suffix))


with open(os.path.join(meta_dire, 'cid2pub_year.pkl'), 'rb') as f:
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

    return disp_index, \
        size_dev, \
        size_disp, \
        size_K


def disp_analysis(pid_list, batch_index):

    # only the reference COUNT is kept (the full id list is no longer persisted)
    pid2up_refer_count = {}

    # year_offset is a single scalar, so each paper maps to a single value
    # (not a per-offset list) -- mirrors the faster WoS get_disruption.py.
    pid2year_disp_index = {}

    pid2year_I_count = {}
    pid2year_J_count = {}
    pid2year_K_count = {}

    for focal_pid in tqdm(pid_list, desc='batch %s' % batch_index, position=3, leave=False):
        focal_pub_year = pid2year[focal_pid]
        if focal_pub_year >= max_year or focal_pub_year < 1945:
            continue
        # drop focal papers whose year_offset window is not fully observed
        if skip_incomplete_window and year_offset != -1 and focal_pub_year >= max_year - year_offset:
            continue
        focal_reference_set = set(citing_pid2cited_works.get(focal_pid, ()))
        pid2up_refer_count[focal_pid] = len(focal_reference_set)

        if year_offset == -1:
            # all future citations, but still bounded by the data horizon (max_year - 1)
            max_year_threshold = max_year - 1
        else:
            max_year_threshold = min(focal_pub_year + year_offset, max_year - 1)

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

        # every paper in citing_focal_papers is already within the year window
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

    return pid2up_refer_count, \
        pid2year_disp_index, \
        pid2year_I_count, \
        pid2year_J_count, \
        pid2year_K_count


save_dire = os.path.join(meta_dire, 'disruption_division')
os.makedirs(save_dire, exist_ok=True)

citing_pid2cited_pids_path = os.path.join(meta_dire, f'citing2cited_cid_list_remove_circle{network_suffix}.pkl')
cited_pid2citing_pids_path = os.path.join(meta_dire, f'cited2citing_cid_list_remove_circle{network_suffix}.pkl')

print("citing_pid2cited_pids_path", citing_pid2cited_pids_path)
print("cited_pid2citing_pids_path", cited_pid2citing_pids_path)

with open(citing_pid2cited_pids_path, 'rb') as f:
    citing_pid2cited_works = pickle.load(f)
print("citing_pid2cited_works load finished")

with open(cited_pid2citing_pids_path, 'rb') as f:
    cited_pid2citing_works = pickle.load(f)
print("cited_pid2citing_works load finished")


# round-robin division assignment, matching get_efg.py so the two S2 metrics
# partition the corpus identically
cid_list_M = sorted(list(pid2year.keys()))
division2cid_list = {division_index: [] for division_index in range(division_count)}
for i, cid in enumerate(cid_list_M):
    division2cid_list[i % division_count].append(cid)


computed_pid_count = 0

for division_index in tqdm(range(division_start, division_end), position=1):

    pid_list = division2cid_list[division_index]
    computed_pid_count += len(pid_list)

    finish_dire = os.path.join(
        save_dire, f'disp_finished{year_tag}{network_suffix}{window_suffix}_d_{division_index}_max_{division_count}.pkl')
    if os.path.exists(finish_dire) == True:
        continue

    if process_count > 1:

        batches = [[] for _ in range(process_count)]
        batch_indexs = [i for i in range(process_count)]
        pid_index = 0
        for pid in tqdm(pid_list, desc='process batch data s1', position=2, leave=False):
            batch_index = pid_index % process_count
            batches[batch_index].append(pid)
            pid_index += 1

        pid2up_refer_count_M = {}
        pid2year_disp_index_M = {}
        pid2year_I_count_M = {}
        pid2year_J_count_M = {}
        pid2year_K_count_M = {}

        with ProcessPoolExecutor(max_workers=process_count) as executor:

            with tqdm(total=process_count, desc='disp_analysis', position=2, leave=False) as pbar:
                for res in executor.map(disp_analysis, batches, batch_indexs):
                    pbar.update(1)

                    pid2up_refer_count, \
                        pid2year_disp_index, \
                        pid2year_I_count, \
                        pid2year_J_count, \
                        pid2year_K_count \
                        = res

                    pid2up_refer_count_M.update(pid2up_refer_count)
                    pid2year_disp_index_M.update(pid2year_disp_index)
                    pid2year_I_count_M.update(pid2year_I_count)
                    pid2year_J_count_M.update(pid2year_J_count)
                    pid2year_K_count_M.update(pid2year_K_count)
    else:
        pid2up_refer_count_M, \
            pid2year_disp_index_M, \
            pid2year_I_count_M, \
            pid2year_J_count_M, \
            pid2year_K_count_M \
            = disp_analysis(pid_list, 0)

    df_res = []
    traverse_list = list(pid2up_refer_count_M)
    for pid in tqdm(traverse_list, desc='process pid2disp', position=2, leave=False):

        orderdict = OrderedDict()
        orderdict['pid'] = pid
        orderdict['pub_year'] = pid2year[pid]

        orderdict['up_refer_count'] = pid2up_refer_count_M[pid]

        J = pid2year_J_count_M[pid]
        I = pid2year_I_count_M[pid]
        K = pid2year_K_count_M[pid]

        orderdict['J_count_y%s' % year_offset] = J
        orderdict['I_count_y%s' % year_offset] = I
        orderdict['K_count_y%s' % year_offset] = K

        citation = I + J

        orderdict['citation_count_y%s' % year_offset] = citation

        orderdict['disp_y%s' % year_offset] = pid2year_disp_index_M[pid]

        if I == 0:
            orderdict['disp_no_J_numer_y%s' % year_offset] = 0
        else:
            orderdict['disp_no_J_numer_y%s' % year_offset] = I / (I + J + K)

        df_res.append(orderdict)

    df_res = pd.DataFrame(df_res)
    df_res.to_csv(os.path.join(
        save_dire, f'disp_paper{year_tag}{network_suffix}{window_suffix}_d_{division_index}_max_{division_count}.csv'), index=False)

    with open(finish_dire, 'wb') as f:
        pickle.dump(tuple(), f, pickle.HIGHEST_PROTOCOL)


print("computed_pid_count", computed_pid_count)
print("len(pid2year)", len(pid2year))
# Only a full run (all divisions) is expected to cover every pid exactly once.
if division_start == 0 and division_end == division_count:
    assert len(cid_list_M) == computed_pid_count, 'Computation error'

print("finished")
