"""
Classify each S2ORC in-text reference instance into an E/F/G citation type plus a
disruptive / non_disruptive axis, using the de-circularized S2AG citation network.
For a focal (citing) paper A and a cited paper B (both matched to a corpusid via
their DOI), with up(X) = papers X cites and down(X) = papers citing X:
    up_count   = |up(A) & up(B)|     (references of B that A also cites)
    down_count = |up(A) & down(B)|   (citers of B that A also cites)
    disruptive : up_count == 0 (A cites none of B's references), else non_disruptive
    g : up_count == 0 and down_count == 0
    f : down_count >= up_count and down_count > 0
    e : down_count <  up_count
On top of the disruptive/non_disruptive axis we also tag each disruptive instance
with the "disruptive foundation" (disp_f) variants, mirroring
openalex/get_efg_v2_disp_f.py: a citation is a disruptive foundation at threshold T
when up_count == 0 AND down_count > T, for T in DISP_F_DOWN_THRESHOLDS = [0, 1, 2].
Each instance stores disp_f_types = ['disp_f_{T}' for every T it exceeds] (always []
for non_disruptive instances, since those have up_count > 0).
Only references whose cited paper is among A's recorded references are kept (the
rest counted as error_instance_count). The citing paper's publication year is
attached to each kept instance for longitudinal analysis.

Input  : META_DIRE/{cid2DOI, cid2pub_year}.pkl,
         META_DIRE/{citing2cited, cited2citing}_cid_list_remove_circle.pkl
         S2ORC_META_DIRE/DOI2refer_instance_list_V2.pkl   (from get_s2orc_meta_merge.py)
Output : S2ORC_META_DIRE/DOI2refer_instance_list_cid_parsed_V2.pkl
"""

import os
import sys
import pickle
from collections import defaultdict

from tqdm import tqdm

# utils.py is in the parent directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
from utils import S2ORC_META_DIRE, META_DIRE  # noqa: E402

meta_original = META_DIRE
meta_dire = S2ORC_META_DIRE

# disruptive-foundation thresholds, mirroring openalex/get_efg_v2_disp_f.py: a disruptive
# citation (up_count == 0) is additionally a "disruptive foundation" at threshold T when
# down_count > T. T=0 is the plain foundation-among-disruptive case (efg_type == 'f').
DISP_F_DOWN_THRESHOLDS = [0, 1, 2]


def disp_f_types_for(down_count):
    """Labels of the disp_f variants a disruptive (up_count == 0) instance qualifies for."""
    return [f'disp_f_{t}' for t in DISP_F_DOWN_THRESHOLDS if down_count > t]

with open(os.path.join(meta_original,'cid2DOI.pkl'),'rb') as f:
    cid2DOI = pickle.load(f)

doi2pid_list = defaultdict(list)

for cid, doi in tqdm(cid2DOI.items(), desc="Building doi2pid_list"):

    doi2pid_list[doi].append(cid)

doi2pid = {doi: pid_list[0] for doi, pid_list in tqdm(doi2pid_list.items()) if len(pid_list) == 1}

with open(os.path.join(meta_original, 'citing2cited_cid_list_remove_circle.pkl'), 'rb') as f:
    pid2up_list = pickle.load(f)

print("pid2up_list load finished")

with open(os.path.join(meta_original, 'cited2citing_cid_list_remove_circle.pkl'), 'rb') as f:
    # ordered version for optimization
    pid2down_list = pickle.load(f)
print("pid2down_list load finished")

with open(os.path.join(meta_original, 'cid2pub_year.pkl'), 'rb') as f:
    cid2pub_year = pickle.load(f)   # citing paper's publication year (for longitudinal analysis)
print("cid2pub_year load finished")

with open(os.path.join(meta_dire, 'DOI2refer_instance_list_V2.pkl'), 'rb') as f:
    DOI2refer_instances_M = pickle.load(f)
print("DOI2refer_instances_M load finished")

DOI2refer_instances_pid_parsed = {}
both_pid_matched_count = 0
error_instance_count = 0
for DOI, refer_instance_list in tqdm(DOI2refer_instances_M.items()):
    new_refer_instance_list = []
    if DOI in doi2pid:

        focal_pid = doi2pid[DOI]
        focal_up_pid_set = set(pid2up_list.get(focal_pid, []))
        citing_year = cid2pub_year.get(focal_pid)   # citing paper's publication year
        for refer_instance in refer_instance_list:
            refer_doi = refer_instance['doi']

            
            if refer_doi is not None and refer_doi in doi2pid:

                up_pid = doi2pid[refer_doi]
                
                refer_instance['cid'] = up_pid
                both_pid_matched_count += 1
                if up_pid not in focal_up_pid_set:

                    error_instance_count += 1
                    continue


                up_of_up_set = set(pid2up_list.get(up_pid, []))
                down_of_up_set = set(pid2down_list.get(up_pid, []))

                # disruptive axis (independent of E/F/G; stored separately):
                #   up_count == 0 (focal cites NONE of the cited paper's references)
                #     -> 'disruptive', else 'non_disruptive'.
                if up_of_up_set.isdisjoint(focal_up_pid_set):
                    # no overlap in up  ->  up_count == 0  ->  disruptive
                    refer_instance['disp_type'] = 'disruptive'

                    down_count = len(focal_up_pid_set & down_of_up_set)
                    if down_count == 0:
                        # gen citation
                        refer_instance['efg_type'] = 'g'
                    else:
                        refer_instance['efg_type'] = 'f'
                    # disruptive-foundation variants (up_count == 0 & down_count > T)
                    refer_instance['disp_f_types'] = disp_f_types_for(down_count)
                else:
                    # has overlap in up  ->  up_count > 0  ->  non_disruptive
                    refer_instance['disp_type'] = 'non_disruptive'

                    up_count = len(focal_up_pid_set & up_of_up_set)
                    down_count = len(focal_up_pid_set & down_of_up_set)

                    if down_count >= up_count:
                        refer_instance['efg_type'] = 'f'
                    else:
                        refer_instance['efg_type'] = 'e'
                    # up_count > 0 -> not a disruptive foundation under any threshold
                    refer_instance['disp_f_types'] = []

                # citing paper's publication year (for longitudinal analysis)
                refer_instance['citing_year'] = citing_year
                new_refer_instance_list.append(refer_instance)

        if len(new_refer_instance_list) > 0:
            DOI2refer_instances_pid_parsed[DOI] = new_refer_instance_list



with open(os.path.join(meta_dire, 'DOI2refer_instance_list_cid_parsed_V2.pkl'), 'wb') as f:
    pickle.dump(DOI2refer_instances_pid_parsed, f, pickle.HIGHEST_PROTOCOL)
print("size of DOI2refer_instances_pid_parsed", len(DOI2refer_instances_pid_parsed))
print("size of both_pid_matched_count", both_pid_matched_count)
print("size of error_instance_count", error_instance_count)
print("finished")