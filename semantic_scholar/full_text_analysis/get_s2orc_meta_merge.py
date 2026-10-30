"""
Merge the per-batch S2ORC full-text metadata written by get_s2orc_meta.py into two
single pickles. Reads every batch's section-header and reference-instance pickles
(asserting each batch's finished-flag exists), drops the now-unneeded 'end' offset
from each reference instance, and concatenates them across all batches.

Input  : S2ORC_META_DIRE/meta_batched_v2/reference/
           DOI2section_header_list_b_<batch>_total_<N>.pkl
           DOI2refer_instance_list_b_<batch>_total_<N>.pkl   (from get_s2orc_meta.py)
Outputs: S2ORC_META_DIRE/DOI2section_header_list_V2.pkl
         S2ORC_META_DIRE/DOI2refer_instance_list_V2.pkl
"""

import os
import sys
import pickle

from tqdm import tqdm

# utils.py is in the parent directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
from utils import S2ORC_META_DIRE  # noqa: E402

meta_dire = S2ORC_META_DIRE


batch_count = 32
print("batch_count",batch_count)

load_dire = os.path.join(meta_dire, 'meta_batched_v2', 'reference')


DOI2section_header_M = {}
DOI2refer_instances_M = {}

for batch in tqdm(range(batch_count)):
    finished_dire = os.path.join(load_dire,  f'batch_{batch}_total_{batch_count}_finished.pkl')
    assert os.path.exists(finished_dire)

    with open(os.path.join(load_dire, f'DOI2section_header_list_b_{batch}_total_{batch_count}.pkl'), 'rb') as f:
        DOI2section_header = pickle.load(f)
    with open(os.path.join(load_dire, f'DOI2refer_instance_list_b_{batch}_total_{batch_count}.pkl'), 'rb') as f:
        DOI2refer_instances = pickle.load(f)

    DOI2refer_instances_P = {}
    for DOI, refer_instances in tqdm(DOI2refer_instances.items(), position = 1, leave = False):

        
        for refer_instance in refer_instances:
            refer_instance.pop('end')
        DOI2refer_instances_P[DOI] = refer_instances
    DOI2section_header_M.update(DOI2section_header)
    DOI2refer_instances_M.update(DOI2refer_instances_P)


with open(os.path.join(meta_dire, 'DOI2section_header_list_V2.pkl'), 'wb') as f:
    pickle.dump(DOI2section_header_M, f, pickle.HIGHEST_PROTOCOL)


with open(os.path.join(meta_dire, 'DOI2refer_instance_list_V2.pkl'), 'wb') as f:
    pickle.dump(DOI2refer_instances_M, f, pickle.HIGHEST_PROTOCOL)

print("size of DOI2section_header_M", len(DOI2section_header_M))
print("size of DOI2refer_instances_M", len(DOI2refer_instances_M))
print("finished")
