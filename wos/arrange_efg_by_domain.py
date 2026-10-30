"""
Split the merged per-paper E/F/G table (from get_efg_s2_merge.py) into one CSV per
domain. Loads concept2pid_list.pkl (domain -> [pid], from get_meta_data.py) and, for
each domain, keeps the rows of gen_V3_y{year_offset}.csv whose 'pid' belongs to that
domain, writing a per-domain CSV (a paper in several domains appears in each).
Resumable via a per-domain finished-flag. `year_offset` must match get_efg.py /
get_efg_s2_merge.py.

Input  : META_DIRE/concept2pid_list.pkl                       (domain -> [pid], from get_meta_data.py)
         META_DIRE/efg_computation/gen_V3_y{year_offset}.csv  (from get_efg_s2_merge.py)
Output : META_DIRE/efg_computation/efg_by_domain/<domain>/efg_<domain>_y{year_offset}.csv
         (+ efg_arrange_finished.pkl markers)
"""

import os
import sys
import pickle

import pandas as pd
from tqdm import tqdm

# utils.py is in this same directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from utils import META_DIRE  # noqa: E402

meta_dire = META_DIRE
print("meta_dire", meta_dire)

year_offset = 5  # must match get_efg.py / get_efg_s2_merge.py
print("year_offset", year_offset)

load_dire = os.path.join(meta_dire, 'efg_computation')
save_dire = os.path.join(load_dire, 'efg_by_domain')
os.makedirs(save_dire, exist_ok=True)

with open(os.path.join(meta_dire, 'concept2pid_list.pkl'), 'rb') as f:
    topic2P_list = pickle.load(f)
print("topic2P_list load finished; domains:", len(topic2P_list))

merged_csv = os.path.join(load_dire, f'gen_V3_y{year_offset}.csv')
# pid must stay a string: WoS ids are digit-like and pandas would otherwise infer
# them as ints (leading zeros lost), so the isin() against the string pids of
# concept2pid_list.pkl below would silently drop those papers.
df_combined = pd.read_csv(merged_csv, dtype={'pid': str})
print("merged efg table:", df_combined.shape)

topic_list = list(topic2P_list)

for topic in tqdm(topic_list, desc='process by domain'):

    save_address = os.path.join(save_dire, topic)
    os.makedirs(save_address, exist_ok=True)
    if os.path.exists(os.path.join(save_address, 'efg_arrange_finished.pkl')):
        # already processed
        continue

    valid_pid_set = set(topic2P_list[topic])
    df_selected = df_combined[df_combined['pid'].isin(valid_pid_set)]

    df_selected.to_csv(
        os.path.join(save_address, f'efg_{topic}_y{year_offset}.csv'), index=False)

    with open(os.path.join(save_address, 'efg_arrange_finished.pkl'), 'wb') as f:
        pickle.dump(tuple(), f, pickle.HIGHEST_PROTOCOL)


print("finished")
