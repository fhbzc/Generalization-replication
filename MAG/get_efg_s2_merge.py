import os
import pickle
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import pandas as pd
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)               # utils.py is in this same directory
from utils import META_DIRE


meta_dire = META_DIRE
print("meta_dire", meta_dire)

division_count = 20
print("division_count", division_count)

load_dire = os.path.join(meta_dire, 'efg_computation')

year_offset = -1  # -1 for no offset; must match get_efg.py
print("year_offset", year_offset)

# Which E/F/G run to merge:
#   False -> get_efg.py output       (gen_V3_y{year_offset}...)
#   True  -> get_efg_v2_disp_f.py output, which additionally carries the
#            disruption_f_count / disruption_f columns (gen_V3_disp_f_y{year_offset}...)
# The choice is encoded in both the per-division input filenames and the merged
# output filename via file_prefix, so the two runs never collide.
use_disp_f = True
file_prefix = 'gen_V3_disp_f' if use_disp_f else 'gen_V3'
print("use_disp_f", use_disp_f)
print("file_prefix", file_prefix)

# Must match get_efg.py: selects which per-division network output to merge.
#   'normal' -> gen_V3_y{year_offset}_*
# The choice is encoded in both the per-division input filenames and the merged
# output filename via network_suffix, so future variants never collide.
network_mode = 'normal'
NETWORK_SUFFIX_MAP = {
    'normal': '',
}
assert network_mode in NETWORK_SUFFIX_MAP, network_mode
network_suffix = NETWORK_SUFFIX_MAP[network_mode]
print("network_mode", network_mode)
print("network_suffix", repr(network_suffix))

# ==================================================
# 1. load & concat all divisions
# ==================================================
df_list = []

for division_index in tqdm(range(division_count), desc="loading divisions"):
    finish_dire = os.path.join(
        load_dire,
        f'{file_prefix}_y{year_offset}{network_suffix}_finished_d_{division_index}_max_{division_count}.pkl'
    )
    assert os.path.exists(finish_dire)

    csv_path = os.path.join(
        load_dire,
        f'{file_prefix}_y{year_offset}{network_suffix}_d_{division_index}_max_{division_count}.csv'
    )
    assert os.path.exists(csv_path)

    df = pd.read_csv(csv_path)
    df_list.append(df)

df_res = pd.concat(df_list, ignore_index=True)
print("concat df shape:", df_res.shape)

# ==================================================
# 2. save
# ==================================================
# No filtering here: this step only concatenates the per-division output.
# Downstream scripts apply their own pub_year / reference / citation thresholds.
save_path = os.path.join(
    load_dire,
    f'{file_prefix}_y{year_offset}{network_suffix}.csv'
)

df_res.to_csv(save_path, index=False)
print("saved to:", save_path)
print("finished")
