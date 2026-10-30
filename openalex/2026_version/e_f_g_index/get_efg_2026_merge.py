import os
import pickle
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from  utils import META_DIRE_2026


meta_dire = META_DIRE_2026   # 2026 metadata (separate from the main META_DIRE)
print("meta_dire", meta_dire)

division_count = 20
print("division_count", division_count)

load_dire = os.path.join(meta_dire, 'gen_V3_keep_disp')

year_offset = 5  # -1 for no offset
print("year_offset", year_offset)

# ==================================================
# 1. load & concat all divisions
# ==================================================
df_list = []

for division_index in tqdm(range(division_count), desc="loading divisions"):
    finish_dire = os.path.join(
        load_dire,
        f'gen_V3_y{year_offset}_finished_d_{division_index}_max_{division_count}.pkl'
    )
    assert os.path.exists(finish_dire)

    csv_path = os.path.join(
        load_dire,
        f'gen_V3_y{year_offset}_d_{division_index}_max_{division_count}.csv'
    )
    assert os.path.exists(csv_path)

    df = pd.read_csv(csv_path)
    df_list.append(df)

df_res = pd.concat(df_list, ignore_index=True)
print("concat df shape:", df_res.shape)

# ==================================================
# 2. rename pid -> cid
# ==================================================
df_res = df_res.rename(columns={'pid': 'cid'})

# ==================================================
# 3. normalize counts -> index shares  e/f/g_y{off} = *_count_y{off} / citation_count_y{off}
#    (0 when citation_count == 0). get_efg_2026.py writes raw counts; downstream
#    (plot_longitudional_change.py) expects the normalized e_y/f_y/g_y shares, same
#    schema as the WoS gen_V3 table.
# ==================================================
citation_col = f'citation_count_y{year_offset}'
for letter in ('e', 'f', 'g'):
    count_col = f'{letter}_count_y{year_offset}'
    share_col = f'{letter}_y{year_offset}'
    df_res[share_col] = (df_res[count_col] / df_res[citation_col]).where(
        df_res[citation_col] > 0, 0.0)

# ==================================================
# 4. save (no filtering -- thresholds are applied downstream)
# ==================================================
save_path = os.path.join(
    load_dire,
    f'gen_V3_y{year_offset}.csv'
)

df_res.to_csv(save_path, index=False)
print("saved df shape:", df_res.shape)
print("saved to:", save_path)
print("finished")