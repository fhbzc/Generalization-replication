import pandas as pd
import os
from tqdm import tqdm
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)               # utils.py is in this same directory
from utils import META_DIRE

# Must match get_disruption.py: the review-inclusive meta (get_meta_data_preprint_review.py)
# lives in META_DIRE + '_with_review', so switching the base directory makes this step merge
# that variant's per-division disruption output and write disp_full under the same separate
# tree (no collision with the article-only run).
use_with_review = False
main_dire = META_DIRE + '_with_review' if use_with_review else META_DIRE
print("use_with_review", use_with_review)
print("main_dire",main_dire)
load_dire = os.path.join(main_dire, 'disruption_division')
save_dire = os.path.join(main_dire, 'disruption')
os.makedirs(save_dire, exist_ok = True)
division_count = 20
print("division_count", division_count)

# Must match get_disruption.py's year_offset_main: it tags every per-division file it writes with
# year_tag, so this step must look for the same tag (and write its disp_full under it) or it would
# merge a DIFFERENT citation window's output. BACKWARD COMPATIBILITY: year_offset_main = 5 keeps
# the untagged names, so the already-saved y5 files and disp_full.csv resolve exactly as before.
# (-1 is get_disruption.py's "no citation window" sentinel: everything up to 2025.)
year_offset_main = -1
year_tag = '' if year_offset_main == 5 else f'_y{year_offset_main}'
print("year_offset_main", year_offset_main, "| year_tag", repr(year_tag))

list_of_df = []
for division_index in tqdm(range(division_count), desc = 'load data'):

    csv_path = os.path.join(load_dire, 'disp_paper%s_d_%s_max_%s.csv'%(year_tag, division_index, division_count))
    assert os.path.exists(csv_path), (
        "%s\nRun get_disruption.py with year_offset_main=%s for every division first."
        % (csv_path, year_offset_main))
    df = pd.read_csv(csv_path)
    list_of_df.append(df)


df_combined = pd.concat(list_of_df, ignore_index=True)

save_path = os.path.join(save_dire, 'disp_full%s.csv'%year_tag)
if os.path.exists(save_path) == False:
    df_combined.to_csv(save_path, index = False)
print("saved to:", save_path)
assert len(df_combined) == len(set(df_combined['pid']))
print("save full finished, size", len(df_combined))

print("finished")
