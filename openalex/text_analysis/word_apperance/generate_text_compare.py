import pickle
import pandas as pd
import os
from collections import OrderedDict
from tqdm import tqdm
from openpyxl.styles import PatternFill
from openpyxl import load_workbook
import numpy as np
import sys
# utils.py lives two levels up (openalex/), not in this word_apperance/ directory;
# add it to the path so the import resolves regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import META_DIRE

meta_dire = META_DIRE
print("meta_dire",meta_dire)

# input: the merged F/E/G table produced by get_efg_s2_merge.py (same source and
# split as text_appear_ratio_division.py, so the top/bottom halves match)
load_dire = os.path.join(meta_dire, 'efg_computation')

# these must match get_efg.py / get_efg_s2_merge.py / text_appear_ratio_division.py
year_offset = 5
reference_threshold = 1
citation_threshold = 5
# normal mode only (reshuffle variant lives in atypicality/get_efg_reshuffle.py)

# F = foundation (pillar), E = extension, G = generalization; *_y{off} are the
# citation-normalized shares emitted by get_efg.py
foundation_col = f'f_y{year_offset}'
extension_col = f'e_y{year_offset}'
generalization_col = f'g_y{year_offset}'
citation_col = f'citation_count_y{year_offset}'

year_threshold = 2020  # matches the title token-list file (before_2020)
print("citation_threshold",citation_threshold)
print("reference_threshold",reference_threshold)
print("year_threshold",year_threshold)

merged_csv = os.path.join(
    load_dire,
    f'gen_V3_y{year_offset}.csv'
)
df_gen = pd.read_csv(merged_csv, usecols = [citation_col, 'up_refer_count', 'pub_year',
                                            foundation_col, extension_col, generalization_col])
df_gen = df_gen[df_gen['up_refer_count'] >= reference_threshold]
df_gen = df_gen[df_gen[citation_col] >= citation_threshold]
df_gen = df_gen[df_gen['pub_year'] < year_threshold]

rng = np.random.default_rng(seed=42)
noise = rng.uniform(low=-1e-9, high=1e-9, size=len(df_gen))

keyword2top_size = {}
keyword2bottom_size = {}

for key_word in tqdm([foundation_col, extension_col, generalization_col]):


    df_gen[key_word] = df_gen[key_word] + noise

    median = df_gen[key_word].median()
    top = df_gen[df_gen[key_word] > median]
    bottom = df_gen[df_gen[key_word] <= median]

    keyword2top_size[key_word] = len(top)
    keyword2bottom_size[key_word] = len(bottom)


load_address = os.path.join(meta_dire, 'text_appear_gen_foundation_ext')

print("citation_threshold",citation_threshold)
print("reference_threshold",reference_threshold)

word_list = ['device', 'tool', 'software', 'review','guideline', 'tutorial', 'new', 'novel', 'innovative',
                'theory', 'metric', 'hypothesis', 'discovery', 'dataset', 'observation']


print("word_list",word_list)
index2word2ratio = {}
index_name_list = [foundation_col, extension_col, generalization_col]


index2word2top_fre = {}
index2word2bottom_fre = {}

for index_name in tqdm(index_name_list, desc = 'load data'):


    df_res = pd.read_csv(os.path.join(load_address, '%s_c_t_%s_r_t_%s_V3.csv'%(index_name, citation_threshold, reference_threshold)))
    word2index_ratio = dict(zip(df_res['word'], df_res['top_bottom_ratio']))
    word2top_fre = dict(zip(df_res['word'], df_res['top_fre']))
    word2bottom_fre = dict(zip(df_res['word'], df_res['bottom_fre']))

    index2word2ratio[index_name] = word2index_ratio
    index2word2top_fre[index_name] = word2top_fre
    index2word2bottom_fre[index_name] = word2bottom_fre


def fast_bootstrap_mean_ci(data, num_samples=100, ci=95, random_state=None):
    data = np.asarray(data)
    n = len(data)
    rng = np.random.default_rng(seed=random_state)

    # Generate bootstrap indices: shape (n, num_samples)
    bootstrap_indices = rng.integers(0, n, size=(num_samples, n))

    # Use advanced indexing to create bootstrap samples and compute means
    bootstrap_samples = data[bootstrap_indices]  # shape (num_samples, n)
    means = bootstrap_samples.mean(axis=1)

    # Compute confidence interval
    lower = np.percentile(means, (100 - ci) / 2)
    upper = np.percentile(means, 100 - (100 - ci) / 2)
    mean = means.mean()

    return mean, lower, upper



df_res = []
for word in tqdm(word_list, desc = 'iterate over words'):
    orderdict = OrderedDict()
    orderdict['word'] = word
    for index_name in tqdm(index_name_list, desc = 'iterate over index', position = 1, leave = False):
        orderdict['%s_word_ratio'%index_name] = index2word2ratio[index_name][word] if word in index2word2ratio[index_name] else -1
        top_n_fre = keyword2top_size[index_name] - index2word2top_fre[index_name][word]
        top_fre_list = [1 for _ in range(index2word2top_fre[index_name][word])]
        top_fre_list.extend([0 for _ in range(top_n_fre)])
        assert len(top_fre_list) == keyword2top_size[index_name]

        top_fre_mean, top_lower, top_upper = fast_bootstrap_mean_ci(top_fre_list)

        bottom_n_fre = keyword2bottom_size[index_name] - index2word2bottom_fre[index_name][word]
        bottom_fre_list = [1 for _ in range(index2word2bottom_fre[index_name][word])]
        bottom_fre_list.extend([0 for _ in range(bottom_n_fre)])
        assert len(bottom_fre_list) == keyword2bottom_size[index_name]

        bottom_fre_mean, bottom_lower, bottom_upper = fast_bootstrap_mean_ci(bottom_fre_list)
        orderdict['%s_top_ratio_mean'%index_name] = top_fre_mean
        orderdict['%s_top_ratio_lower'%index_name] = top_lower
        orderdict['%s_top_ratio_upper'%index_name] = top_upper

        orderdict['%s_bottom_ratio_mean'%index_name] = bottom_fre_mean
        orderdict['%s_bottom_ratio_lower'%index_name] = bottom_lower
        orderdict['%s_bottom_ratio_upper'%index_name] = bottom_upper

    df_res.append(orderdict)

df_res = pd.DataFrame(df_res)

output_path = os.path.join(load_address, 'manual_evaluation_word_c_t_%s_r_t_%s_V3.xlsx'%(citation_threshold, reference_threshold))


df_res.to_excel(output_path, index = False)


# assign_color
wb = load_workbook(output_path)
ws = wb.active

# Define a blue fill
blue_fill = PatternFill(start_color="ADD8E6", end_color="ADD8E6", fill_type="solid")  # light blue
red_fill = PatternFill(start_color="FF9999", end_color="FF9999", fill_type="solid")    # light red
# Apply to all cells with value < 1.1
for row in ws.iter_rows(min_row=2, min_col=0, max_row=ws.max_row, max_col=ws.max_column):
    for cell in row:
        try:
            if isinstance(cell.value, (float)):
                if  cell.value < 0.8:
                    cell.fill = blue_fill
                elif cell.value > 1.2:
                    cell.fill = red_fill

        except:
            pass  # ignore errors for non-numeric cells

# Save workbook
wb.save(output_path)

print("finished")
