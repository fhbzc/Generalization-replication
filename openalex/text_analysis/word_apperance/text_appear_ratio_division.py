import pickle
import pandas as pd
import os
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import numpy as np
import sys
# utils.py lives two levels up (openalex/), not in this word_apperance/ directory;
# add it to the path so the import resolves regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import META_DIRE

meta_dire = META_DIRE
print("meta_dire",meta_dire)

# input: the merged F/E/G table produced by get_efg_s2_merge.py
load_dire = os.path.join(meta_dire, 'efg_computation')

save_dire = os.path.join(meta_dire, 'text_appear_gen_foundation_ext')
os.makedirs(save_dire, exist_ok = True)

# these must match get_efg.py / get_efg_s2_merge.py so the merged file name resolves
year_offset = 5
reference_threshold = 1
citation_threshold = 5
# normal mode only (reshuffle variant lives in atypicality/get_efg_reshuffle.py)

print("year_offset",year_offset)
print("citation_threshold",citation_threshold)
print("reference_threshold",reference_threshold)

# F = foundation (pillar), E = extension, G = generalization; *_y{off} are the
# citation-normalized shares emitted by get_efg.py
foundation_col = f'f_y{year_offset}'
extension_col = f'e_y{year_offset}'
generalization_col = f'g_y{year_offset}'
citation_col = f'citation_count_y{year_offset}'

year_threshold = 2020  # matches the title token-list file (before_2020)
print("year_threshold",year_threshold)

merged_csv = os.path.join(
    load_dire,
    f'gen_V3_y{year_offset}.csv'
)
df_gen = pd.read_csv(merged_csv, usecols = ['pid', citation_col, 'up_refer_count', 'pub_year',
                                            foundation_col, extension_col, generalization_col])
df_gen = df_gen[df_gen['up_refer_count'] >= reference_threshold]
df_gen = df_gen[df_gen[citation_col] >= citation_threshold]
df_gen = df_gen[df_gen['pub_year'] < year_threshold]


method_string = 'lemma' # [lemma, raw]
print("method_string",method_string)
with open(os.path.join(meta_dire, 'pid2eng_token_list_%s_before_%s.pkl'%(method_string, year_threshold)), 'rb') as f:
    pid2token_list = pickle.load(f)
print("load pid2token_list finished")
rng = np.random.default_rng(seed=42)
noise = rng.uniform(low=-1e-9, high=1e-9, size=len(df_gen))


for key_word in tqdm([foundation_col, extension_col, generalization_col]):


    df_gen[key_word] = df_gen[key_word] + noise


    median = df_gen[key_word].median()
    top = df_gen[df_gen[key_word] > median]
    bottom = df_gen[df_gen[key_word] <= median]

    top_valid_pid_count = 0
    bottom_valid_pid_count = 0
    top_word2fre = defaultdict(int)
    bottom_word2fre = defaultdict(int)


    for pid in set(top['pid']):
        if pid in pid2token_list:
            top_valid_pid_count += 1
            token_list = pid2token_list[pid]
            valid_token_set = set(token_list)

            for token in valid_token_set:
                top_word2fre[token] += 1


    for pid in set(bottom['pid']):
        if pid in pid2token_list:
            bottom_valid_pid_count += 1
            token_list = pid2token_list[pid]
            valid_token_set = set(token_list)


            for token in valid_token_set:
                bottom_word2fre[token] += 1


    appear_threshold = (top_valid_pid_count + bottom_valid_pid_count) * 0.0001
    word_tb_ratio = []
    for word in set(top_word2fre) & set(bottom_word2fre):
        top_fre = top_word2fre[word]
        bottom_fre = bottom_word2fre[word]

        total_fre = top_fre + bottom_fre
        if total_fre < appear_threshold:
            continue

        top_fre_ratio = top_fre / top_valid_pid_count
        bottom_fre_ratio = bottom_fre / bottom_valid_pid_count

        top_bottom_ratio = top_fre_ratio / bottom_fre_ratio
        word_tb_ratio.append([word, top_bottom_ratio])

    word_tb_ratio = sorted(word_tb_ratio, key = lambda x:-x[1])


    df_res = []
    rank = 0
    for word, ratio in word_tb_ratio:
        orderdict = OrderedDict()
        orderdict['word'] = word
        orderdict['rank'] = rank
        orderdict['top_bottom_ratio'] = ratio
        orderdict['top_fre'] = top_word2fre[word]
        orderdict['bottom_fre'] = bottom_word2fre[word]
        df_res.append(orderdict)

        rank += 1


    df_res = pd.DataFrame(df_res)
    df_res.to_csv(os.path.join(save_dire, '%s_c_t_%s_r_t_%s_V3.csv'%(key_word, citation_threshold, reference_threshold)), index = False)
print("finished")
