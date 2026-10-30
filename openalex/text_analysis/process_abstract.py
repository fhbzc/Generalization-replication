import os
import pickle
from tqdm import tqdm
from collections import OrderedDict, defaultdict
from langdetect import detect, detect_langs, DetectorFactory
import string
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import random
import re
import html
import time
import fasttext
import sys
# utils.py lives one level up (openalex/), not in this text_analysis/ directory;
# add the parent directory to the path so the import resolves regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import META_DIRE

tag_pattern = re.compile(r'<[^>]+>')
def remove_html_fast(text):

    return tag_pattern.sub('', html.unescape(text))

meta_dire = META_DIRE
process_count = 1

print("meta_dire",meta_dire)
print("process_count",process_count)


trans_table_obj = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
def process_abstract_s1(pid_list):

    pid2abstract_P = {}
    for pid in tqdm(pid_list, position = 1, leave = False, desc = 'process title P'):

        abstract = remove_html_fast(pid2abstract[pid])
        pid2abstract_P[pid] = abstract.replace('\n', ' ').replace('\xa0', ' ')

    return pid2abstract_P



year_before = 2025
year_after = 2020
print("year_before",year_before)
print("year_after",year_after)

abstract_no_html_directory = os.path.join(meta_dire, 'pid2abstract_no_html_method_lemma_before_%s_after_%s.pkl'%(year_before, year_after))

# this lemma is for the method to process title, not for abstract
if os.path.exists(abstract_no_html_directory) == False:



    with open(os.path.join(meta_dire, 'pid2abstract_method_lemma_before_%s_after_%s.pkl'%(year_before, year_after)), 'rb') as f:
        pid2abstract = pickle.load(f)
    print("pid2abstract load finished", len(pid2abstract))

    pid_list = list(pid2abstract)
    print("size of pid_list", len(pid_list))
    if process_count > 1:

        batched_pid_list = [[] for _ in range(process_count)]
        for index, pid in enumerate(pid_list):
            batch_index = index % process_count
            batched_pid_list[batch_index].append(pid)

        pid2abstract_P = {}
        with ProcessPoolExecutor(max_workers=process_count) as executor:
            with tqdm(total=len(batched_pid_list), desc = 'process abstract P main') as pbar:
                for pid2title_P_sig in executor.map(process_abstract_s1, batched_pid_list):
                    pbar.update(1)
                    pid2abstract_P.update(pid2title_P_sig)
    else:
        pid2abstract_P = process_abstract_s1(pid_list)

    with open(abstract_no_html_directory, 'wb') as f:
        pickle.dump(pid2abstract_P, f, pickle.HIGHEST_PROTOCOL)

print("finished")