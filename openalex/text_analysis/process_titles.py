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
import urllib.request
import sys
# utils.py lives one level up (openalex/), not in this text_analysis/ directory;
# add the parent directory to the path so the import resolves regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import META_DIRE, LID_MODEL_DIRE


tag_pattern = re.compile(r'<[^>]+>')
def remove_html_fast(title):

    return tag_pattern.sub('', html.unescape(title))


meta_dire = META_DIRE
process_count = 4

print("meta_dire",meta_dire)
print("process_count",process_count)

FASTTEXT_LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
lid_model_path = os.path.join(LID_MODEL_DIRE, "lid.176.bin")

def ensure_lid_model(model_path):
    if os.path.exists(model_path):
        return model_path
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print("language detection model not found, downloading from %s ..." % FASTTEXT_LID_URL)
    tmp_path = model_path + ".part"
    urllib.request.urlretrieve(FASTTEXT_LID_URL, tmp_path)
    os.replace(tmp_path, model_path)
    print("download finished:", model_path)
    return model_path

model = fasttext.load_model(ensure_lid_model(lid_model_path))
def is_english(title):

    if title is None:
        return False

    pred = model.predict(title)[0][0]
    return pred == '__label__en'

trans_table_obj = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
def process_title_s1(pid_list):

    pid2title_P = {}
    for pid in tqdm(pid_list, position = 1, leave = False, desc = 'process title P'):

        title = remove_html_fast(pid2title_M[pid])
        pid2title_P[pid] = title.replace('\n', ' ')

    return pid2title_P

def detect_english_title(pid_list):
    pid2eng = {}
    for pid in tqdm(pid_list, position = 1, leave = False, desc = 'process tokens'):

        title = pid2title_P[pid]

        pid2eng[pid] = is_english(title)
    return pid2eng



year_threshold = 2025
print("year_threshold",year_threshold)

title_no_html_directory = os.path.join(meta_dire, 'pid2title_no_html_before_%s.pkl'%year_threshold)
if os.path.exists(title_no_html_directory) == False:


    with open(os.path.join(meta_dire, 'pid2title.pkl'), 'rb') as f:
        pid2title_M = pickle.load(f)
    print("pid2title_M load finished")

    with open(os.path.join(meta_dire, 'pid2pub_year.pkl'), 'rb') as f:
        pid2year = pickle.load(f)
    print("pid2year load finished")

    pid2title_M = {pid: title for pid, title in tqdm(pid2title_M.items(), desc = 'get pid list') if pid2year[pid] < year_threshold and title is not None}
    del pid2year
    pid_list = list(pid2title_M)
    print("size of pid_list", len(pid_list))
    if process_count > 1:

        batched_pid_list = [[] for _ in range(process_count)]
        for index, pid in enumerate(pid_list):
            batch_index = index % process_count
            batched_pid_list[batch_index].append(pid)

        pid2title_P = {}
        with ProcessPoolExecutor(max_workers=process_count) as executor:
            with tqdm(total=len(batched_pid_list), desc = 'process title P main') as pbar:
                for pid2title_P_sig in executor.map(process_title_s1, batched_pid_list):
                    pbar.update(1)
                    pid2title_P.update(pid2title_P_sig)
    else:
        pid2title_P = process_title_s1(pid_list)

    with open(title_no_html_directory, 'wb') as f:
        pickle.dump(pid2title_P, f, pickle.HIGHEST_PROTOCOL)

    del pid2title_M



else:
    with open(title_no_html_directory, 'rb') as f:
        pid2title_P = pickle.load(f)

# detect english title

if os.path.exists(os.path.join(meta_dire, 'pid2eng_flag_before_%s.pkl'%year_threshold)) == False:
    pid_list = list(pid2title_P)
    if process_count > 1:

        batched_pid_list = [[] for _ in range(process_count)]
        for index, pid in enumerate(pid_list):
            batch_index = index % process_count
            batched_pid_list[batch_index].append(pid)

        pid2eng_flag = {}
        with ProcessPoolExecutor(max_workers=process_count) as executor:
            with tqdm(total=len(batched_pid_list), desc = 'detect english titles') as pbar:
                for pid2eng_flag_sig in executor.map(detect_english_title, batched_pid_list):
                    pbar.update(1)
                    pid2eng_flag.update(pid2eng_flag_sig)
    else:
        pid2eng_flag = detect_english_title(pid_list)

    with open(os.path.join(meta_dire, 'pid2eng_flag_before_%s.pkl'%year_threshold), 'wb') as f:
        pickle.dump(pid2eng_flag, f, pickle.HIGHEST_PROTOCOL)

print("size of pid2eng_flag", len(pid2eng_flag))

print("finished")