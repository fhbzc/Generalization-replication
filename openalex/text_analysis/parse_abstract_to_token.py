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
import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex
import sys
# utils.py lives one level up (openalex/), not in this text_analysis/ directory;
# add the parent directory to the path so the import resolves regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import META_DIRE

meta_dire = META_DIRE
print("meta_dire",meta_dire)
save_dire = os.path.join(meta_dire, 'abstract_yearly')
os.makedirs(save_dire, exist_ok = True)

process_count = 1
year_before = 2025
year_after = 2020


method_string = 'raw' # [lemma, raw]
print("meta_dire",meta_dire)
print("process_count",process_count)
print("year_before",year_before)
print("year_after",year_after)
print("method_string",method_string)


punctuations = string.punctuation.replace('-', '')

trans_table_obj = str.maketrans(punctuations, ' ' * len(punctuations))
def process_title_m1(pid_title_list, trans_table):
    # replace punctuations, no modifications
    pid2token_list = {}

    for pid, title_P in tqdm(pid_title_list, position = 2, leave = False, desc = 'process tokens'):


        tokens = title_P.translate(trans_table).lower().split()

        if len(tokens) > 0:
            pid2token_list[pid] = tokens

    return pid2token_list


with open(os.path.join(meta_dire, 'pid2pub_year.pkl'), 'rb') as f:
    pid2pub_year = pickle.load(f)
print("load pid2pub_year finished")

pub_year2pid_list = defaultdict(list)
for pid, pub_year in tqdm(pid2pub_year.items(), desc = 'process pub_year'):

    pub_year2pid_list[pub_year].append(pid)


def load_spacy_model(model_name="en_core_web_sm"):
    try:
        return spacy.load(model_name)
    except OSError:
        print("spaCy model %s not found, downloading ..." % model_name)
        from spacy.cli import download
        download(model_name)
        return spacy.load(model_name)

nlp = load_spacy_model("en_core_web_sm")
infixes = nlp.Defaults.infixes
infixes = [x for x in infixes if '-' not in x]  # Remove patterns with '-'
infix_regex = compile_infix_regex(infixes)

# Re-assign tokenizer
nlp.tokenizer = Tokenizer(nlp.vocab, 
                          rules=nlp.Defaults.tokenizer_exceptions,
                          prefix_search=nlp.tokenizer.prefix_search,
                          suffix_search=nlp.tokenizer.suffix_search,
                          infix_finditer=infix_regex.finditer,
                          token_match=nlp.tokenizer.token_match)


def process_title_m2(pid_title_list, trans_table):
    # replace punctuations, lemmatize words
    pid2token_list = {}

    for pid, title_P in tqdm(pid_title_list, position = 1, leave = False, desc = 'process tokens'):


        title_P = title_P.translate(trans_table).lower()
        tokens = []
        for token in nlp(title_P):
            P_token = token.lemma_.strip()
            if len(P_token) > 0:
                tokens.append(P_token)


        if len(tokens) > 0:
            pid2token_list[pid] = tokens

    return pid2token_list



title_no_html_directory = os.path.join(meta_dire, 'pid2abstract_no_html_method_lemma_before_%s_after_%s.pkl'%(year_before, year_after))


with open(title_no_html_directory, 'rb') as f:
    pid2title_P = pickle.load(f)

if method_string == 'lemma':
    process_title_m = process_title_m2
elif method_string == 'raw':
    process_title_m = process_title_m1
else:
    assert False

pid_set_main = set(pid2title_P)

for target_year in tqdm(range(year_after, year_before), position = 0):

    pid_list = list(set(pub_year2pid_list[target_year]) & pid_set_main)

    save_finish_dire = os.path.join(save_dire, '%s_abs_%s_finished.pkl'%(target_year, method_string))
    if os.path.exists(save_finish_dire):
        continue


    if process_count > 1:

        batched_pid_list = [[] for _ in range(process_count)]
        for index, pid in enumerate(pid_list):
            batch_index = index % process_count
            batched_pid_list[batch_index].append((pid, pid2title_P[pid]))
        pid2token_list = {}
        with ProcessPoolExecutor(max_workers=process_count) as executor:
            with tqdm(total=len(batched_pid_list), desc = 'process tokens main', position = 1, leave = False) as pbar:
                for pid2token_list_sig in executor.map(process_title_m, batched_pid_list, [trans_table_obj for _ in range(process_count)]):
                    pid2token_list.update(pid2token_list_sig)
                    pbar.update(1)

    else:
        pid_title_list = [(pid, pid2title_P[pid]) for pid in pid_list]
        pid2token_list = process_title_m(pid_title_list, trans_table_obj)


    with open(os.path.join(save_dire, 'pid2abstract_eng_token_list_%s_y_%s.pkl'%(method_string, target_year)), 'wb') as f:
        pickle.dump(pid2token_list, f, pickle.HIGHEST_PROTOCOL)


    with open(save_finish_dire, 'wb') as f:
        pickle.dump([], f, pickle.HIGHEST_PROTOCOL)

print("finished")