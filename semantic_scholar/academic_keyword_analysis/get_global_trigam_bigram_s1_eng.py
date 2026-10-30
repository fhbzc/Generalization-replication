import json
import os
import sys
import gzip
from tqdm import tqdm
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict, OrderedDict
import pandas as pd
from pathlib import Path
import re
from nltk.corpus import stopwords
import numpy as np
import networkx as nx
import spacy
# from langdetect import detect, DetectorFactory
import langid
from multiprocessing import Pool
import fasttext

# utils.py lives in the parent (semantic_scholar) directory; make it importable regardless of CWD
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
from utils import META_DIRE, LANG_DETECT_MODEL



def detect_chunk_worker(cid_list):
    out_cid = []
    out_text = []
    for cid in cid_list:
        title = cid2title[cid]
        abstract = cid2abs.get(cid, '')
        text = f"{title} {abstract}".lower().translate(_TRANS_TABLE).strip()

        detect = is_english_fast_text(text)
        if detect:
            out_cid.append(cid)
            out_text.append(text)
    return out_cid, out_text
model = fasttext.load_model(LANG_DETECT_MODEL)

def is_english_fast_text(text):

    if len(text) <= 2:
        return False

    pred = model.predict(text)[0][0]
    return pred == '__label__en'

def is_english_langid(text):
    if len(text) <= 2:
        return False, None, None
    lang, prob = langid.classify(text[:1000])
    return lang == 'en'

det_process_count = 8
n_cpu = 16
batch_size = 1000
print("det_process_count",det_process_count)

print("process count", n_cpu)
print("batch_size", batch_size)
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

print("V1")
# stop_words_set = set(stopwords.words('english'))
meta_dire = META_DIRE
load_dire = os.path.join(meta_dire, 'cid2abs_year')
save_dire = os.path.join(meta_dire, 'global_tribigram_eng')
os.makedirs(save_dire, exist_ok = True)
number_str = '0123456789'

start_year = 2025
end_year = 2026 # 2026

print("start_year",start_year)
print("end_year",end_year)

assert len(number_str) == 10

# 中文 + 英文常见标点
_TRANS_TABLE = str.maketrans({
    '[': ' ',
    ']': ' ',
    '(': ' ',
    ')': ' ',
    '\n': ' ',
    '\r': ' ',
})

with open(os.path.join(meta_dire,'cid2title.pkl'),'rb') as f:
    cid2title = pickle.load(f)

print("load cid2title finished")

with open(os.path.join(meta_dire,'cid2pub_year.pkl'),'rb') as f:
    cid2pub_year = pickle.load(f)
print("load cid2pub_year finished")

pub_year2cid_list = defaultdict(list)

for cid, pub_year in tqdm(cid2pub_year.items()):

    pub_year2cid_list[pub_year].append(cid)

for year in tqdm(range(start_year, end_year)):
    finish_dire = os.path.join(save_dire,f'finished_y_{year}.pkl')
    if os.path.exists(finish_dire):
        continue
    with open(os.path.join(load_dire,f'cid2pub_date_y_{year}.pkl'),'rb') as f:
        cid2abs = pickle.load(f)


    token2fre_single = defaultdict(int)
    token2fre_bigram = defaultdict(int)
    token2fre_trigram = defaultdict(int)
    
    cid_list = pub_year2cid_list[year]
    texts = []
    valid_cids = []

    

    if det_process_count > 1:

        cid_split = [[] for _ in range(det_process_count)]
        for i, cid in enumerate(cid_list):
            cid_split[i%det_process_count].append(cid)

        with Pool(processes=det_process_count) as pool:
            for out_cid_list, out_text_list in pool.map(detect_chunk_worker, cid_split):
                valid_cids.extend(out_cid_list)
                texts.extend(out_text_list)
    else:
        valid_cids, texts = detect_chunk_worker(cid_list)

    cid2parsed_text = {}
    for cid, doc in tqdm(zip(
        valid_cids,
        nlp.pipe(
            texts,
            batch_size=batch_size,
            n_process=n_cpu   # 👈 按机器核数调
        )),
        total=len(cid_list),
        desc="spaCy lemmatizing", leave = False, position = 1):



        list_of_token_list = []
        new_list = []
        for token in doc:
            if (
                token.is_space
                or token.is_punct
                or token.is_stop
                or token.lemma_ == "-PRON-"
                or (len(token.text) == 1 and token.text not in number_str)
            ):
                if len(new_list) > 0:
                    list_of_token_list.append(new_list)
                new_list = []
            else:
                new_list.append(token.lemma_)

        if len(new_list) > 0:
            list_of_token_list.append(new_list)

        cid2parsed_text[cid] = list_of_token_list
        for token_list in list_of_token_list:
            l = len(token_list)
            for focal_token_index in range(l):
                focal_token = token_list[focal_token_index]
                token_item = focal_token
                token2fre_single[token_item] += 1

                if focal_token_index <= l -2:
                    next_token = token_list[focal_token_index + 1]
                    token_item = (focal_token, next_token)
                    token2fre_bigram[token_item] += 1
                    if focal_token_index <= l - 3:
                        last_token = token_list[focal_token_index + 2]
                        token_item = ((focal_token, next_token), (next_token, last_token))
                        token2fre_trigram[token_item] += 1

    with open(os.path.join(save_dire,f'single_token2fre_y_{year}.pkl'),'wb') as f:
        pickle.dump(token2fre_single, f, pickle.HIGHEST_PROTOCOL)


    with open(os.path.join(save_dire,f'bigram_token2fre_y_{year}.pkl'),'wb') as f:
        pickle.dump(token2fre_bigram, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_dire,f'trigram_token2fre_y_{year}.pkl'),'wb') as f:
        pickle.dump(token2fre_trigram, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_dire,f'cid2parsed_token_list_{year}.pkl'),'wb') as f:
        pickle.dump(cid2parsed_text, f, pickle.HIGHEST_PROTOCOL)


    with open(finish_dire,'wb') as f:
        pickle.dump([], f, pickle.HIGHEST_PROTOCOL)




    # assert False
print("finished")
