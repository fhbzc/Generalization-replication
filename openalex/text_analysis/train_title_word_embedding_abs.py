import os
import pickle
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import random
import time
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import sys
# utils.py lives one level up (openalex/), not in this text_analysis/ directory;
# add the parent directory to the path so the import resolves regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import META_DIRE


def get_english_stopwords():
    try:
        return set(stopwords.words('english'))
    except LookupError:
        print("NLTK stopwords corpus not found, downloading ...")
        nltk.download('stopwords')
        return set(stopwords.words('english'))

stop_words_set = get_english_stopwords()
print("V1")
meta_dire = META_DIRE
print("meta_dire",meta_dire)

embed_size = 128
seed = 42
window_size = 2
epochs = 2
min_count = 0
parse_method = 'raw'
end_year_start = 2020
end_year_finish = 2025
period_length = 5
print("embed_size", embed_size)
print("seed", seed)
print("window_size", window_size)
print("epochs", epochs)
print("min_count", min_count)
print("parse_method", parse_method)


print("end_year_start", end_year_start)
print("end_year_finish", end_year_finish)
print("period_length", period_length)


year2token_list = {}


embedding_load_dire = os.path.join(meta_dire, 'abstract_yearly')


save_directory = os.path.join(meta_dire, 'text_embedding_abs')

os.makedirs(save_directory, exist_ok = True)

for embedding_year_before in tqdm(range(end_year_start, end_year_finish)):


    if period_length > 0:
        embedding_year_after = embedding_year_before - period_length - 1 # extra year

        save_file_name = os.path.join(save_directory, 'parse_%s_word_embedding_before_%s_after_%s_dim_%s_min_count_%s_epoch_%s_seed_%s.model'%(parse_method, embedding_year_before, embedding_year_after, embed_size, min_count, epochs, seed))
    else:
        assert False

    if os.path.exists(save_file_name) == True:
        continue

    required_year_set = set([year for year in range(embedding_year_after, embedding_year_before)])
    contained_year_set = set(year2token_list)
    all_year_set = required_year_set | contained_year_set
    for year in tqdm(all_year_set, desc = 'process yearly data s1', position = 1, leave = False):
        if year in required_year_set and year not in contained_year_set:
            with open(os.path.join(embedding_load_dire, 'pid2abstract_eng_token_list_%s_y_%s.pkl'%(parse_method, year)), 'rb') as f:
                pid2token_list = pickle.load(f)

            token_list_full = []
            for token_list in tqdm(pid2token_list.values(), desc = 'process yearly data s2', position = 2, leave = False):
                valid_token_list = [token for token in token_list if token not in stop_words_set and len(token) > 1]
                if len(valid_token_list) > 1:
                    token_list_full.append(valid_token_list)
            year2token_list[year] = token_list_full

        if year in contained_year_set and year not in required_year_set:
            del year2token_list[year]


    list_of_token_list = []

    for year in tqdm(required_year_set, desc = 'process yearly token merge', position = 1, leave = False):
        list_of_token_list.extend(year2token_list[year])




    model = Word2Vec(sentences=list_of_token_list, 
                    vector_size=embed_size, 
                    window=window_size, 
                    min_count=min_count, 
                    sg=1, 
                    seed = seed,
                    workers=16,
                    epochs = epochs)

    model.save(save_file_name)


print("finished")