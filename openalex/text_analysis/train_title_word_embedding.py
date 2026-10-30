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

meta_dire = META_DIRE
print("meta_dire",meta_dire)

year_before = 2025



period_length = 5



print("period_length", period_length)


embed_size = 128
seed = 100
epochs = 5
min_count = 0

window_size = 2

parse_method = 'lemma'
end_year_start = 1951
end_year_finish = 2025
print("embed_size", embed_size)
print("seed", seed)
print("window_size", window_size)
print("epochs", epochs)
print("min_count", min_count)
print("year_before", year_before)
print("parse_method", parse_method)


print("end_year_start", end_year_start)
print("end_year_finish", end_year_finish)
with open(os.path.join(meta_dire, 'pid2eng_token_list_%s_before_%s.pkl'%(parse_method, year_before)), 'rb') as f:
    pid2token_list = pickle.load(f)
print("load pid2token_list finished")

with open(os.path.join(meta_dire, 'pid2pub_year.pkl'), 'rb') as f:
    pid2pub_year = pickle.load(f)
print("load pid2pub_year finished")


for embedding_year_before in tqdm(range(end_year_start, end_year_finish)):

    save_directory = os.path.join(meta_dire, 'text_embedding')
    os.makedirs(save_directory, exist_ok = True)

    if period_length > 0:
        embedding_year_after = embedding_year_before - period_length - 1 # extra year

        save_file_name = os.path.join(save_directory, 'word_embedding_before_%s_after_%s_dim_%s_min_count_%s_epoch_%s_seed_%s.model'%(embedding_year_before, embedding_year_after, embed_size, min_count, epochs, seed))

    else:
        embedding_year_after = -1
        save_file_name = os.path.join(save_directory, 'word_embedding_before_%s_dim_%s_min_count_%s_epoch_%s_seed_%s.model'%(embedding_year_before, embed_size, min_count, epochs, seed))


    if os.path.exists(save_file_name) == True:
        continue


    list_of_token_list = []
    for pid, token_list in tqdm(pid2token_list.items(), desc = 'process tokens', leave = False, position = 1):
        pub_year = pid2pub_year[pid]
        if pub_year < embedding_year_before:


            if pub_year < embedding_year_after:
                continue

            valid_token_list = [token for token in token_list if token not in stop_words_set and len(token) > 1]
            if len(valid_token_list) > 1:
                list_of_token_list.append(valid_token_list)


    print("train %s"%embedding_year_before)

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