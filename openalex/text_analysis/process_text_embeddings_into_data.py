from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm
import sys
# utils.py lives one level up (openalex/), not in this text_analysis/ directory;
# add the parent directory to the path so the import resolves regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import META_DIRE

meta_dire = META_DIRE


train_title_text_flag = True

end_year_start = 2020
end_year_finish = 2025
period_length = 5
print("end_year_start", end_year_start)
print("end_year_finish", end_year_finish)
print("period_length", period_length)
print("train_title_text_flag",train_title_text_flag)


if train_title_text_flag:
    # for title
    embed_size = 128
    seed = 100
    epochs = 5
    min_count = 0

    window_size = 2
    
    parse_method = 'lemma'
    save_directory = os.path.join(meta_dire, 'text_embedding')
else:
    # for abstract
    embed_size = 128
    seed = 42
    window_size = 2
    epochs = 2
    min_count = 0
    parse_method = 'raw'
    save_directory = os.path.join(meta_dire, 'text_embedding_abs')

print("embed_size", embed_size)
print("seed", seed)
print("window_size", window_size)
print("epochs", epochs)
print("min_count", min_count)
print("parse_method",parse_method)

for embedding_year_before in tqdm(range(end_year_start, end_year_finish)):


    if period_length == -1:
        if train_title_text_flag == False:
            fild_load_dire = os.path.join(save_directory, 'parse_%s_word_embedding_before_%s_dim_%s_min_count_%s_epoch_%s_seed_%s.model'%(parse_method, embedding_year_before, embed_size, min_count, epochs, seed))
            file_save_dire = os.path.join(save_directory, 'token2embed_parse_%s_before_%s_dim_%s_min_count_%s_epoch_%s_seed_%s.pkl'%(parse_method, embedding_year_before, embed_size, min_count, epochs, seed))

        else:

            fild_load_dire = os.path.join(save_directory, 'word_embedding_before_%s_dim_%s_min_count_%s_epoch_%s_seed_%s.model'%(embedding_year_before, embed_size, min_count, epochs, seed))
            file_save_dire = os.path.join(save_directory, 'token2embed_before_%s_dim_%s_min_count_%s_epoch_%s_seed_%s.pkl'%(embedding_year_before, embed_size, min_count, epochs, seed))

    else:

        embedding_year_after = embedding_year_before - period_length - 1 # extra year


        if train_title_text_flag == False:
            fild_load_dire = os.path.join(save_directory, 'parse_%s_word_embedding_before_%s_after_%s_dim_%s_min_count_%s_epoch_%s_seed_%s.model'%(parse_method, embedding_year_before, embedding_year_after, embed_size, min_count, epochs, seed))
            file_save_dire = os.path.join(save_directory, 'token2embed_parse_%s_before_%s_after_%s_dim_%s_min_count_%s_epoch_%s_seed_%s.pkl'%(parse_method, embedding_year_before, embedding_year_after, embed_size, min_count, epochs, seed))
        
        else:
            fild_load_dire = os.path.join(save_directory, 'word_embedding_before_%s_after_%s_dim_%s_min_count_%s_epoch_%s_seed_%s.model'%(embedding_year_before, embedding_year_after, embed_size, min_count, epochs, seed))
            file_save_dire = os.path.join(save_directory, 'token2embed_before_%s_after_%s_dim_%s_min_count_%s_epoch_%s_seed_%s.pkl'%(embedding_year_before, embedding_year_after, embed_size, min_count, epochs, seed))
    if os.path.exists(file_save_dire) == True:
        continue
    model_embedding = Word2Vec.load(fild_load_dire)

    text_keys = set(model_embedding.wv.index_to_key)



    text2embedding = {text: model_embedding.wv[text] for text in tqdm(text_keys, desc = 'process embeddings', leave = False, position = 1)}
    with open(file_save_dire, 'wb') as f:
        pickle.dump(text2embedding, f, pickle.HIGHEST_PROTOCOL)

print("finished")