from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)             # utils.py is in the parent directory
from utils import META_DIRE

meta_dire = META_DIRE

embed_size = 128
seed = 42
window_size = 100
epochs = 5
min_count = 0
print("embed_size", embed_size)
print("seed", seed)
print("window_size", window_size)
print("epochs", epochs)
print("min_count", min_count)

end_year_start = 1950
end_year_finish = 2025
period_length = 5
print("end_year_start", end_year_start)
print("end_year_finish", end_year_finish)
print("period_length", period_length)

save_directory = os.path.join(meta_dire, 'reference_atypicality')


for embedding_year_before in tqdm(range(end_year_start, end_year_finish)):


    if period_length == -1:


            fild_load_dire = os.path.join(save_directory, 'location_embedding_before_%s_dim_%s_min_count_%s_epoch_%s_seed_%s.model'%(embedding_year_before, embed_size, min_count, epochs, seed))
            file_save_dire = os.path.join(save_directory, 'location2embedding_before_%s_dim_%s_min_count_%s_epoch_%s_seed_%s.model'%(embedding_year_before, embed_size, min_count, epochs, seed))

    else:

        embedding_year_after = embedding_year_before - period_length - 1 # extra year


        fild_load_dire = os.path.join(save_directory, 'location_embedding_before_%s_after_%s_dim_%s_min_count_%s_epoch_%s_seed_%s.model'%(embedding_year_before, embedding_year_after, embed_size, min_count, epochs, seed))
        file_save_dire = os.path.join(save_directory, 'location2embedding_before_%s_after_%s_dim_%s_min_count_%s_epoch_%s_seed_%s.model'%(embedding_year_before, embedding_year_after, embed_size, min_count, epochs, seed))
    if os.path.exists(file_save_dire) == True:
        continue
    model_embedding = Word2Vec.load(fild_load_dire)

    text_keys = set(model_embedding.wv.index_to_key)



    text2embedding = {text: model_embedding.wv[text] for text in tqdm(text_keys, desc = 'process embeddings', leave = False, position = 1)}
    with open(file_save_dire, 'wb') as f:
        pickle.dump(text2embedding, f, pickle.HIGHEST_PROTOCOL)

print("finished")