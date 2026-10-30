"""
Extract a SINGLE concept per paper from the OpenAlex snapshot.

Same concept logic as get_meta_data_preprint.py (level-0 concepts mapped to a
display name via CONCEPT_DIRE), but instead of keeping the full concept list per
paper this keeps exactly ONE concept per pid:

  * a paper's "valid" concepts are its level-0 concepts that resolve to a display
    name in Concepts_definition.xlsx;
  * the valid concept with the HIGHEST score is kept; if several share the top
    score, ONE of them is chosen at random;
  * if a paper has no valid concept, it is assigned 'Others'.

The RNG (used only for top-score ties) is seeded once at startup; reproducibility
across runs is NOT guaranteed (and not needed).

Outputs (to META_DIRE):
  pid2single_concept.pkl       pid  -> chosen concept display_name (or 'Others')
  concept2pid_list_single.pkl  concept display_name -> [pids assigned to it]
                               (each pid appears under exactly ONE concept)
"""

import json
import os
import gzip
import pickle
import random
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import pandas as pd
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)               # utils.py is in this same directory
from utils import OPENALEX_DIRE, META_DIRE, CONCEPT_DIRE

path_openalex = OPENALEX_DIRE
# save to the with-review directory (same pid selection as
# get_meta_data_preprint_review.py), so it is not mixed with the
# article/preprint-only meta files
save_dire = META_DIRE + '_with_review'
os.makedirs(save_dire, exist_ok=True)
process_count = 8

# seed once; the per-paper concept pick is then a plain (fast) random.choice.
# We do NOT care about reproducibility, so no per-pid re-seeding.
random.seed(42)

# work types kept (same as get_meta_data_preprint_review.py -- includes 'review')
KEEP_TYPES = {'article', 'preprint', 'review'}

print("save_dire", save_dire)
print("process_count", process_count)
print("KEEP_TYPES", KEEP_TYPES)

# level-0 concept id -> display name (the only concepts considered "valid")
df_concept = pd.read_excel(CONCEPT_DIRE)
df_concept = df_concept[df_concept['level'] == 0]
concept2name = dict(zip(df_concept['openalex_id'], df_concept['display_name']))


def get_single_concept(file_path):
    """File-level worker: return pid -> ONE concept display_name for this .gz file.

    For each valid paper, collect its level-0 (valid) concepts and keep the one
    with the HIGHEST score; if several share the top score, pick among them at
    random (global RNG, seeded once at startup). Papers with no valid concept are
    assigned 'Others'.
    """
    pid2single_concept_single = {}
    with gzip.open(file_path, 'rb') as f:
        for line in f:
            line = json.loads(line)

            id_ = line['id'][21:]

            if line and line['publication_year'] and line['authorships'] and line['type'] and (line['type'] in KEEP_TYPES):
                # collect the paper's valid (level-0) concepts as (name, score)
                valid_concepts = []
                for concept in line['concepts']:
                    if concept['level'] == 0:
                        concept_name = concept2name[concept['id'].lower()]
                        valid_concepts.append((concept_name, concept['score']))

                if valid_concepts:
                    # highest score wins; random tie-break among equal top scores.
                    max_score = max(score for _, score in valid_concepts)
                    top = [name for name, score in valid_concepts if score == max_score]
                    pid2single_concept_single[id_] = top[0] if len(top) == 1 else random.choice(top)
                else:
                    pid2single_concept_single[id_] = 'Others'

    return pid2single_concept_single


path_work = os.path.join(path_openalex, 'openalex-snapshot/data/works/')
dirs = sorted(os.listdir(path_work))

# ===============================
# Collect all valid gz files (file-level parallelism, like get_meta_data_preprint.py)
# ===============================
gz_files = []
for d in dirs:
    if '_updated_date' not in d and 'updated_date' in d:
        dpath = os.path.join(path_work, d)
        for fname in sorted(os.listdir(dpath)):
            if fname.endswith('.gz') and '_part' not in fname:
                gz_files.append(os.path.join(dpath, fname))
print("Total gz files:", len(gz_files))


if os.path.exists(os.path.join(save_dire, 'pid2single_concept.pkl')) == False:
    pid2single_concept = {}

    if process_count > 1:
        with ProcessPoolExecutor(max_workers=process_count) as executor:
            with tqdm(total=len(gz_files), desc='process concept') as pbar:
                for pid2single_concept_single in executor.map(get_single_concept, gz_files):
                    pbar.update(1)
                    pid2single_concept.update(pid2single_concept_single)
    else:
        for file_path in tqdm(gz_files, desc='process concept'):
            pid2single_concept_single = get_single_concept(file_path)
            pid2single_concept.update(pid2single_concept_single)

    # invert: concept -> list of pids (each pid under exactly one concept)
    concept2pid_list_single = defaultdict(list)
    for pid, concept in tqdm(pid2single_concept.items(), desc='invert concept2pid'):
        concept2pid_list_single[concept].append(pid)
    concept2pid_list_single = dict(concept2pid_list_single)

    with open(os.path.join(save_dire, 'pid2single_concept.pkl'), 'wb') as f:
        pickle.dump(pid2single_concept, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_dire, 'concept2pid_list_single.pkl'), 'wb') as f:
        pickle.dump(concept2pid_list_single, f, pickle.HIGHEST_PROTOCOL)

    print("size of pid2single_concept", len(pid2single_concept))
    print("number of concepts", len(concept2pid_list_single))
else:
    print("pid2single_concept.pkl already exists, nothing to do.")

print("finished")
