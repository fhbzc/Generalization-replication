import json
import os
import gzip
from tqdm import tqdm
import pickle
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict, OrderedDict
import numpy as np
import pandas as pd
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)             # utils.py is in the parent directory
from utils import META_DIRE, AI_USAGE_DIRE
import random
from datetime import datetime, timedelta
import math
print("V1")

# ── Outcome selection ─────────────────────────────────────────────────────────
# "g_ratio" | "cross_domain_ratio" | "refer_dist" | "title_dist"
dataset = "title_dist"
print("dataset:", dataset)

def generate_random_date(start_date, end_date):
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    random_date = start_date + timedelta(days=random_days)
    date_string = random_date.strftime("%Y-%m-%d")
    return date_string


def get_relative_time(date_str, adopted_time_str, date_format="%Y-%m-%d"):
    """
    Compute floor((date - adopted_time).days / 30)
    """
    date = datetime.strptime(date_str, date_format)
    adopted_time = datetime.strptime(adopted_time_str, date_format)
    diff_days = (date - adopted_time).days
    return math.floor(diff_days / 30)


load_dire = AI_USAGE_DIRE  # external dataset: {arxiv,biorxiv,ssrn}_alpha_share.parquet (see utils.py)
save_dire = os.path.join(load_dire, 'processed')
os.makedirs(save_dire, exist_ok=True)

dataset_name_list = ['arxiv', 'biorxiv', 'ssrn']

# for dataset_name in dataset_name_list:
#     if dataset_name == 'arxiv':
#         df = pd.read_parquet(os.path.join(load_dire, f"{dataset_name}_alpha_share.parquet"))
#         doi2ai_shared = dict(zip(df['id'], df['estimated_alpha_new']))
#     else:
#         df = pd.read_parquet(os.path.join(load_dire, f"{dataset_name}_alpha_share.parquet"))
#         doi2ai_shared = dict(zip(df['doi'], df['estimated_alpha_new']))

#     print("dataset_name", dataset_name, len(doi2ai_shared))
# assert False

ai_threshold_list  = [0.1]
RANDOM_SEED        = 42
print("dataset_name_list", dataset_name_list)
print("ai_threshold_list", ai_threshold_list)
print("RANDOM_SEED", RANDOM_SEED)

year_offset         = -1   # -1 for no offset
reference_threshold = 1
max_year            = 2026
print("year_offset", year_offset)
print("reference_threshold", reference_threshold)
print("max_year", max_year)

# ── Load shared data (once) ────────────────────────────────────────────────────
df_gen = pd.read_csv(
    os.path.join(META_DIRE, 'generalization_up',
                 f'gen_V3_up_y{year_offset}_'
                 f'rt_{reference_threshold}_'
                 f'ybefore_{max_year}.csv'),
    usecols=['pid', f'up_refer_count_y{year_offset}', f'disp_g_count_y{year_offset}']
)
pid2up_refer_count = dict(zip(df_gen['pid'], df_gen[f'up_refer_count_y{year_offset}']))
pid2g_count        = dict(zip(df_gen['pid'], df_gen[f'disp_g_count_y{year_offset}']))

with open(os.path.join(META_DIRE, 'pid2pub_date.pkl'), 'rb') as f:
    pid2pub_date = pickle.load(f)

with open(os.path.join(META_DIRE, 'pid2author_list.pkl'), 'rb') as f:
    pid2author_list = pickle.load(f)

# author2pid_list_before / author2augmented_pid_set are independent of dataset_name
author2pid_list_before  = defaultdict(list)
author2augmented_pid_set = defaultdict(set)
for pid, author_list in tqdm(pid2author_list.items(), desc='build author2pid_list_before'):
    if 2018 <= int(pid2pub_date[pid][:4]) < 2022:
        for author in author_list:
            author2pid_list_before[author].append((pid, pid2pub_date[pid], 0))
            author2augmented_pid_set[author].add(pid)

# ── Pre-load arxiv pid2doi (needed inside the loop) ───────────────────────────
with open(os.path.join(META_DIRE, 'pid2arxiv_id_list.pkl'), 'rb') as f:
    pid2arxiv_id_list = pickle.load(f)

with open(os.path.join(META_DIRE, 'pid2doi.pkl'), 'rb') as f:
    pid2doi_all = pickle.load(f)

# ── Precompute pid2metric (outcome-dependent, dataset_name-independent) ────────
# Will be populated once; pid2metric is recomputed per dataset_name below
# because all_panel_pids includes doi2pid_valid which differs per dataset_name.
# The heavy shared resources for non-g_ratio outcomes are loaded here once.

if dataset == 'cross_domain_ratio':
    with open(os.path.join(META_DIRE, 'citing_pid2cited_pids_list.pkl'), 'rb') as f:
        citing_pid2cited_pids = pickle.load(f)
    print(f"citing_pid2cited_pids size: {len(citing_pid2cited_pids)}")
    with open(os.path.join(META_DIRE, 'pid2root_concepts_level_0.pkl'), 'rb') as f:
        _raw = pickle.load(f)
    pid2root_concept = {pid: next(iter(c)) for pid, c in _raw.items()
                        if c is not None and len(c) == 1}
    del _raw
    print(f"pids with single root concept: {len(pid2root_concept)}")

elif dataset in ('refer_dist', 'title_dist'):
    _period_length = 5
    _embed_size    = 128

    print("\nLoading citing_pid2cited_pids ...")
    with open(os.path.join(META_DIRE, 'citing_pid2cited_pids_list.pkl'), 'rb') as f:
        _citing2cited = pickle.load(f)
    print(f"  _citing2cited size: {len(_citing2cited)}")

    if dataset == 'refer_dist':
        _embed_seed  = 42
        _token_pkl   = os.path.join(META_DIRE, 'citing_pid2cited_locations_list_before_2025.pkl')
        def _embed_path(yr):
            yr_after = yr - _period_length - 1
            return os.path.join(META_DIRE, 'reference_atypicality',
                                f'location2embedding_before_{yr}_after_{yr_after}_dim_{_embed_size}'
                                f'_min_count_0_epoch_5_seed_{_embed_seed}.model')
    else:  # title_dist
        _embed_seed   = 100
        _parse_method = 'lemma'
        _year_before  = 2025
        _token_pkl    = os.path.join(META_DIRE,
                                     f'pid2eng_token_list_{_parse_method}_before_{_year_before}.pkl')
        def _embed_path(yr):
            yr_after = yr - _period_length - 1
            return os.path.join(META_DIRE, 'text_embedding',
                                f'token2embed_before_{yr}_after_{yr_after}_dim_{_embed_size}'
                                f'_min_count_0_epoch_5_seed_{_embed_seed}.pkl')

    print(f"\nLoading pid2token_list for {dataset} ...")
    with open(_token_pkl, 'rb') as f:
        _pid2token_list = pickle.load(f)
    print(f"  pid2token_list size: {len(_pid2token_list)}")


# ── Main loop over dataset_name × ai_threshold ────────────────────────────────
for dataset_name in tqdm(dataset_name_list, desc='dataset_name', position=0):
    print(f"\n{'='*60}")
    print(f"dataset_name: {dataset_name}")

    # ── Build doi2ai_shared ────────────────────────────────────────────────────
    if dataset_name == 'arxiv':
        df = pd.read_parquet(os.path.join(load_dire, f"{dataset_name}_alpha_share.parquet"))
        doi2ai_shared = dict(zip(df['id'], df['estimated_alpha_new']))
    else:
        df = pd.read_parquet(os.path.join(load_dire, f"{dataset_name}_alpha_share.parquet"))
        doi2ai_shared = dict(zip(df['doi'], df['estimated_alpha_new']))

    valid_doi_set = set(doi2ai_shared)

    # ── Build doi2pid_valid ────────────────────────────────────────────────────
    doi2pid_list = defaultdict(list)

    if dataset_name == 'arxiv':
        for pid, doi_list in tqdm(pid2arxiv_id_list.items(), desc='arxiv doi2pid', position=1, leave=False):
            pub_year = int(pid2pub_date[pid][:4])
            if pub_year < 2022 or pub_year >= 2025:
                continue
            for doi in doi_list:
                doi2pid_list[doi].append(pid)

        doi2pid_valid = {}
        for doi, pid_list in doi2pid_list.items():
            if doi not in valid_doi_set:
                continue
            l = len(pid_list)
            assert l > 0
            if l > 1:
                pid_order_seq = sorted(
                    [(pid, pid2up_refer_count.get(pid, 0)) for pid in pid_list],
                    key=lambda x: -x[1]
                )
                doi2pid_valid[doi] = pid_order_seq[0][0]
            else:
                doi2pid_valid[doi] = pid_list[0]
    else:
        for pid, doi in tqdm(pid2doi_all.items(), desc=f'{dataset_name} doi2pid', position=1, leave=False):
            pub_year = int(pid2pub_date[pid][:4])
            if pub_year < 2022 or pub_year >= 2025:
                continue
            doi2pid_list[doi].append(pid)

        doi2pid_valid = {
            doi: doi2pid_list[doi][0]
            for doi in tqdm(sorted(valid_doi_set), position=1, leave=False)
            if len(doi2pid_list.get(doi, [])) == 1
        }

    print(f"original paper count: {len(doi2ai_shared)}")
    print(f"identify pid paper count: {len(doi2pid_valid)}")

    # ── Build author2pid_sequence ──────────────────────────────────────────────
    author2pid_sequence = defaultdict(list)
    for doi, pid in tqdm(doi2pid_valid.items(), desc='build author2pid_sequence', position=1, leave=False):
        for author in pid2author_list.get(pid, []):
            author2pid_sequence[author].append((pid, pid2pub_date[pid], doi2ai_shared[doi]))

    # ── Compute pid2metric for this dataset_name's panel pids ─────────────────
    pid2metric = {}

    if dataset != 'g_ratio':
        all_panel_pids = (
            set(doi2pid_valid.values()) |
            {pid for pid_seq in author2pid_list_before.values() for pid, _, _ in pid_seq}
        )
        print(f"total panel pids: {len(all_panel_pids)}")

        if dataset == 'cross_domain_ratio':
            for p in tqdm(all_panel_pids, desc='cross_domain_ratio', position=1, leave=False):
                p_root = pid2root_concept.get(p)
                if p_root is None:
                    continue
                eligible = [r for r in citing_pid2cited_pids.get(p, [])
                            if r in pid2root_concept]
                if not eligible:
                    continue
                cross = sum(1 for r in eligible if pid2root_concept[r] != p_root)
                pid2metric[p] = cross / len(eligible)

        elif dataset in ('refer_dist', 'title_dist'):
            year2pids_panel = defaultdict(list)
            for pid in all_panel_pids:
                yr = int(pid2pub_date[pid][:4]) if pid in pid2pub_date else None
                if yr is not None:
                    year2pids_panel[yr].append(pid)

            def _compute_center(pid, embed):
                tokens = _pid2token_list.get(pid)
                if not tokens:
                    return None
                vecs = [embed[t] for t in tokens if t in embed]
                if not vecs:
                    return None
                return np.mean(np.array(vecs, dtype=np.float32), axis=0)

            for pub_year in tqdm(sorted(year2pids_panel.keys()), desc=f'computing {dataset}', position=1, leave=False):
                epath = _embed_path(pub_year)
                if not os.path.exists(epath):
                    assert False, pub_year
                    continue
                with open(epath, 'rb') as _ef:
                    _embed = pickle.load(_ef)

                # Collect all pids needed this year: focal + their references
                focal_pids  = year2pids_panel[pub_year]
                needed_pids = set(focal_pids)
                for pid in focal_pids:
                    needed_pids.update(_citing2cited.get(pid, []))

                # Pre-compute all centers at once
                pid2center_yr = {}
                for pid in needed_pids:
                    c = _compute_center(pid, _embed)
                    if c is not None:
                        pid2center_yr[pid] = c

                # For each focal paper, batch-compute mean cosine distance to references
                for pid in tqdm(focal_pids, desc=f'  yr={pub_year} dist', position=2, leave=False):
                    if pid not in pid2center_yr:
                        continue
                    p_center = pid2center_yr[pid]
                    refs = _citing2cited.get(pid, [])
                    ref_centers = np.array(
                        [pid2center_yr[r] for r in refs if r in pid2center_yr],
                        dtype=np.float32
                    )
                    if len(ref_centers) == 0:
                        continue
                    # Normalise
                    p_norm   = p_center / (np.linalg.norm(p_center) + 1e-10)
                    r_norms  = ref_centers / (np.linalg.norm(ref_centers, axis=1, keepdims=True) + 1e-10)
                    # All cosine distances in one matrix multiply
                    mean_dist = float(np.mean(1.0 - r_norms @ p_norm))
                    pid2metric[pid] = mean_dist

        print(f"pids with {dataset} computed: {len(pid2metric)}")

    # ── Loop over ai_threshold ─────────────────────────────────────────────────
    for ai_threshold in tqdm(ai_threshold_list, desc='ai_threshold', position=1, leave=False):
        print(f"\n  ai_threshold: {ai_threshold}")
        random.seed(RANDOM_SEED)

        df_res = []
        for author, pid_sequence in tqdm(author2pid_sequence.items(), desc='build panel', position=2, leave=False):
            if len(author2pid_list_before[author]) < 4:
                continue
            pid_sequence = list(pid_sequence)  # copy to avoid mutation across thresholds
            pid_sequence.extend(author2pid_list_before[author])
            pid_sequence = sorted(pid_sequence, key=lambda x: x[1])

            adopted_time = None
            has_adopted  = False
            for pid, date, ai_share in pid_sequence:
                if ai_share >= ai_threshold:
                    adopted_time = date
                    has_adopted  = True
                    break
            if not has_adopted:
                adopted_time = generate_random_date(datetime(2023, 1, 1), datetime(2024, 6, 30))

            if dataset == 'g_ratio':
                pid_sequence = [item for item in pid_sequence if pid2up_refer_count.get(item[0], 0) > 0]
            else:
                pid_sequence = [item for item in pid_sequence
                                if pid2up_refer_count.get(item[0], 0) > 0 and item[0] in pid2metric]

            i = 0
            adoption_index = None
            for pid, date, ai_share in pid_sequence:
                if date >= adopted_time:
                    adoption_index = i
                    break
                i += 1
            if adoption_index is None:
                continue

            before_adoption_paper_count = sum(1 for _, date, _ in pid_sequence if date < adopted_time)
            after_adoption_paper_count  = sum(1 for _, date, _ in pid_sequence if date >= adopted_time)

            i = 0
            aug_pid_set = author2augmented_pid_set[author]
            for pid, date, ai_share in pid_sequence:
                relative_month = get_relative_time(date, adopted_time)
                od = OrderedDict()
                od['author']                      = author
                od['before_adoption_paper_count'] = before_adoption_paper_count
                od['after_adoption_paper_count']  = after_adoption_paper_count
                od['augmented']                   = pid in aug_pid_set
                od['relative_month']              = relative_month
                od['relative_index']              = i - adoption_index
                od['treatment_flag']              = has_adopted
                od['pid']                         = pid
                od['pub_date']                    = date
                od['pub_year']                    = int(date[:4])
                od['pub_month']                   = date[:7]
                od['ai_share']                    = ai_share
                up_refer_count                    = pid2up_refer_count[pid]
                od['up_refer_count']              = up_refer_count
                if dataset == 'g_ratio':
                    g_count       = pid2g_count.get(pid, 0)
                    od['g_count'] = g_count
                    od['g_ratio'] = g_count / up_refer_count if g_count > 0 else 0
                else:
                    od[dataset] = pid2metric.get(pid)
                df_res.append(od)
                i += 1

        df_res = pd.DataFrame(df_res)
        ai_threshold_str = str(ai_threshold).replace('.', 'p')
        out_path = os.path.join(
            save_dire,
            f'{dataset_name}_ait{ai_threshold_str}_seed{RANDOM_SEED}_{dataset}_panel_data.csv'
        )
        df_res.to_csv(out_path, index=False)
        print(f"  size of df_res: {len(df_res)}")
        print(f"  saved to: {out_path}")

print("\nfinished")
