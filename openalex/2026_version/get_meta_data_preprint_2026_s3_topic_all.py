import json
import os
import gzip
import pickle
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)             # utils.py is in the parent directory
from utils import OPENALEX_DIRE_2026, META_DIRE_2026


print("OPENALEX_DIRE_2026", OPENALEX_DIRE_2026)
print("META_DIRE_2026", META_DIRE_2026)

path_openalex = OPENALEX_DIRE_2026
save_dire = META_DIRE_2026   # 2026 metadata kept separate from the main META_DIRE
os.makedirs(save_dire, exist_ok=True)

process_count = 16
print("save_dire", save_dire)
print("process_count", process_count)

# =========================
# 收集所有 file-level task
# =========================
path_work = os.path.join(path_openalex, 'data/works/')
dirs = sorted(os.listdir(path_work))

all_files = []
for dir_ in tqdm(dirs):
    if '_updated_date' not in dir_ and 'updated_date' in dir_:
        full_dir = os.path.join(path_work, dir_)
        for file in os.listdir(full_dir):
            if file.endswith('.gz') and '_part' not in file:
                all_files.append(os.path.join(full_dir, file))

print("Total gz files:", len(all_files))


def _strip_prefix(openalex_url):
    # 去掉 'https://openalex.org/' 前缀，和 pid 处理一致
    # topic -> 'T13497'
    return openalex_url[21:]


# =========================
# worker: 一个 paper 的全部 topics（只要 topics，存 topic 的 id）
# =========================
def get_topics_from_file(file_path):
    pid2topics = {}

    # topic id -> display_name 映射
    topic_id2display_name = {}

    with gzip.open(file_path, 'rb') as f:
        for line in f:
            line = json.loads(line)
            if 'id' not in line:
                continue

            # 只保留 article / preprint，且带有基本元数据，和 metadata pipeline 对齐
            if not (
                line.get('publication_year')
                and line.get('publication_date')
                and line.get('title')
                and line.get('authorships')
                and line.get('type') in {'article', 'preprint'}
            ):
                continue

            topics = line.get('topics')
            if not topics:
                continue

            id_ = line['id'][21:]

            # 收集这个 paper 的全部 topic id（按原始顺序，score 从高到低）
            topic_ids = []
            for topic in topics:
                topic_id = topic.get('id')
                if not topic_id:
                    continue
                topic_id = _strip_prefix(topic_id)
                topic_ids.append(topic_id)
                topic_id2display_name[topic_id] = topic.get('display_name')

            if topic_ids:
                pid2topics[id_] = topic_ids

    return pid2topics, topic_id2display_name


# =========================
# 主流程
# =========================
if not os.path.exists(os.path.join(save_dire, 'pid2topics.pkl')):

    pid2topics = {}
    topic_id2display_name = {}

    with ProcessPoolExecutor(max_workers=process_count) as executor:
        with tqdm(total=len(all_files), desc='process topics (file-level)') as pbar:
            for topics, t_id2name in executor.map(get_topics_from_file, all_files):
                pid2topics.update(topics)
                topic_id2display_name.update(t_id2name)
                pbar.update(1)

    with open(os.path.join(save_dire, 'pid2topics.pkl'), 'wb') as f:
        pickle.dump(pid2topics, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_dire, 'topic_id2display_name.pkl'), 'wb') as f:
        pickle.dump(topic_id2display_name, f, pickle.HIGHEST_PROTOCOL)

    print("size of pid2topics:", len(pid2topics))
    print("size of topic_id2display_name:", len(topic_id2display_name))

print("finished")
