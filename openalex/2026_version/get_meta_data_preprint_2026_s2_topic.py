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
    # topic     -> 'T13497'
    # subfield  -> 'subfields/1211'
    # field     -> 'fields/12'
    # domain    -> 'domains/2'
    return openalex_url[21:]


# =========================
# worker: primary_topic
# =========================
def get_primary_topic_from_file(file_path):
    pid2primary_topic = {}
    pid2primary_subfield = {}
    pid2primary_field = {}
    pid2primary_domain = {}

    # 每一层各自维护一个 id -> display_name 映射
    topic_id2display_name = {}
    subfield_id2display_name = {}
    field_id2display_name = {}
    domain_id2display_name = {}

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

            primary_topic = line.get('primary_topic')
            if not primary_topic:
                continue

            id_ = line['id'][21:]

            # 各级都存 id，同时把 id -> display_name 记进对应映射；缺失的层级留 None
            for pid2x, id2name, node in (
                (pid2primary_topic, topic_id2display_name, primary_topic),
                (pid2primary_subfield, subfield_id2display_name, primary_topic.get('subfield')),
                (pid2primary_field, field_id2display_name, primary_topic.get('field')),
                (pid2primary_domain, domain_id2display_name, primary_topic.get('domain')),
            ):
                node = node or {}
                node_id = node.get('id')
                if node_id:
                    node_id = _strip_prefix(node_id)
                    pid2x[id_] = node_id
                    id2name[node_id] = node.get('display_name')
                else:
                    pid2x[id_] = None

    return (pid2primary_topic, pid2primary_subfield,
            pid2primary_field, pid2primary_domain,
            topic_id2display_name, subfield_id2display_name,
            field_id2display_name, domain_id2display_name)


# =========================
# 主流程
# =========================
if not os.path.exists(os.path.join(save_dire, 'pid2primary_topic.pkl')):

    pid2primary_topic = {}
    pid2primary_subfield = {}
    pid2primary_field = {}
    pid2primary_domain = {}

    topic_id2display_name = {}
    subfield_id2display_name = {}
    field_id2display_name = {}
    domain_id2display_name = {}

    with ProcessPoolExecutor(max_workers=process_count) as executor:
        with tqdm(total=len(all_files), desc='process primary_topic (file-level)') as pbar:
            for (topic, subfield, field, domain,
                 t_id2name, s_id2name, f_id2name, d_id2name) in executor.map(get_primary_topic_from_file, all_files):
                pid2primary_topic.update(topic)
                pid2primary_subfield.update(subfield)
                pid2primary_field.update(field)
                pid2primary_domain.update(domain)
                topic_id2display_name.update(t_id2name)
                subfield_id2display_name.update(s_id2name)
                field_id2display_name.update(f_id2name)
                domain_id2display_name.update(d_id2name)
                pbar.update(1)

    with open(os.path.join(save_dire, 'pid2primary_topic.pkl'), 'wb') as f:
        pickle.dump(pid2primary_topic, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_dire, 'pid2primary_subfield.pkl'), 'wb') as f:
        pickle.dump(pid2primary_subfield, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_dire, 'pid2primary_field.pkl'), 'wb') as f:
        pickle.dump(pid2primary_field, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_dire, 'pid2primary_domain.pkl'), 'wb') as f:
        pickle.dump(pid2primary_domain, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_dire, 'topic_id2display_name.pkl'), 'wb') as f:
        pickle.dump(topic_id2display_name, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_dire, 'subfield_id2display_name.pkl'), 'wb') as f:
        pickle.dump(subfield_id2display_name, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_dire, 'field_id2display_name.pkl'), 'wb') as f:
        pickle.dump(field_id2display_name, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_dire, 'domain_id2display_name.pkl'), 'wb') as f:
        pickle.dump(domain_id2display_name, f, pickle.HIGHEST_PROTOCOL)

    print("size of pid2primary_topic:", len(pid2primary_topic))
    print("size of pid2primary_subfield:", len(pid2primary_subfield))
    print("size of pid2primary_field:", len(pid2primary_field))
    print("size of pid2primary_domain:", len(pid2primary_domain))
    print("size of topic_id2display_name:", len(topic_id2display_name))
    print("size of subfield_id2display_name:", len(subfield_id2display_name))
    print("size of field_id2display_name:", len(field_id2display_name))
    print("size of domain_id2display_name:", len(domain_id2display_name))

print("finished")
