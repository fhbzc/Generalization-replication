import os
import pickle
from collections import defaultdict
from tqdm import tqdm
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)               # utils.py is in this same directory
from utils import META_DIRE

meta_dire = META_DIRE
save_dire = meta_dire
os.makedirs(save_dire, exist_ok=True)
print("meta_dire", meta_dire)

# keep authors with at least this many (journal) papers
MIN_PAPERS = 3
print("MIN_PAPERS", MIN_PAPERS)

# ---------------------------------------------------------------------------
# load inputs (produced by get_paper_meta.py)
# ---------------------------------------------------------------------------
with open(os.path.join(meta_dire, 'pid2author_list.pkl'), 'rb') as f:
    pid2author_list = pickle.load(f)
print("pid2author_list load finished, size", len(pid2author_list))

with open(os.path.join(meta_dire, 'pid2pub_year.pkl'), 'rb') as f:
    pid2pub_year = pickle.load(f)
print("pid2pub_year load finished, size", len(pid2pub_year))

# ---------------------------------------------------------------------------
# 1. invert pid -> authors  into  author -> papers
# ---------------------------------------------------------------------------
author2pid_list = defaultdict(list)
for pid, author_list in tqdm(pid2author_list.items(), desc='invert pid->authors'):
    if pid not in pid2pub_year:      # only papers with a known publication year
        continue
    for author_id in author_list:
        author2pid_list[author_id].append(pid)

print("total distinct authors:", len(author2pid_list))

# ---------------------------------------------------------------------------
# 2. keep authors with >= MIN_PAPERS papers; compute per-paper academic age
#    academic_age(paper) = pub_year(paper) - min(pub_year over the author's papers)
# ---------------------------------------------------------------------------
author2paper_list = {}                    # author -> [pid, ...]  (>= MIN_PAPERS)
author2first_pub_year = {}                # author -> earliest pub_year
author_paper_academic_age = {}            # author -> {pid: academic_age}

for author_id, pid_list in tqdm(author2pid_list.items(), desc='compute academic age'):
    # de-duplicate papers (an author should not count the same paper twice)
    pid_list = list(set(pid_list))
    if len(pid_list) < MIN_PAPERS:
        continue

    first_year = min(pid2pub_year[pid] for pid in pid_list)
    author2paper_list[author_id] = pid_list
    author2first_pub_year[author_id] = first_year
    author_paper_academic_age[author_id] = {
        pid: pid2pub_year[pid] - first_year for pid in pid_list
    }

print("authors with >= %d papers: %d" % (MIN_PAPERS, len(author2paper_list)))


def dump(obj, name):
    path = os.path.join(save_dire, name)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    print("saved", path, "size", len(obj))


dump(author2paper_list, 'author2paper_list.pkl')
dump(author2first_pub_year, 'author2first_pub_year.pkl')
dump(author_paper_academic_age, 'author_paper_academic_age.pkl')

print("finished")
