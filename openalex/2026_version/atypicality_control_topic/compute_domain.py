"""
Coverage check for the primary-topic hierarchy over the pid2pub_year universe.

For every pid in pid2pub_year.pkl (produced by get_meta_data_preprint_2026.py), report how
many of them have a VALID (non-None) value at each level of the OpenAlex topic hierarchy
(domain / field / subfield / topic), using the pid2primary_<level>.pkl maps produced by
get_meta_data_preprint_2026_s2_topic.py.

Also, for each level, write a CSV listing how many pids fall into each group (each
domain / field / subfield / topic). Counts are over the pid2pub_year universe only.

Input  : META_DIRE_2026/pid2pub_year.pkl
         META_DIRE_2026/pid2primary_{domain,field,subfield,topic}.pkl
         META_DIRE_2026/{domain,field,subfield,topic}_id2display_name.pkl   (optional, for labels)
Output : ./{domain,field,subfield,topic}_pid_count.csv
"""

import os
import sys
import pickle
from collections import Counter

import pandas as pd

# utils.py is in openalex/, two levels up from this file
# (openalex/2026_version/atypicality_control_topic/<this file>).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OPENALEX_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.insert(0, OPENALEX_DIR)
from utils import META_DIRE_2026  # noqa: E402

meta_dire = META_DIRE_2026
print("meta_dire", meta_dire)

LEVELS = ['domain', 'field', 'subfield', 'topic']


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# ---- the pid universe = every pid in pid2pub_year -------------------------------------
pid2year = load_pkl(os.path.join(meta_dire, 'pid2pub_year.pkl'))
all_pids = list(pid2year.keys())
total = len(all_pids)
print("total pids in pid2pub_year:", total)

for level in LEVELS:
    pid2group = load_pkl(os.path.join(meta_dire, f'pid2primary_{level}.pkl'))

    # optional id -> display_name map for nicer labels
    id2name_path = os.path.join(meta_dire, f'{level}_id2display_name.pkl')
    id2name = load_pkl(id2name_path) if os.path.exists(id2name_path) else {}

    # count valid (non-None) coverage + per-group pid counts, over the pid2pub_year universe.
    # groups are keyed by display_name (fall back to the raw id if a name is missing).
    valid_count = 0
    group_counter = Counter()
    for pid in all_pids:
        group = pid2group.get(pid)
        if group is None:
            continue
        valid_count += 1
        group_counter[id2name.get(group, group)] += 1

    # pids in the pid2pub_year universe with NO valid value at this level (None / missing)
    invalid_count = total - valid_count

    pct = 100.0 * valid_count / total if total else 0.0
    print(f"[{level:>8}] valid (non-None): {valid_count} / {total} ({pct:.2f}%)"
          f" | no valid value: {invalid_count}"
          f" | distinct {level}s: {len(group_counter)}")

    # ---- per-group CSV: how many pids in each domain/field/subfield/topic ----
    rows = [
        {'display_name': name, 'pid_count': cnt}
        for name, cnt in group_counter.items()
    ]
    df = pd.DataFrame(rows).sort_values('pid_count', ascending=False).reset_index(drop=True)

    # append a final row with the count of pids that have no valid value at this level
    df = pd.concat(
        [df, pd.DataFrame([{'display_name': f'(no valid {level})', 'pid_count': invalid_count}])],
        ignore_index=True,
    )

    out_path = os.path.join(BASE_DIR, f'{level}_pid_count.csv')
    df.to_csv(out_path, index=False)
    print(f"    wrote {out_path}  ({len(df)} rows)")

    del pid2group, id2name, group_counter

print("finished")
