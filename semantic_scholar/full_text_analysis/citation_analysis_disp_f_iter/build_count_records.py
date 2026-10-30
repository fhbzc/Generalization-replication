"""
Standalone add-on to full_text_analysis (does NOT touch / share files with the main pipeline).

Builds its OWN per-citation mention-position records that additionally carry the raw disruption
counts up_count and down_count for every citation, so downstream plotting can bucket citations
by them (e.g. disruptive citations by down_count, or by up_count+down_count).

For a citing paper A citing a paper B (both matched to a corpusid via DOI, B among A's recorded
references), with up(X)=papers X cites and down(X)=papers citing X:
    up_count   = |up(A) & up(B)|    (references of B that A also cites)
    down_count = |up(A) & down(B)|  (downstream citers of B that A also cites)
    disp_type  = 'disruptive' if up_count == 0 else 'non_disruptive'
    efg        = g (up_count==0 & down_count==0), f (down_count>=up_count>0 or up_count==0 &
                 down_count>0), e (down_count<up_count)
This mirrors identify_citation_property.py's classification but ALSO records up_count/down_count.

Each in-text reference is assigned to a document region (before_intro / intro / method / result /
res_dis / discussion / conclusion / after_conclusion) exactly as get_refer_mention_position.py
does, using the canonical section separators from identify_paper_template.py. Mentions of the same
cited paper within one citing document are merged into a single record ({position: frequency}).

This reads the SAME upstream inputs as the main pipeline but writes to its OWN output file, so it
never overwrites the main pipeline's DOI2refer_instance_list_cid_parsed_V2.pkl /
refer_mention_records_V2.pkl.

Input  : META_DIRE/{cid2DOI, cid2pub_year}.pkl,
         META_DIRE/{citing2cited, cited2citing}_cid_list_remove_circle.pkl
         S2ORC_META_DIRE/DOI2refer_instance_list_V2.pkl      (from get_s2orc_meta_merge.py)
         S2ORC_META_DIRE/DOI2parsed_section_seperator_V2.pkl (from identify_paper_template.py)
Output : S2ORC_META_DIRE/citation_analysis_disp_f_iter/refer_mention_count_records_V2.pkl
             (flat list of dicts: efg, disp, up_count, down_count, cid, year, pos)
"""

import os
import sys
import bisect
import pickle
from collections import defaultdict

from tqdm import tqdm

# utils.py lives two levels up (semantic_scholar/); make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEM_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.insert(0, SEM_DIR)
from utils import S2ORC_META_DIRE, META_DIRE  # noqa: E402

meta_original = META_DIRE
meta_dire = S2ORC_META_DIRE
out_dire = os.path.join(meta_dire, 'citation_analysis_disp_f_iter')
os.makedirs(out_dire, exist_ok=True)

# ---- load the DOI -> corpusid map (unique DOIs only, mirroring identify_citation_property.py) ----
with open(os.path.join(meta_original, 'cid2DOI.pkl'), 'rb') as f:
    cid2DOI = pickle.load(f)
doi2pid_list = defaultdict(list)
for cid, doi in tqdm(cid2DOI.items(), desc='Building doi2pid_list'):
    doi2pid_list[doi].append(cid)
doi2pid = {doi: pids[0] for doi, pids in doi2pid_list.items() if len(pids) == 1}
print('doi2pid (unique):', len(doi2pid))

with open(os.path.join(meta_original, 'citing2cited_cid_list_remove_circle.pkl'), 'rb') as f:
    pid2up_list = pickle.load(f)       # cid -> papers it cites (references / up)
print('pid2up_list loaded')
with open(os.path.join(meta_original, 'cited2citing_cid_list_remove_circle.pkl'), 'rb') as f:
    pid2down_list = pickle.load(f)     # cid -> papers citing it (downstream / down)
print('pid2down_list loaded')
with open(os.path.join(meta_original, 'cid2pub_year.pkl'), 'rb') as f:
    cid2pub_year = pickle.load(f)
print('cid2pub_year loaded')

with open(os.path.join(meta_dire, 'DOI2refer_instance_list_V2.pkl'), 'rb') as f:
    DOI2refer_instances_M = pickle.load(f)
print('DOI2refer_instances_M loaded')
with open(os.path.join(meta_dire, 'DOI2parsed_section_seperator_V2.pkl'), 'rb') as f:
    DOI2parsed_section_header = pickle.load(f)
print('DOI2parsed_section_header loaded')


def section_boundaries(parsed_section_list):
    """(sorted boundary starts, labels, intro_start) for one paper's canonical sections,
    replicating get_refer_mention_position.py (merged Results&Discussion -> 'res_dis';
    an 'after_conclusion' boundary when the conclusion region has an end)."""
    type2start = {item['type']: item['start'] for item in parsed_section_list}
    conclusion_end = parsed_section_list[4].get('conclusion_end')
    if type2start['result'] == type2start['discussion']:
        boundaries = [
            (type2start['intro'], 'intro'),
            (type2start['method'], 'method'),
            (type2start['result'], 'res_dis'),
            (type2start['conclusion'], 'conclusion'),
        ]
    else:
        boundaries = [
            (type2start['intro'], 'intro'),
            (type2start['method'], 'method'),
            (type2start['result'], 'result'),
            (type2start['discussion'], 'discussion'),
            (type2start['conclusion'], 'conclusion'),
        ]
    if conclusion_end is not None:
        boundaries.append((conclusion_end, 'after_conclusion'))
    boundaries.sort(key=lambda b: b[0])
    return [b[0] for b in boundaries], [b[1] for b in boundaries], type2start['intro']


records = []
both_pid_matched_count = 0
error_instance_count = 0
for DOI, parsed_section_list in tqdm(DOI2parsed_section_header.items(), desc='DOIs'):
    if DOI not in DOI2refer_instances_M or DOI not in doi2pid:
        continue
    assert len(parsed_section_list) == 5, len(parsed_section_list)

    focal_pid = doi2pid[DOI]
    focal_up_pid_set = set(pid2up_list.get(focal_pid, []))
    citing_year = cid2pub_year.get(focal_pid)
    bound_starts, bound_labels, intro_start = section_boundaries(parsed_section_list)

    # accumulate per cited paper (refer_id) within this citing document
    refer_id2pos = {}
    refer_id2fields = {}
    for refer_instance in DOI2refer_instances_M[DOI]:
        refer_doi = refer_instance.get('doi')
        if refer_doi is None or refer_doi not in doi2pid:
            continue
        up_pid = doi2pid[refer_doi]                 # the cited paper B
        both_pid_matched_count += 1
        if up_pid not in focal_up_pid_set:          # B not among A's references -> skip
            error_instance_count += 1
            continue

        up_of_up_set = set(pid2up_list.get(up_pid, []))
        down_of_up_set = set(pid2down_list.get(up_pid, []))
        up_count = len(focal_up_pid_set & up_of_up_set)
        down_count = len(focal_up_pid_set & down_of_up_set)
        if up_count == 0:
            disp_type = 'disruptive'
            efg = 'g' if down_count == 0 else 'f'
        else:
            disp_type = 'non_disruptive'
            efg = 'f' if down_count >= up_count else 'e'

        # mention position (nearest canonical region start <= offset; before_intro if earlier)
        refer_start = refer_instance['start']
        if refer_start < intro_start:
            mention_position = 'before_intro'
        else:
            idx = bisect.bisect_right(bound_starts, refer_start) - 1
            mention_position = bound_labels[idx]

        refer_id = refer_instance['refer_id']
        assert refer_id is not None
        refer_id2fields[refer_id] = {
            'efg': efg, 'disp': disp_type,
            'up_count': up_count, 'down_count': down_count, 'cid': up_pid,
        }
        refer_id2pos.setdefault(refer_id, {})
        refer_id2pos[refer_id][mention_position] = refer_id2pos[refer_id].get(mention_position, 0) + 1

    for refer_id, pos in refer_id2pos.items():
        rec = dict(refer_id2fields[refer_id])
        rec['year'] = citing_year
        rec['pos'] = pos
        records.append(rec)

out_path = os.path.join(out_dire, 'refer_mention_count_records_V2.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(records, f, pickle.HIGHEST_PROTOCOL)
print('records:', len(records))
print('both_pid_matched_count:', both_pid_matched_count)
print('error_instance_count:', error_instance_count)
print('saved:', out_path)
print('finished')
