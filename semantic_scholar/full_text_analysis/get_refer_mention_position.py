"""
Build per-citation mention-position records, supporting BOTH classification
systems (efg and disp) at once.

For every citation (one citing paper -> one cited paper), we record:
  efg    : 'e' / 'f' / 'g'                 (from identify_citation_property.py)
  disp   : 'disruptive' / 'non_disruptive' (independent axis, same source)
  disp_f : list of 'disp_f_{T}' variants this citation qualifies for
           (disruptive foundation at down_count > T, T in {0,1,2}; [] otherwise)
  cid    : the CITED paper's corpus id     (for per-domain grouping downstream)
  year : citing paper's publication year
  pos  : {mention_position: frequency}   over before_intro / intro / method /
         result / res_dis / discussion / conclusion / after_conclusion

identify_paper_template.py guarantees the five canonical sections appear in
order (intro -> method -> result -> discussion -> conclusion), though OTHER
sections may sit in between (e.g. intro -> "Section 1" -> method). The document
is therefore partitioned into REGIONS by their starts, and a citation is
assigned to whichever region it falls in: the canonical section with the largest
`start` that is <= the citation's offset. A citation inside an in-between or
sub-section header still maps to the preceding canonical region (e.g. something
between intro and method -> 'intro'; a Methods sub-section -> 'method').

Two extra buckets capture text outside the canonical body:
  before_intro     : citation offset is BEFORE the intro header (e.g. abstract)
  after_conclusion : citation offset is at/after the first top-level header that
                     follows conclusion (References / Acknowledgements / ...),
                     i.e. the conclusion region's end (conclusion_end from the
                     template). If conclusion is the last section there is no such
                     boundary and everything after it stays 'conclusion'.

When result and discussion are a single merged "Results and Discussion" section
(they share the same start), that section's label is 'res_dis'.

Output: refer_mention_records_V2.pkl -> a flat list of such record dicts. Both
analyze_refer_mention_position.py (aggregate stats) and the longitudinal script
consume this single list.
"""

import os
import sys
import bisect
import pickle

from tqdm import tqdm

# utils.py is in the parent directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
from utils import S2ORC_META_DIRE  # noqa: E402

meta_dire = S2ORC_META_DIRE

with open(os.path.join(meta_dire, 'DOI2parsed_section_seperator_V2.pkl'), 'rb') as f:
    DOI2parsed_section_header = pickle.load(f)
print("load DOI2parsed_section_header finished")
with open(os.path.join(meta_dire, 'DOI2refer_instance_list_cid_parsed_V2.pkl'), 'rb') as f:
    DOI2refer_instances_M = pickle.load(f)
print("load DOI2refer_instances_M finished")

records = []          # one dict per (citing DOI, cited refer_id) citation
for DOI, parsed_section_list in tqdm(DOI2parsed_section_header.items()):
    if DOI not in DOI2refer_instances_M:
        continue

    assert len(parsed_section_list) == 5, len(parsed_section_list)
    type2start = {item['type']: item['start'] for item in parsed_section_list}
    # end of the conclusion region (start of the first top header AFTER
    # conclusion, e.g. References); None if conclusion is the last section
    conclusion_end = parsed_section_list[4].get('conclusion_end')

    # build (start, position_label) boundaries. result & discussion collapse to
    # a single 'res_dis' boundary when they are the SAME (merged) section.
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

    # citations after the conclusion region (>= conclusion_end) are labelled
    # 'after_conclusion'; add that boundary so they fall into their own bucket
    if conclusion_end is not None:
        boundaries.append((conclusion_end, 'after_conclusion'))

    # sort by document position; the section a citation falls in is the one whose
    # start is the largest start <= the citation offset (order NOT assumed)
    boundaries.sort(key=lambda b: b[0])
    bound_starts = [b[0] for b in boundaries]
    bound_labels = [b[1] for b in boundaries]
    intro_start = type2start['intro']

    # accumulate per cited paper (refer_id) within this citing document
    refer_id2pos = {}
    refer_id2efg = {}
    refer_id2disp = {}
    refer_id2disp_f = {}
    refer_id2cid = {}
    doi_year = None
    for refer_instance in DOI2refer_instances_M[DOI]:
        if 'efg_type' not in refer_instance:
            continue
        refer_id = refer_instance['refer_id']
        assert refer_id is not None
        refer_start = refer_instance['start']

        if refer_start < intro_start:
            # citation precedes the intro header -> 'before_intro'
            metion_position = 'before_intro'
        else:
            # nearest canonical region header at or before this citation offset
            idx = bisect.bisect_right(bound_starts, refer_start) - 1
            metion_position = bound_labels[idx]

        refer_id2efg[refer_id] = refer_instance['efg_type']
        refer_id2disp[refer_id] = refer_instance.get('disp_type')
        refer_id2disp_f[refer_id] = refer_instance.get('disp_f_types', [])
        refer_id2cid[refer_id] = refer_instance.get('cid')  # cited paper's cid
        doi_year = refer_instance.get('citing_year')

        refer_id2pos.setdefault(refer_id, {})
        refer_id2pos[refer_id].setdefault(metion_position, 0)
        refer_id2pos[refer_id][metion_position] += 1

    for refer_id, pos in refer_id2pos.items():
        records.append({
            'efg': refer_id2efg[refer_id],
            'disp': refer_id2disp[refer_id],
            'disp_f': refer_id2disp_f[refer_id],   # list of disp_f_{T} variants (or [])
            'cid': refer_id2cid[refer_id],   # cited paper cid (for domain grouping)
            'year': doi_year,
            'pos': pos,
        })

with open(os.path.join(meta_dire, 'refer_mention_records_V2.pkl'), 'wb') as f:
    pickle.dump(records, f, pickle.HIGHEST_PROTOCOL)
print("size of records", len(records))
print("finished")
