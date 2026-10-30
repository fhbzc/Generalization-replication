"""
Identify each paper's canonical "template" from its S2ORC section headers: keep
only papers whose TOP-LEVEL section titles contain, in document order, the five
canonical sections intro -> method -> result -> discussion -> conclusion (a merged
"Results and Discussion" header fills both the result and discussion slots).
Matching papers yield a fixed-order length-5 section list (index 1 == method,
relied on downstream); papers that fail are logged per rejection reason.

Input  : S2ORC_META_DIRE/DOI2section_header_list_V2.pkl   (from get_s2orc_meta_merge.py)
Outputs: S2ORC_META_DIRE/DOI2parsed_section_seperator_V2.pkl
         S2ORC_META_DIRE/analysis_V2/template_rejected/rejected_<reason>_V2.jsonl
"""

import json
import os
import sys
import pickle
import random
from collections import defaultdict

from tqdm import tqdm

# utils.py is in the parent directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
from utils import S2ORC_META_DIRE  # noqa: E402

meta_dire = S2ORC_META_DIRE

# keyword buckets for the five canonical sections (kept exactly as before)
intro_keywords = ['introduction', 'background']
method_keywords = ['method', 'data', 'procedure', 'experiment']
result_keywords = ['result', 'outcome']
discussion_keywords = ['discussion', 'implication']
conclusion_keywords = ['conclusion']

print("intro_keywords", intro_keywords)
print("method_keywords", method_keywords)
print("result_keywords", result_keywords)
print("discussion_keywords", discussion_keywords)
print("conclusion_keywords", conclusion_keywords)

# fixed order; index 1 must remain 'method' (downstream depends on it)
TYPE2KEYWORDS = [
    ('intro', intro_keywords),
    ('method', method_keywords),
    ('result', result_keywords),
    ('discussion', discussion_keywords),
    ('conclusion', conclusion_keywords),
]

# fixed order for the output list; index 1 must remain 'method'
TYPE_ORDER = ['intro', 'method', 'result', 'discussion', 'conclusion']
ALLOWED_TYPES = set(TYPE_ORDER)


def check_keyword(keyword_list, target_string):
    for key in keyword_list:
        if key in target_string:
            return True
    return False


def is_top_level(n):
    """A header is top-level (level-1) if its section number 'n' is a single
    numeric segment (e.g. '1', '1.', '2.') OR is absent/empty. A multi-segment
    'n' (e.g. '1.1', '3.2.1') is a sub-section and is NOT top-level."""
    n = str(n if n is not None else '').strip().rstrip('.')
    return n == '' or n.isdigit()


def matched_types(title):
    """Return the SET of canonical types whose keywords appear in `title`."""
    low = (title or '').lower()
    return {cat for cat, kws in TYPE2KEYWORDS if check_keyword(kws, low)}


def identify_template(DOI2section_header_M):
    """Decide each paper's canonical template from its TOP-LEVEL section titles
    by case-insensitive keyword match, requiring the five canonical types to
    appear IN ORDER.

    A paper is KEPT iff its top-level headers contain, in document order,
    intro -> method -> result -> discussion -> conclusion (the first header
    matching each type, scanning forward). OTHER sections in between are fine:
        intro, method, conclusion                          -> invalid (no result/discussion)
        intro, <section 1>, method, results, discussion, conclusion -> valid (extra section ignored)
        intro, method, result, discussion, conclusion      -> valid
        intro, method, result&discussion, conclusion       -> valid (merged R&D fills both)
    result & discussion may be a single merged "Results and Discussion" header
    (it fills both the result and discussion slots, sharing one start). Headers
    that match no keyword, repeated types, or anything after conclusion simply do
    not advance the match and are ignored.

    Returns (DOI2parsed_section_header, rejected). DOI2parsed_section_header is
    {DOI -> length-5 list in the fixed order [intro, method, result, discussion,
    conclusion]}; downstream consumes it and relies on index 1 == 'method'. For a
    merged Results-and-Discussion section, the result and discussion entries
    share that section's start (-> res_dis downstream).

    `rejected` is {reason -> [record, ...]} for every paper NOT kept, where each
    record is {doi, reason, detail, top_headers:[{title,n,start}],
    matched_types:[...]}. Reasons:
      empty_sections     no section headers at all
      missing_intro      could not find an intro header
      missing_method     found intro but no later method
      missing_result     found intro/method but no later result (or merged R&D)
      missing_discussion found result but no later discussion
      missing_conclusion found discussion but no later conclusion
    `detail` lists the types that WERE matched in order before the chain broke.
    """
    DOI2parsed_section_header = {}
    rejected = defaultdict(list)

    def record_reject(DOI, reason, top_headers, matched, detail=None):
        rejected[reason].append({
            'doi': DOI,
            'reason': reason,
            'detail': detail,   # e.g. which type was duplicated
            'top_headers': [
                {'title': h.get('title'), 'n': h.get('n'), 'start': h.get('start')}
                for h in top_headers
            ],
            'matched_types': sorted(matched),
        })

    DOI_list = list(DOI2section_header_M.keys())
    random.shuffle(DOI_list)
    for DOI in tqdm(DOI_list):
        section_header_list = DOI2section_header_M[DOI]
        if len(section_header_list) == 0:
            record_reject(DOI, 'empty_sections', [], set())
            continue

        # only TOP-LEVEL (level-1) headers participate; sub-sections skipped,
        # ordered by their position in the document. Headers whose title contains
        # "journal of" (a journal name leaking into the headers, not a real
        # section) are dropped outright.
        top_headers = sorted(
            (h for h in section_header_list
             if is_top_level(h.get('n'))
             and 'journal of' not in (h.get('title') or '').lower()),
            key=lambda h: h['start'])

        # the raw annotations sometimes store the SAME header several times
        # (identical title + identical start); collapse such duplicate instances
        # so they don't look like a real duplicate section. A merged
        # "Results and Discussion" header is a single distinct title, so it is
        # unaffected.
        deduped = []
        seen = set()
        for h in top_headers:
            key = (h.get('start'), h.get('title'))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(h)
        top_headers = deduped

        # GREEDY ORDERED subsequence match of TYPE_ORDER over the top-level
        # headers (already sorted by document position): find intro, then a later
        # method, then a later result, then a later discussion, then a later
        # conclusion. Headers in between (other sections, unmatched titles, even a
        # repeated type) are simply skipped -- only the ordered appearance of the
        # five types matters. result & discussion may be ONE merged
        # "Results and Discussion" header (it fills both slots, same start).
        seq = [None] * len(TYPE_ORDER)   # chosen header per TYPE_ORDER slot
        ptr = 0
        for h in top_headers:
            if ptr >= len(TYPE_ORDER):
                break
            matched = matched_types(h.get('title'))
            if not matched:
                continue
            target = TYPE_ORDER[ptr]
            if target == 'result':
                if 'result' in matched:
                    seq[2] = h                       # result
                    if 'discussion' in matched:
                        seq[3] = h                   # merged R&D -> discussion too
                        ptr = 4                      # next: conclusion
                    else:
                        ptr = 3                      # next: discussion
            elif target in matched:
                seq[ptr] = h
                ptr += 1

        if ptr < len(TYPE_ORDER):
            # the next type we still needed never appeared (in order) afterwards
            matched_so_far = {TYPE_ORDER[i] for i in range(len(TYPE_ORDER))
                              if seq[i] is not None}
            record_reject(DOI, 'missing_' + TYPE_ORDER[ptr], top_headers,
                          matched_so_far,
                          detail='matched_in_order=' + ','.join(sorted(matched_so_far)))
            continue

        # the conclusion REGION ends at the first top-level header AFTER the
        # conclusion header (e.g. References / Acknowledgements). Anything beyond
        # that start is "after conclusion". None if conclusion is the last header.
        conclusion_start = seq[4]['start']
        conclusion_end = None
        for h in top_headers:
            if h['start'] > conclusion_start:
                conclusion_end = h['start']
                break

        parsed_section_header_list = [
            {
                'type': t,
                'start': seq[i]['start'],
                'name': (seq[i]['title'] or '').lower(),
            }
            for i, t in enumerate(TYPE_ORDER)
        ]
        # carry the conclusion region's end on the conclusion entry (index 4)
        parsed_section_header_list[4]['conclusion_end'] = conclusion_end

        DOI2parsed_section_header[DOI] = parsed_section_header_list

    return DOI2parsed_section_header, dict(rejected)


if __name__ == "__main__":
    with open(os.path.join(meta_dire, 'DOI2section_header_list_V2.pkl'), 'rb') as f:
        DOI2section_header_M = pickle.load(f)

    DOI2parsed_section_header, rejected = identify_template(DOI2section_header_M)

    with open(os.path.join(meta_dire, 'DOI2parsed_section_seperator_V2.pkl'), 'wb') as f:
        pickle.dump(DOI2parsed_section_header, f, pickle.HIGHEST_PROTOCOL)

    print("size of DOI2parsed_section_header", len(DOI2parsed_section_header))

    # ---- save rejected papers as JSONL, ONE FILE PER REJECTION REASON ----
    # each reason's list is reshuffled so the first lines are a random sample
    reject_dir = os.path.join(meta_dire, 'analysis_V2', 'template_rejected')
    os.makedirs(reject_dir, exist_ok=True)
    print("rejected papers per reason:")
    for reason, recs in sorted(rejected.items()):
        random.shuffle(recs)
        out_path = os.path.join(reject_dir, f'rejected_{reason}_V2.jsonl')
        with open(out_path, 'w', encoding='utf-8') as f:
            for rec in tqdm(recs, desc=reason, leave=False):
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f"  {reason:18s} {len(recs):>8d}  -> {out_path}")

    print("finished")
