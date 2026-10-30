"""
Extract per-paper full-text metadata from the Semantic Scholar S2ORC dump (gzipped
JSONL shards under S2AG_DIRE/s2orc, one paper per line). For each paper that has a
DOI and a unified `content` block, parse the section headers and the in-text
reference (bibref) instances, attaching each reference's DOI and venue from the
bibliography (bibentry / bibvenue) annotations. Shards are split into 32 batches
and processed in parallel; each batch writes its own pickles plus a finished-flag,
so the job is resumable.

Input  : S2AG_DIRE/s2orc/*.gz   (S2ORC full-text dump; S2AG_DIRE set in ../utils.py)
Outputs: S2ORC_META_DIRE/meta_batched_v2/reference/
           DOI2section_header_list_b_<batch>_total_<N>.pkl
           DOI2refer_instance_list_b_<batch>_total_<N>.pkl
           batch_<batch>_total_<N>_finished.pkl   (resume marker)
"""

import json
import os
import sys
import gzip
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

# utils.py is in the parent directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
from utils import S2AG_DIRE, S2ORC_META_DIRE  # noqa: E402

file_dire = os.path.join(S2AG_DIRE, 's2orc')
meta_dire = S2ORC_META_DIRE


def _load_ann(annotations, key):
    """Annotation values are JSON strings (list of {start,end,attributes}).
    Return the decoded list, or None if the key is absent/empty."""
    raw = annotations.get(key)
    if raw is None:
        return None
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def get_meta(file_path):
    DOI2section_header = {}
    DOI2refer_instances = {}

    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            # ---- new S2ORC format: externalids at top level (lowercase doi),
            #      single unified `content` block (text + annotations) ----
            externalids = obj.get('externalids')
            content = obj.get('content')
            if (
                not isinstance(externalids, dict) or
                externalids.get('doi') is None or
                not isinstance(content, dict) or
                content.get('text') is None
            ):
                continue

            DOI = externalids['doi']

            if DOI is None:
                continue

            annotations = content.get('annotations')
            if (
                not isinstance(annotations, dict) or
                annotations.get('sectionheader') is None or
                annotations.get('bibref') is None
            ):
                continue

            content_text = content['text']

            # ---------- section headers ----------
            section_headers = _load_ann(annotations, 'sectionheader')
            parsed_section_header_list = []

            for section_item in section_headers:
                start = int(section_item['start'])
                end = int(section_item['end'])
                title = content_text[start:end]

                parsed_section_header_list.append({
                    'start': start,
                    'end': end,
                    'title': title,
                    'n': (section_item.get('attributes') or {}).get('n')
                })

            DOI2section_header[DOI] = parsed_section_header_list

            # ---------- reference instances ----------
            bib_ref = _load_ann(annotations, 'bibref')
            refer_id2index = defaultdict(list)
            refer_instance_list = []

            for i, bib_item in enumerate(bib_ref):
                refer_id = None
                attributes = bib_item.get('attributes')
                if attributes is not None:
                    refer_id = attributes.get('ref_id')

                refer_instance_list.append({
                    'refer_id': refer_id,
                    'start': int(bib_item['start']),
                    'end': int(bib_item['end']),
                    'doi': None,
                    'venue': None
                })

                if refer_id is not None:
                    refer_id2index[refer_id].append(i)

            refer_id2doi = {}
            refer_id2text = {}
            refer_id2venue = {}
            # ---------- bibliography (now part of the unified content) ----------
            bib_entries = _load_ann(annotations, 'bibentry')
            if bib_entries is not None:
                bili_text = content_text

                valid_entry_flag = False
                venue_entries = _load_ann(annotations, 'bibvenue')
                if (
                    venue_entries is not None and
                    len(venue_entries) == len(bib_entries)
                ):
                    valid_entry_flag = True
                for bib_index, bib_entry in enumerate(bib_entries):
                    attributes = bib_entry.get('attributes')
                    if attributes is None:
                        continue

                    refer_id = attributes.get('id')
                    if refer_id is None:
                        continue

                    # NEW format already provides a clean extracted DOI; use it
                    # directly instead of regex-matching the entry text.
                    entry_doi = attributes.get('doi')
                    if entry_doi is not None:
                        refer_id2doi[refer_id] = entry_doi

                    if valid_entry_flag:
                        venue_start = int(venue_entries[bib_index]['start'])
                        venue_end = int(venue_entries[bib_index]['end'])
                        venue = bili_text[venue_start: venue_end]
                        refer_id2venue[refer_id] = venue
            for refer_id, doi in refer_id2doi.items():
                for idx in refer_id2index.get(refer_id, []):
                    refer_instance_list[idx]['doi'] = doi

            for refer_id, venue in refer_id2venue.items():
                for idx in refer_id2index.get(refer_id, []):
                    refer_instance_list[idx]['venue'] = venue

            DOI2refer_instances[DOI] = refer_instance_list

    return DOI2section_header, DOI2refer_instances


if __name__ == "__main__":

    file_list = sorted(os.listdir(file_dire))
    batch_count = 32
    print("batch_count:", batch_count)

    batch2file = defaultdict(list)
    for i, file in enumerate(file_list):
        batch2file[i % batch_count].append(file)

    save_dire = os.path.join(meta_dire, 'meta_batched_v2', 'reference')
    os.makedirs(save_dire, exist_ok=True)

    max_workers = 9
    print("Using workers:", max_workers)

    if max_workers > 1:
        for batch, files in tqdm(batch2file.items(), desc="Batches"):

            finished_flag = os.path.join(
                save_dire,
                f'batch_{batch}_total_{batch_count}_finished.pkl'
            )
            if os.path.exists(finished_flag):
                continue

            DOI2section_header_M = {}
            DOI2refer_instances_M = {}

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        get_meta,
                        os.path.join(file_dire, file)
                    )
                    for file in files
                ]

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Batch {batch}",
                    leave=False
                ):
                    sec_h, ref_i = future.result()
                    DOI2section_header_M.update(sec_h)
                    DOI2refer_instances_M.update(ref_i)

            with open(
                os.path.join(
                    save_dire,
                    f'DOI2section_header_list_b_{batch}_total_{batch_count}.pkl'
                ),
                'wb'
            ) as f:
                pickle.dump(DOI2section_header_M, f, pickle.HIGHEST_PROTOCOL)

            with open(
                os.path.join(
                    save_dire,
                    f'DOI2refer_instance_list_b_{batch}_total_{batch_count}.pkl'
                ),
                'wb'
            ) as f:
                pickle.dump(DOI2refer_instances_M, f, pickle.HIGHEST_PROTOCOL)

            with open(finished_flag, 'wb') as f:
                pickle.dump([], f)
    else:

        for batch, files in tqdm(batch2file.items(), desc="Batches"):

            finished_flag = os.path.join(
                save_dire,
                f'batch_{batch}_total_{batch_count}_finished.pkl'
            )
            if os.path.exists(finished_flag):
                continue

            DOI2section_header_M = {}
            DOI2refer_instances_M = {}

            for file in files:


                sec_h, ref_i = get_meta(os.path.join(file_dire, file))

                DOI2section_header_M.update(sec_h)
                DOI2refer_instances_M.update(ref_i)

            with open(
                os.path.join(
                    save_dire,
                    f'DOI2section_header_list_b_{batch}_total_{batch_count}.pkl'
                ),
                'wb'
            ) as f:
                pickle.dump(DOI2section_header_M, f, pickle.HIGHEST_PROTOCOL)

            with open(
                os.path.join(
                    save_dire,
                    f'DOI2refer_instance_list_b_{batch}_total_{batch_count}.pkl'
                ),
                'wb'
            ) as f:
                pickle.dump(DOI2refer_instances_M, f, pickle.HIGHEST_PROTOCOL)

            with open(finished_flag, 'wb') as f:
                pickle.dump([], f)


    print("finished")
