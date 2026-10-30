import os
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)               # utils.py is in this same directory
from utils import MAG_DATA_DIRE, MAG_ADVANCED_DIRE, META_DIRE

# ---------------------------------------------------------------------------
# Paths of the raw MAG tables (schema: see the bundled
# "Microsoft Academic Graph data schema ... .html")
# ---------------------------------------------------------------------------
PAPERS_PATH                    = os.path.join(MAG_DATA_DIRE,     'Papers.txt')
PAPER_REFERENCES_PATH          = os.path.join(MAG_DATA_DIRE,     'PaperReferences.txt')
PAPER_AUTHOR_AFFILIATIONS_PATH = os.path.join(MAG_DATA_DIRE,     'PaperAuthorAffiliations.txt')
AFFILIATIONS_PATH              = os.path.join(MAG_DATA_DIRE,     'Affiliations.txt')
PAPER_FIELDS_OF_STUDY_PATH     = os.path.join(MAG_ADVANCED_DIRE, 'PaperFieldsOfStudy.txt')
FIELDS_OF_STUDY_PATH           = os.path.join(MAG_ADVANCED_DIRE, 'FieldsOfStudy.txt')

save_dire = META_DIRE
os.makedirs(save_dire, exist_ok=True)

# each big single-file table is split into byte-range chunks processed in parallel
process_count = min(16, os.cpu_count() or 1)
print("save_dire", save_dire)
print("process_count", process_count)

# ---------------------------------------------------------------------------
# Papers.txt column indices (0-based, tab-separated)
#   0 PaperId | 1 Rank | 2 Doi | 3 DocType | 4 PaperTitle | 5 OriginalTitle
#   6 BookTitle | 7 Year | 8 Date | 9 OnlineDate | 10 Publisher | 11 JournalId | ...
# We keep only journal papers (DocType == 'Journal').
# ---------------------------------------------------------------------------
COL_PAPER_ID       = 0
COL_DOC_TYPE       = 3
COL_PAPER_TITLE    = 4
COL_ORIGINAL_TITLE = 5
COL_YEAR           = 7

# PaperReferences.txt : 0 PaperId (citing) | 1 PaperReferenceId (cited)
COL_REF_CITING = 0
COL_REF_CITED  = 1

# PaperAuthorAffiliations.txt : 0 PaperId | 1 AuthorId | 2 AffiliationId | 3 AuthorSequenceNumber | ...
COL_PAA_PAPER_ID       = 0
COL_PAA_AUTHOR_ID      = 1
COL_PAA_AFFILIATION_ID = 2
COL_PAA_AUTHOR_SEQ     = 3

# Affiliations.txt : 0 AffiliationId | ... | 10 Iso3166Code (two-letter country code) | ...
COL_AFF_ID   = 0
COL_AFF_ISO  = 10

# PaperFieldsOfStudy.txt : 0 PaperId | 1 FieldOfStudyId | 2 Score
COL_PFOS_PAPER_ID = 0
COL_PFOS_FOS_ID   = 1

# FieldsOfStudy.txt : 0 FieldOfStudyId | 1 Rank | 2 NormalizedName | 3 DisplayName | 4 MainType | 5 Level | ...
COL_FOS_ID          = 0
COL_FOS_DISPLAYNAME = 3
COL_FOS_LEVEL       = 5

# only keep the top-level (Level 0) fields of study, analogous to the
# level-0 concepts used in the OpenAlex pipeline (19 broad disciplines)
FOS_LEVEL_TO_KEEP = 0


# ===========================================================================
# Byte-range chunking: split a single big text file into `n_chunks` ranges
# that are aligned to line boundaries, so each worker owns whole lines.
# ===========================================================================
def make_chunks(path, n_chunks):
    file_size = os.path.getsize(path)
    if file_size == 0:
        return []
    chunk_size = max(1, file_size // n_chunks)
    chunks = []
    with open(path, 'rb') as f:
        start = 0
        while start < file_size:
            f.seek(min(start + chunk_size, file_size))
            f.readline()               # advance to the end of the current line
            end = f.tell()
            if end <= start:
                end = file_size
            chunks.append((path, start, end))
            start = end
    return chunks


def iter_range(path, start, end):
    """Yield decoded lines of `path` within the byte range [start, end)."""
    with open(path, 'rb') as f:
        f.seek(start)
        pos = start
        while pos < end:
            raw = f.readline()
            if not raw:
                break
            pos = f.tell()
            yield raw.decode('utf-8', 'ignore')


# ===========================================================================
# Step 1: Papers.txt -> pid2title, pid2pub_year  (journal papers only)
# ===========================================================================
def worker_papers(arg):
    path, start, end = arg
    pid2title_single = {}
    pid2pub_year_single = {}
    for line in iter_range(path, start, end):
        fields = line.rstrip('\n').split('\t')
        if len(fields) <= COL_YEAR:
            continue
        if fields[COL_DOC_TYPE] != 'Journal':          # keep journal papers only
            continue

        pid = int(fields[COL_PAPER_ID])
        # prefer the OriginalTitle (proper casing); fall back to PaperTitle
        pid2title_single[pid] = fields[COL_ORIGINAL_TITLE] or fields[COL_PAPER_TITLE]

        year = fields[COL_YEAR]
        if year:
            pid2pub_year_single[pid] = int(year)
    return pid2title_single, pid2pub_year_single


def build_paper_meta():
    pid2title = {}
    pid2pub_year = {}
    chunks = make_chunks(PAPERS_PATH, process_count)
    with ProcessPoolExecutor(max_workers=process_count) as executor:
        for pid2title_single, pid2pub_year_single in tqdm(
                executor.map(worker_papers, chunks), total=len(chunks), desc='Papers.txt'):
            pid2title.update(pid2title_single)
            pid2pub_year.update(pid2pub_year_single)
    return pid2title, pid2pub_year


# ===========================================================================
# Step 2: FieldsOfStudy.txt + PaperFieldsOfStudy.txt -> pid2field_of_study
#         (level-0 field-of-study display names, journal papers only)
# ===========================================================================
def build_fosid2name_level0():
    fosid2name_level0 = {}
    # FieldsOfStudy.txt is small; a single pass is enough
    with open(FIELDS_OF_STUDY_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='FieldsOfStudy.txt'):
            fields = line.rstrip('\n').split('\t')
            if len(fields) <= COL_FOS_LEVEL:
                continue
            if fields[COL_FOS_LEVEL] and int(fields[COL_FOS_LEVEL]) == FOS_LEVEL_TO_KEEP:
                fosid2name_level0[int(fields[COL_FOS_ID])] = fields[COL_FOS_DISPLAYNAME]
    return fosid2name_level0


def worker_fields_of_study(arg):
    # relies on module globals `valid_pid_set` and `fosid2name_level0`,
    # inherited by fork on Linux (avoids pickling the big set to every task)
    path, start, end = arg
    pid2field_of_study_single = defaultdict(list)
    for line in iter_range(path, start, end):
        fields = line.rstrip('\n').split('\t')
        if len(fields) <= COL_PFOS_FOS_ID:
            continue
        pid = int(fields[COL_PFOS_PAPER_ID])
        if pid not in valid_pid_set:
            continue
        fos_id = int(fields[COL_PFOS_FOS_ID])
        if fos_id in fosid2name_level0:
            pid2field_of_study_single[pid].append(fosid2name_level0[fos_id])
    return dict(pid2field_of_study_single)


def build_field_of_study():
    pid2field_of_study = defaultdict(list)
    chunks = make_chunks(PAPER_FIELDS_OF_STUDY_PATH, process_count)
    with ProcessPoolExecutor(max_workers=process_count) as executor:
        for single in tqdm(executor.map(worker_fields_of_study, chunks),
                           total=len(chunks), desc='PaperFieldsOfStudy.txt'):
            for pid, names in single.items():
                pid2field_of_study[pid].extend(names)
    return dict(pid2field_of_study)


# ===========================================================================
# Step 2a0: Affiliations.txt -> affid2country (AffiliationId -> ISO country code)
# ===========================================================================
def build_affid2country():
    affid2country = {}
    with open(AFFILIATIONS_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='Affiliations.txt'):
            fields = line.rstrip('\n').split('\t')
            if len(fields) <= COL_AFF_ISO:
                continue
            iso = fields[COL_AFF_ISO].strip()
            if iso:                            # keep only affiliations with a country code
                affid2country[int(fields[COL_AFF_ID])] = iso
    return affid2country


# ===========================================================================
# Step 2b: PaperAuthorAffiliations.txt -> pid2author_list, pid2author_country_list
#          (author ids ordered by AuthorSequenceNumber, journal papers only; each author's
#           affiliation countries as a de-duplicated list -- an author can have several
#           affiliations in different countries on one paper; empty list [] when none)
# ===========================================================================
def worker_authors(arg):
    # relies on module globals `valid_pid_set` and `affid2country` (inherited by fork)
    path, start, end = arg
    # pid -> list of (sequence_number, author_id, country); sorted/deduped after merge
    pid2seq_author_single = defaultdict(list)
    for line in iter_range(path, start, end):
        fields = line.rstrip('\n').split('\t')
        if len(fields) <= COL_PAA_AUTHOR_SEQ:
            continue
        pid = int(fields[COL_PAA_PAPER_ID])
        if pid not in valid_pid_set:
            continue
        if not fields[COL_PAA_AUTHOR_ID]:      # skip rows with no AuthorId
            continue
        author_id = int(fields[COL_PAA_AUTHOR_ID])
        seq = int(fields[COL_PAA_AUTHOR_SEQ]) if fields[COL_PAA_AUTHOR_SEQ] else 0
        # affiliation may be missing (nullable AffiliationId) -> country stays None
        aff = fields[COL_PAA_AFFILIATION_ID]
        country = affid2country.get(int(aff)) if aff else None
        pid2seq_author_single[pid].append((seq, author_id, country))
    return dict(pid2seq_author_single)


def build_author_list():
    pid2seq_author = defaultdict(list)
    chunks = make_chunks(PAPER_AUTHOR_AFFILIATIONS_PATH, process_count)
    with ProcessPoolExecutor(max_workers=process_count) as executor:
        for single in tqdm(executor.map(worker_authors, chunks),
                           total=len(chunks), desc='PaperAuthorAffiliations.txt'):
            for pid, triples in single.items():
                pid2seq_author[pid].extend(triples)

    # order authors by sequence number and drop duplicate authors (a paper can list the same
    # author on multiple rows / affiliations). Authors are ALWAYS kept even when the affiliation
    # is missing; each author's countries are collected as a de-duplicated, sorted list (an
    # author with several affiliations may span multiple countries), and [] when there is none.
    pid2author_list = {}
    pid2author_country_list = {}   # parallel to pid2author_list: each author's list of countries
    for pid, triples in tqdm(pid2seq_author.items(), desc='order authors'):
        triples.sort(key=lambda x: x[0])
        pos = {}                    # author_id -> index in the authors list
        authors = []
        country_sets = []           # index -> set of countries for that author
        for _seq, author_id, country in triples:
            if author_id not in pos:
                pos[author_id] = len(authors)
                authors.append(author_id)
                country_sets.append(set())
            if country is not None:
                country_sets[pos[author_id]].add(country)
        pid2author_list[pid] = authors
        pid2author_country_list[pid] = [sorted(s) for s in country_sets]
    return pid2author_list, pid2author_country_list


# ===========================================================================
# Step 3: PaperReferences.txt -> citing2cited_paper, cited2citing_paper
#         (both endpoints must be journal papers; drop future references)
# ===========================================================================
def worker_references(arg):
    # relies on module globals `valid_pid_set` and `pid2pub_year` (inherited by fork)
    path, start, end = arg
    citing2cited_single = defaultdict(list)
    cited2citing_single = defaultdict(list)
    n_total = n_journal = n_removed = n_kept = 0
    for line in iter_range(path, start, end):
        fields = line.rstrip('\n').split('\t')
        if len(fields) <= COL_REF_CITED:
            continue
        n_total += 1
        citing = int(fields[COL_REF_CITING])
        cited = int(fields[COL_REF_CITED])

        # both endpoints must be journal papers
        if citing not in valid_pid_set or cited not in valid_pid_set:
            continue
        n_journal += 1

        # drop references to papers published after the citing paper
        # (mirrors the temporal filter in the OpenAlex pipeline)
        citing_year = pid2pub_year.get(citing)
        cited_year = pid2pub_year.get(cited)
        if citing_year is not None and cited_year is not None and cited_year > citing_year:
            n_removed += 1
            continue

        n_kept += 1
        citing2cited_single[citing].append(cited)
        cited2citing_single[cited].append(citing)
    return dict(citing2cited_single), dict(cited2citing_single), (n_total, n_journal, n_removed, n_kept)


def build_citation():
    citing2cited_paper = defaultdict(list)   # citing_pid -> [cited_pid, ...]
    cited2citing_paper = defaultdict(list)   # cited_pid  -> [citing_pid, ...]
    n_total_edges = n_journal_edges = n_removed_by_year = n_kept_edges = 0

    chunks = make_chunks(PAPER_REFERENCES_PATH, process_count)
    with ProcessPoolExecutor(max_workers=process_count) as executor:
        for citing_single, cited_single, counts in tqdm(
                executor.map(worker_references, chunks), total=len(chunks), desc='PaperReferences.txt'):
            for p, v in citing_single.items():
                citing2cited_paper[p].extend(v)
            for p, v in cited_single.items():
                cited2citing_paper[p].extend(v)
            n_total_edges     += counts[0]
            n_journal_edges   += counts[1]
            n_removed_by_year += counts[2]
            n_kept_edges      += counts[3]

    print("total edges in PaperReferences.txt:      ", n_total_edges)
    print("edges with both endpoints journal papers:", n_journal_edges)
    print("edges removed because cited_year>citing_year:", n_removed_by_year)
    print("edges kept (before dedup):               ", n_kept_edges)

    # de-duplicate reference / citation lists
    citing2cited_paper = {p: list(set(v)) for p, v in tqdm(citing2cited_paper.items(), desc='dedup citing')}
    cited2citing_paper = {p: list(set(v)) for p, v in tqdm(cited2citing_paper.items(), desc='dedup cited')}
    return citing2cited_paper, cited2citing_paper


def dump(obj, name):
    path = os.path.join(save_dire, name)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    print("saved", path, "size", len(obj))


if __name__ == '__main__':
    # ---- Step 1 ----
    if not os.path.exists(os.path.join(save_dire, 'pid2pub_year.pkl')):
        pid2title, pid2pub_year = build_paper_meta()
        dump(pid2title, 'pid2title.pkl')
        dump(pid2pub_year, 'pid2pub_year.pkl')
    else:
        with open(os.path.join(save_dire, 'pid2pub_year.pkl'), 'rb') as f:
            pid2pub_year = pickle.load(f)

    # module globals so forked workers inherit them without per-task pickling
    valid_pid_set = set(pid2pub_year)
    print("number of journal papers with a year:", len(valid_pid_set))

    # ---- Step 2 ----
    if not os.path.exists(os.path.join(save_dire, 'pid2field_of_study.pkl')):
        fosid2name_level0 = build_fosid2name_level0()
        print("number of level-0 fields of study:", len(fosid2name_level0))
        pid2field_of_study = build_field_of_study()
        dump(pid2field_of_study, 'pid2field_of_study.pkl')

    # ---- Step 2b ----
    if not os.path.exists(os.path.join(save_dire, 'pid2author_country_list.pkl')):
        # module global so forked workers inherit the affiliation->country map
        affid2country = build_affid2country()
        print("number of affiliations with a country code:", len(affid2country))
        pid2author_list, pid2author_country_list = build_author_list()
        dump(pid2author_list, 'pid2author_list.pkl')
        dump(pid2author_country_list, 'pid2author_country_list.pkl')

    # ---- Step 3 ----
    if not os.path.exists(os.path.join(save_dire, 'cited_pid2citing_pids_list.pkl')):
        citing2cited_paper, cited2citing_paper = build_citation()
        dump(citing2cited_paper, 'citing_pid2cited_pids_list.pkl')
        dump(cited2citing_paper, 'cited_pid2citing_pids_list.pkl')

    print("finished")
