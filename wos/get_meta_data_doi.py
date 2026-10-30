"""
Build the WoS paper -> DOI map (pid2doi.pkl).

The WoS counterpart of openalex/get_meta_data_preprint_doi.py. Nothing in the F/E/G pipeline
needs it -- it exists so Nobel-prize papers, which the external prize records identify by DOI,
can be located in the WoS corpus. It is the map award_analysis/award_quantile_analysis.py
joins on, and the only extra input that analysis needs.

Reads the same WoS papers parquet shards as get_meta_data.py, applies the SAME filter
(pubtype == 'Journal' and doctypes == 'article') and the SAME id convention (the 'WOS:' prefix
stripped), so the keys line up one-to-one with pid2title.pkl / pid2pub_year.pkl and therefore
with every downstream table.

The parquet's DOI column is 'doi'. It holds the BARE DOI -- no resolver prefix -- in whatever
case the publisher registered it ('10.1103/PhysRevC.59.2728', '10.3866/PKU.WHXB19981118'), and a
paper with NO DOI carries an EMPTY STRING rather than null, so .notna() alone does not filter
those out.

DOIs are therefore stored CANONICALISED -- stripped, lower-cased, and any resolver prefix
removed should one ever appear -- so consumers can compare without normalising again. DOIs are
case-insensitive by spec and both this column and the Nobel records carry mixed-case ones, so a
raw comparison would silently lose papers. Papers whose DOI is empty are left out of the map.

Input  : WOS_DIRE/wos_papers.parquet/*.parquet   (EXTERNAL WoS dataset; WOS_DIRE in utils.py)
Output (to META_DIRE/): pid2doi.pkl   {pid: '10.1103/physrevlett.50.1153', ...}
         An existing output is left alone; delete it to rebuild.

Usage  : python get_meta_data_doi.py
"""

import os
import sys
import pickle
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# utils.py is in this same directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from utils import META_DIRE, WOS_DIRE  # noqa: E402

max_workers = 6

paper_dire_raw = os.path.join(WOS_DIRE, 'wos_papers.parquet')
output_dire = META_DIRE
os.makedirs(output_dire, exist_ok=True)
doi_out_path = os.path.join(output_dire, 'pid2doi.pkl')

print("paper_dire_raw", paper_dire_raw)
print("output_dire", output_dire)
print("max_workers", max_workers)

# The column holding the DOI, verified against the papers parquet.
DOI_COLUMN = 'doi'

_DOI_PREFIXES = ('https://doi.org/', 'http://doi.org/', 'https://dx.doi.org/',
                 'http://dx.doi.org/', 'doi:')


def norm_doi(doi):
    """Canonical form for DOI comparison: stripped, lower-cased, no resolver prefix."""
    s = str(doi).strip().lower()
    for prefix in _DOI_PREFIXES:
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s or None


# ==================================================
# 1. locate the shards and resolve the DOI column name from the schema
# ==================================================
assert os.path.isdir(paper_dire_raw), (
    "%s\nSet WOS_DIRE in utils.py to the WoS dataset root." % paper_dire_raw)
files = sorted(f for f in os.listdir(paper_dire_raw) if f.endswith('.parquet'))
assert files, "no .parquet shard in %s" % paper_dire_raw
print("parquet shards:", len(files))

schema_names = pq.ParquetFile(os.path.join(paper_dire_raw, files[0])).schema.names
doi_col = DOI_COLUMN
assert doi_col in schema_names, (
    "'%s' is not a column of %s.\nThe shard's columns are:\n  %s"
    % (doi_col, files[0], '\n  '.join(schema_names)))
print("DOI column:", doi_col)

read_cols = ['UID', 'pubtype', 'doctypes', doi_col]


# ==================================================
# 2. one worker per shard: pid -> doi for the journal articles in it
# ==================================================
def extract_doi(file_name):
    df = pd.read_parquet(os.path.join(paper_dire_raw, file_name), columns=read_cols)
    # exactly get_meta_data.py's paper filter, so the keys match the rest of the pipeline
    df = df[(df['pubtype'] == 'Journal') & (df['doctypes'] == 'article')]

    uid = df['UID'].astype(str)
    keep = uid.str.startswith('WOS:')          # same 'WOS:' convention as get_meta_data.py
    pids = uid[keep].str[4:].to_numpy()

    pid2doi = {}
    for pid, v in zip(pids, df.loc[keep, doi_col].to_numpy()):
        if v is None or pd.isna(v):
            continue
        # a paper with no DOI carries '' here; norm_doi turns that into None -> skipped
        key = norm_doi(v)
        if key:
            pid2doi[pid] = key
    return pid2doi


if __name__ == '__main__':
    if os.path.exists(doi_out_path):
        print("output already exists, nothing to do:", doi_out_path)
        print("(delete it to rebuild)")
        sys.exit(0)

    pid2doi_M = {}
    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for part in tqdm(executor.map(extract_doi, files), total=len(files),
                             desc='extracting pid->doi'):
                pid2doi_M.update(part)
    else:
        for file_name in tqdm(files, desc='extracting pid->doi'):
            pid2doi_M.update(extract_doi(file_name))

    print("pid2doi size:", len(pid2doi_M))
    # How often one DOI is shared by several papers -- award_quantile_analysis.py drops exactly
    # those (a winner is only accepted through a one-to-one mapping), so this number is the
    # price of that rule.
    print("papers whose DOI is shared with another paper:",
          len(pid2doi_M) - len(set(pid2doi_M.values())))

    with open(doi_out_path, 'wb') as f:
        pickle.dump(pid2doi_M, f, pickle.HIGHEST_PROTOCOL)
    print("saved:", doi_out_path)
    print("finished")
