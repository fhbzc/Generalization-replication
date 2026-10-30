
import os

# anchor relative paths to this file's directory so scripts run regardless of CWD
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# may be written as either a relative path (anchored to this file's directory) or an absolute path
OPENALEX_DIRE = os.path.join(_BASE_DIR, '/path/to/openalex_snapshot') # the directory where openalex dataset is saved

# the 2026 version uses a separate OpenAlex snapshot directory
OPENALEX_DIRE_2026 = os.path.join(_BASE_DIR, '/path/to/openalex_snapshot_2026_01') # the directory where the 2026 openalex dataset is saved

META_DIRE = os.path.join(_BASE_DIR, 'data') # the directory where you want to save the processed files

LID_MODEL_DIRE = os.path.join(_BASE_DIR, 'models') # the directory where the fastText language detection model is saved/downloaded

# may be written as either a relative path (anchored to this file's directory) or an absolute path
META_DIRE_2026 = os.path.join(_BASE_DIR, 'data/release_2026_01_meta') # the directory where you want to save the 2026 processed files


# EXTERNAL DATASET (not produced by this pipeline): per-paper AI-usage (LLM) share
# estimates, i.e. the files {arxiv,biorxiv,ssrn}_alpha_share.parquet. Point this at
# wherever those parquet files are stored.
AI_USAGE_DIRE = os.path.join(_BASE_DIR, 'data', 'llm_usage_dection')

# EXTERNAL DATASET (not produced by this pipeline): the FSOe-2026 source table
# (directory containing FSOe-2026.txt). Point this at wherever that file is stored.
FSO_DIRE = os.path.join(_BASE_DIR, 'data', 'FSO_data')

# EXTERNAL DATASET (not produced by this pipeline): the major-prize winner
# spreadsheets (Medicine/Biology/Chemistry prize lists). Point this at wherever
# those files are stored.
PRIZE_DATA_DIRE = os.path.join(_BASE_DIR, 'data', 'major_prize_data')

# EXTERNAL DATASET (not produced by this pipeline): the Nobel-prize publication
# records ({Medicine,Chemistry,Physics} publication record.csv). Point this at
# wherever those files are stored.
NOBEL_DATA_DIRE = os.path.join(_BASE_DIR, 'data', 'nobel_prize')

# EXTERNAL DATASET (not produced by this pipeline): the ACL-ARC citation-purpose
# dataset (acl-arc.citation-graph.with-functions.tsv, arc-paper-ids.tsv). Point this
# at wherever those files are stored.
NLP_PURPOSE_DIRE = os.path.join(_BASE_DIR, 'data', 'nlp_purpose')

# EXTERNAL DATASET (not produced by this pipeline): the ACT2 citation-purpose
# dataset (ACT2_dataset.tsv). Point this at wherever that file is stored.
ACT2_PURPOSE_DIRE = os.path.join(_BASE_DIR, 'data', 'nlp_purpose_act')


CONCEPT_DIRE = os.path.join(_BASE_DIR, 'data', 'Concepts_definition.xlsx')

'''

This replication package provides the dataset

Original source:
Downloaded from
https://docs.google.com/spreadsheets/d/1LBFHjPt4rj_9r0t0TTAlT68NwOtNH8Z21lBMsJDMoZg/edit?gid=575855905#gid=575855905
and rename to Concepts_definition.xlsx
'''