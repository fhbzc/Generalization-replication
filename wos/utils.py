import os

# anchor relative paths to this file's directory so scripts run regardless of CWD
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# may be written as either a relative path (anchored to this file's directory) or an absolute path
WOS_DIRE = os.path.join(_BASE_DIR, '/path/to/wos') # the directory where the Web of Science (WoS) dataset is stored

META_DIRE = os.path.join(_BASE_DIR, 'data') # the directory where you want to save the processed files

# EXTERNAL DATASET (not produced by this pipeline): the Nobel-prize publication
# records ({Medicine,Chemistry,Physics} publication record.csv). They are kept
# under META_DIRE alongside the processed files; repoint this if you store them
# elsewhere. Read by award_analysis/award_quantile_analysis.py.
NOBEL_DATA_DIRE = os.path.join(META_DIRE, 'nobel_prize')
