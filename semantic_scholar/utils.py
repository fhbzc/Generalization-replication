
import os

# anchor relative paths to this file's directory so scripts run regardless of CWD
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# may be written as either a relative path (anchored to this file's directory) or an absolute path
S2AG_DIRE = os.path.join(_BASE_DIR, '/path/to/s2ag') # the directory where the Semantic Scholar Academic Graph (S2AG) dataset is saved

META_DIRE = os.path.join(_BASE_DIR, 'data', 's2ag_processed') # the directory where you want to save the processed files

S2ORC_META_DIRE = os.path.join(_BASE_DIR, 'data', 's2orc_processed') # the directory where the S2ORC-derived full-text metadata is saved

# fastText language-identification model (lid.176.bin), used by academic_keyword_analysis to
# keep only English title+abstract text. Download from
# https://fasttext.cc/docs/en/language-identification.html
# may be written as either a relative path (anchored to this file's directory) or an absolute path
LANG_DETECT_MODEL = os.path.join(_BASE_DIR, '/path/to/lid.176.bin')

# Semantic Scholar API key, required by download_s2ag.py to download the S2AG / S2ORC
# bulk datasets. Get one at https://www.semanticscholar.org/product/api
SEMANTIC_SCHOLAR_API_KEY = '' # replace with your Semantic Scholar API key
