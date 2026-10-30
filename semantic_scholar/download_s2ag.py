"""
Download a Semantic Scholar Academic Graph (S2AG) dataset (e.g. "papers",
"citations", "s2orc", ...) from the Semantic Scholar bulk-datasets API into
S2AG_DIRE/<DATASET_NAME>/. Looks up the latest release, lists the dataset's file
shards, and downloads each (skipping any already present, so it is resumable).

Provide your own API key by setting SEMANTIC_SCHOLAR_API_KEY in utils.py, and set
DATASET_NAME to the dataset you want; the downstream pipeline needs "papers"
(get_s2ag_meta.py), "citations" (get_s2ag_citation.py), and "s2orc"
(full_text_analysis/get_s2orc_meta.py). Run once per dataset.
"""

import os
import sys
from urllib.parse import urlparse

import requests
import wget
from tqdm import tqdm

# utils.py is in this same directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from utils import S2AG_DIRE, SEMANTIC_SCHOLAR_API_KEY  # noqa: E402

# Semantic Scholar API key — set it in utils.py (SEMANTIC_SCHOLAR_API_KEY).
API_KEY = SEMANTIC_SCHOLAR_API_KEY

# which S2AG dataset to download. Options include:
#   papers, citations, authors, abstracts, tldrs, embeddings-specter_v2, s2orc
# The downstream pipeline needs "papers" and "citations".
DATASET_NAME = "s2orc"
print("DATASET_NAME", DATASET_NAME)

LOCAL_PATH = os.path.join(S2AG_DIRE, DATASET_NAME)
os.makedirs(LOCAL_PATH, exist_ok=True)

# 1. latest release
response = requests.get(
    "https://api.semanticscholar.org/datasets/v1/release/latest"
).json()
RELEASE_ID = response["release_id"]
print(f"Latest release ID: {RELEASE_ID}")

# 2. dataset metadata
response = requests.get(
    f"https://api.semanticscholar.org/datasets/v1/release/{RELEASE_ID}/dataset/{DATASET_NAME}/",
    headers={"x-api-key": API_KEY}
).json()

print(f"Starting download of {DATASET_NAME} dataset...")

for url in tqdm(response["files"]):
    filename = os.path.basename(urlparse(url).path)
    save_path = os.path.join(LOCAL_PATH, filename)

    if os.path.exists(save_path):
        # already downloaded, skip
        continue

    try:
        wget.download(url, out=save_path)
    except Exception as e:
        print(f"\nError downloading {url}: {e}")
        raise

print("\nDownloaded all shards (skipped existing files).")
