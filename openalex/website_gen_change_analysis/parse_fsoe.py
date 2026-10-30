"""
parse_fsoe.py
Parses data/FSOe-2026/FSOe-2026.txt into data/FSOe-2026/FSOe-2026.csv

The source .txt is already comma-delimited with double-quote enclosure (21 fields).
This script:
  1. Reads it via Python's csv module (handles embedded newlines / quotes correctly).
  2. Strips the leading space from Vendor values like " https://" -> "https://".
  3. When Vendor is a URL prefix (http:// or https://), reconstructs the full URL
     by concatenating Vendor + Datafle into a new `URL` column, and clears Vendor/Datafle.
  4. Writes a clean CSV to the output path.
"""

import csv
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)             # utils.py is in the parent directory
from utils import FSO_DIRE

FSO_data = "FSOe-2026.txt"
INPUT_PATH  = os.path.join(FSO_DIRE, FSO_data)                          # external dataset (see utils.py)
OUTPUT_PATH = os.path.join(FSO_DIRE, FSO_data.replace(".txt", ".csv"))

VENDOR_URL_PREFIXES = ("http://", "https://")


def is_url_vendor(vendor: str) -> bool:
    v = vendor.strip()
    return v.startswith(VENDOR_URL_PREFIXES)


def main():
    rows_written = 0

    with (
        open(INPUT_PATH,  "r", encoding="utf-8", errors="replace", newline="") as fin,
        open(OUTPUT_PATH, "w", encoding="utf-8", newline="")                    as fout,
    ):
        reader = csv.reader(fin)
        writer = csv.writer(fout, quoting=csv.QUOTE_ALL)

        header = next(reader)
        # Add URL column after Datafle (index 8); original cols stay for traceability
        out_header = header + ["URL"]
        writer.writerow(out_header)

        for row in reader:
            vendor  = row[7]   # Vendor field
            datafle = row[8]   # Datafle field

            # Strip leading/trailing whitespace from Vendor
            row[7] = vendor.strip()

            # Reconstruct full URL when Vendor is a protocol prefix
            if is_url_vendor(vendor):
                url = vendor.strip() + datafle.strip()
            else:
                url = ""

            row.append(url)
            writer.writerow(row)
            rows_written += 1

    print(f"Done. {rows_written:,} data rows written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
