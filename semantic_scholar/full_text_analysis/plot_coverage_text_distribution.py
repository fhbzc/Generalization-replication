"""
Fig S4: per-domain in-text citation coverage bar chart. Reads the coverage table
from get_coverage_text_distribution.py, computes coverage_ratio =
covered_citation_count / total_citation_count per domain (0 when the denominator is
0), renames a couple of long domain names, and draws one bar per domain (sorted
alphabetically).

Input  : S2ORC_META_DIRE/text_coverage_V2/coverage.csv   (from get_coverage_text_distribution.py)
Outputs: ./figures/coverage_text_distribution/Fig S4.svg (+ .csv)
"""

import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless / cluster-safe
import matplotlib.pyplot as plt  # noqa: E402

# utils.py is in the parent directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
from utils import S2ORC_META_DIRE  # noqa: E402

# try the reference font but fall back silently if unavailable (e.g. on Linux)
try:
    import matplotlib.font_manager as fm
    if any("Gill Sans MT" in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = "Gill Sans MT"
except Exception:
    pass

save_dire = os.path.join(BASE_DIR, 'figures', 'coverage_text_distribution')
os.makedirs(save_dire, exist_ok=True)
csv_path = os.path.join(S2ORC_META_DIRE, 'text_coverage_V2', "coverage.csv")
out_svg = os.path.join(save_dire, "Fig S4.svg")

df = pd.read_csv(csv_path)

# coverage ratio (guard against division by zero)
df["coverage_ratio"] = df["covered_citation_count"] / df["total_citation_count"].replace(0, pd.NA)
df["coverage_ratio"] = df["coverage_ratio"].fillna(0.0)

# category as string (for lexicographic sorting below)
df["category"] = df["category"].astype(str)

# rename a couple of long domain names
df["category"] = df["category"].replace({
    "Agricultural and Food Sciences": "Agriculture",
    "Environmental Science": "Environment",
})


df = df.sort_values("category")

# plot
plt.figure(figsize=(max(12, 0.35 * len(df)), 6))
plt.bar(df["category"], df["coverage_ratio"])
plt.ylabel("Coverage", fontsize=18)
plt.xlabel("Domain", fontsize=18)

plt.xticks(rotation=90, fontsize=14)
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(out_svg, format="svg", transparent = True)
plt.close()

out_csv = out_svg.replace('.svg', '.csv')
df.to_csv(out_csv, index=False)

print("Saved:", out_svg)
print("Saved:", out_csv)
