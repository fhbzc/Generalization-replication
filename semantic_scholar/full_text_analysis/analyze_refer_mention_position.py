"""
Aggregate mention-position stats for BOTH classification systems (efg + disp),
OVERALL and PER cited-paper DOMAIN.

Reads refer_mention_records_V2.pkl (one record per citation: efg, disp, cid,
year, pos) and, for each panel (Frequency + each mention position), computes the
per-type mean and pairwise two-sided Welch t-tests, for citation types:

    e, f, g                       (efg system)
    disruptive, non_disruptive    (disp system, independent axis)
    disp_f_0, disp_f_1, disp_f_2  (disruptive-foundation variants: up_count == 0 and
                                   down_count > 0/1/2; each citation may join several)

Domains come from the CITED paper's cid via META_DIRE/cid2category_list.pkl,
normalised the same way as get_coverage_text_distribution.py ('And' -> 'and').
A citation whose cited paper has several domains contributes to each of them.

Binary toggle (REQUIRE_REFS)
----------------------------
Set REQUIRE_REFS = True (below) to run the alternative version in which every
citing OR cited paper with NO references in the S2AG citation network is removed,
keeping only citations where BOTH have >= 1 reference. The CITING side is already
guaranteed: identify_citation_property.py only keeps an instance when the citing
paper cites the cited one, so its reference set is non-empty. The CITED side is
enforced here using each record's cid against the reference map
META_DIRE/citing2cited_cid_list_remove_circle.pkl (the same map used to build the
records; loaded only when REQUIRE_REFS). Outputs go to a separate, suffixed
directory (analysis_V2_hasref) so the two versions coexist.

Outputs (in S2ORC_META_DIRE/analysis_V2[_hasref]/):
  efgd_full_stats.csv                      OVERALL (all citations; back-compat)
  efgd_by_domain/<domain>_stats.csv        one per domain

before_intro and after_conclusion ARE counted in each citation's total reference
count (the Frequency denominator) and get their own mean/ratio rows in the CSV,
but the figure does not draw them.
"""

import os
import sys
import re
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import ttest_ind
from tqdm import tqdm

# utils.py is in the parent directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
from utils import S2ORC_META_DIRE, META_DIRE  # noqa: E402

meta_dire = S2ORC_META_DIRE
pkl_path = os.path.join(meta_dire, 'refer_mention_records_V2.pkl')

# ---------------------------------------------------------------------------
# Binary toggle (see module docstring). When True, only keep citations where
# BOTH the citing and the cited paper have at least one reference in the S2AG
# citation network. The citing side is guaranteed by construction (records only
# exist for citing papers that cite the cited one); the cited side is enforced
# here via the record's cid against citing2cited_cid_list_remove_circle.pkl.
# Results go to a separate, suffixed directory so the two versions coexist.
# ---------------------------------------------------------------------------
REQUIRE_REFS = True
SUFFIX = "_hasref" if REQUIRE_REFS else ""
CITING2CITED_PATH = os.path.join(META_DIRE,
                                 'citing2cited_cid_list_remove_circle.pkl')

save_dir = os.path.join(meta_dire, 'analysis_V2' + SUFFIX)
os.makedirs(save_dir, exist_ok=True)
out_csv = os.path.join(save_dir, 'efgd_full_stats.csv')          # overall
by_domain_dir = os.path.join(save_dir, 'efgd_by_domain')
os.makedirs(by_domain_dir, exist_ok=True)

# minimum citations of a type within a domain for that type to be reported
MIN_CITATIONS_PER_DOMAIN = 5

# before_intro & after_conclusion are INCLUDED so they count toward each
# citation's total reference count (the Frequency denominator) and get their own
# ratio in the CSV. The plot script has its own hard-coded PANELS and simply does
# not draw these two.
positions = ["before_intro", "intro", "method", "result", "res_dis",
             "discussion", "conclusion", "after_conclusion"]
# disruptive-foundation variants (up_count == 0 & down_count > T), mirroring
# openalex/get_efg_v2_disp_f.py's DISP_F_DOWN_THRESHOLDS = [0, 1, 2]. Each record carries a
# 'disp_f' list of the variants it belongs to; a citation can be in several (disp_f_2 ⊂
# disp_f_1 ⊂ disp_f_0), and disp_f_0 is exactly the 'f' subset of the disruptive citations.
DISP_F_TYPES = ["disp_f_0", "disp_f_1", "disp_f_2"]

# types drawn in the figure (e/f/g from efg, disruptive from disp; non_disruptive
# kept for the disp-system comparison; the disp_f variants are extra bars)
mean_types = ["e", "f", "g", "disruptive", "non_disruptive"] + DISP_F_TYPES
welch_pairs = [
    ("e", "f"), ("e", "g"), ("e", "disruptive"),
    ("f", "g"), ("f", "disruptive"), ("g", "disruptive"),
    ("disruptive", "non_disruptive"),
    ("disruptive", "disp_f_0"), ("disp_f_0", "disp_f_1"), ("disp_f_1", "disp_f_2"),
]


def edges_to_matrix(edges, positions):
    n = len(edges)
    fre_mat = np.zeros((n, len(positions)), dtype=float)
    for i, d in enumerate(edges):
        for j, p in enumerate(positions):
            fre_mat[i, j] = float(d.get(p, 0.0))
    mention_sum = fre_mat.sum(axis=1)
    ratio_mat = np.zeros_like(fre_mat)
    mask = mention_sum > 0
    ratio_mat[mask] = fre_mat[mask] / mention_sum[mask, None]
    return mention_sum, ratio_mat


def welch_test(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan"), float("nan")
    res = ttest_ind(a, b, equal_var=False)
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    n1, n2 = len(a), len(b)
    denom = ((s1 ** 2) / ((n1 ** 2) * (n1 - 1)) +
             (s2 ** 2) / ((n2 ** 2) * (n2 - 1)))
    df = (s1 / n1 + s2 / n2) ** 2 / denom if denom > 0 else float("nan")
    return res.statistic, df, res.pvalue


def compute_stats(records, min_citations=0):
    """Build the long-format stats rows for one set of records. A type with
    fewer than `min_citations` records is dropped (its bars/tests omitted)."""
    type2edges = defaultdict(list)
    for r in records:
        type2edges[r['efg']].append(r['pos'])
        if r.get('disp') is not None:
            type2edges[r['disp']].append(r['pos'])
        # a citation contributes to every disp_f variant it qualifies for
        for dft in (r.get('disp_f') or []):
            type2edges[dft].append(r['pos'])

    type2mat = {t: edges_to_matrix(type2edges[t], positions) for t in mean_types}
    usable = {t for t in mean_types if len(type2edges[t]) >= min_citations}

    rows = []
    for panel in ["Frequency"] + positions:
        group_data = {}
        for t in mean_types:
            if t not in usable:
                continue
            mention_sum, ratio_mat = type2mat[t]
            if panel == "Frequency":
                arr = mention_sum
            else:
                arr = ratio_mat[:, positions.index(panel)]
            group_data[t] = arr
            rows.append({"panel": panel, "type": "mean", "group1": t,
                         "group2": "",
                         "value": arr.mean() if len(arr) else float("nan"),
                         # sample variance (ddof=1) so the plot can draw a 95% CI of the mean
                         "var": float(np.var(arr, ddof=1)) if len(arr) > 1 else float("nan"),
                         "n": len(arr), "t_value": "", "df": "", "p_value": ""})

        for a, b in welch_pairs:
            if a not in group_data or b not in group_data:
                continue
            t_val, df_val, p_val = welch_test(group_data[a], group_data[b])
            rows.append({"panel": panel, "type": "welch", "group1": a,
                         "group2": b, "value": "", "var": "", "n": "",
                         "t_value": t_val, "df": df_val, "p_value": p_val})
    return rows


def safe_name(domain):
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(domain)).strip("_") or "domain"


if __name__ == "__main__":
    with open(pkl_path, 'rb') as f:
        records = pickle.load(f)
    print("records:", len(records))
    print("REQUIRE_REFS =", REQUIRE_REFS,
          "(only citations where BOTH papers have references)"
          if REQUIRE_REFS else "(all citations)")

    # ---- binary toggle: drop citations whose CITED paper has no references ----
    # (the citing paper always has references for records that exist; see docstring)
    if REQUIRE_REFS:
        with open(CITING2CITED_PATH, 'rb') as f:
            pid2up_list = pickle.load(f)   # cid -> [references]
        print("  loaded reference map:", len(pid2up_list), "entries")
        n_before = len(records)
        records = [r for r in records if pid2up_list.get(r.get('cid'))]
        print(f"  kept {len(records)} / {n_before} records "
              f"(dropped {n_before - len(records)} with a cited paper lacking references)")

    # ---- OVERALL (all citations) -> back-compatible efgd_full_stats.csv ----
    pd.DataFrame(compute_stats(records, min_citations=0)).to_csv(out_csv,
                                                                 index=False)
    print("Saved (overall):", out_csv)

    # ---- per cited-paper domain ----
    with open(os.path.join(META_DIRE, 'cid2category_list.pkl'), 'rb') as f:
        cid2category_list = pickle.load(f)
    # normalise domain names like get_coverage_text_distribution.py
    cid2domains = {cid: {c.replace('And', 'and') for c in cats}
                   for cid, cats in cid2category_list.items()}

    domain2records = defaultdict(list)
    for r in tqdm(records, desc="Grouping by domain"):
        for domain in cid2domains.get(r.get('cid'), ()):
            domain2records[domain].append(r)
    print("domains:", len(domain2records))

    n_written = 0
    for domain in tqdm(sorted(domain2records), desc="Domains"):
        rows = compute_stats(domain2records[domain],
                             min_citations=MIN_CITATIONS_PER_DOMAIN)
        if not rows:
            continue  # no type reached the threshold in this domain
        path = os.path.join(by_domain_dir, f'{safe_name(domain)}_stats.csv')
        pd.DataFrame(rows).to_csv(path, index=False)
        n_written += 1

    print(f"Saved {n_written} per-domain stats CSVs in {by_domain_dir}")
    print("finished")
