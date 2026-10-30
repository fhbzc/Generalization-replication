"""
F / E / G / D citation-purpose analysis on the ACT2 dataset, via the OpenAlex
citation network -- mirrors efg_citation_purpose_openalex.py but the purpose
comes from ACT2's citation_class_label and the paper ids come from
match_act2_openalex.py.

Pipeline
--------
1. Load act2_match.pkl (from match_act2_openalex.py). Keep only instances where
   BOTH the citing and cited paper matched EXACTLY ONE OpenAlex id.
2. For each such instance A (citing) -> B (cited), classify with the OpenAlex
   citation network (citing_pid2cited = references, cited_pid2citing = citations):
       A_refs   = references of A
       B_refs   = references of B
       B_cites  = citations of B
       up   = |A_refs & B_refs|     down = |A_refs & B_cites|
       F : down >= up and down > 0       E : down < up
       G : up == 0 and down == 0
       D (disruptive) : up == 0          non_disruptive : up >= 1
   The edge is only classified if OpenAlex records A -> B (oa_B in A_refs).
3. Purpose = ACT2 citation_class_label, mapped 0-5 ->
       0 BACKGROUND, 1 USES, 2 CompareOrContrast, 3 MOTIVATION,
       4 EXTENSION, 5 FUTURE
4. Write crosstabs / Welch t-tests / per-citation frequency CSVs in the schema
   efg_citation_purpose_act2_s2_plot.py expects.

Binary toggle (REQUIRE_REFS)
----------------------------
Set REQUIRE_REFS = True (below) to run the alternative version in which every
citing OR cited paper with NO references in the OpenAlex network is removed:
an instance A -> B is only counted when BOTH A and B have >= 1 reference, and
only the remaining instances are tallied. Its outputs are written to a separate,
suffixed directory (nlp_purpose_analysis_act2_hasref) so the two versions
coexist. The shared act2_match.pkl is always read from the base (unsuffixed)
directory because matching is reference-independent.

Outputs (META_DIRE/nlp_purpose_analysis_act2[_hasref]):
  efg_purpose_crosstab.csv          type, purpose, instance_count, ratio_within_type  (e/f/g)
  disruptive_purpose_crosstab.csv   same, for disruptive / non_disruptive
  avg_instances_per_citation.csv    scheme, type, n_instances, n_citations, avg.., var.., std..
  efg_purpose_welch_ttest.csv       per-purpose pairwise Welch tests (e/f/g)
  disruptive_purpose_welch_ttest.csv
  act2_efgd_stats.csv               run summary
"""

import os
import sys
import csv
import math
import pickle
from collections import defaultdict, Counter

from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

# utils.py is in the parent directory; make it importable regardless of CWD.
sys.path.insert(0, PARENT_DIR)
from utils import META_DIRE  # noqa: E402

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))

meta_dire = META_DIRE

# ---------------------------------------------------------------------------
# Binary toggle (see module docstring). When True, only count citation
# instances where BOTH the citing and the cited paper have at least one
# reference in the OpenAlex network -- papers with no references are removed and
# only the remaining instances are tallied. Results go to a separate, suffixed
# output directory so the two versions coexist; the shared act2_match.pkl is
# always read from the base (unsuffixed) directory because matching is
# reference-independent.
# ---------------------------------------------------------------------------
REQUIRE_REFS = True
SUFFIX = "_hasref" if REQUIRE_REFS else ""

BASE_OUT_DIR = os.path.join(meta_dire, "nlp_purpose_analysis_act2")
OUT_DIR = os.path.join(meta_dire, "nlp_purpose_analysis_act2" + SUFFIX)
MATCH_PATH = os.path.join(BASE_OUT_DIR, "act2_match.pkl")
CITING2CITED_PATH = os.path.join(meta_dire, "citing_pid2cited_pids_list.pkl")
CITED2CITING_PATH = os.path.join(meta_dire, "cited_pid2citing_pids_list.pkl")

# ACT2 citation_class_label -> purpose name
LABEL2PURPOSE = {
    "0": "Background",
    "1": "Uses",
    "2": "CompareOrContrast",
    "3": "Motivation",
    "4": "Extends",
    "5": "Future",
}

EFG_TYPES = ("f", "e", "g")
EFG_PAIRS = (("f", "e"), ("f", "g"), ("e", "g"))
DISR_TYPES = ("disruptive", "non_disruptive")
DISR_PAIRS = (("disruptive", "non_disruptive"),)


def welch_ttest_binary(n1, k1, n2, k2):
    """Two-sided Welch t-test between two 0/1 samples given as (size, #ones).
    Returns (t_stat, df, p_value)."""
    if n1 < 2 or n2 < 2:
        return 0.0, 0.0, 1.0
    p1, p2 = k1 / n1, k2 / n2
    v1 = n1 * p1 * (1 - p1) / (n1 - 1)
    v2 = n2 * p2 * (1 - p2) / (n2 - 1)
    se1, se2 = v1 / n1, v2 / n2
    denom = math.sqrt(se1 + se2)
    if denom == 0:
        return 0.0, 0.0, 1.0
    t = (p1 - p2) / denom
    num = (se1 + se2) ** 2
    dd = (se1 ** 2) / (n1 - 1) + (se2 ** 2) / (n2 - 1)
    df = num / dd if dd > 0 else 0.0
    try:
        from scipy import stats
        p = 2.0 * float(stats.t.sf(abs(t), df))
    except Exception:
        p = math.erfc(abs(t) / math.sqrt(2.0))
    return t, df, p


def significance(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def write_crosstab(path, types, type_counts, type_purpose_counts, purposes):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "purpose", "instance_count", "ratio_within_type"])
        for t in types:
            t_total = type_counts.get(t, 0)
            for p in purposes:
                cnt = type_purpose_counts[t].get(p, 0)
                ratio = cnt / t_total if t_total else 0.0
                w.writerow([t, p, cnt, f"{ratio:.6f}"])
    print(f"Wrote cross-tab CSV: {path}")


def write_welch(path, types, pairs, type_counts, type_purpose_counts, purposes):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["purpose", "type_a", "type_b", "n_a", "ones_a", "n_b",
                    "ones_b", "ratio_a", "ratio_b", "t_stat", "df", "p_value",
                    "significance"])
        for p in purposes:
            for a, b in pairs:
                na, nb = type_counts.get(a, 0), type_counts.get(b, 0)
                ka = type_purpose_counts[a].get(p, 0)
                kb = type_purpose_counts[b].get(p, 0)
                t, df, pv = welch_ttest_binary(na, ka, nb, kb)
                ra = ka / na if na else 0.0
                rb = kb / nb if nb else 0.0
                w.writerow([p, a, b, na, ka, nb, kb, f"{ra:.6f}", f"{rb:.6f}",
                            f"{t:.6f}", f"{df:.4f}", f"{pv:.6g}",
                            significance(pv)])
    print(f"Wrote Welch t-test CSV: {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Writing results to: {OUT_DIR}")
    print(f"REQUIRE_REFS = {REQUIRE_REFS} "
          f"({'only instances where BOTH papers have references' if REQUIRE_REFS else 'all instances'})")

    # ---- matches: keep instances where BOTH sides map to exactly one pid ----
    with open(MATCH_PATH, "rb") as f:
        match = pickle.load(f)
    print(f"  ACT2 instances in match map: {len(match)}")

    # ---- OpenAlex citation network ----
    print("Loading OpenAlex reference / citation network ...")
    with open(CITING2CITED_PATH, "rb") as f:
        citing2cited = pickle.load(f)   # oa id -> [references]
    with open(CITED2CITING_PATH, "rb") as f:
        cited2citing = pickle.load(f)   # oa id -> [citations]
    print(f"  citing_pid2cited: {len(citing2cited)}  "
          f"cited_pid2citing: {len(cited2citing)}")

    refs_cache = {}
    edge_cache = {}

    def get_refs(oa):
        r = refs_cache.get(oa)
        if r is None:
            r = set(citing2cited.get(oa, ()))
            refs_cache[oa] = r
        return r

    def classify(oa_a, oa_b):
        a_refs = get_refs(oa_a)
        if oa_b not in a_refs:        # OpenAlex does not record A -> B
            return None
        b_refs = get_refs(oa_b)
        b_cites = set(cited2citing.get(oa_b, ()))
        up = len(a_refs & b_refs)
        down = len(a_refs & b_cites)
        if down == 0 and up == 0:
            e = "g"
        elif down >= up:
            e = "f"
        else:
            e = "e"
        d = "non_disruptive" if up >= 1 else "disruptive"
        return e, d

    # ---- tallies ----
    type_counts = Counter()
    purpose_counts = Counter()
    type_purpose_counts = defaultdict(Counter)
    disr_counts = Counter()
    disr_purpose_counts = defaultdict(Counter)
    pair_type = {}
    pair_disr = {}
    pair_inst_count = Counter()

    total = len(match)
    skip_not_both_unique = 0
    both_unique = 0
    skip_unknown_label = 0
    skip_no_oa_edge = 0
    skip_no_refs = 0
    counted = 0

    for uid, m in tqdm(match.items(), desc="Classifying ACT2 via OpenAlex"):
        citing_pids = m.get("citing_pids", [])
        cited_pids = m.get("cited_pids", [])
        if len(citing_pids) != 1 or len(cited_pids) != 1:
            skip_not_both_unique += 1
            continue
        both_unique += 1

        purpose = LABEL2PURPOSE.get(str(m.get("class_label", "")).strip())
        if purpose is None:
            skip_unknown_label += 1
            continue

        oa_a, oa_b = citing_pids[0], cited_pids[0]

        # Binary toggle: drop the instance if either the citing or the cited
        # paper has NO references in the OpenAlex network. (For counted
        # instances the citing paper always has references -- oa_b must be in
        # A_refs -- so in practice this removes cited papers with no references,
        # but both sides are checked for clarity.)
        if REQUIRE_REFS and (not get_refs(oa_a) or not get_refs(oa_b)):
            skip_no_refs += 1
            continue

        edge = (oa_a, oa_b)
        cls = edge_cache.get(edge, "NA")
        if cls == "NA":
            cls = classify(oa_a, oa_b)
            edge_cache[edge] = cls
        if cls is None:
            skip_no_oa_edge += 1
            continue
        efg_t, disr_t = cls

        counted += 1
        purpose_counts[purpose] += 1
        type_counts[efg_t] += 1
        type_purpose_counts[efg_t][purpose] += 1
        disr_counts[disr_t] += 1
        disr_purpose_counts[disr_t][purpose] += 1

        pair_type[edge] = efg_t
        pair_disr[edge] = disr_t
        pair_inst_count[edge] += 1

    purposes = [LABEL2PURPOSE[k] for k in sorted(LABEL2PURPOSE)]

    # ---------------------------------------------------------------- summary
    print("\n" + "=" * 70)
    print("ACT2 OPENALEX-NETWORK F/E/G/D - SUMMARY")
    print("=" * 70)
    stats = [
        ("require_refs", int(REQUIRE_REFS)),
        ("total_instances", total),
        ("skip_not_both_exactly_one", skip_not_both_unique),
        ("both_uniquely_mapped", both_unique),
        ("skip_unknown_class_label", skip_unknown_label),
        ("skip_no_references", skip_no_refs),
        ("skip_no_openalex_edge", skip_no_oa_edge),
        ("counted_instances", counted),
        ("unique_citation_pairs", len(pair_type)),
    ]
    for name, val in stats:
        print(f"  {name:<28s}: {val}")
    stats_path = os.path.join(OUT_DIR, "act2_efgd_stats.csv")
    with open(stats_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(stats)
    print(f"Wrote dataset stats CSV: {stats_path}")

    # ---- per-citation frequency (mean/var/std of #instances per pair) ----
    efg_pc = Counter(); efg_sum = Counter(); efg_sq = Counter()
    disr_pc = Counter(); disr_sum = Counter(); disr_sq = Counter()
    for edge in pair_type:
        n = pair_inst_count[edge]
        et, dt = pair_type[edge], pair_disr[edge]
        efg_pc[et] += 1; efg_sum[et] += n; efg_sq[et] += n * n
        disr_pc[dt] += 1; disr_sum[dt] += n; disr_sq[dt] += n * n
    avg_rows = []
    for scheme, pc, sm, sq, types in (
        ("EFG", efg_pc, efg_sum, efg_sq, EFG_TYPES),
        ("disruptive", disr_pc, disr_sum, disr_sq, DISR_TYPES),
    ):
        for t in types:
            n_cit = pc.get(t, 0); sx = sm.get(t, 0); sx2 = sq.get(t, 0)
            mean = sx / n_cit if n_cit else 0.0
            var = (sx2 - sx * sx / n_cit) / (n_cit - 1) if n_cit > 1 else 0.0
            std = math.sqrt(var) if var > 0 else 0.0
            avg_rows.append([scheme, t, sx, n_cit, f"{mean:.6f}",
                             f"{var:.6f}", f"{std:.6f}"])
    avg_path = os.path.join(OUT_DIR, "avg_instances_per_citation.csv")
    with open(avg_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheme", "type", "n_instances", "n_citations",
                    "avg_instances_per_citation", "var_instances_per_citation",
                    "std_instances_per_citation"])
        w.writerows(avg_rows)
    print(f"Wrote avg-instances CSV: {avg_path}")

    # ---- crosstabs + Welch t-tests ----
    write_crosstab(os.path.join(OUT_DIR, "efg_purpose_crosstab.csv"),
                   EFG_TYPES, type_counts, type_purpose_counts, purposes)
    write_crosstab(os.path.join(OUT_DIR, "disruptive_purpose_crosstab.csv"),
                   DISR_TYPES, disr_counts, disr_purpose_counts, purposes)
    write_welch(os.path.join(OUT_DIR, "efg_purpose_welch_ttest.csv"),
                EFG_TYPES, EFG_PAIRS, type_counts, type_purpose_counts, purposes)
    write_welch(os.path.join(OUT_DIR, "disruptive_purpose_welch_ttest.csv"),
                DISR_TYPES, DISR_PAIRS, disr_counts, disr_purpose_counts, purposes)

    print("\nPurpose distribution (counted instances):")
    for p in purposes:
        print(f"  {p:<18s}: {purpose_counts.get(p, 0)}")
    print("finished")


if __name__ == "__main__":
    main()
