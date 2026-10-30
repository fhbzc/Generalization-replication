"""
Count how many papers appearing in the ACL-ARC citation graph can be matched to
an OpenAlex id, via TITLE matching.

Why title matching (not id lookup)?
-----------------------------------
pid2title.pkl (built by ../get_meta_data_preprint.py) maps OpenAlex work id ->
title. The ARC paper ids (P13-1037, External_89521, ...) live in a different id
space, so the only join key is the paper *title*. A title can be shared by
several OpenAlex works, which is exactly why a paper can have 0, exactly 1, or
multiple OpenAlex matches.

Pipeline
--------
1. Collect every paper id appearing in EITHER dataset (their UNION): the citation
   graph acl-arc.citation-graph.with-functions.tsv (citing + cited columns) and
   arc-paper-ids.tsv. Internal AND external.
2. Look up each paper's title in arc-paper-ids.tsv (id -> year, title, authors).
   These titles are already lower-cased / normalised. (Graph-only ids absent from
   arc-paper-ids have no title and therefore cannot be matched.)
3. Invert pid2title.pkl into  normalised_title -> {openalex_id, ...}  (restricted
   to the titles we actually need, to keep memory small).
4. For each paper, normalise its title and count how many OpenAlex ids it maps to.

Title normalisation: lower-case, keep only [a-z0-9] (drop spaces & punctuation).
This makes "Coarse-to-fine", "coarse to fine" and "coarsetofine" all collapse to
the same key, so the ARC (already-normalised) titles and the raw OpenAlex titles
line up robustly.

Note: paper ids are matched case-insensitively (ids are lower-cased on the join).

Binary toggle (REQUIRE_REFS)
----------------------------
Matching is reference-independent, so arc_id2openalex_pids.pkl /
openalex_match_summary.csv / openalex_match_per_paper.csv are ALWAYS written to
the base directory (nlp_purpose_analysis_openalex) and shared by both versions.
When REQUIRE_REFS = True (below) the script additionally loads the OpenAlex
reference map and writes openalex_match_summary.csv into the suffixed directory
(nlp_purpose_analysis_openalex_hasref) with an extra exactly_one_with_refs
column -- how many uniquely-matched papers also carry >= 1 reference, the
per-paper precondition for the instance-level "both papers have references"
filter used by the REQUIRE_REFS run of efg_citation_purpose_openalex.py.
"""

import os
import sys
import csv
import re
import pickle
from collections import defaultdict, Counter

from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

# utils.py is in the parent directory; make it importable regardless of CWD.
sys.path.insert(0, PARENT_DIR)
from utils import META_DIRE, NLP_PURPOSE_DIRE  # noqa: E402

meta_dire = META_DIRE
DATA_DIR = NLP_PURPOSE_DIRE                                         # external input data (see utils.py)

# ---------------------------------------------------------------------------
# Binary toggle (see module docstring). Matching is reference-independent, so
# the match outputs always go to the base directory; when True we additionally
# load the reference map and write a per-group exactly_one_with_refs count
# (uniquely-matched papers carrying >= 1 reference) to the suffixed directory.
# ---------------------------------------------------------------------------
REQUIRE_REFS = True
SUFFIX = "_hasref" if REQUIRE_REFS else ""

OUT_DIR = os.path.join(meta_dire, "nlp_purpose_analysis_openalex")  # output (same as efg_citation_purpose_openalex.py)
ANALYSIS_OUT_DIR = os.path.join(meta_dire, "nlp_purpose_analysis_openalex" + SUFFIX)
GRAPH_PATH = os.path.join(DATA_DIR, "acl-arc.citation-graph.with-functions.tsv")
PIDS_PATH = os.path.join(DATA_DIR, "arc-paper-ids.tsv")     # id -> year, title, authors
PID2TITLE_PATH = os.path.join(meta_dire, "pid2title.pkl")   # openalex id -> title
CITING2CITED_PATH = os.path.join(meta_dire, "citing_pid2cited_pids_list.pkl")  # references

EXTERNAL_PREFIX = "External_"

_keep = re.compile(r"[^a-z0-9]+")


def norm_title(s):
    """lower-case and keep only [a-z0-9] (drop spaces & punctuation)."""
    if not s:
        return ""
    return _keep.sub("", s.lower())


def is_external(pid):
    return pid.startswith(EXTERNAL_PREFIX)


def count_lines(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in f)


def collect_graph_paper_ids(path):
    """Set of every paper id appearing in the citation graph.

    Ids are matched STRICTLY (case-sensitive); only the title is normalised /
    lower-cased (in norm_title), since arc-paper-ids titles are already lower-case.
    """
    n_lines = count_lines(path)
    ids = set()
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in tqdm(reader, total=n_lines, desc="Scanning citation graph"):
            if len(row) < 2:
                continue
            ids.add(row[0])
            ids.add(row[1])
    return ids


def load_paper_titles(path):
    """From arc-paper-ids.tsv (col0=id, col2=title) return id -> normalised_title
    (id kept verbatim; every id retained, even with an empty title)."""
    n_lines = count_lines(path)
    arc = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in tqdm(reader, total=n_lines, desc="Loading ARC paper titles"):
            if not row:
                continue
            pid = row[0]
            nt = norm_title(row[2]) if len(row) >= 3 else ""
            if pid not in arc:
                arc[pid] = nt
            elif not arc[pid] and nt:
                arc[pid] = nt  # fill in a title if the first occurrence had none
    return arc


def build_openalex_title_index(path, wanted_titles):
    """norm_title -> set(openalex_id), restricted to titles in `wanted_titles`."""
    print(f"Loading OpenAlex pid2title from: {path}")
    with open(path, "rb") as f:
        oa_pid2title = pickle.load(f)
    print(f"  OpenAlex titles loaded: {len(oa_pid2title)}")

    index = defaultdict(set)
    for oa_id, title in tqdm(oa_pid2title.items(), desc="Indexing OpenAlex titles"):
        nt = norm_title(title)
        if nt and nt in wanted_titles:
            index[nt].add(oa_id)
    return index


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. paper ids in the citation graph (verbatim, case-sensitive)
    graph_ids = collect_graph_paper_ids(GRAPH_PATH)
    print(f"  paper ids in citation graph: {len(graph_ids)}")

    # 2. ids + titles from arc-paper-ids.tsv
    arc = load_paper_titles(PIDS_PATH)  # id -> norm_title
    print(f"  paper ids in arc-paper-ids.tsv: {len(arc)}")

    # 3. UNION of ids from either dataset (strict, case-sensitive). Title (if any)
    #    comes from arc-paper-ids.
    in_arc = set(arc)
    universe = set(graph_ids) | in_arc

    def source_of(pid):
        g, a = pid in graph_ids, pid in in_arc
        return "both" if (g and a) else ("graph_only" if g else "arc_only")

    n_total = len(universe)
    n_internal = sum(1 for pid in universe if not is_external(pid))
    print(f"  union of ids: {n_total} "
          f"(internal {n_internal}, external {n_total - n_internal})")

    paper_title = {pid: arc[pid] for pid in universe if arc.get(pid)}
    wanted_titles = set(paper_title.values())
    print(f"  ids with a usable title: {len(paper_title)}; "
          f"distinct titles: {len(wanted_titles)}")

    # 4. invert OpenAlex pid2title for just those titles
    oa_index = build_openalex_title_index(PID2TITLE_PATH, wanted_titles)

    # 4b. reference map (only needed for the REQUIRE_REFS version)
    citing2cited = None
    if REQUIRE_REFS:
        print(f"Loading {CITING2CITED_PATH} (reference filter) ...")
        with open(CITING2CITED_PATH, "rb") as f:
            citing2cited = pickle.load(f)   # oa id -> [references]
        print(f"  citing_pid2cited entries: {len(citing2cited)}")

    def has_refs(oa_id):
        """True if the paper has >= 1 reference in the OpenAlex network."""
        return bool(citing2cited.get(oa_id))

    # 5. count matches, split by internal/external; build the match map
    groups = {"all": Counter(), "internal": Counter(), "external": Counter()}
    per_paper_rows = []
    id2pid_list = {}   # arc/graph paper id -> list of matched OpenAlex pids (may be empty)
    for pid in tqdm(sorted(universe), desc="Matching papers to OpenAlex"):
        grp = "external" if is_external(pid) else "internal"
        nt = paper_title.get(pid)
        oa_ids = oa_index.get(nt, set()) if nt else set()
        n_match = len(oa_ids)
        # post-filter: uniquely-matched paper that also carries references
        one_with_refs = (REQUIRE_REFS and n_match == 1
                         and has_refs(next(iter(oa_ids))))

        for g in ("all", grp):
            c = groups[g]
            c["total"] += 1
            if nt:
                c["has_title"] += 1
            if n_match >= 1:
                c["matched"] += 1
            if n_match == 1:
                c["exactly_one"] += 1
            if n_match > 1:
                c["multiple"] += 1
            if one_with_refs:
                c["exactly_one_with_refs"] += 1

        # match map: keep every paper that had a title (empty list = searched, no match)
        if nt:
            id2pid_list[pid] = sorted(oa_ids)

        per_paper_rows.append([
            pid, grp, source_of(pid), 1 if nt else 0, n_match,
            "|".join(sorted(oa_ids)),
        ])

    # ----------------------------------------------------------------- report
    print("\n" + "=" * 70)
    print("OPENALEX TITLE-MATCH SUMMARY")
    print("=" * 70)
    header = ("group", "total", "has_title", "matched(>=1)",
              "exactly_one", "multiple(>1)")
    print(f"  {header[0]:<10}{header[1]:>10}{header[2]:>12}{header[3]:>14}"
          f"{header[4]:>13}{header[5]:>14}")
    for g in ("all", "internal", "external"):
        c = groups[g]
        print(f"  {g:<10}{c['total']:>10}{c['has_title']:>12}{c['matched']:>14}"
              f"{c['exactly_one']:>13}{c['multiple']:>14}")

    # summary CSV
    summary_path = os.path.join(OUT_DIR, "openalex_match_summary.csv")
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "total", "has_title", "matched_ge1",
                         "exactly_one", "multiple_gt1"])
        for g in ("all", "internal", "external"):
            c = groups[g]
            writer.writerow([g, c["total"], c["has_title"], c["matched"],
                             c["exactly_one"], c["multiple"]])
    print(f"\nWrote summary CSV: {summary_path}")

    # ---- REQUIRE_REFS: per-group exactly_one_with_refs in the suffixed dir ----
    if REQUIRE_REFS:
        os.makedirs(ANALYSIS_OUT_DIR, exist_ok=True)
        hasref_summary_path = os.path.join(ANALYSIS_OUT_DIR,
                                           "openalex_match_summary.csv")
        with open(hasref_summary_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["group", "total", "has_title", "matched_ge1",
                             "exactly_one", "multiple_gt1", "exactly_one_with_refs"])
            for g in ("all", "internal", "external"):
                c = groups[g]
                writer.writerow([g, c["total"], c["has_title"], c["matched"],
                                 c["exactly_one"], c["multiple"],
                                 c["exactly_one_with_refs"]])
        print(f"Wrote REQUIRE_REFS summary CSV: {hasref_summary_path}")

    # per-paper detail CSV
    detail_path = os.path.join(OUT_DIR, "openalex_match_per_paper.csv")
    with open(detail_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["paper_id", "group", "source", "has_title",
                         "n_openalex_matches", "openalex_ids"])
        writer.writerows(per_paper_rows)
    print(f"Wrote per-paper CSV: {detail_path}")

    # match map: arc paper id -> list of matched OpenAlex pids (pkl)
    match_path = os.path.join(OUT_DIR, "arc_id2openalex_pids.pkl")
    with open(match_path, "wb") as f:
        pickle.dump(id2pid_list, f, pickle.HIGHEST_PROTOCOL)
    n_nonempty = sum(1 for v in id2pid_list.values() if v)
    print(f"Wrote match-map PKL: {match_path} "
          f"({len(id2pid_list)} keys, {n_nonempty} with >=1 match)")


if __name__ == "__main__":
    main()
