"""
Match every citation instance in ACT2_dataset.tsv to OpenAlex paper ids.

For each row (one citation instance = one citing paper -> one cited paper):
  * CITED paper : match by cited_doi FIRST (exact DOI lookup); if there is no
                  doi or the doi finds nothing, fall back to cited_title.
  * CITING paper: match by citing_title (ACT2 has no citing DOI).

A paper title/DOI may map to 0, exactly 1, or several OpenAlex ids -- which is
why a paper can have 0 / exactly-one / multiple matches. We report how many
instances find EXACTLY ONE match for the cited paper, the citing paper, and BOTH
(only the "both exactly one" instances are usable for the F/E/G/D analysis).

Binary toggle (REQUIRE_REFS)
----------------------------
Matching is reference-independent, so act2_match.pkl / act2_match_summary.csv /
act2_match_per_instance.csv are ALWAYS written to the base directory
(nlp_purpose_analysis_act2) and shared by both versions. When REQUIRE_REFS = True
(below) the script additionally loads the OpenAlex reference map and writes
act2_match_summary.csv into the suffixed directory (nlp_purpose_analysis_act2_hasref)
with the post-filter usable count -- i.e. how many "both exactly one" instances
also have BOTH papers carrying >= 1 reference (papers with no references removed).
That suffixed count matches the usable set of the REQUIRE_REFS runs of
efg_citation_purpose_act2.py / both_match_cited_concept_distribution.py.

Inputs:
  ACT2_PURPOSE_DIRE/ACT2_dataset.tsv   (EXTERNAL dataset; path set in ../utils.py)
  META_DIRE/pid2doi.pkl     openalex id -> bare DOI (already prefix-stripped, "10.xxx/...")
  META_DIRE/pid2title.pkl   openalex id -> title
  META_DIRE/citing_pid2cited_pids_list.pkl   oa id -> [references]  (only when REQUIRE_REFS)
Outputs (in META_DIRE/nlp_purpose_analysis_act2):
  act2_match.pkl              {unique_id -> {citing_pids:[...], cited_pids:[...],
                                             cited_match_by: 'doi'|'title'|None}}
  act2_match_summary.csv      counts (cited / citing / both, by exactly-one etc.)
  act2_match_per_instance.csv per-row detail
  (REQUIRE_REFS only) META_DIRE/nlp_purpose_analysis_act2_hasref/act2_match_summary.csv
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
from utils import META_DIRE, ACT2_PURPOSE_DIRE  # noqa: E402

meta_dire = META_DIRE
DATA_DIR = ACT2_PURPOSE_DIRE                            # external input data (see utils.py)

# ---------------------------------------------------------------------------
# Binary toggle (see module docstring). Matching is reference-independent, so
# the match outputs always go to the base directory; when True we additionally
# load the reference map and write a post-filter usable count (both papers with
# >= 1 reference) to the suffixed directory.
# ---------------------------------------------------------------------------
REQUIRE_REFS = True
SUFFIX = "_hasref" if REQUIRE_REFS else ""

OUT_DIR = os.path.join(meta_dire, "nlp_purpose_analysis_act2")
ANALYSIS_OUT_DIR = os.path.join(meta_dire, "nlp_purpose_analysis_act2" + SUFFIX)
TSV_PATH = os.path.join(DATA_DIR, "ACT2_dataset.tsv")
PID2DOI_PATH = os.path.join(meta_dire, "pid2doi.pkl")
PID2TITLE_PATH = os.path.join(meta_dire, "pid2title.pkl")
CITING2CITED_PATH = os.path.join(meta_dire, "citing_pid2cited_pids_list.pkl")

# allow huge single-field cells (abstracts) in the tsv
csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))

_keep = re.compile(r"[^a-z0-9]+")


def norm_title(s):
    """lower-case, keep only [a-z0-9] (drop spaces & punctuation)."""
    if not s:
        return ""
    return _keep.sub("", s.lower())


def norm_doi(s):
    """lower-case + strip a leading https://doi.org/ or doi.org/ prefix."""
    if not s:
        return ""
    s = s.strip().lower()
    for pref in ("https://doi.org/", "http://doi.org/", "doi.org/"):
        if s.startswith(pref):
            s = s[len(pref):]
            break
    return s


def load_instances(path):
    """Yield (unique_id, citing_title, cited_title, cited_doi, class_label)."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            yield (row["unique_id"], row.get("citing_title", ""),
                   row.get("cited_title", ""), row.get("cited_doi", ""),
                   row.get("citation_class_label", ""))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- 1. read ACT2: collect the titles / dois we need to resolve ----
    rows = list(load_instances(TSV_PATH))
    print(f"ACT2 instances: {len(rows)}")

    wanted_titles = set()
    wanted_dois = set()
    for _uid, citing_title, cited_title, cited_doi, _lbl in rows:
        if citing_title:
            wanted_titles.add(norm_title(citing_title))
        if cited_title:
            wanted_titles.add(norm_title(cited_title))
        d = norm_doi(cited_doi)
        if d:
            wanted_dois.add(d)
    wanted_titles.discard("")
    print(f"  distinct titles wanted: {len(wanted_titles)}; "
          f"distinct cited DOIs wanted: {len(wanted_dois)}")

    # ---- 2. build doi -> {pid} index (invert pid2doi, restricted to wanted) ----
    print(f"Loading {PID2DOI_PATH} ...")
    with open(PID2DOI_PATH, "rb") as f:
        pid2doi = pickle.load(f)
    print(f"  pid2doi entries: {len(pid2doi)}")
    doi2pids = defaultdict(set)
    for pid, doi in tqdm(pid2doi.items(), desc="Indexing OpenAlex DOIs"):
        nd = norm_doi(doi)
        if nd and nd in wanted_dois:
            doi2pids[nd].add(pid)
    del pid2doi
    print(f"  cited DOIs found in OpenAlex: {len(doi2pids)}")

    # ---- 3. build norm_title -> {pid} index (invert pid2title, restricted) ----
    print(f"Loading {PID2TITLE_PATH} ...")
    with open(PID2TITLE_PATH, "rb") as f:
        pid2title = pickle.load(f)
    print(f"  pid2title entries: {len(pid2title)}")
    title2pids = defaultdict(set)
    for pid, title in tqdm(pid2title.items(), desc="Indexing OpenAlex titles"):
        nt = norm_title(title)
        if nt and nt in wanted_titles:
            title2pids[nt].add(pid)
    del pid2title
    print(f"  wanted titles found in OpenAlex: {len(title2pids)}")

    # ---- 3b. reference map (only needed for the REQUIRE_REFS version) ----
    citing2cited = None
    if REQUIRE_REFS:
        print(f"Loading {CITING2CITED_PATH} (reference filter) ...")
        with open(CITING2CITED_PATH, "rb") as f:
            citing2cited = pickle.load(f)   # oa id -> [references]
        print(f"  citing_pid2cited entries: {len(citing2cited)}")

    def has_refs(pid):
        """True if the paper has >= 1 reference in the OpenAlex network."""
        return bool(citing2cited.get(pid))

    # ---- 4. resolve each instance ----
    match = {}
    per_instance = []
    cited_c = Counter()    # exactly_one / multiple / none for cited
    citing_c = Counter()   # ... for citing
    both_exactly_one = 0
    both_exactly_one_with_refs = 0   # both-exactly-one AND both papers have refs
    both_exactly_one_dropped = 0     # both-exactly-one but a paper has no refs
    cited_by = Counter()   # 'doi' / 'title' / 'none'

    for uid, citing_title, cited_title, cited_doi, label in tqdm(
            rows, desc="Matching ACT2 instances"):
        # cited: DOI first, else title
        cited_pids = set()
        by = None
        nd = norm_doi(cited_doi)
        if nd and nd in doi2pids:
            cited_pids = doi2pids[nd]
            by = "doi"
        if not cited_pids:
            nt_cited = norm_title(cited_title)
            if nt_cited and nt_cited in title2pids:
                cited_pids = title2pids[nt_cited]
                by = "title"
        if not cited_pids:
            by = None

        # citing: title only
        citing_pids = set()
        nt_citing = norm_title(citing_title)
        if nt_citing and nt_citing in title2pids:
            citing_pids = title2pids[nt_citing]

        nc, ng = len(cited_pids), len(citing_pids)

        def bucket(n):
            return "exactly_one" if n == 1 else ("multiple" if n > 1 else "none")

        cited_c[bucket(nc)] += 1
        citing_c[bucket(ng)] += 1
        cited_by[by or "none"] += 1
        if nc == 1 and ng == 1:
            both_exactly_one += 1
            # post-filter usable count for the REQUIRE_REFS version: keep only
            # the both-exactly-one instances where BOTH papers carry references.
            if REQUIRE_REFS:
                # nc == ng == 1, so each set holds exactly one pid.
                if has_refs(next(iter(cited_pids))) and \
                        has_refs(next(iter(citing_pids))):
                    both_exactly_one_with_refs += 1
                else:
                    both_exactly_one_dropped += 1

        match[uid] = {
            "citing_pids": sorted(citing_pids),
            "cited_pids": sorted(cited_pids),
            "cited_match_by": by,
            "class_label": label,
        }
        per_instance.append([
            uid, label, by or "none", nc, ng,
            "|".join(sorted(cited_pids)), "|".join(sorted(citing_pids)),
        ])

    # ----------------------------------------------------------------- report
    n = len(rows)
    print("\n" + "=" * 70)
    print("ACT2 -> OPENALEX MATCH SUMMARY")
    print("=" * 70)
    print(f"  total instances                 : {n}")
    print(f"  cited matched by DOI            : {cited_by['doi']}")
    print(f"  cited matched by title          : {cited_by['title']}")
    print(f"  cited not matched               : {cited_by['none']}")
    print(f"  cited exactly-one / multiple / none : "
          f"{cited_c['exactly_one']} / {cited_c['multiple']} / {cited_c['none']}")
    print(f"  citing exactly-one / multiple / none: "
          f"{citing_c['exactly_one']} / {citing_c['multiple']} / {citing_c['none']}")
    print(f"  BOTH exactly-one (usable)       : {both_exactly_one}")
    if REQUIRE_REFS:
        print(f"  ... with refs (REQUIRE_REFS)    : {both_exactly_one_with_refs}")
        print(f"  ... dropped (a paper no refs)   : {both_exactly_one_dropped}")

    # summary CSV
    summary_path = os.path.join(OUT_DIR, "act2_match_summary.csv")
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["total_instances", n])
        w.writerow(["cited_matched_by_doi", cited_by["doi"]])
        w.writerow(["cited_matched_by_title", cited_by["title"]])
        w.writerow(["cited_not_matched", cited_by["none"]])
        for who, c in (("cited", cited_c), ("citing", citing_c)):
            w.writerow([f"{who}_exactly_one", c["exactly_one"]])
            w.writerow([f"{who}_multiple", c["multiple"]])
            w.writerow([f"{who}_none", c["none"]])
        w.writerow(["both_exactly_one", both_exactly_one])
    print(f"\nWrote summary CSV: {summary_path}")

    # ---- REQUIRE_REFS: post-filter usable count in the suffixed dir ----
    if REQUIRE_REFS:
        os.makedirs(ANALYSIS_OUT_DIR, exist_ok=True)
        hasref_summary_path = os.path.join(ANALYSIS_OUT_DIR,
                                           "act2_match_summary.csv")
        with open(hasref_summary_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            w.writerow(["require_refs", 1])
            w.writerow(["total_instances", n])
            w.writerow(["both_exactly_one", both_exactly_one])
            w.writerow(["both_exactly_one_with_refs", both_exactly_one_with_refs])
            w.writerow(["both_exactly_one_dropped_no_refs",
                        both_exactly_one_dropped])
        print(f"Wrote REQUIRE_REFS summary CSV: {hasref_summary_path}")

    # per-instance detail CSV
    detail_path = os.path.join(OUT_DIR, "act2_match_per_instance.csv")
    with open(detail_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unique_id", "class_label", "cited_match_by",
                    "n_cited_matches", "n_citing_matches",
                    "cited_pids", "citing_pids"])
        w.writerows(per_instance)
    print(f"Wrote per-instance CSV: {detail_path}")

    # match map pkl
    match_path = os.path.join(OUT_DIR, "act2_match.pkl")
    with open(match_path, "wb") as f:
        pickle.dump(match, f, pickle.HIGHEST_PROTOCOL)
    print(f"Wrote match-map PKL: {match_path}")


if __name__ == "__main__":
    main()
