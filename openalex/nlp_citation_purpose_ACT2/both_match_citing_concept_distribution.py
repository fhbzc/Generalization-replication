"""
Root-concept distribution of the CITING papers in all "both-match" ACT2 citation
instances. (Companion to both_match_cited_concept_distribution.py, which tallies
the CITED paper B instead; this script tallies the CITING paper A.)

A "both-match" instance is kept on the SOLE criterion that BOTH the citing and
the cited paper matched EXACTLY ONE OpenAlex id. Unlike the cited-paper companion
(both_match_cited_concept_distribution.py) and efg_citation_purpose_act2.py, this
script does NOT require the ACT2 class_label to be known and does NOT require the
citation to exist in the OpenAlex network (no A -> B edge check): both-match
alone is enough. An ACT2 instance A (citing) -> B (cited) is kept when:
  1. BOTH the citing and the cited paper matched EXACTLY ONE OpenAlex id.
  2. (REQUIRE_REFS only) BOTH A and B have >= 1 reference in the network.
NOTE: "both-match", not "both-math" -- there is no Mathematics / domain filter
here; the script file just keeps its historical name.

For every such instance we look at the CITING paper A and tally its domain
label(s), via a single rule (domains_of) applied everywhere we read
pid2root_concepts: a paper with NO entry, a None entry, or MORE THAN MAX_DOMAINS
(=2) root concepts is bucketed as the single label 'Others'; otherwise its 1-2
root concept ids are used. A 2-domain tie paper is counted once for each of its
fields. The result is the distribution of the citing papers' root concepts
(level-0 domains) across the both-match citation instances.

Inputs
------
  META_DIRE/pid2root_concepts_level_0.pkl            pid -> [root concept ids] (or None)
                                                     (loaded only on a cache miss)
  META_DIRE/nlp_purpose_analysis_act2/act2_match.pkl uid -> {citing_pids, cited_pids, ...}
                                                     (loaded only on a cache miss)
  META_DIRE/citing_pid2cited_pids_list.pkl           oa id -> [references]
                                                     (loaded only on a cache miss, and only when REQUIRE_REFS)
  CONCEPT_DIRE (Concepts_definition.xlsx)            openalex_id, display_name, level, ...
                                                     (always read, for display names)

Binary toggle (REQUIRE_REFS)
----------------------------
Set REQUIRE_REFS = True (below) to run the alternative version in which every
citing OR cited paper with NO references in the OpenAlex network is removed: a
both-match instance A -> B is only kept when BOTH A and B have >= 1 reference,
and only the remaining instances feed the distribution. Its outputs are written
to a separate, suffixed directory (nlp_purpose_analysis_act2_hasref); the shared
act2_match.pkl is always read from the base (unsuffixed) directory.

Outputs (META_DIRE/nlp_purpose_analysis_act2[_hasref])
  both_match_citing_root_concept_distribution.csv  root_concept_id, root_concept_name,
                                                   citing_paper_count, instance_count,
                                                   ratio_papers, ratio_instances
  both_match_citing_root_concept_stats.csv         run summary
  both_match_citing_root_concept_distribution.svg/.png  bar chart of the per-domain
                                                   citation count (instance_count)
"""

import os
import sys
import csv
import pickle
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")  # headless / cluster-safe
import matplotlib.pyplot as plt  # noqa: E402

# try the reference font but fall back silently if unavailable (e.g. on Linux)
try:
    import matplotlib.font_manager as fm
    if any("Gill Sans MT" in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = "Gill Sans MT"
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

# utils.py is in the parent directory; make it importable regardless of CWD.
sys.path.insert(0, PARENT_DIR)
from utils import META_DIRE, CONCEPT_DIRE  # noqa: E402

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))

ROOT_LEVEL = 0

# a paper with no root-concept entry, a None entry, or more than MAX_DOMAINS
# root concepts is bucketed under this single label.
OTHERS_LABEL = "Others"
MAX_DOMAINS = 2

meta_dire = META_DIRE

# ---------------------------------------------------------------------------
# Binary toggle (see module docstring). When True, only keep both-match
# instances where BOTH the citing and the cited paper have at least one
# reference in the OpenAlex network -- papers with no references are removed and
# only the remaining instances feed the distribution. Results go to a separate,
# suffixed output directory so the two versions coexist; the shared
# act2_match.pkl is always read from the base (unsuffixed) directory because
# matching is reference-independent.
# ---------------------------------------------------------------------------
REQUIRE_REFS = False
SUFFIX = "_hasref" if REQUIRE_REFS else ""

BASE_OUT_DIR = os.path.join(meta_dire, "nlp_purpose_analysis_act2")
OUT_DIR = os.path.join(meta_dire, "nlp_purpose_analysis_act2" + SUFFIX)
# Figures go to ./figures[_hasref]/concept_distribution next to this script (the
# CSV/stats/cache stay in OUT_DIR). Mirrors efg_citation_purpose_act2_s2_plot.py,
# which also writes its figures to ./figures[_hasref].
FIG_DIR = os.path.join(BASE_DIR, "figures" + SUFFIX, "concept_distribution")
ROOT_CONCEPTS_PATH = os.path.join(meta_dire,
                                  "pid2root_concepts_level_%s.pkl" % ROOT_LEVEL)
MATCH_PATH = os.path.join(BASE_OUT_DIR, "act2_match.pkl")
CITING2CITED_PATH = os.path.join(meta_dire, "citing_pid2cited_pids_list.pkl")

# Cache of the (expensive) instance-selection step -- the per-citing-paper
# instance counts plus the run stats (incl. the total number of citation
# instances kept). When present and valid it is loaded directly, so the big
# references map + match map and the per-instance loop are skipped on reruns.
# Distinct filename from the cited-paper script so the two caches never collide.
# Set REBUILD_CACHE = True to force a fresh selection (the cache is rewritten).
# bump when the selection logic changes so stale caches are auto-invalidated.
CACHE_VERSION = 3
USE_CACHE = True
REBUILD_CACHE = False
CACHE_PATH = os.path.join(OUT_DIR,
                          "both_match_citing_instance_count_cache.pkl")

# pid2root_concepts stores root ids as openalex_id[21:].lower()  (e.g. "c33923547")
URL_PREFIX_LEN = len("https://openalex.org/")


def strip_prefix(openalex_id):
    """'https://openalex.org/C33923547' -> 'C33923547' (case preserved)."""
    return str(openalex_id)[URL_PREFIX_LEN:]


def plot_domain_citation_counts(rows, out_dir):
    """Bar chart of the per-domain citation count (instance_count) across the
    both-match citing papers. `rows` is the same list written to the CSV:
    [root_concept_id, root_concept_name, citing_paper_count, instance_count,
     ratio_papers, ratio_instances], already sorted by citing_paper_count desc.

    Bars are ordered alphabetically by domain label (x ticks A->Z).
    """
    if not rows:
        print("No rows to plot; skipping figure.")
        return

    os.makedirs(out_dir, exist_ok=True)

    # label = display name when available, else the concept id; instance_count
    # is the citation count (a citing paper is weighted by how many both-match
    # instances it makes).
    items = []
    for r in rows:
        root_cid, name, _paper_cnt, inst_cnt = r[0], r[1], r[2], r[3]
        label = name if name else root_cid
        items.append((label, int(inst_cnt), root_cid == OTHERS_LABEL))

    # x ticks ordered alphabetically by domain label (case-insensitive).
    items.sort(key=lambda it: it[0].lower())

    labels = [it[0] for it in items]
    counts = [it[1] for it in items]
    is_other = [it[2] for it in items]

    fig_w = max(7.0, 0.55 * len(labels) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, 5.0))
    x = range(len(labels))
    colors = ["#BFBFBF" if o else "#4C72B0" for o in is_other]
    bars = ax.bar(x, counts, color=colors, width=0.82,
                  edgecolor="white", linewidth=0.6)

    # value labels on top of each bar
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                "{:,}".format(c), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Citation instance count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(y=0.12)
    fig.tight_layout()

    out_svg = os.path.join(out_dir,
                           "both_match_citing_root_concept_distribution.svg")
    out_png = os.path.join(out_dir,
                           "both_match_citing_root_concept_distribution.png")
    fig.savefig(out_svg, format="svg", transparent=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("Wrote domain citation-count figure:", out_svg)
    print("Wrote domain citation-count figure:", out_png)


def build_distribution():
    """Run the whole expensive pipeline -- the part worth caching.

    Loads the ACT2 match map (and, only when REQUIRE_REFS, the OpenAlex
    references map) plus pid2root_concepts, keeps every instance where BOTH
    papers matched exactly one OpenAlex id (no class_label / citation-exists
    filter), tallies the CITING paper A, and aggregates those papers' level-0
    root concepts into the per-domain distribution. Returns
    (root_paper_count, root_instance_count, stats). Because the per-domain counts
    are cached, a cache hit needs NEITHER the match/references maps NOR the (large)
    pid2root_concepts map -- only the small concept-name Excel for display names."""
    print("Loading", MATCH_PATH, "...")
    with open(MATCH_PATH, "rb") as f:
        match = pickle.load(f)
    print("  ACT2 instances in match map:", len(match))

    # References are only needed for the REQUIRE_REFS toggle now -- there is no
    # citation-exists (A->B edge) filter here, so skip the big load when off.
    citing2cited = None
    if REQUIRE_REFS:
        print("Loading", CITING2CITED_PATH, "(reference filter) ...")
        with open(CITING2CITED_PATH, "rb") as f:
            citing2cited = pickle.load(f)   # oa id -> [references]
        print("  citing_pid2cited entries:", len(citing2cited))

    def has_refs(pid):
        """True if the paper has >= 1 reference in the OpenAlex network. Always
        True when REQUIRE_REFS is off (the reference map is not loaded)."""
        return True if citing2cited is None else bool(citing2cited.get(pid))

    total = len(match)
    skip_not_both_match = 0
    skip_no_refs = 0
    both_match_instances = 0
    # citing pid -> how many both-match instances it makes
    citing_pid2instance_count = defaultdict(int)

    for uid, m in tqdm(match.items(), desc="Selecting both-match instances"):
        citing_pids = m.get("citing_pids", [])
        cited_pids = m.get("cited_pids", [])
        # (1) "both-match": both the citing and cited paper map to exactly one id
        if len(citing_pids) != 1 or len(cited_pids) != 1:
            skip_not_both_match += 1
            continue

        oa_a, oa_b = citing_pids[0], cited_pids[0]

        # (2) Binary toggle: drop the instance if either the citing or the cited
        # paper has NO references in the OpenAlex network.
        if REQUIRE_REFS and (not has_refs(oa_a) or not has_refs(oa_b)):
            skip_no_refs += 1
            continue

        both_match_instances += 1
        citing_pid2instance_count[oa_a] += 1   # tally the CITING paper A

    n_citing_papers = len(citing_pid2instance_count)
    print("  both-match citation instances:", both_match_instances)
    print("  distinct both-match citing papers:", n_citing_papers)

    # ---- aggregate the citing papers' root concepts into the distribution ----
    print("Loading", ROOT_CONCEPTS_PATH, "...")
    with open(ROOT_CONCEPTS_PATH, "rb") as f:
        pid2root_concepts = pickle.load(f)
    print("  pid2root_concepts entries:", len(pid2root_concepts))

    def domains_of(pid):
        """Single source of truth for a paper's domain label(s).
        No entry / None entry / more than MAX_DOMAINS root concepts -> ['Others'];
        otherwise the paper's 1..MAX_DOMAINS root concept ids."""
        rc = pid2root_concepts.get(pid)
        if rc is None or len(rc) > MAX_DOMAINS:
            return [OTHERS_LABEL]
        return list(rc)

    root_paper_count = defaultdict(int)     # # distinct citing papers w/ this root
    root_instance_count = defaultdict(int)  # instance-weighted (paper counted n times)
    for citing_pid, inst_n in tqdm(citing_pid2instance_count.items(),
                                   desc="Aggregating citing-paper root concepts"):
        for root_cid in domains_of(citing_pid):  # 'Others' for None/missing/>MAX_DOMAINS
            root_paper_count[root_cid] += 1
            root_instance_count[root_cid] += inst_n

    stats = {
        "cache_version": CACHE_VERSION,
        "require_refs": REQUIRE_REFS,
        "total_instances": total,
        "skip_not_both_match": skip_not_both_match,
        "skip_no_refs": skip_no_refs,
        # the headline number: how many citation instances survived all filters
        "both_match_instances": both_match_instances,
        "distinct_citing_papers": n_citing_papers,
    }
    return dict(root_paper_count), dict(root_instance_count), stats


def load_or_build_distribution():
    """Return (root_paper_count, root_instance_count, stats), preferring the
    on-disk cache. The cache stores the aggregated per-domain counts plus the run
    stats (including the total number of citation instances kept), so a cache hit
    skips loading the match/references maps AND pid2root_concepts entirely. Reused
    only when CACHE_VERSION and REQUIRE_REFS match; otherwise rebuilt and the
    cache rewritten. Force a rebuild with REBUILD_CACHE = True."""
    if USE_CACHE and not REBUILD_CACHE and os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "rb") as f:
                cached = pickle.load(f)
            stats = cached["stats"]
            if (stats.get("cache_version") == CACHE_VERSION
                    and stats.get("require_refs") == REQUIRE_REFS):
                print("Loaded distribution from cache:", CACHE_PATH)
                print("  total citation instances kept (from cache):",
                      stats["both_match_instances"])
                return (cached["root_paper_count"],
                        cached["root_instance_count"], stats)
            print("Cache present but stale (cache_version/require_refs mismatch);"
                  " rebuilding ...")
        except Exception as e:
            print("Failed to read cache (%s); rebuilding ..." % e)

    root_paper_count, root_instance_count, stats = build_distribution()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump({"stats": stats,
                     "root_paper_count": root_paper_count,
                     "root_instance_count": root_instance_count}, f)
    print("Wrote distribution cache:", CACHE_PATH,
          "(total citation instances kept = %d)"
          % stats["both_match_instances"])
    return root_paper_count, root_instance_count, stats


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Writing results to:", OUT_DIR)
    print("REQUIRE_REFS =", REQUIRE_REFS,
          "(only instances where BOTH papers have references)"
          if REQUIRE_REFS else "(all both-match instances)")

    # ---- 1. concept metadata: root id -> display name ------------------------
    print("Loading concept definitions:", CONCEPT_DIRE)
    df_concept = pd.read_excel(CONCEPT_DIRE)
    # Like get_meta_data_preprint.py's concept2name (read from the same Excel),
    # but keyed by the LOWER-cased, prefix-stripped concept id so it matches the
    # lower-cased root ids stored in pid2root_concepts.
    cid2name = {strip_prefix(oid).lower(): name
                for oid, name in zip(df_concept["openalex_id"],
                                     df_concept["display_name"])}

    # ---- 2. both-match citing-paper root-concept distribution (cached) -------
    # The whole expensive pipeline (load match + references + pid2root_concepts,
    # run the filter loop, aggregate per-domain) is cached to CACHE_PATH. On a
    # cache hit pid2root_concepts is NOT loaded -- only the small concept Excel
    # above (for display names) is read. The cache also records the total number
    # of citation instances kept (sel_stats['both_match_instances']).
    root_paper_count, root_instance_count, sel_stats = load_or_build_distribution()
    total = sel_stats["total_instances"]
    skip_not_both_match = sel_stats["skip_not_both_match"]
    skip_no_refs = sel_stats["skip_no_refs"]
    both_match_instances = sel_stats["both_match_instances"]
    n_citing_papers = sel_stats["distinct_citing_papers"]

    # ---- 6. write the distribution CSV (sorted by citing_paper_count desc) ----
    rows = []
    for root_cid, paper_cnt in root_paper_count.items():
        inst_cnt = root_instance_count[root_cid]
        ratio_papers = paper_cnt / n_citing_papers if n_citing_papers else 0.0
        ratio_inst = (inst_cnt / both_match_instances
                      if both_match_instances else 0.0)
        name = (OTHERS_LABEL if root_cid == OTHERS_LABEL
                else cid2name.get(root_cid.lower(), ""))
        rows.append([
            root_cid,
            name,
            paper_cnt,
            inst_cnt,
            "%.6f" % ratio_papers,
            "%.6f" % ratio_inst,
        ])
    # sort: most common root concept among citing papers first, then by name
    rows.sort(key=lambda r: (-r[2], -r[3], str(r[1])))

    dist_path = os.path.join(OUT_DIR,
                             "both_match_citing_root_concept_distribution.csv")
    with open(dist_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["root_concept_id", "root_concept_name",
                    "citing_paper_count", "instance_count",
                    "ratio_papers", "ratio_instances"])
        w.writerows(rows)
    print("Wrote root-concept distribution CSV:", dist_path)

    # ---- 6b. bar chart of the per-domain citation count ----------------------
    plot_domain_citation_counts(rows, FIG_DIR)

    # ---- 7. summary ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("BOTH-MATCH CITING-PAPER ROOT-CONCEPT DISTRIBUTION - SUMMARY")
    print("=" * 70)
    stats = [
        ("require_refs", int(REQUIRE_REFS)),
        ("total_instances", total),
        ("skip_not_both_exactly_one", skip_not_both_match),
        ("skip_no_references", skip_no_refs),
        ("both_match_instances", both_match_instances),
        ("distinct_both_match_citing_papers", n_citing_papers),
        ("distinct_root_concepts", len(root_paper_count)),
    ]
    for name, val in stats:
        print("  %-32s: %s" % (name, val))
    stats_path = os.path.join(OUT_DIR,
                              "both_match_citing_root_concept_stats.csv")
    with open(stats_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(stats)
    print("Wrote stats CSV:", stats_path)

    print("\nRoot-concept distribution among both-match citing papers:")
    for r in rows:
        print("  %-12s %-30s papers=%-6s inst=%-6s ratio_papers=%s"
              % (r[0], (str(r[1])[:28]), r[2], r[3], r[4]))
    print("finished")


if __name__ == "__main__":
    main()
