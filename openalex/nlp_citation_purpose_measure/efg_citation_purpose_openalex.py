"""
F / E / G / D citation-purpose analysis using the OPENALEX citation network.

This script classifies each ACL-ARC citation with the OpenAlex citation network
(rather than the ACL-ARC citation graph itself):

  1. Maps each ACL-ARC paper id to an OpenAlex id via arc_id2openalex_pids.pkl
     (produced by count_openalex_matches.py). ONLY citations where BOTH the
     citing and cited paper map to EXACTLY ONE OpenAlex id are used.
  2. Classifies each such citation A->B with the OpenAlex citation network
     (citing_pid2cited_pids_list.pkl = references, cited_pid2citing_pids_list.pkl
     = citations, both from get_meta_data_preprint.py):

         A_refs   = OpenAlex references of A   (citing_pid2cited[oa_A])
         B_refs   = OpenAlex references of B   (citing_pid2cited[oa_B])
         B_cites  = OpenAlex citations of B    (cited_pid2citing[oa_B])
         up_count   = |A_refs & B_refs|
         down_count = |A_refs & B_cites|

         F : down_count >= up_count and down_count > 0
         E : down_count <  up_count
         G : up_count == 0 and down_count == 0
         D (disruptive)     : up_count == 0   (A cites none of B's references)
         non_disruptive     : up_count >= 1
         disp_f_{0,1,2}     : disruptive foundation, up_count == 0 & down_count > 0/1/2
                              (extra scheme; a citation can join several, disp_f_0 = the
                              F subset of D). Mirrors openalex/get_efg_v2_disp_f.py; written
                              to disp_f_purpose_crosstab.csv + disp_f rows in
                              avg_instances_per_citation.csv.

     A citation is only classified if the OpenAlex network confirms the edge
     (oa_B is in A's OpenAlex references); otherwise it is skipped and counted.

The purpose of each mention comes from the ACL-ARC tsv (function column). The
loaders / stats / temporal helpers are inlined here so the script is
self-contained. Outputs (crosstabs, Welch t-tests, per-citation frequency,
longitudinal purpose-ratio CSV + plots) are written to a separate directory
(META_DIRE/nlp_purpose_analysis_openalex) so
efg_citation_purpose_openalex_s2_plot.py can plot them.

Binary toggle (REQUIRE_REFS)
----------------------------
Set REQUIRE_REFS = True (below) to run the alternative version in which every
citing OR cited paper with NO references in the OpenAlex network is removed: a
citation A -> B is only counted when BOTH A and B have >= 1 reference, and only
the remaining instances are tallied. Its outputs go to a separate, suffixed
directory (META_DIRE/nlp_purpose_analysis_openalex_hasref) so the two versions
coexist. The shared arc_id2openalex_pids.pkl is always read from the base
(unsuffixed) directory because matching is reference-independent.
"""

import os
import sys
import csv
import math
import pickle
from collections import defaultdict, Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

# utils.py is in the parent directory; make it importable regardless of CWD.
sys.path.insert(0, PARENT_DIR)
from utils import META_DIRE, NLP_PURPOSE_DIRE  # noqa: E402

DATA_DIR = NLP_PURPOSE_DIRE                                         # external input tsvs (see utils.py)

# ---------------------------------------------------------------------------
# Binary toggle (see module docstring). When True, only count citations where
# BOTH the citing and the cited paper have at least one reference in the
# OpenAlex network -- papers with no references are removed and only the
# remaining instances are tallied. Results go to a separate, suffixed output
# directory so the two versions coexist; the shared arc_id2openalex_pids.pkl is
# always read from the base (unsuffixed) directory because matching is
# reference-independent.
# ---------------------------------------------------------------------------
REQUIRE_REFS = True
SUFFIX = "_hasref" if REQUIRE_REFS else ""

BASE_OUT_DIR = os.path.join(META_DIRE, "nlp_purpose_analysis_openalex")          # holds arc_id2openalex_pids.pkl from count_openalex_matches.py
OUT_DIR = os.path.join(META_DIRE, "nlp_purpose_analysis_openalex" + SUFFIX)      # outputs

TSV_PATH = os.path.join(DATA_DIR, "acl-arc.citation-graph.with-functions.tsv")
PIDS_PATH = os.path.join(DATA_DIR, "arc-paper-ids.tsv")          # arc id -> year
MATCH_PATH = os.path.join(BASE_OUT_DIR, "arc_id2openalex_pids.pkl")
CITING2CITED_PATH = os.path.join(META_DIRE, "citing_pid2cited_pids_list.pkl")  # references
CITED2CITING_PATH = os.path.join(META_DIRE, "cited_pid2citing_pids_list.pkl")  # citations

# ---- temporal analysis settings ----
BOOTSTRAP_N = 1000        # bootstrap resamples for the 95% CI
BOOT_SEED = 42            # reproducible bootstrap
MIN_VALID_YEAR = 1960     # drop malformed publication years outside this range
MAX_VALID_YEAR = 2016
MIN_YEAR_N_PLOT = 1000    # for each type, only plot years with MORE THAN this many instances (N)
# Citing-paper years to drop from ALL statistics (empty = keep every year).
EXCLUDE_YEARS = set()

EFG_TYPES = ("f", "e", "g")
EFG_PAIRS = (("f", "e"), ("f", "g"), ("e", "g"))
DISR_TYPES = ("disruptive", "non_disruptive")
DISR_PAIRS = (("disruptive", "non_disruptive"),)

# disruptive-foundation variants, mirroring openalex/get_efg_v2_disp_f.py: a disruptive
# citation (up_count == 0) is additionally a "disruptive foundation" at threshold T when
# down_count > T. A citation can join several (disp_f_2 ⊂ disp_f_1 ⊂ disp_f_0), and disp_f_0
# is exactly the F subset of the disruptive citations.
DISP_F_DOWN_THRESHOLDS = (0, 1, 2)
DISP_F_TYPES = tuple(f"disp_f_{t}" for t in DISP_F_DOWN_THRESHOLDS)
# only compare WITHIN the disp_f scheme: report_level(scheme="disp_f") only tallies the
# disp_f_* types, so a pair naming any other type (e.g. "disruptive") would have n == 0
# and divide by zero. Cross-scheme D-vs-disp_f comparisons are available in the plot script,
# which loads all types together.
DISP_F_PAIRS = (("disp_f_0", "disp_f_1"), ("disp_f_1", "disp_f_2"),
                ("disp_f_0", "disp_f_2"))


def load_instances(path):
    """Yield (citing, cited, instance_id, section, function) tuples."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 5:
                continue
            citing, cited, inst_id, section, function = row[0], row[1], row[2], row[3], row[4]
            yield citing, cited, inst_id, section, function


def load_paper_years(path):
    """paper id -> publication year (int), keeping only sane years.

    arc-paper-ids.tsv columns: id, year, title, authors. Some rows carry
    malformed years (e.g. '0170'); we keep only MIN_VALID_YEAR..MAX_VALID_YEAR.
    """
    pid2year = {}
    n_bad = 0
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in tqdm(reader, desc="Loading paper years"):
            if len(row) < 2:
                continue
            pid, year_str = row[0], row[1]
            try:
                year = int(year_str)
            except ValueError:
                n_bad += 1
                continue
            if MIN_VALID_YEAR <= year <= MAX_VALID_YEAR:
                pid2year[pid] = year
            else:
                n_bad += 1
    print(f"  paper years loaded: {len(pid2year)} (dropped {n_bad} malformed/out-of-range)")
    return pid2year


def count_lines(path):
    """Total number of lines in the file (for tqdm progress totals)."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in f)


def _t_sf(t_abs, df):
    """Upper-tail survival function P(T > t_abs) for a Student-t with `df` dof.

    Uses scipy if available, otherwise a normal approximation (df is huge here -
    n is in the tens of thousands - so the normal approx is essentially exact).
    """
    try:
        from scipy import stats
        return float(stats.t.sf(t_abs, df))
    except Exception:
        # standard-normal survival function
        return 0.5 * math.erfc(t_abs / math.sqrt(2.0))


def welch_ttest_binary(n1, k1, n2, k2):
    """Two-sided Welch's t-test between two 0/1 samples given as (size, #ones).

    Group 1: n1 instances, k1 of which are ones (the rest zeros). Likewise group 2.
    This is exactly scipy.stats.ttest_ind(arr1, arr2, equal_var=False) where each
    arr is the binary indicator array, but computed from the counts directly.

    Returns dict with proportions, t statistic, Welch-Satterthwaite df, p-value.
    """
    # a group with no instances (n == 0) cannot be tested -> return a null (non-significant)
    # result instead of dividing by zero.
    if n1 < 1 or n2 < 1:
        return {
            "p1": (k1 / n1 if n1 else float("nan")),
            "p2": (k2 / n2 if n2 else float("nan")),
            "n1": n1, "n2": n2,
            "t_stat": 0.0, "df": float(max(n1 + n2 - 2, 0)), "p_value": 1.0,
        }
    p1 = k1 / n1
    p2 = k2 / n2
    # sample variance with ddof=1 for a 0/1 array: n*p*(1-p)/(n-1)
    var1 = n1 * p1 * (1.0 - p1) / (n1 - 1) if n1 > 1 else 0.0
    var2 = n2 * p2 * (1.0 - p2) / (n2 - 1) if n2 > 1 else 0.0
    se1 = var1 / n1
    se2 = var2 / n2
    denom = math.sqrt(se1 + se2)
    if denom == 0.0:
        # both groups constant (e.g. all zeros) -> no difference detectable
        t_stat = 0.0
        df = float(n1 + n2 - 2)
        p_value = 1.0
    else:
        t_stat = (p1 - p2) / denom
        df = (se1 + se2) ** 2 / (
            (se1 ** 2) / (n1 - 1) + (se2 ** 2) / (n2 - 1)
        )
        p_value = 2.0 * _t_sf(abs(t_stat), df)
    return {
        "p1": p1, "p2": p2, "n1": n1, "n2": n2,
        "t_stat": t_stat, "df": df, "p_value": p_value,
    }


def report_level(title, unit, scheme, types, pairs, type_counts,
                 type_purpose_counts, purposes, total_units,
                 crosstab_path, ttest_path, multilabel):
    """Print + write the type x purpose cross-tab, ratios, and pairwise Welch
    t-tests for one classification `scheme` (e.g. EFG or disruptive) at one
    aggregation level (instance-level or pair-level).

    `types` is the ordered list of class labels; `pairs` the list of (a, b)
    label pairs to compare in the t-test.

    `ratio_within_type` = fraction of that type's units having the purpose.
    When `multilabel` is True (pair level) a unit may carry several purposes, so
    the per-type purpose ratios do NOT sum to 1.
    """
    type_w = max([len(t) for t in types] + [len("type"), len("ALL")]) + 2

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print(f"\n{scheme} {unit} counts")
    print("-" * 40)
    for t in types:
        n = type_counts.get(t, 0)
        pct = 100.0 * n / total_units if total_units else 0.0
        print(f"  {t:<{type_w}}: {n:>10d}  ({pct:5.2f}%)")

    # overall purpose totals (derived so it works for both levels)
    purpose_totals = Counter()
    for t in types:
        for p in purposes:
            purpose_totals[p] += type_purpose_counts[t].get(p, 0)

    note = " (multi-label: a pair can have several purposes)" if multilabel else ""
    print(f"\nCross-tab: {scheme} type x purpose ({unit} counts){note}")
    print("-" * 70)
    print("type".ljust(type_w) + "".join(p[:14].rjust(16) for p in purposes) + "total".rjust(16))
    for t in types:
        row = t.ljust(type_w)
        for p in purposes:
            row += str(type_purpose_counts[t].get(p, 0)).rjust(16)
        row += str(type_counts.get(t, 0)).rjust(16)
        print(row)
    total_row = "ALL".ljust(type_w)
    for p in purposes:
        total_row += str(purpose_totals.get(p, 0)).rjust(16)
    total_row += str(total_units).rjust(16)
    print(total_row)

    print("\nRatio of each purpose WITHIN each type "
          f"(ratio_within_type = fraction of that type's {unit}s having the purpose)")
    print("-" * 70)
    print("type".ljust(type_w) + "".join(p[:14].rjust(16) for p in purposes))
    for t in types:
        row = t.ljust(type_w)
        t_total = type_counts.get(t, 0)
        for p in purposes:
            ratio = type_purpose_counts[t].get(p, 0) / t_total if t_total else 0.0
            row += f"{ratio:.4f}".rjust(16)
        print(row)

    with open(crosstab_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "purpose", f"{unit}_count", "ratio_within_type"])
        for t in types:
            t_total = type_counts.get(t, 0)
            for p in purposes:
                cnt = type_purpose_counts[t].get(p, 0)
                ratio = cnt / t_total if t_total else 0.0
                writer.writerow([t, p, cnt, f"{ratio:.6f}"])
    print(f"\nWrote cross-tab CSV: {crosstab_path}")

    # ----------------------------------------------- pairwise Welch's t-tests
    # For each purpose, compare the per-type "has this purpose" indicator (1/0)
    # between each pair of types. For type A the sample is every A unit scored 1
    # if it has the purpose else 0; likewise B. A two-sided Welch's t-test asks
    # whether ratio_within_type differs.
    pair_w = max([len(f"{a}-{b}") for a, b in pairs] + [len("pair")]) + 2
    print("\n" + "-" * 70)
    print(f"PAIRWISE TWO-SIDED WELCH'S t-TEST  ({scheme}, {unit} level, per purpose)")
    print("ratio = fraction of that type's units having the purpose")
    print("-" * 70)
    ttest_rows = []
    for p in tqdm(purposes, desc=f"Welch t-tests ({scheme} {unit})"):
        print(f"\nPurpose: {p}")
        print(f"  {'pair':<{pair_w}}{'ratio_a':>10}{'ratio_b':>10}"
              f"{'t_stat':>12}{'df':>12}{'p_value':>14}  sig")
        for a, b in pairs:
            na, ka = type_counts.get(a, 0), type_purpose_counts[a].get(p, 0)
            nb, kb = type_counts.get(b, 0), type_purpose_counts[b].get(p, 0)
            res = welch_ttest_binary(na, ka, nb, kb)
            pv = res["p_value"]
            sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else "ns"
            print(f"  {a + '-' + b:<{pair_w}}{res['p1']:>10.4f}{res['p2']:>10.4f}"
                  f"{res['t_stat']:>12.4f}{res['df']:>12.1f}{pv:>14.3e}  {sig}")
            ttest_rows.append([
                p, a, b, na, ka, nb, kb,
                f"{res['p1']:.6f}", f"{res['p2']:.6f}",
                f"{res['t_stat']:.6f}", f"{res['df']:.4f}",
                f"{pv:.6e}", sig,
            ])

    with open(ttest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "purpose", "type_a", "type_b",
            "n_a", "ones_a", "n_b", "ones_b",
            "ratio_a", "ratio_b", "t_stat", "df", "p_value", "significance",
        ])
        writer.writerows(ttest_rows)
    print(f"\nWrote Welch t-test CSV: {ttest_path}")


def temporal_analysis(scheme, types, year_purpose, year_total, purposes,
                      purpose_colors, type_colors, rng):
    """Build per-(type, year) purpose ratios with bootstrap 95% CIs, return CSV
    rows, and save two families of line plots (both restricted to (type, year)
    cells with MORE THAN MIN_YEAR_N_PLOT instances; N is in every filename):

      A) one plot per citation type: x=year, line+CI per purpose
         -> purpose_ratio_by_year_{scheme}_{type}_minN{N}.png
      B) one plot per purpose:       x=year, line+CI per citation type
         -> type_ratio_by_year_{scheme}_{purpose}_minN{N}.png

    Returns list of rows: [scheme, type, year, purpose, n_total, count,
                           ratio, ci_low, ci_high].
    """
    N = MIN_YEAR_N_PLOT
    rows = []
    # est[(t, year)] = (probs, lo, hi) arrays aligned with `purposes`
    est = {}
    years_by_type = {}
    for t in types:
        years = sorted(y for (tt, y) in year_total if tt == t)
        years_by_type[t] = years
        for year in years:
            total = year_total[(t, year)]
            counts = year_purpose[(t, year)]
            probs = np.array([counts.get(p, 0) / total for p in purposes])
            # nonparametric bootstrap of a categorical proportion == resampling
            # n draws from the empirical distribution == Multinomial(n, p_hat).
            boot = rng.multinomial(total, probs, size=BOOTSTRAP_N) / total
            lo = np.percentile(boot, 2.5, axis=0)
            hi = np.percentile(boot, 97.5, axis=0)
            est[(t, year)] = (probs, lo, hi)
            for i, p in enumerate(purposes):
                rows.append([scheme, t, year, p, total, counts.get(p, 0),
                             f"{probs[i]:.6f}", f"{lo[i]:.6f}", f"{hi[i]:.6f}"])

    # ---- A) one plot per citation type (lines = purposes) -------------------
    for t in types:
        plot_years = [y for y in years_by_type[t] if year_total[(t, y)] > N]
        if not plot_years:
            print(f"  [plot skipped] {scheme}/{t}: no year with > {N} instances")
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        xs = np.array(plot_years)
        for i, p in enumerate(purposes):
            ys = np.array([est[(t, y)][0][i] for y in plot_years])
            los = np.array([est[(t, y)][1][i] for y in plot_years])
            his = np.array([est[(t, y)][2][i] for y in plot_years])
            color = purpose_colors[p]
            ax.plot(xs, ys, "-o", ms=3, color=color, label=p)
            ax.fill_between(xs, los, his, color=color, alpha=0.18)
        ax.set_title(f"{scheme} - '{t}' citations: purpose ratio by year "
                     f"(citing-paper year; 95% bootstrap CI; year n>{N})")
        ax.set_xlabel("Year (citing paper publication year)")
        ax.set_ylabel("Purpose ratio within type")
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
        ax.legend(title="purpose", fontsize=9, ncol=2)
        fig.tight_layout()
        out_png = os.path.join(
            OUT_DIR, f"purpose_ratio_by_year_{scheme}_{t}_minN{N}.png")
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"  Wrote plot: {out_png}")

    # ---- B) one plot per purpose (lines = citation types) -------------------
    for i, p in enumerate(purposes):
        fig, ax = plt.subplots(figsize=(10, 6))
        any_line = False
        for t in types:
            plot_years = [y for y in years_by_type[t] if year_total[(t, y)] > N]
            if not plot_years:
                continue
            any_line = True
            xs = np.array(plot_years)
            ys = np.array([est[(t, y)][0][i] for y in plot_years])
            los = np.array([est[(t, y)][1][i] for y in plot_years])
            his = np.array([est[(t, y)][2][i] for y in plot_years])
            color = type_colors.get(t)
            ax.plot(xs, ys, "-o", ms=3, color=color, label=t)
            ax.fill_between(xs, los, his, color=color, alpha=0.18)
        if not any_line:
            plt.close(fig)
            print(f"  [plot skipped] {scheme}/purpose '{p}': no year with > {N}")
            continue
        ax.set_title(f"{scheme} - purpose '{p}': ratio by year, by citation type "
                     f"(citing-paper year; 95% bootstrap CI; year n>{N})")
        ax.set_xlabel("Year (citing paper publication year)")
        ax.set_ylabel(f"Ratio of '{p}' within citation type")
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
        ax.legend(title="citation type", fontsize=9)
        fig.tight_layout()
        out_png = os.path.join(
            OUT_DIR, f"type_ratio_by_year_{scheme}_{p}_minN{N}.png")
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"  Wrote plot: {out_png}")

    return rows


def load_unique_arc2oa(path):
    """arc paper id -> single OpenAlex id, keeping only ids with EXACTLY ONE match."""
    with open(path, "rb") as f:
        match = pickle.load(f)
    arc2oa = {}
    n_multi = 0
    for arc_id, pids in match.items():
        if len(pids) == 1:
            arc2oa[arc_id] = pids[0]
        elif len(pids) > 1:
            n_multi += 1
    print(f"  arc ids uniquely mapped: {len(arc2oa)} (dropped {n_multi} multi-match)")
    return arc2oa


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Writing results to: {OUT_DIR}")
    print(f"REQUIRE_REFS = {REQUIRE_REFS} "
          f"({'only citations where BOTH papers have references' if REQUIRE_REFS else 'all citations'})")

    # ---- mappings & network ----
    arc2oa = load_unique_arc2oa(MATCH_PATH)
    pid2year = load_paper_years(PIDS_PATH)
    print("Loading OpenAlex reference / citation network ...")
    with open(CITING2CITED_PATH, "rb") as f:
        citing2cited = pickle.load(f)   # oa id -> [references]
    with open(CITED2CITING_PATH, "rb") as f:
        cited2citing = pickle.load(f)   # oa id -> [citations]
    print(f"  citing_pid2cited: {len(citing2cited)}  cited_pid2citing: {len(cited2citing)}")

    n_lines = count_lines(TSV_PATH)

    # ---- tallies ----
    type_counts = Counter(); purpose_counts = Counter()
    type_purpose_counts = defaultdict(Counter)
    disr_counts = Counter(); disr_purpose_counts = defaultdict(Counter)
    dispf_counts = Counter(); dispf_purpose_counts = defaultdict(Counter)
    pair_type = {}; pair_disr = {}; pair_dispf = {}; pair_inst_count = Counter()
    efg_year_purpose = defaultdict(Counter); efg_year_total = Counter()
    disr_year_purpose = defaultdict(Counter); disr_year_total = Counter()
    year_both_mapped = Counter(); year_classified = Counter()

    edge_cache = {}     # (oa_a, oa_b) -> (efg, disr, dispf_tuple) or None
    refs_cache = {}     # oa_a -> set(references)

    total = 0
    skip_citing_unmapped = 0
    skip_cited_unmapped = 0
    both_mapped = 0
    skip_no_refs = 0
    skip_no_oa_edge = 0
    skip_excluded_year = 0
    counted = 0
    counted_with_year = 0

    def get_refs(oa):
        r = refs_cache.get(oa)
        if r is None:
            r = set(citing2cited.get(oa, ()))
            refs_cache[oa] = r
        return r

    def classify(oa_a, oa_b):
        a_refs = get_refs(oa_a)
        if oa_b not in a_refs:          # OpenAlex does not record A -> B
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
        # disruptive-foundation variants: only disruptive (up == 0) citations qualify,
        # each for every threshold T with down > T.
        dispf = tuple(f"disp_f_{t}" for t in DISP_F_DOWN_THRESHOLDS
                      if up == 0 and down > t)
        return e, d, dispf

    for citing, cited, _inst, _sec, function in tqdm(
        load_instances(TSV_PATH), total=n_lines,
        desc="Classifying via OpenAlex network"
    ):
        total += 1
        oa_a = arc2oa.get(citing)
        if oa_a is None:
            skip_citing_unmapped += 1
            continue
        oa_b = arc2oa.get(cited)
        if oa_b is None:
            skip_cited_unmapped += 1
            continue
        both_mapped += 1

        # Binary toggle: drop the citation if either the citing or the cited
        # paper has NO references in the OpenAlex network. (For counted
        # instances the citing paper always has references -- oa_b must be in
        # A_refs -- so in practice this removes cited papers with no references,
        # but both sides are checked for clarity.)
        if REQUIRE_REFS and (not get_refs(oa_a) or not get_refs(oa_b)):
            skip_no_refs += 1
            continue

        year = pid2year.get(citing)   # citing paper's publication year
        if year is not None:
            year_both_mapped[year] += 1

        edge = (oa_a, oa_b)
        cls = edge_cache.get(edge, "NA")
        if cls == "NA":
            cls = classify(oa_a, oa_b)
            edge_cache[edge] = cls
        if cls is None:
            skip_no_oa_edge += 1
            continue
        efg_t, disr_t, dispf_ts = cls

        if year is not None:
            year_classified[year] += 1   # raw (all years, incl. excluded)

        if year in EXCLUDE_YEARS:        # drop 2015 from analysis stats
            skip_excluded_year += 1
            continue

        counted += 1
        purpose_counts[function] += 1
        type_counts[efg_t] += 1
        type_purpose_counts[efg_t][function] += 1
        disr_counts[disr_t] += 1
        disr_purpose_counts[disr_t][function] += 1
        # a citation contributes to every disp_f variant it qualifies for
        for dft in dispf_ts:
            dispf_counts[dft] += 1
            dispf_purpose_counts[dft][function] += 1

        pair_type[edge] = efg_t
        pair_disr[edge] = disr_t
        pair_dispf[edge] = dispf_ts
        pair_inst_count[edge] += 1

        if year is not None:
            counted_with_year += 1
            efg_year_purpose[(efg_t, year)][function] += 1
            efg_year_total[(efg_t, year)] += 1
            disr_year_purpose[(disr_t, year)][function] += 1
            disr_year_total[(disr_t, year)] += 1

    purposes = sorted(purpose_counts.keys())

    # ---------------------------------------------------------------- stats
    print("\n" + "=" * 70)
    print("OPENALEX-NETWORK F/E/G/D  -  SUMMARY")
    print("=" * 70)
    stats = [
        ("require_refs", int(REQUIRE_REFS)),
        ("total_instances", total),
        ("skip_citing_unmapped", skip_citing_unmapped),
        ("skip_cited_unmapped", skip_cited_unmapped),
        ("both_uniquely_mapped", both_mapped),
        ("skip_no_references", skip_no_refs),
        ("skip_no_openalex_edge", skip_no_oa_edge),
        ("classified_all_years", both_mapped - skip_no_refs - skip_no_oa_edge),
        ("excluded_years", "|".join(str(y) for y in sorted(EXCLUDE_YEARS))),
        ("dropped_excluded_year", skip_excluded_year),
        ("counted_instances", counted),
        ("counted_with_known_year", counted_with_year),
        ("unique_citation_pairs", len(pair_type)),
    ]
    for name, val in stats:
        print(f"  {name:<28s}: {val}")
    stats_path = os.path.join(OUT_DIR, "efg_dataset_stats.csv")
    with open(stats_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["metric", "value"]); w.writerows(stats)
    print(f"Wrote dataset stats CSV: {stats_path}")

    # citation instances by year (all years, no exclusion)
    by_year_path = os.path.join(OUT_DIR, "citation_instances_by_year.csv")
    all_years = sorted(set(year_both_mapped) | set(year_classified))
    with open(by_year_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["year", "n_both_mapped", "n_classified"])
        for y in all_years:
            w.writerow([y, year_both_mapped.get(y, 0), year_classified.get(y, 0)])
    print(f"Wrote citation-instances-by-year CSV: {by_year_path}")

    # ---- per-citation frequency (mean/var/std of #instances per pair) ----
    efg_pc = Counter(); efg_sum = Counter(); efg_sq = Counter()
    disr_pc = Counter(); disr_sum = Counter(); disr_sq = Counter()
    dispf_pc = Counter(); dispf_sum = Counter(); dispf_sq = Counter()
    for edge in tqdm(pair_type, desc="Aggregating per-citation frequency"):
        n = pair_inst_count[edge]
        et, dt = pair_type[edge], pair_disr[edge]
        efg_pc[et] += 1; efg_sum[et] += n; efg_sq[et] += n * n
        disr_pc[dt] += 1; disr_sum[dt] += n; disr_sq[dt] += n * n
        # a citation pair contributes to every disp_f variant it qualifies for
        for dft in pair_dispf.get(edge, ()):
            dispf_pc[dft] += 1; dispf_sum[dft] += n; dispf_sq[dft] += n * n
    avg_rows = []
    for scheme, pc, sm, sq, types in (
        ("EFG", efg_pc, efg_sum, efg_sq, EFG_TYPES),
        ("disruptive", disr_pc, disr_sum, disr_sq, DISR_TYPES),
        ("disp_f", dispf_pc, dispf_sum, dispf_sq, DISP_F_TYPES),
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

    # ---- purpose distribution (instance level) ----
    report_level(
        title="EFG (OpenAlex net) - INSTANCE LEVEL",
        unit="instance", scheme="EFG", types=EFG_TYPES, pairs=EFG_PAIRS,
        type_counts=type_counts, type_purpose_counts=type_purpose_counts,
        purposes=purposes, total_units=counted,
        crosstab_path=os.path.join(OUT_DIR, "efg_purpose_crosstab.csv"),
        ttest_path=os.path.join(OUT_DIR, "efg_purpose_welch_ttest.csv"),
        multilabel=False,
    )
    report_level(
        title="DISRUPTIVE (OpenAlex net) - INSTANCE LEVEL (disruptive = cites NONE of B's references)",
        unit="instance", scheme="disruptive", types=DISR_TYPES, pairs=DISR_PAIRS,
        type_counts=disr_counts, type_purpose_counts=disr_purpose_counts,
        purposes=purposes, total_units=counted,
        crosstab_path=os.path.join(OUT_DIR, "disruptive_purpose_crosstab.csv"),
        ttest_path=os.path.join(OUT_DIR, "disruptive_purpose_welch_ttest.csv"),
        multilabel=False,
    )
    report_level(
        title="DISP_F (OpenAlex net) - INSTANCE LEVEL (disruptive foundation: up_count == 0 & down_count > T)",
        unit="instance", scheme="disp_f", types=DISP_F_TYPES, pairs=DISP_F_PAIRS,
        type_counts=dispf_counts, type_purpose_counts=dispf_purpose_counts,
        purposes=purposes, total_units=counted,
        crosstab_path=os.path.join(OUT_DIR, "disp_f_purpose_crosstab.csv"),
        ttest_path=os.path.join(OUT_DIR, "disp_f_purpose_welch_ttest.csv"),
        multilabel=False,
    )

    # ---- longitudinal purpose ratio ----
    purpose_colors = dict(zip(purposes, plt.get_cmap("tab10").colors))
    type_colors = {"f": "C0", "e": "C1", "g": "C2",
                   "disruptive": "C3", "non_disruptive": "C4"}
    rng = np.random.default_rng(BOOT_SEED)
    temporal_rows = []
    temporal_rows += temporal_analysis(
        "EFG", EFG_TYPES, efg_year_purpose, efg_year_total,
        purposes, purpose_colors, type_colors, rng)
    temporal_rows += temporal_analysis(
        "disruptive", DISR_TYPES, disr_year_purpose, disr_year_total,
        purposes, purpose_colors, type_colors, rng)
    temporal_path = os.path.join(OUT_DIR, "purpose_ratio_by_year.csv")
    with open(temporal_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheme", "type", "year", "purpose", "n_total",
                    "count", "ratio", "ci_low", "ci_high"])
        w.writerows(temporal_rows)
    print(f"\nWrote temporal purpose-ratio CSV: {temporal_path}")


if __name__ == "__main__":
    main()
