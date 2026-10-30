"""
S2-style panel figure for the ACT2 F / E / G / D citation-purpose analysis.

Self-contained plotting script. Identical plotting to
../nlp_citation_purpose_measure/efg_citation_purpose_s2_plot.py (grid of bar
panels with the citation type on the x-axis and Welch significance brackets, for
citation types E / F / G / D where D = "disruptive"; the complementary
"non_disruptive" type is intentionally NOT drawn), but reads the CSVs produced by
efg_citation_purpose_act2.py and draws ONLY the per-purpose ratio panels (the
leading "Frequency" panel is dropped; the 6 ACT2 purpose panels lay out as a 2x3
grid).

Two figures are produced:
  1. efg_purpose_ratio_s2.*         the original 6 ACT2 purposes
  2. efg_purpose_ratio_s2_merged.*  the 6 purposes collapsed into 5 categories:
       Background        = Background          (kept separate)
       Motivation        = Motivation          (kept separate)
       Intent to Modify  = CompareOrContrast + Extends
       Future            = Future
       Uses              = Uses
     (2x3 grid). Merging only sums the per-purpose instance counts, so the per-type
     ratios, bootstrap CIs and Welch tests all follow from the same counts.

Inputs (produced by efg_citation_purpose_act2.py, in
META_DIRE/nlp_purpose_analysis_act2; META_DIRE imported from ../utils.py):
  efg_purpose_crosstab.csv            type, purpose, instance_count, ratio_within_type   (e/f/g)
  disruptive_purpose_crosstab.csv     same, for disruptive/non_disruptive
  avg_instances_per_citation.csv      scheme, type, n_instances, n_citations, avg/var_instances_per_citation

The per-purpose Welch p-values are recomputed here straight from the crosstab
counts, so efg_citation_purpose_act2.py's *_welch_ttest.csv files are not read.

Run match_act2_openalex.py, then efg_citation_purpose_act2.py, then this script.
The figure is written to META_DIRE/nlp_purpose_analysis_act2/s2_figures.

Significance:
  * On each purpose panel ALL 6 pairwise comparisons among E/F/G/D are tested and
    drawn as brackets (E-F, E-G, E-D, F-G, F-D, G-D). Each pair's two-sided Welch
    t-test on the per-purpose 0/1 indicator is computed straight from the crosstab
    counts (type total n + per-purpose count k), so the cross-scheme D-vs-E/F/G
    pairs are available too.
  * Caveat: D (disruptive) and E/F/G are different partitions of the SAME
    instances (e.g. every G citation is also disruptive), so the D-vs-E/F/G tests
    compare overlapping samples and are approximate.
"""

import os
import sys
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless / cluster-safe
import matplotlib.pyplot as plt  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

# utils.py is in the parent directory; make it importable regardless of CWD.
sys.path.insert(0, PARENT_DIR)
from utils import META_DIRE  # noqa: E402

# Binary toggle -- must match the value used when running
# efg_citation_purpose_act2.py. When True, read the CSVs produced by the
# "papers must have references" version (suffixed input dir) and write the
# figures to a matching suffixed directory so the two versions coexist.
REQUIRE_REFS = True
SUFFIX = "_hasref" if REQUIRE_REFS else ""

OUT_DIR = os.path.join(META_DIRE, "nlp_purpose_analysis_act2" + SUFFIX)  # csv inputs
# all outputs (figures + the value/p-value csv "caches") go to ./figures[_hasref] next to this script
FIG_DIR = os.path.join(BASE_DIR, "figures" + SUFFIX)

# try the reference font but fall back silently if unavailable (e.g. on Linux)
try:
    import matplotlib.font_manager as fm
    if any("Gill Sans MT" in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = "Gill Sans MT"
except Exception:
    pass

# x-axis: citation types to draw (non_disruptive excluded)
X_TYPES = ["f", "e", "g", "disruptive"]
X_LABELS = ["F", "E", "G", "D"]

# ACT2 figure: draw only the purpose-ratio panels (drop the leading "Frequency"
# panel) and lay the 6 purpose panels out as a 2x3 grid.
INCLUDE_FREQUENCY = False

# number of subplot columns in the panel grid.
NCOLS = 3

# Second figure: regroup the 6 ACT2 purposes into 5 categories. The ONLY merge is
# "Intent to Modify" = CompareOrContrast + Extends; Background, Motivation, Future
# and Uses stay as-is. The 5 merged panels are laid out as a 2x3 grid (NCOLS_MERGED).
MERGE_MAP = {
    "Background": ["Background"],
    "Motivation": ["Motivation"],
    "Intent to Modify": ["CompareOrContrast", "Extends"],
    "Future": ["Future"],
    "Uses": ["Uses"],
}
MERGED_PURPOSE_ORDER = ["Background", "Motivation", "Intent to Modify", "Future", "Uses"]
NCOLS_MERGED = 3

COLOR_MAP = {
    "Frequency": "#4C72B0",
    "Background": "#55A868",
    "CompareOrContrast": "#F28E2B",
    "Extends": "#59A14F",
    "Future": "#E15759",
    "Motivation": "#B07AA1",
    "Uses": "#9C755F",
    "Intent to Modify": "#F28E2B",
}


# all 6 pairwise comparisons among the 4 positions (E,F,G,D)
ALL_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

# ---- bootstrap 95% CI settings (drawn as error bars on every bar) ----
N_BOOT = 1000
BOOT_SEED = 42
CI = 95


def bootstrap_ci_ratio(n, k, rng):
    """Nonparametric bootstrap 95% CI for a proportion k/n. The underlying sample
    is n 0/1 indicators with k ones, so a bootstrap resample's mean is exactly
    Binomial(n, k/n)/n -- drawn directly (the exact bootstrap distribution of the
    mean of a 0/1 sample), which avoids materialising the n-length array."""
    if n is None or n < 1:
        return np.nan, np.nan
    p = k / n
    boot = rng.binomial(n, p, size=N_BOOT) / n
    lo = float(np.percentile(boot, (100 - CI) / 2))
    hi = float(np.percentile(boot, 100 - (100 - CI) / 2))
    return lo, hi


def bootstrap_ci_mean(m, v, n, rng):
    """Parametric (CLT) bootstrap 95% CI for a mean given only its summary stats
    (mean m, sample variance v, size n): bootstrap means ~ Normal(m, sqrt(v/n)).
    Used for the Frequency panel, where only the per-type mean/var/n are stored
    (the raw per-citation instance counts are not available to resample directly)."""
    if n is None or n < 2 or not np.isfinite(m) or not np.isfinite(v):
        return m, m
    se = math.sqrt(v / n)
    if se == 0:
        return m, m
    boot = rng.normal(m, se, size=N_BOOT)
    lo = float(np.percentile(boot, (100 - CI) / 2))
    hi = float(np.percentile(boot, 100 - (100 - CI) / 2))
    return lo, hi


def get_stars(p):
    if p is None:
        return None
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "ns(.)"   # marginal: significant at the 90% level only
    return None


def _t_sf(t_abs, df):
    try:
        from scipy import stats
        return float(stats.t.sf(t_abs, df))
    except Exception:
        return 0.5 * math.erfc(t_abs / math.sqrt(2.0))


def welch_p_from_summary(m1, v1, n1, m2, v2, n2):
    """Two-sided Welch t-test p-value from summary stats (means, sample vars, ns)."""
    if n1 < 2 or n2 < 2:
        return 1.0
    se1, se2 = v1 / n1, v2 / n2
    denom = math.sqrt(se1 + se2)
    if denom == 0:
        return 1.0
    t = (m1 - m2) / denom
    df = (se1 + se2) ** 2 / ((se1 ** 2) / (n1 - 1) + (se2 ** 2) / (n2 - 1))
    return 2.0 * _t_sf(abs(t), df)


def welch_p_binary(n1, k1, n2, k2):
    """Welch t-test p-value between two 0/1 samples given as (size, #ones)."""
    if n1 < 2 or n2 < 2:
        return 1.0
    p1, p2 = k1 / n1, k2 / n2
    v1 = n1 * p1 * (1 - p1) / (n1 - 1)
    v2 = n2 * p2 * (1 - p2) / (n2 - 1)
    return welch_p_from_summary(p1, v1, n1, p2, v2, n2)


def add_sig_brackets(ax, xpos, pairs_p, bar_top, ymin):
    """Draw ALL pairwise brackets (significant -> stars, otherwise "ns"), packing
    non-overlapping ones on the same vertical level. pairs_p: list of ((i,j), p)."""
    items = []
    for (i, j), p in pairs_p:
        label = get_stars(p) or "ns"
        items.append((abs(i - j), min(i, j), max(i, j), label))
    if not items:
        return bar_top
    items.sort(key=lambda z: (z[0], z[1]))  # short spans low, then left-to-right

    total = max(bar_top - ymin, 1e-9)
    base_gap = total * 0.06
    layer_gap = total * 0.12
    line_h = total * 0.02
    lab_off = total * 0.005

    levels = []   # levels[k] = list of (lo, hi) intervals already placed there
    placed = []
    for _, lo, hi, s in items:
        lvl = 0
        while True:
            if lvl >= len(levels):
                levels.append([])
            if all(hi <= a or lo >= b for (a, b) in levels[lvl]):
                levels[lvl].append((lo, hi))
                placed.append((lvl, lo, hi, s))
                break
            lvl += 1

    top = bar_top
    for lvl, lo, hi, s in placed:
        y = bar_top + base_gap + lvl * layer_gap
        x1, x2 = xpos[lo], xpos[hi]
        ax.plot([x1, x1, x2, x2], [y, y + line_h, y + line_h, y], lw=0.9, c="black")
        fs = 7 if s in ("ns", "ns(.)") else 9
        ax.text((x1 + x2) / 2, y + line_h + lab_off, s,
                ha="center", va="bottom", fontsize=fs)
        top = max(top, y + line_h + lab_off)
    return top


def load_inputs():
    def rd(name):
        path = os.path.join(OUT_DIR, name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing input: {path}\nRun efg_citation_purpose_act2.py first.")
        return pd.read_csv(path)

    efg_cross = rd("efg_purpose_crosstab.csv")
    disr_cross = rd("disruptive_purpose_crosstab.csv")
    avg = rd("avg_instances_per_citation.csv")
    if "var_instances_per_citation" not in avg.columns:
        raise KeyError(
            "avg_instances_per_citation.csv is missing 'var_instances_per_citation'.\n"
            "It was produced by an older efg_citation_purpose_act2.py. Please re-run\n"
            "efg_citation_purpose_act2.py to regenerate the analysis CSVs, then re-run "
            "this script.")
    return efg_cross, disr_cross, avg


def render_figure(purposes, kcount, type_n, fmean, fvar, fn,
                  out_stub, include_frequency, ncols):
    """Draw one S2 panel figure for a given purpose set and write its value /
    p-value CSVs. kcount[(type, purpose)] is the instance count and type_n[type]
    the per-type total, so the same renderer serves both the original 6-purpose
    figure and the merged 4-category figure (the caller just passes a different
    purpose list + kcount). Every bar gets a bootstrap 95% CI error bar and all 6
    pairwise Welch brackets. Outputs: {out_stub}.svg/.png/_values.csv/_pvalues.csv."""
    # ratio recomputed from integer counts (exact; the CSV ratio is rounded)
    ratio = {key: k / type_n[key[0]] for key, k in kcount.items()}

    # each type's purpose ratios must sum to 1 (each instance has exactly 1 purpose)
    for t in X_TYPES:
        s = sum(ratio.get((t, p), 0.0) for p in purposes)
        assert abs(s - 1.0) < 1e-8, (
            f"purpose ratios for type '{t}' sum to {s!r}, not 1 (within 1e-8)")

    def pair_pvalue(panel, i, j):
        a, b = X_TYPES[i], X_TYPES[j]
        if panel == "Frequency":
            return welch_p_from_summary(fmean[a], fvar[a], fn[a],
                                        fmean[b], fvar[b], fn[b])
        return welch_p_binary(type_n[a], kcount[(a, panel)],
                              type_n[b], kcount[(b, panel)])

    # ---- panels: Frequency first (optional), then one per purpose ----
    panels = (["Frequency"] if include_frequency else []) + list(purposes)
    nrows = int(np.ceil(len(panels) / ncols))
    # match the per-panel size of the openalex combined (2x3) figure in
    # ../nlp_citation_purpose_measure/efg_citation_purpose_openalex_s2_plot.py
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(4.9 * ncols, 4.3 * nrows))
    axes = np.atleast_1d(axes).flatten()
    x = np.arange(len(X_TYPES))

    plot_rows, pval_rows = [], []
    boot_rng = np.random.default_rng(BOOT_SEED)
    for idx, panel in enumerate(panels):
        ax = axes[idx]
        if panel == "Frequency":
            values = [fmean.get(t, np.nan) for t in X_TYPES]
            cis = [bootstrap_ci_mean(fmean.get(t, np.nan), fvar.get(t, np.nan),
                                     fn.get(t, 0), boot_rng) for t in X_TYPES]
            ylabel = "Frequency"
        else:
            values = [ratio.get((t, panel), np.nan) for t in X_TYPES]
            cis = [bootstrap_ci_ratio(type_n.get(t, 0), kcount.get((t, panel), 0),
                                      boot_rng) for t in X_TYPES]
            ylabel = "Ratio"
        ci_lo = np.array([c[0] for c in cis], dtype=float)
        ci_hi = np.array([c[1] for c in cis], dtype=float)
        values_arr = np.array(values, dtype=float)
        # asymmetric error bars; clip tiny negatives from percentile rounding
        yerr = np.vstack([np.clip(values_arr - ci_lo, 0, None),
                          np.clip(ci_hi - values_arr, 0, None)])
        for t, v, lo, hi in zip(X_TYPES, values, ci_lo, ci_hi):
            plot_rows.append([panel, t, v, lo, hi])

        bars = ax.bar(x, values, color=COLOR_MAP.get(panel, "#4C72B0"), width=0.82,
                      yerr=yerr, capsize=3,
                      error_kw={"elinewidth": 0.9, "ecolor": "black"})
        # distinguish D (disruptive, last bar) with diagonal hatching
        d_idx = X_TYPES.index("disruptive")
        bars[d_idx].set_hatch("///")
        bars[d_idx].set_edgecolor("white")
        bars[d_idx].set_linewidth(0.0)

        # include the error-bar extents in the axis / significance-bracket geometry
        vmin = float(np.nanmin(ci_lo))
        vmax = float(np.nanmax(ci_hi))
        ymin = max(0.0, vmin - 0.4 * (vmax - vmin + 1e-9))

        # all 6 pairwise comparisons -- still computed and written to the p-value CSV
        # cache, but NOT drawn as brackets on the panel (no comparison lines wanted).
        for (i, j) in ALL_PAIRS:
            p = pair_pvalue(panel, i, j)
            pval_rows.append([panel, X_LABELS[i], X_LABELS[j], p,
                              get_stars(p) or "ns"])

        ax.set_ylim(ymin, vmax + 0.04 * (vmax - ymin + 1e-9))
        # font sizes match the openalex combined figure in
        # ../nlp_citation_purpose_measure/efg_citation_purpose_openalex_s2_plot.py
        ax.set_title(panel, fontsize=26, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(X_LABELS, fontsize=16)
        ax.tick_params(axis="y", labelsize=14)
        # the Frequency panel keeps its own distinct y-label; the Ratio panels share a single
        # figure-level "Ratio" label (added once after the loop), so no per-panel label here.
        if panel == "Frequency":
            ax.set_ylabel(ylabel, fontsize=20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j in range(len(panels), len(axes)):
        axes[j].set_visible(False)

    # single shared x-axis label, centred across the figure
    fig.supxlabel("Citation type", fontsize=20)
    # single shared y-axis label for the Ratio panels, centred on the y-axis
    fig.supylabel("Ratio", fontsize=20)

    fig.tight_layout()

    out_svg = out_stub + ".svg"
    out_png = out_stub + ".png"
    fig.savefig(out_svg, format="svg", transparent=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    out_csv = out_stub + "_values.csv"
    pd.DataFrame(plot_rows,
                 columns=["panel", "type", "value", "ci_low", "ci_high"]).to_csv(
        out_csv, index=False)
    pval_csv = out_stub + "_pvalues.csv"
    pd.DataFrame(pval_rows,
                 columns=["panel", "type_a", "type_b", "p_value", "stars"]).to_csv(
        pval_csv, index=False)

    print("Saved:", out_svg)
    print("Saved:", out_png)
    print("Saved:", out_csv)
    print("Saved:", pval_csv)


def build_merged_kcount(kcount, type_n):
    """Sum the original per-purpose instance counts into the 4 merged categories
    (MERGE_MAP) for every citation type, returning a kcount keyed by
    (type, merged_category)."""
    merged = {}
    for t in type_n:
        for cat, comps in MERGE_MAP.items():
            merged[(t, cat)] = sum(kcount.get((t, c), 0) for c in comps)
    return merged


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    efg_cross, disr_cross, avg = load_inputs()

    purposes = sorted(efg_cross["purpose"].unique())
    cross = pd.concat([efg_cross, disr_cross], ignore_index=True)

    # ---- ratio + per-(type,purpose) instance count, and per-type totals ----
    kcount = {}
    for _, r in cross.iterrows():
        kcount[(r["type"], r["purpose"])] = int(r["instance_count"])
    type_n = cross.groupby("type")["instance_count"].sum().to_dict()  # total instances/type

    # ---- frequency stats per type: mean, var, n(=citations) ----
    fmean = {r["type"]: r["avg_instances_per_citation"] for _, r in avg.iterrows()}
    fvar = {r["type"]: r["var_instances_per_citation"] for _, r in avg.iterrows()}
    fn = {r["type"]: int(r["n_citations"]) for _, r in avg.iterrows()}

    # ---- report how many qualifying citation links / purpose instances feed the figure ----
    # type_n[type] = purpose instances per type (the ratio/test denominator); n_citations /
    # n_instances come from avg_instances_per_citation.csv (a citation link can yield several
    # instances). E/F/G is one partition of the citations and disruptive/non_disruptive another,
    # so the two partitions each cover the same underlying citation links.
    n_citations_by_type = {r["type"]: int(r["n_citations"]) for _, r in avg.iterrows()}
    n_instances_by_type = {r["type"]: int(r["n_instances"]) for _, r in avg.iterrows()}
    print("-" * 60)
    print("qualifying citation links / purpose instances per citation type:")
    for t, lab in zip(X_TYPES, X_LABELS):
        print(f"    {lab:>2} ({t:>13}): {n_citations_by_type.get(t, 0):>10} citation links, "
              f"{n_instances_by_type.get(t, 0):>10} instances "
              f"(crosstab instances: {int(type_n.get(t, 0)):>10})")
    efg_types = ["f", "e", "g"]
    print(f"    total (E/F/G partition): "
          f"{sum(n_citations_by_type.get(t, 0) for t in efg_types):>10} citation links, "
          f"{sum(int(type_n.get(t, 0)) for t in efg_types):>10} instances")
    print("-" * 60)

    # ---- figure 1: the original per-purpose panels ----
    render_figure(
        purposes, kcount, type_n, fmean, fvar, fn,
        os.path.join(FIG_DIR, "efg_purpose_ratio_s2"),
        INCLUDE_FREQUENCY, NCOLS,
    )

    # ---- figure 2: the 4 merged categories (Background+Motivation -> Background,
    # CompareOrContrast+Extends -> "Intent to Modify", Future, Uses) ----
    merge_components = {c for comps in MERGE_MAP.values() for c in comps}
    missing = set(purposes) - merge_components
    assert not missing, (
        f"MERGE_MAP does not cover these ACT2 purposes: {sorted(missing)} "
        f"(update MERGE_MAP so every purpose is assigned to a merged category)")
    merged_kcount = build_merged_kcount(kcount, type_n)
    render_figure(
        MERGED_PURPOSE_ORDER, merged_kcount, type_n, fmean, fvar, fn,
        os.path.join(FIG_DIR, "efg_purpose_ratio_s2_merged"),
        INCLUDE_FREQUENCY, NCOLS_MERGED,
    )


if __name__ == "__main__":
    main()
