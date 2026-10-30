"""
S2-style panel figure for the OPENALEX-NETWORK F / E / G / D citation-purpose analysis.

Self-contained plotting script: a grid of bar panels with the citation type on the
x-axis and Welch significance brackets, for citation types E / F / G / D where
D = "disruptive" (the complementary "non_disruptive" type is intentionally NOT drawn).
Set INCLUDE_DISP_F = True (below) to append the three disruptive-foundation variants
Df>0 / Df>1 / Df>2 (up_count == 0 & down_count > 0/1/2) as extra bars; those outputs carry
a "_dispf" filename suffix so both versions coexist. Reads the CSVs produced by
efg_citation_purpose_openalex.py (OpenAlex-network classification).

Panels:
  * Frequency           -> avg #citation instances per citation (mention frequency)
  * one per purpose      -> ratio_within_type (fraction of that type's instances
                            having the purpose)

Inputs (produced by efg_citation_purpose_openalex.py, in
META_DIRE/nlp_purpose_analysis_openalex; META_DIRE imported from ../utils.py):
  efg_purpose_crosstab.csv            type, purpose, instance_count, ratio_within_type   (e/f/g)
  disruptive_purpose_crosstab.csv     same, for disruptive/non_disruptive
  disp_f_purpose_crosstab.csv         same, for disp_f_0/1/2 (disruptive-foundation variants)
  avg_instances_per_citation.csv      scheme, type, n_instances, n_citations, avg/var_instances_per_citation

The per-purpose Welch p-values are recomputed here straight from the crosstab
counts, so efg_citation_purpose_openalex.py's *_welch_ttest.csv files are not read.

Run efg_citation_purpose_openalex.py first, then this script. The figure (and the
value / p-value CSV "caches") are written to ./figures[_hasref] next to this script.

At the paper's settings (REQUIRE_REFS = False, COMBINE_PURPOSES = True,
INCLUDE_DISP_F = False) this figure IS 'Fig. 2f' (is_fig2f): the svg/png AND the two CSV
caches beside them are named after the panel; every other combination keeps the descriptive
stem efg_purpose_ratio_openalex_s2{_combined}{_dispf}. That same combination is the one
reproduced at full page width, so it is also what turns on the enlarged axis fonts
(BIG_AXIS_FONTS). Outputs:
  ./figures[_hasref]/{stem}.svg / {stem}.png
  ./figures[_hasref]/{stem}_values.csv    (panel, type, value, ci_low, ci_high)
  ./figures[_hasref]/{stem}_pvalues.csv   (panel, type_a, type_b, p_value, stars)
with stem = 'Fig. 2f' at the deliverable setting.

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
# efg_citation_purpose_openalex.py. When True, read the CSVs produced by the
# "papers must have references" version (suffixed input dir) and write the
# figures to a matching suffixed directory so the two versions coexist.
REQUIRE_REFS = False
SUFFIX = "_hasref" if REQUIRE_REFS else ""

# When True, draw the COMBINED version: the CompareOrContrast and Extends purposes are
# merged into a single "Intent To Modify" category (their per-(type) instance counts are
# summed before ratios / Welch tests are computed), which drops the panel count from 7 to 6
# and lays them out as a 2x3 grid. Outputs carry a "_combined" suffix so the two versions
# coexist.
COMBINE_PURPOSES = True
COMBINED_SUFFIX = "_combined" if COMBINE_PURPOSES else ""
MERGED_LABEL = "Intent To Modify"
MERGE_MAP = {"CompareOrContrast": MERGED_LABEL, "Extends": MERGED_LABEL}

# When True, add the three disruptive-foundation variants (Df>0/Df>1/Df>2, i.e.
# up_count == 0 & down_count > 0/1/2) as extra bars after D; requires the disp_f crosstab
# from efg_citation_purpose_openalex.py. When False, draw only the F/E/G/D bars. Outputs
# carry a "_dispf" suffix so the two versions coexist.
INCLUDE_DISP_F = False
DISPF_SUFFIX = "_dispf" if INCLUDE_DISP_F else ""

OUT_DIR = os.path.join(META_DIRE, "nlp_purpose_analysis_openalex" + SUFFIX)  # csv inputs
# all outputs (figures + the value/p-value csv "caches") go to ./figures[_hasref] next to
# this script. The OpenAlex-network outputs carry an "_openalex" stem in their filenames.
FIG_DIR = os.path.join(BASE_DIR, "figures" + SUFFIX)

# try the reference font but fall back silently if unavailable (e.g. on Linux)
try:
    import matplotlib.font_manager as fm
    if any("Gill Sans MT" in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = "Gill Sans MT"
except Exception:
    pass

# x-axis: citation types to draw (non_disruptive excluded). Always F/E/G/D; when
# INCLUDE_DISP_F is on, the three disruptive-foundation variants (up_count == 0 &
# down_count > 0/1/2) are appended as extra bars (disp_f_0 is the F subset of D).
DISP_F_TYPES = ["disp_f_0", "disp_f_1", "disp_f_2"]
DISP_F_LABELS = ["Df>0", "Df>1", "Df>2"]
X_TYPES = ["f", "e", "g", "disruptive"] + (DISP_F_TYPES if INCLUDE_DISP_F else [])
X_LABELS = ["F", "E", "G", "D"] + (DISP_F_LABELS if INCLUDE_DISP_F else [])
# disruption-family bars drawn with a hatch to set them apart from the E/F/G bars
HATCH_TYPES = {"disruptive"} | (set(DISP_F_TYPES) if INCLUDE_DISP_F else set())

# whether to draw the leading "Frequency" (mention-frequency) panel.
INCLUDE_FREQUENCY = True

# number of subplot columns in the panel grid (3 -> 2x3 for the combined version).
NCOLS = 3 if COMBINE_PURPOSES else 4

# ---- deliverable naming ----------------------------------------------------------------
# The REQUIRE_REFS=False + COMBINE_PURPOSES=True + INCLUDE_DISP_F=False variant IS Fig. 2f,
# so its figure AND the value / p-value CSV "caches" beside it are named after the panel;
# every other combination keeps the descriptive stem. The gate lists all three toggles --
# the only parameters that change what the bars show (which CSVs are read, whether
# CompareOrContrast+Extends are merged, and which bars are drawn) -- so no other run can
# land on this name.
FIG_NAME = "Fig. 2f"
is_fig2f = (not REQUIRE_REFS) and COMBINE_PURPOSES and (not INCLUDE_DISP_F)
print("is_fig2f", is_fig2f, ("-> %s" % FIG_NAME) if is_fig2f else "")

# Axis-label / tick font sizes. The deliverable variant is the one reproduced at full page
# width, so it uses enlarged fonts; every other combination keeps the original sizes.
# (tweakable)
BIG_AXIS_FONTS = is_fig2f
XLABEL_FS = 28 if BIG_AXIS_FONTS else 24
YLABEL_FS = 28 if BIG_AXIS_FONTS else 20
XTICK_FS = 22 if BIG_AXIS_FONTS else (13 if INCLUDE_DISP_F else 16)
YTICK_FS = 20 if BIG_AXIS_FONTS else 14

COLOR_MAP = {
    "Frequency": "#4C72B0",
    "Background": "#55A868",
    "CompareOrContrast": "#F28E2B",
    "Extends": "#59A14F",
    "Future": "#E15759",
    "Motivation": "#B07AA1",
    "Uses": "#9C755F",
    "Intent To Modify": "#F28E2B",   # combined CompareOrContrast + Extends
}


# all pairwise comparisons among the drawn types (written to the p-value CSV, not drawn)
ALL_PAIRS = [(i, j) for i in range(len(X_TYPES)) for j in range(i + 1, len(X_TYPES))]

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
        label = get_stars(p)
        # only draw genuinely-significant brackets; skip non-significant ("ns") and
        # marginal ("ns(.)") pairs so they don't clutter the panel.
        if label is None or label == "ns(.)":
            continue
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
                f"Missing input: {path}\nRun efg_citation_purpose_openalex.py first.")
        return pd.read_csv(path)

    efg_cross = rd("efg_purpose_crosstab.csv")
    disr_cross = rd("disruptive_purpose_crosstab.csv")
    dispf_cross = rd("disp_f_purpose_crosstab.csv") if INCLUDE_DISP_F else None
    avg = rd("avg_instances_per_citation.csv")
    if "var_instances_per_citation" not in avg.columns:
        raise KeyError(
            "avg_instances_per_citation.csv is missing 'var_instances_per_citation'.\n"
            "It was produced by an older efg_citation_purpose_openalex.py. Please re-run\n"
            "efg_citation_purpose_openalex.py to regenerate the analysis CSVs, then re-run "
            "this script.")
    return efg_cross, disr_cross, dispf_cross, avg


def combine_purposes(df):
    """Merge the MERGE_MAP purposes (CompareOrContrast + Extends -> 'Intent To Modify')
    by summing their instance_count within each type. ratio_within_type is dropped here
    because main() recomputes ratios from the integer counts anyway."""
    df = df.copy()
    df["purpose"] = df["purpose"].replace(MERGE_MAP)
    return df.groupby(["type", "purpose"], as_index=False)["instance_count"].sum()


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    efg_cross, disr_cross, dispf_cross, avg = load_inputs()

    if COMBINE_PURPOSES:
        efg_cross = combine_purposes(efg_cross)
        disr_cross = combine_purposes(disr_cross)
        if dispf_cross is not None:
            dispf_cross = combine_purposes(dispf_cross)

    purposes = sorted(efg_cross["purpose"].unique())
    cross_parts = [efg_cross, disr_cross] + ([dispf_cross] if dispf_cross is not None else [])
    cross = pd.concat(cross_parts, ignore_index=True)

    # ---- ratio + per-(type,purpose) instance count, and per-type totals ----
    kcount = {}
    for _, r in cross.iterrows():
        kcount[(r["type"], r["purpose"])] = int(r["instance_count"])
    type_n = cross.groupby("type")["instance_count"].sum().to_dict()  # total instances/type
    # ratio recomputed from integer counts (exact; the CSV ratio is rounded)
    ratio = {key: k / type_n[key[0]] for key, k in kcount.items()}

    # each type's purpose ratios must sum to 1 (each instance has exactly 1 purpose).
    # Skip types with no instances at all (e.g. a disp_f threshold nobody meets).
    for t in X_TYPES:
        if type_n.get(t, 0) == 0:
            continue
        s = sum(ratio.get((t, p), 0.0) for p in purposes)
        assert abs(s - 1.0) < 1e-8, (
            f"purpose ratios for type '{t}' sum to {s!r}, not 1 (within 1e-8)")

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

    def pair_pvalue(panel, i, j):
        a, b = X_TYPES[i], X_TYPES[j]
        if panel == "Frequency":
            return welch_p_from_summary(fmean[a], fvar[a], fn[a],
                                        fmean[b], fvar[b], fn[b])
        return welch_p_binary(type_n[a], kcount[(a, panel)],
                              type_n[b], kcount[(b, panel)])

    # ---- panels: Frequency first (optional), then one per purpose ----
    panels = (["Frequency"] if INCLUDE_FREQUENCY else []) + list(purposes)
    x = np.arange(len(X_TYPES))

    # Layout. For the non-combined 7-panel version (Frequency + 6 purposes) use a custom
    # 4-row x 4-col GridSpec where every panel spans 2 row-units (so all panels share the
    # SAME width and height). The 6 purpose panels fill the right 3 columns x 2 rows; the
    # Frequency panel sits in the leftmost column offset by one row-unit (rows 1-2), so it
    # is vertically centred between the two purpose rows. Otherwise fall back to a plain
    # nrows x NCOLS grid.
    special_layout = (not COMBINE_PURPOSES) and INCLUDE_FREQUENCY and len(purposes) == 6
    if special_layout:
        ncols = 4
        # 2 visual rows tall; 4 grid-rows so a panel (2 units) can be centred by a 1-unit offset.
        fig = plt.figure(figsize=(4.9 * ncols, 4.3 * 2))
        gs = fig.add_gridspec(4, ncols)
        # axes order MUST match `panels` order: Frequency first, then the 6 purposes.
        # Frequency: left column, rows 1-2 -> same size as a purpose panel, vertically centred.
        axes = [fig.add_subplot(gs[1:3, 0])]
        # purposes: right 3 columns, each spanning 2 row-units (top rows 0-1, bottom rows 2-3).
        purpose_cells = [(0, 1), (0, 2), (0, 3), (2, 1), (2, 2), (2, 3)]
        for (r, c) in purpose_cells:
            axes.append(fig.add_subplot(gs[r:r + 2, c]))
        axes = np.array(axes, dtype=object)
    else:
        ncols = NCOLS
        nrows = int(np.ceil(len(panels) / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                 figsize=(4.9 * ncols, 4.3 * nrows))
        axes = np.atleast_1d(axes).flatten()

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
        # distinguish the disruption-family bars (D + disp_f variants) with diagonal hatching
        for hi in [k for k, t in enumerate(X_TYPES) if t in HATCH_TYPES]:
            bars[hi].set_hatch("///")
            bars[hi].set_edgecolor("white")
            bars[hi].set_linewidth(0.0)

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
        ax.set_title(panel, fontsize=26, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(X_LABELS, fontsize=XTICK_FS)
        ax.tick_params(axis="y", labelsize=YTICK_FS)
        ax.set_ylabel(ylabel, fontsize=YLABEL_FS)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j in range(len(panels), len(axes)):
        axes[j].set_visible(False)

    # single shared x-label, centred across all columns at the very bottom
    fig.supxlabel("Citation type", fontsize=XLABEL_FS)
    fig.tight_layout()

    # named after the panel at the deliverable setting, descriptive otherwise; the value /
    # p-value CSVs follow the same stem, so figure and data never drift apart.
    stem = (FIG_NAME if is_fig2f
            else f"efg_purpose_ratio_openalex_s2{COMBINED_SUFFIX}{DISPF_SUFFIX}")
    out_svg = os.path.join(FIG_DIR, f"{stem}.svg")
    out_png = os.path.join(FIG_DIR, f"{stem}.png")
    fig.savefig(out_svg, format="svg", transparent=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    out_csv = os.path.join(FIG_DIR, f"{stem}_values.csv")
    pd.DataFrame(plot_rows,
                 columns=["panel", "type", "value", "ci_low", "ci_high"]).to_csv(
        out_csv, index=False)
    pval_csv = os.path.join(FIG_DIR, f"{stem}_pvalues.csv")
    pd.DataFrame(pval_rows,
                 columns=["panel", "type_a", "type_b", "p_value", "stars"]).to_csv(
        pval_csv, index=False)

    print("Saved:", out_svg)
    print("Saved:", out_png)
    print("Saved:", out_csv)
    print("Saved:", pval_csv)


if __name__ == "__main__":
    main()
