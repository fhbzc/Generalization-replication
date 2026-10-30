"""
E / F / G / D (+ optional disp_f) paneled bar figure of mention-position ratios + mention
frequency, in the style of ../../openalex/nlp_citation_purpose_measure/efg_citation_purpose_openalex_s2_plot.py
(D = disruptive added to E/F/G; set INCLUDE_DISP_F = True to also append the three
disruptive-foundation variants Df>0/Df>1/Df>2 as extra bars, in which case the outputs carry
a "_dispf" filename suffix so both versions coexist; each bar carries a 95% CI whisker; the
disruption-family bars are hatched; NO pairwise-comparison brackets; a single shared
"Citation type" x-label). The 95%
CI needs the per-(panel, type) sample variance, which the updated
analyze_refer_mention_position.py writes as a 'var' column -- rerun it if your
stats CSVs predate that column (otherwise the bars simply have no whiskers).

Draws ONE figure for the OVERALL set (analysis_V2/efgd_full_stats.csv) AND one
figure per cited-paper DOMAIN (analysis_V2/efgd_by_domain/<domain>_stats.csv),
both produced by analyze_refer_mention_position.py.

At the paper's settings (INCLUDE_DISP_F = False and REQUIRE_REFS = False) the OVERALL
figure IS 'Fig. 2g' (is_fig2g) and is named after the panel; every other combination keeps
the descriptive efgd_mention_position{dispf} name. That same combination is the one
reproduced at full page width, so it is also what turns on the enlarged axis fonts
(BIG_AXIS_FONTS). The PER-DOMAIN figures are never a named deliverable and always keep
their <domain> filenames.

Output ({suffix} = "_hasref" when REQUIRE_REFS else ""; {dispf} = "_dispf" when
INCLUDE_DISP_F, else ""):
  ./figures/refer_mention_position{suffix}/
      Fig. 2g.svg/.png                              (overall, at the deliverable setting)
      efgd_mention_position{dispf}.svg/.png         (overall, every other setting)
      by_domain/<domain>{dispf}.svg/.png            (per domain, always)
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# utils.py is in the parent directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
from utils import S2ORC_META_DIRE  # noqa: E402

meta_dire = S2ORC_META_DIRE

# Binary toggle -- must match the value used when running
# analyze_refer_mention_position.py. When True, read the stats produced by the
# "papers must have references" version (suffixed input dir) and write the
# figures to a matching suffixed directory so the two versions coexist.
REQUIRE_REFS = False
SUFFIX = "_hasref" if REQUIRE_REFS else ""

# When True, append the three disruptive-foundation variants (Df>0/Df>1/Df>2, i.e.
# up_count == 0 & down_count > 0/1/2) as extra bars after D; needs the disp_f_* rows in
# efgd_full_stats.csv (analyze_refer_mention_position.py always writes them). When False,
# draw only the F/E/G/D bars. Outputs carry a "_dispf" suffix so the two versions coexist.
INCLUDE_DISP_F = False
DISPF_SUFFIX = "_dispf" if INCLUDE_DISP_F else ""

stats_path = os.path.join(meta_dire, "analysis_V2" + SUFFIX, "efgd_full_stats.csv")
by_domain_stats_dir = os.path.join(meta_dire, "analysis_V2" + SUFFIX, "efgd_by_domain")
out_dir = os.path.join(BASE_DIR, "figures", "refer_mention_position" + SUFFIX)
by_domain_out_dir = os.path.join(out_dir, "by_domain")
os.makedirs(out_dir, exist_ok=True)
os.makedirs(by_domain_out_dir, exist_ok=True)

# panels: (display title, key used in efgd_full_stats.csv)
# Conclusion is NOT drawn (and before_intro / after_conclusion are likewise only
# in the CSV, never plotted).
PANELS = [
    ("Frequency", "Frequency"),
    ("Introduction", "intro"),
    ("Method", "method"),
    ("Result", "result"),
    ("Result & Discussion", "res_dis"),
    ("Discussion", "discussion"),
]
COLOR_MAP = {
    "Frequency": "#4C72B0", "Introduction": "#55A868", "Method": "#F28E2B",
    "Result": "#59A14F", "Result & Discussion": "#E15759", "Discussion": "#B07AA1",
}

# Always F/E/G/D; when INCLUDE_DISP_F, append the three disruptive-foundation variants
# (up_count == 0 & down_count > 0/1/2) as extra bars (disp_f_0 is the 'f' subset of D).
DISP_F_TYPES = ["disp_f_0", "disp_f_1", "disp_f_2"]
DISP_F_LABELS = ["Df>0", "Df>1", "Df>2"]
X_TYPES = ["f", "e", "g", "disruptive"] + (DISP_F_TYPES if INCLUDE_DISP_F else [])
X_LABELS = ["F", "E", "G", "D"] + (DISP_F_LABELS if INCLUDE_DISP_F else [])
# disruption-family bars drawn with a hatch to set them apart from the E/F/G bars
HATCH_TYPES = {"disruptive"} | (set(DISP_F_TYPES) if INCLUDE_DISP_F else set())

# ---- deliverable naming ----------------------------------------------------------------
# The REQUIRE_REFS=False + INCLUDE_DISP_F=False overall figure IS Fig. 2g, so it is named
# after the panel; every other combination keeps the descriptive name. The gate lists both
# toggles -- the only parameters that change what the bars show (which stats CSVs are read
# and which bars are drawn) -- so no other run can land on this name. The per-domain figures
# are not a named deliverable and keep their <domain> names in every run.
FIG_NAME = "Fig. 2g"
is_fig2g = (not REQUIRE_REFS) and (not INCLUDE_DISP_F)
print("is_fig2g", is_fig2g, ("-> %s" % FIG_NAME) if is_fig2g else "")

# Axis-label / tick font sizes. The deliverable variant is the one reproduced at full page
# width, so it uses enlarged fonts; every other combination keeps the original sizes.
# (tweakable)
BIG_AXIS_FONTS = is_fig2g
XLABEL_FS = 28 if BIG_AXIS_FONTS else 20
YLABEL_FS = 28 if BIG_AXIS_FONTS else 16
XTICK_FS = 22 if BIG_AXIS_FONTS else (13 if INCLUDE_DISP_F else 16)
YTICK_FS = 20 if BIG_AXIS_FONTS else 14

Z_95 = 1.959963985   # two-sided normal quantile for a 95% CI


def ci95_mean(mean, var, n):
    """95% CI of the mean from summary stats (mean, sample variance, n):
    mean +/- z * sqrt(var / n). Returns (nan, nan) when it cannot be formed."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return np.nan, np.nan
    if n < 2 or not np.isfinite(mean) or not np.isfinite(var):
        return np.nan, np.nan
    se = math.sqrt(var / n)
    return mean - Z_95 * se, mean + Z_95 * se


def plot_one(stats_csv, out_svg, out_png, title=None, print_counts=False):
    """Render the F/E/G/D paneled figure from one stats CSV. When print_counts is True
    (only the overall/main figure), print the per-type citation-link / mention-instance
    counts; the per-domain figures print nothing."""
    df = pd.read_csv(stats_csv)
    mean_df = df[df["type"] == "mean"]
    means = {(r["panel"], r["group1"]): r["value"] for _, r in mean_df.iterrows()}
    # 'var' is present only on CSVs regenerated by the updated analyze script; older
    # CSVs fall back to NaN -> no CI bar drawn for that bar (r.get avoids a KeyError).
    vars = {(r["panel"], r["group1"]): r.get("var", np.nan) for _, r in mean_df.iterrows()}
    ns = {(r["panel"], r["group1"]): r.get("n", np.nan) for _, r in mean_df.iterrows()}

    # ---- report how many qualifying citation links / mention instances feed the MAIN figure ----
    # Only the overall/main figure prints this (print_counts=True); per-domain figures skip it.
    # Every panel is averaged over the SAME per-citation array (analyze_refer_mention_position.py:
    # edges_to_matrix -> n = number of citations), so ns[(panel, t)] is the CITATION-LINK count
    # for every panel -- not an instance count. The mention-instance total is recovered from the
    # Frequency panel, whose mean value = mean mentions per citation: instances = freq_mean * n.
    if print_counts:
        def _links(t):
            v = ns.get(("Frequency", t), np.nan)
            return int(v) if pd.notna(v) else 0

        def _instances(t):
            n = ns.get(("Frequency", t), np.nan)
            m = means.get(("Frequency", t), np.nan)
            if pd.notna(n) and pd.notna(m):
                return int(round(float(m) * float(n)))   # total mentions = mean-per-citation * n
            return 0

        print("-" * 60)
        print("qualifying citation links / mention instances per citation type (main figure):")
        for t, lab in zip(X_TYPES, X_LABELS):
            print(f"    {lab:>2} ({t:>13}): {_links(t):>10} citation links, "
                  f"{_instances(t):>10} instances")
        efg_types = ["f", "e", "g"]
        print(f"    total (E/F/G partition): "
              f"{sum(_links(t) for t in efg_types):>10} citation links, "
              f"{sum(_instances(t) for t in efg_types):>10} instances")
        print("-" * 60)

    ncols = 3
    nrows = int(np.ceil(len(PANELS) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(4.9 * ncols, 4.3 * nrows))
    axes = np.atleast_1d(axes).flatten()
    x = np.arange(len(X_TYPES))
    hatch_idxs = [i for i, t in enumerate(X_TYPES) if t in HATCH_TYPES]

    for idx, (panel_title, key) in enumerate(PANELS):
        ax = axes[idx]
        values = np.array([means.get((key, t), np.nan) for t in X_TYPES], dtype=float)
        cis = [ci95_mean(means.get((key, t), np.nan), vars.get((key, t), np.nan),
                         ns.get((key, t), np.nan)) for t in X_TYPES]
        ci_lo = np.array([c[0] for c in cis], dtype=float)
        ci_hi = np.array([c[1] for c in cis], dtype=float)
        # asymmetric error bars; NaN (no var) -> 0 so that bar simply has no whisker
        yerr = np.vstack([np.nan_to_num(np.clip(values - ci_lo, 0, None), nan=0.0),
                          np.nan_to_num(np.clip(ci_hi - values, 0, None), nan=0.0)])

        bars = ax.bar(x, np.nan_to_num(values, nan=0.0),
                      color=COLOR_MAP.get(panel_title, "#4C72B0"), width=0.82,
                      yerr=yerr, capsize=3,
                      error_kw={"elinewidth": 0.9, "ecolor": "black"})
        for hi in hatch_idxs:
            bars[hi].set_hatch("///")
            bars[hi].set_edgecolor("white")
            bars[hi].set_linewidth(0.0)

        # include the CI extents in the y-range
        lows = np.concatenate([values[np.isfinite(values)], ci_lo[np.isfinite(ci_lo)]])
        highs = np.concatenate([values[np.isfinite(values)], ci_hi[np.isfinite(ci_hi)]])
        vmin = float(lows.min()) if lows.size else 0.0
        vmax = float(highs.max()) if highs.size else 1.0
        ymin = max(0.0, vmin - 0.4 * (vmax - vmin + 1e-9))
        ax.set_ylim(ymin, vmax + 0.08 * (vmax - ymin + 1e-9))

        ax.set_title(panel_title, fontsize=22, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(X_LABELS, fontsize=XTICK_FS)
        ax.tick_params(axis="y", labelsize=YTICK_FS)
        ax.set_ylabel("Frequency" if key == "Frequency" else "Ratio", fontsize=YLABEL_FS)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j in range(len(PANELS), len(axes)):
        axes[j].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=16, y=0.99)
    # single shared x-label, centred across all columns at the very bottom
    fig.supxlabel("Citation type", fontsize=XLABEL_FS)
    fig.tight_layout(rect=[0, 0, 1, 0.97 if title else 1])

    fig.savefig(out_svg, format="svg", transparent=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    # ---- overall ----
    # named after the panel at the deliverable setting, descriptive otherwise
    overall_stem = FIG_NAME if is_fig2g else f"efgd_mention_position{DISPF_SUFFIX}"
    overall_svg = os.path.join(out_dir, f"{overall_stem}.svg")
    plot_one(stats_path,
             overall_svg,
             os.path.join(out_dir, f"{overall_stem}.png"),
             print_counts=True)
    print("Saved (overall):", overall_svg)

    # ---- per domain ----
    if os.path.isdir(by_domain_stats_dir):
        n = 0
        for fname in sorted(os.listdir(by_domain_stats_dir)):
            if not fname.endswith('_stats.csv'):
                continue
            domain = fname[:-len('_stats.csv')]
            plot_one(os.path.join(by_domain_stats_dir, fname),
                     os.path.join(by_domain_out_dir, f"{domain}{DISPF_SUFFIX}.svg"),
                     os.path.join(by_domain_out_dir, f"{domain}{DISPF_SUFFIX}.png"),
                     title=domain)
            n += 1
        print(f"Saved {n} per-domain figures in {by_domain_out_dir}")
    print("finished")


if __name__ == "__main__":
    main()
