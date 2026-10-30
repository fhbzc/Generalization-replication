"""
nobel_logistic_regression_sep_and_all_plot.py

Grouped bar chart combining:
  • "All"      : results from nobel_logistic_regression.py   (Chemistry+Medicine+Physics)
  • Per-domain : results from nobel_logistic_regression_sep.py (one domain at a time)

x-axis  : independent variable / metric
y-axis  : logit coefficient (standardised)
colours : one per dataset (All / Chemistry / Medicine / Physics)
hatching: non-significant bars (p >= 0.05) are hatched + lighter

Files are located via glob so that differing rt/ct or year ranges are handled
automatically.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)             # utils.py is in the parent directory
from utils import META_DIRE

# ── Directories ───────────────────────────────────────────────────────────────
meta_dire    = META_DIRE
all_save_dire = os.path.join(meta_dire, 'nobel_analysis')
sep_save_dire = os.path.join(meta_dire, 'nobel_analysis_sep')
FIG_DIRE      = os.path.join('./figures', 'nobel_validation')
os.makedirs(FIG_DIRE, exist_ok=True)

# ── Metric specs (model_name, variable column, display label) ─────────────────
METRIC_SPECS = [
    ('f_index',    'f',               'Foundation'),
    ('e_index',    'e',               'Extension'),
    ('g_index',    'g',               'Generalization'),
    ('disp',       'disp_y5',         'Disruption'),
    ('refer_dist', 'refer_dist_mean', 'Reference\nDistance'),
    ('title_dist', 'title_dist_mean', 'Semantic\nDistance'),
]

# when True, also draw a SEPARATE figure with the three disp_f ("disruptive foundation")
# variants (down>0 / down>1 / down>2) from nobel_logistic_regression_sep.py's disp_f_0/1/2
# models. Same grouped-bar style as the main figure; saved to its own '..._dispf.svg' file.
use_disp_f = True
DISPF_SPECS = [
    ('disp_f_0', 'disruption_f_y5',          'Disruptive\nFoundation\n(>0)'),
    ('disp_f_1', 'disruption_f_down_gt1_y5', 'Disruptive\nFoundation\n(>1)'),
    ('disp_f_2', 'disruption_f_down_gt2_y5', 'Disruptive\nFoundation\n(>2)'),
]

# ── Datasets: (label, ds_key, reference_threshold, citation_threshold) ────────
# ds_key='all' → reads from nobel_analysis/; otherwise reads from nobel_analysis_sep/
# Set rt/ct to match what was used when running the regression script.
DATASETS = [
    ('Chemistry', 'Chemistry', 1, 5),
    ('Medicine',  'Medicine',  1, 5),
    ('Physics',   'Physics',   1, 5),
    ('All',       'all',       1, 5),   # 'All' drawn on the far right of each metric group
]

COLORS = {
    'All':       '#4477AA',
    'Chemistry': '#EE6677',
    'Medicine':  '#228833',
    'Physics':   '#CCBB44',
}

# ── Load one coefficient row ───────────────────────────────────────────────────
def load_coef(fpath, var_name):
    """Return (coef, ci_lo, ci_hi, pvalue, pseudo_r2) or None if not found.
    Raises AssertionError if pseudo_r2 <= 0."""
    try:
        df = pd.read_excel(fpath)
    except Exception as e:
        print(f"  Could not read {fpath}: {e}")
        return None
    row = df[df['variable'] == var_name]
    if row.empty:
        return None
    r = row.iloc[0]
    pseudo_r2 = float(r['pseudo_r2'])
    assert pseudo_r2 > 0, (
        f"pseudo_R2={pseudo_r2:.4f} <= 0 — model did not converge properly.\n"
        f"  file: {fpath}\n  variable: {var_name}"
    )
    return float(r['coef']), float(r['ci_2_5']), float(r['ci_97_5']), float(r['p_value']), pseudo_r2


# ── Font sizes ────────────────────────────────────────────────────────────────
xlabel_size  = 15
ylabel_size  = 15
ytick_size   = 13
legend_size  = 14


def build_figure(metric_specs, svg_filename):
    """Collect the coefficient rows for `metric_specs`, draw the grouped bar chart, and save
    it to FIG_DIRE/`svg_filename`. Reused for the main metric figure and for the
    disp_f-variants figure. (Loads the regression xlsx directly; no data cache is written.)"""
    # results[dataset_label][metric_label] = (coef, ci_lo, ci_hi, pvalue, pseudo_r2)
    results = {ds: {} for ds, _, _, _ in DATASETS}

    for model_name, var_name, display_label in metric_specs:
        for ds_label, ds_key, rt, ct in DATASETS:
            if ds_key == 'all':
                pattern = os.path.join(all_save_dire,
                                       f'nobel_logistic_regression_{model_name}_rt_{rt}_ct_{ct}_*.xlsx')
            else:
                pattern = os.path.join(sep_save_dire,
                                       f'nobel_logistic_sep_{model_name}_dom_{ds_key}_rt_{rt}_ct_{ct}_*.xlsx')

            files = sorted(glob.glob(pattern))
            if not files:
                print(f"  WARNING: no file found for dataset={ds_label} model={model_name}")
                continue
            if len(files) > 1:
                print(f"  WARNING: multiple files for dataset={ds_label} model={model_name}, "
                      f"using latest: {files[-1]}")
            fpath = files[-1]

            rec = load_coef(fpath, var_name)
            if rec is None:
                print(f"  WARNING: variable '{var_name}' not found or pseudo_R2 <= 0 in {fpath}")
                continue

            coef, ci_lo, ci_hi, pval, pseudo_r2 = rec
            results[ds_label][display_label] = (coef, ci_lo, ci_hi, pval, pseudo_r2)
            print(f"  {ds_label:<12} {model_name:<12} coef={coef:+.4f}  "
                  f"[{ci_lo:+.4f}, {ci_hi:+.4f}]  p={pval:.4f}  pseudo_R2={pseudo_r2:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    metric_labels = [spec[2] for spec in metric_specs]
    n_metrics     = len(metric_labels)
    n_ds          = len(DATASETS)
    bar_w         = 0.17
    offsets       = np.array([-(n_ds - 1) / 2 + i for i in range(n_ds)]) * bar_w
    x             = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(11, 5))

    for di, (ds_label, _, _rt, _ct) in enumerate(DATASETS):
        color = COLORS[ds_label]
        for mi, ml in enumerate(metric_labels):
            rec = results[ds_label].get(ml)
            if rec is None:
                continue
            coef, ci_lo, ci_hi, pval, pseudo_r2 = rec
            xpos = x[mi] + offsets[di]

            ax.bar(xpos, coef, width=bar_w,
                   color=color, alpha=0.85,
                   edgecolor='white', linewidth=0.5, zorder=3)
            ax.errorbar(xpos, coef,
                        yerr=[[coef - ci_lo], [ci_hi - coef]],
                        fmt='none', color='black',
                        capsize=2.5, linewidth=1.0, zorder=4)

    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=xlabel_size)
    ax.tick_params(axis='y', labelsize=ytick_size)
    ax.set_ylabel('Scaled(Effect Size)', fontsize=ylabel_size)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend_handles = [Patch(facecolor=COLORS[ds_label], alpha=0.85, label=ds_label)
                      for ds_label, _, _, _ in DATASETS]
    ax.legend(handles=legend_handles, fontsize=legend_size, frameon=False, loc='lower right')

    plt.tight_layout()

    svg_path = os.path.join(FIG_DIRE, svg_filename)
    plt.savefig(svg_path, format='svg', bbox_inches='tight', transparent=True)
    plt.close()
    print(f"\nSaved: {svg_path}")


# ── Main metric figure ─────────────────────────────────────────────────────────
build_figure(METRIC_SPECS, 'Extended Data Fig. 7a.svg')

# ── Separate disp_f-variants figure (down>0 / down>1 / down>2) ─────────────────
if use_disp_f:
    print("\n=== disp_f variants figure ===")
    build_figure(DISPF_SPECS, 'Extended Data Fig. 7a_dispf.svg')

print("Done.")
