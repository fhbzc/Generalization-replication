"""
Per-publication-year share of papers that have NO reference in the citation network
produced by get_meta_data_preprint.py.

The citation network citing_pid2cited_pids_list.pkl maps each citing pid to the list of
cited pids it references (only in-universe references published on/before the citing paper
survive; see get_meta_data_preprint.py). A paper has NO reference iff it is absent from
this mapping or its reference list is empty. The paper universe and each paper's
publication year come from pid2pub_year.pkl (same script).

For every publication year we report
    pct_no_reference = 100 * (#papers with no reference) / (#papers)

and draw it twice:
    (1) overall   -> one line over publication year
    (2) by domain -> a grid of panels, one per OpenAlex level-0 domain, using the
                     domain -> [pid] grouping in concept2pid_list.pkl (also written by
                     get_meta_data_preprint.py; a paper may belong to several domains).

The per-year coverage tables are cached to CSV (./figures/no_reference_coverage_*.csv).
On a redraw, if BOTH caches exist they are loaded directly and the pkl inputs are not
touched at all -- only the figures are rebuilt. Delete the cache CSVs (or set
force_recompute=True) to recompute from the pkls.

Input  : META_DIRE/pid2pub_year.pkl                  (get_meta_data_preprint.py)
         META_DIRE/citing_pid2cited_pids_list.pkl     (get_meta_data_preprint.py)
         META_DIRE/concept2pid_list.pkl               (get_meta_data_preprint.py; by-domain only)
Cache  : ./figures/no_reference_coverage_overall.csv
         ./figures/no_reference_coverage_by_domain.csv
Output : ./figures/no_reference_coverage/no_reference_coverage_overall.svg
         ./figures/no_reference_coverage/no_reference_coverage_by_domain.svg
"""

import os
import sys
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless / cluster-safe
import matplotlib.pyplot as plt  # noqa: E402

# utils.py is in the parent openalex/ directory
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OPENALEX_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, OPENALEX_DIR)
from utils import META_DIRE  # noqa: E402

meta_dire = META_DIRE
print("meta_dire", meta_dire)

# plotting window over publication year (CSV keeps every year; the plot is clipped)
year_min, year_max = 1945, 2024
print("plot year range:", year_min, "-", year_max)

# set True to ignore any existing cache and recompute the coverage tables from the pkls
force_recompute = False

fig_dire = os.path.join(THIS_DIR, 'figures')
os.makedirs(fig_dire, exist_ok=True)
# the rendered figures go into a dedicated subdirectory under ./figures; the cache CSVs
# stay directly in ./figures so a redraw can find them.
img_dire = os.path.join(fig_dire, 'no_reference_coverage')
os.makedirs(img_dire, exist_ok=True)

overall_csv = os.path.join(fig_dire, 'no_reference_coverage_overall.csv')
by_domain_csv = os.path.join(fig_dire, 'no_reference_coverage_by_domain.csv')


def load_pkl(path):
    assert os.path.exists(path), path
    with open(path, 'rb') as f:
        return pickle.load(f)


# ==================================================
# 1. obtain the per-year coverage tables: load from the CSV cache if both are present,
#    otherwise compute them from the pkls and write the cache.
# ==================================================
cache_ready = (not force_recompute
               and os.path.exists(overall_csv)
               and os.path.exists(by_domain_csv))

if cache_ready:
    overall = pd.read_csv(overall_csv)
    by_domain_long = pd.read_csv(by_domain_csv)
    # reconstruct {domain: per-year df} and the domain order from the long-form cache
    domains = sorted(by_domain_long['domain'].unique())
    domain2df = {d: g.drop(columns='domain').reset_index(drop=True)
                 for d, g in by_domain_long.groupby('domain')}
    print("loaded cache:", overall_csv, "and", by_domain_csv,
          "| domains:", len(domains))
else:
    print("cache not found (or force_recompute) -> computing from pkls ...")

    pid2pub_year = load_pkl(os.path.join(meta_dire, 'pid2pub_year.pkl'))
    print("pid2pub_year size:", len(pid2pub_year))

    citing_net = load_pkl(os.path.join(meta_dire, 'citing_pid2cited_pids_list.pkl'))
    print("citing_pid2cited_pids_list size:", len(citing_net))

    # a paper "has a reference" iff it is a key with a non-empty cited list
    has_ref_set = {pid for pid, cited_list in citing_net.items() if cited_list}
    del citing_net
    print("papers with >=1 reference:", len(has_ref_set))

    def coverage_table(pids):
        """Per-year (total, no_reference, pct_no_reference) over the given iterable of pids.
        Returns a DataFrame sorted by pub_year. Pids without a known pub_year are skipped."""
        year2total = Counter()
        year2no_ref = Counter()
        for pid in pids:
            year = pid2pub_year.get(pid)
            if year is None:
                continue
            year2total[year] += 1
            if pid not in has_ref_set:
                year2no_ref[year] += 1

        rows = []
        for year in sorted(year2total):
            total = year2total[year]
            no_ref = year2no_ref[year]
            rows.append({
                'pub_year': year,
                'n_papers': total,
                'n_no_reference': no_ref,
                'pct_no_reference': 100.0 * no_ref / total if total else np.nan,
            })
        return pd.DataFrame(rows)

    # overall coverage
    overall = coverage_table(pid2pub_year.keys())
    overall.to_csv(overall_csv, index=False)
    print("wrote", overall_csv, overall.shape)

    # by-domain coverage (domain -> [pid] from concept2pid_list.pkl)
    concept2pid_list = load_pkl(os.path.join(meta_dire, 'concept2pid_list.pkl'))
    domains = sorted(concept2pid_list)
    print("domains:", len(domains), domains)

    by_domain_rows = []
    domain2df = {}
    for domain in domains:
        df_d = coverage_table(concept2pid_list[domain])
        domain2df[domain] = df_d
        if not df_d.empty:
            tagged = df_d.copy()
            tagged.insert(0, 'domain', domain)
            by_domain_rows.append(tagged)

    if by_domain_rows:
        pd.concat(by_domain_rows, ignore_index=True).to_csv(by_domain_csv, index=False)
        print("wrote", by_domain_csv)

# ==================================================
# 2. overall figure
# ==================================================
plot_df = overall[(overall['pub_year'] >= year_min) & (overall['pub_year'] <= year_max)]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(plot_df['pub_year'], plot_df['pct_no_reference'],
        color='tab:blue', linewidth=2, marker='o', markersize=3)
ax.set_xlabel('Publication year')
ax.set_ylabel('% of papers with no reference')
ax.set_xlim(year_min, year_max)
ax.grid(alpha=0.3)
fig.tight_layout()
overall_svg = os.path.join(img_dire, 'no_reference_coverage_overall.svg')
fig.savefig(overall_svg, format='svg', transparent=True)
plt.close(fig)
print("saved", overall_svg)

# ==================================================
# 3. by-domain figure
# grid of per-domain panels (4 columns), same layout idea as visualize_efg_trend_by_domain.py
# ==================================================
# the 'Others' bucket (papers with no level-0 concept) is not a real domain -> don't plot it
plot_domains = [d for d in domains if d != 'Others']
ncols = 4
nrows = int(np.ceil(len(plot_domains) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.5 * nrows), squeeze=False)

for k, domain in enumerate(plot_domains):
    r, c = divmod(k, ncols)
    ax = axes[r, c]
    df_d = domain2df[domain]
    sub = df_d[(df_d['pub_year'] >= year_min) & (df_d['pub_year'] <= year_max)]
    if not sub.empty:
        ax.plot(sub['pub_year'], sub['pct_no_reference'],
                color='tab:blue', linewidth=1.8)
    # pooled average over ALL papers of this domain (every year): a single horizontal
    # reference line, annotated with its value. Prefer the paper-weighted pooled rate
    # 100 * sum(no_reference) / sum(papers); fall back to the unweighted mean of
    # pct_no_reference for older caches that lack the count columns.
    avg_pct = None
    if {'n_papers', 'n_no_reference'}.issubset(df_d.columns) and df_d['n_papers'].sum() > 0:
        avg_pct = 100.0 * df_d['n_no_reference'].sum() / df_d['n_papers'].sum()
    elif 'pct_no_reference' in df_d.columns and df_d['pct_no_reference'].notna().any():
        avg_pct = float(df_d['pct_no_reference'].mean())
    if avg_pct is not None:
        ax.axhline(avg_pct, color='tab:red', linestyle='--', linewidth=1.3)
        ax.text(0.98, avg_pct, f'avg {avg_pct:.1f}%',
                transform=ax.get_yaxis_transform(), ha='right', va='bottom',
                color='tab:red', fontsize=9)
    # domain name as a text in the top-right corner instead of a subplot title
    ax.text(0.97, 0.97, str(domain).capitalize(), transform=ax.transAxes,
            ha='right', va='top', fontsize=18)
    ax.set_xlim(year_min, year_max)
    # drop the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# turn off any leftover empty axes
for k in range(len(plot_domains), nrows * ncols):
    r, c = divmod(k, ncols)
    axes[r, c].set_axis_off()

# one shared x-label (bottom, centred) and one shared y-label (left, centred) for
# the whole grid, instead of per-row / per-column labels. Reserve a left/bottom
# band via tight_layout's rect, then place the (enlarged) labels in it -- the
# y-label sits well left of the tick numbers so they don't crowd each other.
fig.tight_layout(rect=[0.05, 0.04, 1, 1])
fig.supxlabel('Publication year', fontsize=22, y=0.012)
fig.supylabel('Percentage of papers with zero reference', fontsize=22, x=0.013)
by_domain_svg = os.path.join(img_dire, 'no_reference_coverage_by_domain.svg')
fig.savefig(by_domain_svg, format='svg', transparent=True)
plt.close(fig)
print("saved", by_domain_svg)

print("finished")
