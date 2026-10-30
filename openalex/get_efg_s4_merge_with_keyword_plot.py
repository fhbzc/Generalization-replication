"""
Plot the E/F/G trend over publication year, split by keyword, from the
get_efg_s4_merge_with_keyword.py output.

Layout: a grid of 3 rows (Foundation / Extension / Generalization) by
(n_keywords + 1) columns. Each keyword occupies ONE column (its 3 F/E/G panels),
showing the mean share over pub_year for the papers whose title contains that
keyword, with a 95% bootstrap confidence band. The FINAL column is the
"remaining" papers: those that contain NONE of the keywords.

The keywords to plot are specified in this script (keyword_list below) as a
SUBSET of what get_efg_s4_merge_with_keyword.py produced; the s4 file may carry
more has_kw_* columns than we draw. Each chosen keyword is resolved to its
has_kw_<safe> column (asserted to exist) and "remaining" is computed against the
chosen subset only.

To avoid re-loading / re-bootstrapping the large table on every redraw, the
per-(group, index, pub_year) mean + 95% CI + n is cached to a tidy CSV. If that
cache exists the script draws straight from it and never opens the big file.

Because the "remaining" column depends on WHICH keywords were used, every output
filename carries a keyword tag (kw{N}_{hash} of the keyword set), and the exact
keyword list is written to a sidecar *_keywords.txt and printed in the figure
caption. Different keyword sets therefore never overwrite each other's files.

Input  : META_DIRE/efg_computation/
           gen_V3_y{year_offset}_with_keyword_rt_{reference_threshold}_ct_{citation_threshold}_ybefore_{max_year}.csv
Output : ./figures/keyword/   (name_stub = ay{ay}_ct{ct}_rt{rt}_ybefore{max_year}_{kw_tag})
           efg_keyword_plotcache_{name_stub}.csv     (reused on re-run)
           efg_keyword_trend_{name_stub}.svg
           efg_keyword_{name_stub}_keywords.txt      (the keyword list the tag stands for)
"""

import os
import re
import sys
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# utils.py is in this same directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from utils import META_DIRE  # noqa: E402

# repo plotting convention
try:
    plt.rcParams['font.family'] = 'Gill Sans MT'
except Exception:
    pass
# some fonts lack the Unicode minus glyph; use the ASCII hyphen instead
plt.rcParams['axes.unicode_minus'] = False

meta_dire = META_DIRE
print("meta_dire", meta_dire)

load_dire = os.path.join(meta_dire, 'efg_computation')

# ---- must match get_efg_s4_merge_with_keyword.py so the input filename resolves ----
year_offset = 5
reference_threshold = 1
citation_threshold = 5
max_year = 2025 - year_offset   # same cutoff as s4 (pub_year < 2025 - year_offset)
print("year_offset", year_offset)
print("reference_threshold", reference_threshold)
print("citation_threshold", citation_threshold)
print("max_year", max_year)

# ---- bootstrap settings ----
N_BOOT = 1000
CI = 95
SEED = 42
FORCE_REBUILD = False   # set True to ignore an existing cache and recompute
# only draw a year if its subsample has at least this many papers (plot-time only;
# the cache still stores every year so the threshold can be changed without rebuild)
MIN_PAPERS_PER_YEAR = 10
# whether to draw the trailing "remaining" (no-keyword) column (plot-time only; the
# cache still stores the remaining group regardless, so this can be toggled freely)
DRAW_REMAINING = True

# ---- venue-controlled RESHUFFLE (null-model) overlay ----
# When True, each E/F/G panel overlays the OBSERVED vs RANDOM (reshuffled) trend for
# the SAME pids. The E/F/G values come from the control-venue reshuffle pipeline
# (atypicality_control_venue/get_efg_reshuffle_control_venue.py +
# get_efg_reshuffle_s2_merge_control_venue.py) in efg_computation_control_venue/:
#   obs    = gen_V3_y{ay}_obs_valid_primary_location_before_{upper_year}.csv  (target_iteration=None)
#   random = gen_V3_y{ay}_iter{RESHUFFLE_ITER}_before_{upper_year}.csv        (target_iteration=RESHUFFLE_ITER)
# Keyword membership per pid still comes from the with_keyword source file; the three
# tables are inner-joined on pid so obs and random are compared over an identical pid set.
USE_RESHUFFLE_CONTROL = True
RESHUFFLE_ITER = 0            # which reshuffle iteration is the "random" line (target_iteration)
RESHUFFLE_UPPER_YEAR = 2025   # must match upper_year in get_efg_reshuffle_control_venue.py
print("N_BOOT", N_BOOT, "| CI", CI, "| SEED", SEED, "| MIN_PAPERS_PER_YEAR", MIN_PAPERS_PER_YEAR,
      "| DRAW_REMAINING", DRAW_REMAINING)
print("USE_RESHUFFLE_CONTROL", USE_RESHUFFLE_CONTROL,
      "| RESHUFFLE_ITER", RESHUFFLE_ITER, "| RESHUFFLE_UPPER_YEAR", RESHUFFLE_UPPER_YEAR)

# the three E/F/G index columns (normalized shares) and their row labels
index_cols = [
    ('Foundation', f'f_y{year_offset}'),
    ('Extension',  f'e_y{year_offset}'),
    ('Generalization', f'g_y{year_offset}'),
]
row_order = [name for name, _ in index_cols]
row_colors = {'Foundation': 'tab:red', 'Extension': 'tab:orange', 'Generalization': 'tab:blue'}

REMAINING_ID = '__remaining__'   # group id for the no-keyword column

# ==================================================
# keywords to PLOT (a subset of what get_efg_s4_merge_with_keyword.py produced).
# The s4 output may carry more has_kw_* columns than we want here; only these are
# drawn, and the final "remaining" column = papers containing NONE of THESE.
# ==================================================
keyword_list = ['theory', 'tool', 'software', 'review', 'guideline', 'tutorial',
                'theory', 'metric', 'hypothesis', 'discovery', 'dataset', 'observation', 'overview', 'method', 'device']

# normalize (lower-case, dedup) then SORT, so the hash, column order, plot column
# order and sidecar are all deterministic regardless of how keyword_list is typed
seen = set()
keywords = []
for kw in keyword_list:
    kw = kw.lower().strip()
    if kw and kw not in seen:
        seen.add(kw)
        keywords.append(kw)
keywords = sorted(keywords)
print("keywords to plot:", keywords)

# same keyword -> column-suffix mapping as get_efg_s4_merge_with_keyword.py
def safe_name(kw):
    return re.sub(r'[^0-9a-z]+', '_', kw).strip('_') or 'kw'

figure_dire_save = os.path.join(BASE_DIR, 'figures', 'keyword')
os.makedirs(figure_dire_save, exist_ok=True)

# source = get_efg_s4_merge_with_keyword.py output
src_path = os.path.join(
    load_dire,
    f'gen_V3_y{year_offset}_with_keyword_'
    f'rt_{reference_threshold}_'
    f'ct_{citation_threshold}_'
    f'ybefore_{max_year}.csv'
)
assert os.path.exists(src_path), src_path

# resolve the chosen keywords to their has_kw_* columns and verify they exist in
# the s4 output (read only the header: cheap, no data). The SET of keywords
# defines the "remaining" column, so it is baked into every output filename via a
# short deterministic tag and written in full to a sidecar .txt.
header = pd.read_csv(src_path, nrows=0)
kw_cols = []
for kw in keywords:
    col = f'has_kw_{safe_name(kw)}'
    assert col in header.columns, (
        f"keyword '{kw}' -> column '{col}' not in {src_path}; "
        f"regenerate get_efg_s4_merge_with_keyword.py with this keyword included."
    )
    kw_cols.append(col)
kw_labels = keywords   # display/identity labels = the keywords we chose to plot
print("keyword columns:", kw_cols)

# tag is order-independent: sort so [a, b] and [b, a] map to the same file
kw_blob = ','.join(sorted(kw_labels))
kw_tag = f'kw{len(kw_labels)}_{hashlib.md5(kw_blob.encode()).hexdigest()[:8]}'
print("keyword tag:", kw_tag)

name_stub = f'ay{year_offset}_ct{citation_threshold}_rt{reference_threshold}_ybefore{max_year}_{kw_tag}'
# reshuffle overlay gets its own cache/figure names so it never collides with the
# plain observed-only version
if USE_RESHUFFLE_CONTROL:
    name_stub += f'_reshufctrl_iter{RESHUFFLE_ITER}_before{RESHUFFLE_UPPER_YEAR}'
# The reshuffle-control run over THIS keyword set IS the Extended Data deliverable, so its
# figure + data table are named after the panel instead of carrying the parameter stub. The
# keyword set is spelled out (already normalised + sorted, like `keywords`) rather than matched
# by its md5 tag, so editing keyword_list visibly falls out of the deliverable naming instead of
# silently overwriting it under a stale name.
ED_FIG_NAME = 'Extended Data Fig. 8'
ED_FIG_KEYWORDS = ['dataset', 'device', 'discovery', 'guideline', 'hypothesis', 'method',
                   'metric', 'observation', 'overview', 'review', 'software', 'theory',
                   'tool', 'tutorial']
is_ed_fig = USE_RESHUFFLE_CONTROL and keywords == ED_FIG_KEYWORDS
print("is_ed_fig", is_ed_fig, ("-> %s" % ED_FIG_NAME) if is_ed_fig else "")

if is_ed_fig:
    out_svg = os.path.join(figure_dire_save, f'{ED_FIG_NAME}.svg')
    out_csv = os.path.join(figure_dire_save, f'{ED_FIG_NAME}.csv')
    keywords_path = os.path.join(figure_dire_save, f'{ED_FIG_NAME}_keywords.txt')
else:
    out_svg = os.path.join(figure_dire_save, f'efg_keyword_trend_{name_stub}.svg')
    out_csv = os.path.join(figure_dire_save, f'efg_keyword_trend_{name_stub}.csv')
    keywords_path = os.path.join(figure_dire_save, f'efg_keyword_{name_stub}_keywords.txt')

# ---- per-keyword caching ----
# Each keyword's per-(source, index, year) mean+CI is independent of WHICH other
# keywords are plotted, so it is cached in its own file: re-using a keyword in a
# different combination then loads its cache instead of recomputing. The cache name
# carries everything that DOES change the values -- the year/threshold params, the
# observed-vs-reshuffle mode, and the bootstrap params (N_BOOT/CI/SEED) -- but NOT the
# keyword SET (no kw_tag). The "remaining" group is the one exception: it depends on
# the whole chosen set (= papers matching none of them), so it is cached per kw_tag.
boot_tag = f'b{N_BOOT}_ci{CI}_s{SEED}'
mode_tag = (f'reshufctrl_iter{RESHUFFLE_ITER}_before{RESHUFFLE_UPPER_YEAR}'
            if USE_RESHUFFLE_CONTROL else 'obs')
base_stub = (f'ay{year_offset}_ct{citation_threshold}_rt{reference_threshold}'
             f'_ybefore{max_year}_{mode_tag}_{boot_tag}')


def group_cache_path(kw):
    return os.path.join(figure_dire_save,
                        f'efg_keyword_groupcache_{safe_name(kw)}_{base_stub}.csv')


# the remaining group is specific to the chosen keyword SET, so at the deliverable setting it
# follows the panel name too. (The per-keyword group caches above deliberately do NOT: they are
# shared across every keyword combination, and renaming them would strand that reuse.)
remaining_cache_path = os.path.join(
    figure_dire_save,
    f'{ED_FIG_NAME}_remaining_cache.csv' if is_ed_fig else
    f'efg_keyword_remaining_{kw_tag}_{base_stub}.csv')

# always (re)write the human-readable keyword list this tag stands for
with open(keywords_path, 'w', encoding='utf-8') as f:
    f.write('# keywords used for this efg keyword plot (one per line)\n')
    f.write(f'# tag: {kw_tag}\n')
    f.write(f'# the "remaining" panel = papers containing NONE of these {len(kw_labels)} keywords\n')
    f.write('\n'.join(kw_labels) + '\n')
print("[keywords saved]", keywords_path)


# ==================================================
# bootstrap helper: mean + (lo, hi) percentile CI of the mean
# ==================================================
def bootstrap_ci(values, n_boot, ci, rng):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan
    mean = float(values.mean())
    if n == 1:
        return mean, mean, mean
    # resample means in batches so memory stays bounded for very large groups
    boot_means = np.empty(n_boot, dtype=float)
    max_cells = 5_000_000
    batch = max(1, min(n_boot, max_cells // n))
    done = 0
    while done < n_boot:
        b = min(batch, n_boot - done)
        idx = rng.integers(0, n, size=(b, n))
        boot_means[done:done + b] = values[idx].mean(axis=1)
        done += b
    lo = float(np.percentile(boot_means, (100 - ci) / 2))
    hi = float(np.percentile(boot_means, 100 - (100 - ci) / 2))
    return mean, lo, hi


# ==================================================
# 1. assemble the per-group aggregation, loading each group's cache when present.
#    One cache file PER KEYWORD (independent of the chosen set) + one per-set cache
#    for the set-dependent "remaining" group. The big source file(s) are only opened
#    if at least one needed cache is missing.
# ==================================================
# sources to aggregate: observed always; the reshuffle "random" only in control mode
build_source_order = ['obs', 'random'] if USE_RESHUFFLE_CONTROL else ['obs']


def group_rng(group_id):
    """Deterministic per-group RNG so a keyword's bootstrap CI is reproducible and
    INDEPENDENT of which other keywords are plotted (a single shared rng threaded
    through every group would make a keyword's draws depend on plot order, breaking
    per-keyword caching)."""
    h = int(hashlib.md5(f'{SEED}|{group_id}'.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(h)


# lazily loaded {source -> dataframe}; each frame carries pub_year, the E/F/G index
# columns and the boolean has_kw_* columns. Built once, only when a cache is missing.
_loaded_sources = {}


def get_loaded_sources():
    if _loaded_sources:
        return _loaded_sources
    value_cols = [col for _, col in index_cols]
    if USE_RESHUFFLE_CONTROL:
        # ---- observed vs random (venue-controlled reshuffle), same pids ----
        control_dire = os.path.join(meta_dire, 'efg_computation_control_venue')
        src_by_source = {
            'obs': os.path.join(
                control_dire,
                f'gen_V3_y{year_offset}_obs_valid_primary_location_before_{RESHUFFLE_UPPER_YEAR}.csv'),
            'random': os.path.join(
                control_dire,
                f'gen_V3_y{year_offset}_iter{RESHUFFLE_ITER}_before_{RESHUFFLE_UPPER_YEAR}.csv'),
        }
        for s, p in src_by_source.items():
            assert os.path.exists(p), f"{s} reshuffle E/F/G table not found: {p}"

        # keyword membership per pid from the with_keyword source (booleans)
        kw_member = pd.read_csv(src_path, usecols=['pid'] + kw_cols)
        for col in kw_cols:
            kw_member[col] = kw_member[col].astype(bool)
        kw_member = kw_member.drop_duplicates('pid').set_index('pid')

        efg_by_source = {}
        for s, p in src_by_source.items():
            d = pd.read_csv(p, usecols=['pid', 'pub_year'] + value_cols)
            d = d[d['pub_year'].notna()].copy()
            d['pub_year'] = d['pub_year'].astype(int)
            efg_by_source[s] = d.drop_duplicates('pid').set_index('pid')
            print(f"loaded {s} E/F/G table shape:", efg_by_source[s].shape)

        # pids present in BOTH sources AND the keyword file, so observed and random are
        # aggregated over an identical pid set (this intersection is independent of the
        # chosen keyword subset, so per-keyword caches stay valid across combinations)
        common = kw_member.index
        for s in efg_by_source:
            common = common.intersection(efg_by_source[s].index)
        common = common.unique()
        print(f"common pids across obs/random/keyword: {len(common)}")
        kw_member = kw_member.loc[common]
        for s in build_source_order:
            _loaded_sources[s] = efg_by_source[s].loc[common].join(kw_member)
    else:
        # ---- observed only (original behaviour) ----
        use_cols = ['pub_year'] + value_cols + kw_cols
        df = pd.read_csv(src_path, usecols=use_cols)
        df = df[df['pub_year'].notna()].copy()
        df['pub_year'] = df['pub_year'].astype(int)
        for col in kw_cols:
            df[col] = df[col].astype(bool)
        print("loaded source shape:", df.shape)
        _loaded_sources['obs'] = df
    return _loaded_sources


def aggregate_one_group(group_id, mask_fn):
    """Per (source, index, pub_year) bootstrap mean+CI for ONE group. `mask_fn(df)`
    returns the boolean row mask selecting that group's papers from a source frame."""
    sources = get_loaded_sources()
    rng = group_rng(group_id)
    rows = []
    for s in build_source_order:
        df_s = sources[s]
        sub = df_s[mask_fn(df_s)]
        for index_name, col in index_cols:
            for year, vals in sub.groupby('pub_year')[col]:
                mean, lo, hi = bootstrap_ci(vals.values, N_BOOT, CI, rng)
                rows.append({
                    'source': s,
                    'group': group_id,
                    'index_name': index_name,
                    'pub_year': int(year),
                    'mean_value': mean,
                    'ci_low': lo,
                    'ci_high': hi,
                    'n': int(len(vals)),
                })
        print(f"  [{s}] group '{group_id}': {len(sub)} papers aggregated")
    return rows


group_frames = []

# one cache PER KEYWORD (reused across any combination that includes this keyword)
for kw in keywords:
    gpath = group_cache_path(kw)
    if os.path.exists(gpath) and not FORCE_REBUILD:
        g = pd.read_csv(gpath)
        print(f"[group cache loaded] {kw}:", gpath, "| shape:", g.shape)
    else:
        col_name = f'has_kw_{safe_name(kw)}'
        rows = aggregate_one_group(kw, lambda df, _c=col_name: df[_c].values)
        g = pd.DataFrame(rows)
        g.to_csv(gpath, index=False)
        print(f"[group cache saved] {kw}:", gpath, "| shape:", g.shape)
    group_frames.append(g)

# the "remaining" group depends on the WHOLE chosen set, so it is cached per kw_tag
if DRAW_REMAINING:
    if os.path.exists(remaining_cache_path) and not FORCE_REBUILD:
        g = pd.read_csv(remaining_cache_path)
        print("[remaining cache loaded]", remaining_cache_path, "| shape:", g.shape)
    else:
        rows = aggregate_one_group(REMAINING_ID, lambda df: ~df[kw_cols].any(axis=1).values)
        g = pd.DataFrame(rows)
        g.to_csv(remaining_cache_path, index=False)
        print("[remaining cache saved]", remaining_cache_path, "| shape:", g.shape)
    group_frames.append(g)

agg = pd.concat(group_frames, ignore_index=True)
print("[agg assembled] shape:", agg.shape)

# the tidy table behind the figure: every (group, source, index, year) mean + CI + n actually
# drawn, in one file next to the svg (the per-keyword caches stay separate for reuse)
agg.to_csv(out_csv, index=False)
print("[data table saved]", out_csv)

# safety: every group frame carries 'source'; default to observed if an old cache lacks it
if 'source' not in agg.columns:
    agg['source'] = 'obs'


# ==================================================
# 2. draw the grid (3 rows x (n_keywords + 1) cols)
# ==================================================
# group (column) order: keywords in cache order, then the remaining column last
# (only if DRAW_REMAINING is set)
group_order = [g for g in agg['group'].drop_duplicates().tolist() if g != REMAINING_ID]
if DRAW_REMAINING and REMAINING_ID in agg['group'].values:
    group_order.append(REMAINING_ID)

n_rows = len(row_order)
n_cols = len(group_order)
print("grid:", n_rows, "rows x", n_cols, "cols")

# which sources to overlay (observed always; random only in reshuffle mode)
source_order = [s for s in ('obs', 'random') if s in set(agg['source'])]
# obs keeps the per-index row colour (solid); random is drawn grey + dashed
SOURCE_STYLE = {
    'obs':    {'linestyle': '-',  'linewidth': 1.8, 'band_alpha': 0.25, 'color': None},
    'random': {'linestyle': '--', 'linewidth': 1.6, 'band_alpha': 0.15, 'color': 'tab:gray'},
}

# total valid pids per subsample (per-year n is identical across the 3 indices and
# across sources, so sum n over years for a single index/source to get the size)
totals_source = 'obs' if 'obs' in source_order else source_order[0]
group_totals = (
    agg[(agg['index_name'] == row_order[0]) & (agg['source'] == totals_source)]
    .groupby('group')['n'].sum().astype(int).to_dict()
)
for g in group_order:
    print(f"  subsample '{g}': {group_totals.get(g, 0)} valid pids")

# decade-ish ticks; extend to max_year (e.g. 2020) so that endpoint shows even
# though the data stops at max_year - 1 (pub_year < max_year)
y_min = int(agg['pub_year'].min())
tick_start = y_min - (y_min % 10)
xticks = list(range(tick_start, max_year + 1, 20))

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(3.2 * n_cols, 3.2 * n_rows),
    sharex=True, sharey='row', squeeze=False,
)

for c, group_id in enumerate(group_order):
    n_valid = group_totals.get(group_id, 0)
    base_title = 'Excluding all keyword subsamples' if group_id == REMAINING_ID else group_id.capitalize()
    col_title = f'{base_title}\n(n={n_valid:,})'
    for r, index_name in enumerate(row_order):
        ax = axes[r][c]
        # overlay each source (observed solid in the row colour; random grey dashed)
        for sname in source_order:
            d = agg[(agg['source'] == sname) & (agg['group'] == group_id)
                    & (agg['index_name'] == index_name)].sort_values('pub_year')
            # only keep years whose subsample has at least MIN_PAPERS_PER_YEAR papers
            d = d[d['n'] >= MIN_PAPERS_PER_YEAR]
            if d.empty:
                continue
            st = SOURCE_STYLE[sname]
            color = st['color'] or row_colors[index_name]
            ax.plot(d['pub_year'], d['mean_value'], color=color,
                    linewidth=st['linewidth'], linestyle=st['linestyle'])
            ax.fill_between(d['pub_year'], d['ci_low'], d['ci_high'],
                            color=color, alpha=st['band_alpha'], linewidth=0)

        # only on the top-left panel: label the observed line (in its colour) and the
        # randomized line (in grey), each placed on its own curve
        if r == 0 and c == 0:
            for sname, txt, va, dy in (('obs', 'Observed Network', 'bottom', 40),
                                       ('random', 'Randomized Network', 'top', -8)):
                if sname not in source_order:
                    continue
                d = agg[(agg['source'] == sname) & (agg['group'] == group_id)
                        & (agg['index_name'] == index_name)].sort_values('pub_year')
                d = d[d['n'] >= MIN_PAPERS_PER_YEAR]
                if d.empty:
                    continue
                color = SOURCE_STYLE[sname]['color'] or row_colors[index_name]
                anchor = min(max(int(len(d) * 0.55), 0), len(d) - 1)
                ax.annotate(txt, xy=(d['pub_year'].iloc[anchor], d['mean_value'].iloc[anchor]),
                            xytext=(0, dy), textcoords='offset points',
                            color=color, fontsize=12, fontweight='bold',
                            ha='center', va=va)

        if r == 0:
            # column heading as a text label in the upper-left (no per-subplot title)
            ax.text(0.03, 0.97, col_title, transform=ax.transAxes,
                    ha='left', va='top', fontsize=17)
        if c == 0:
            ax.set_ylabel(index_name, fontsize=20)

        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) for x in xticks], fontsize=14, rotation=45)
        ax.tick_params(axis='y', labelsize=14)
        # extend the right limit to max_year so the (e.g. 2020) tick is visible
        ax.set_xlim(y_min, max_year)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

# single shared x-label, centred under the columns (no figure title)
fig.supxlabel('Publication Year', fontsize=26)

plt.tight_layout()
plt.savefig(out_svg, format='svg', transparent=True)
plt.close()
print("[figure saved]", out_svg)

print("finished")
