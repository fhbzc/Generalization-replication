"""
Country diffusion of top-10% foundation / extension / generalization papers.

Reads get_efg_s3_merge_with_concept.py's output directly -- the get_efg.py -> get_efg_s2_merge.py
table with each paper's single top (root-level) concept already attached and the reference /
citation / pub_year filters already applied, so the concept map is never loaded here.
Given one or more citation windows (e.g. 10-15, 100-150, 1000-1500), each window is processed
on its own: take every paper whose citation_count_y{offset} falls inside it, then rank
e / f / g WITHIN each (pub_year x top_concept) cell of that window -- so a paper is only ever
compared against same-year, same-domain papers sitting in the same citation window. The top
10% of each index is then profiled by how widely its citations spread over countries:

    n_countries       = # distinct countries among the paper's citing papers
    max_country_ratio = (citations from the most-frequent country) / (all valid citations)

A citing paper with authors in several countries counts ONCE FOR EACH of them (a US+CN paper
adds 1 to US and 1 to CN), but the denominator stays the number of citing papers, so with 10
citations of which one is US and one is CN both sit at 10%. Citing papers with NO valid
country are ignored entirely -- they count towards neither numerator nor denominator.

Both stats are averaged over the papers of each top-10% group. The figure draws one point per
(window x index): x = mean n_countries, y = mean max_country_ratio, colour = index
(f red, e orange, g blue), marker size proportional to log10(window lower bound).

Input  : META_DIRE/efg_computation/{file_prefix}_y{offset}{network_suffix}_with_top_concept_
             rt_{reference_threshold}_ct_{citation_threshold}_ybefore_{s3_ybefore}.csv
                                                    (get_efg_s3_merge_with_concept.py)
         META_DIRE/pid2author_country_list.pkl   (get_meta_data_preprint_author_country.py)
         META_DIRE/pid2pub_year.pkl                        (get_meta_data_preprint.py)
         META_DIRE/cited_pid2citing_pids_list{network_suffix}.pkl            "
Output : ./cache/*    (every intermediate is cached per window, so a redraw never recomputes
                       and adding a window only computes that window; top_paper_counts_*.csv
                       is a one-row-per-window table of how many top-X% papers each of
                       e/f/g holds)
                      Every mean in the aggregate CSVs also gets a bootstrap 95% CI
                      (ci_low_* / ci_high_*, with n_boot / seed / boot_mode recording how it
                      was made). The CIs are stored only -- the figure does not draw them.
         ./figures/country_diffusion_*.svg/.png

At the paper's settings (year_offset=5, use_with_review=False, use_disp_f=False,
network_mode='normal', reference_threshold=1, citation_threshold=5,
windows=[(10,15),(100,150),(1000,1500)], top_pct=10, group_cols=['pub_year','top_concept'])
this scatter IS the RIGHT panel of Fig. 2c (is_fig2c_right), so two outputs take the panel
name:
    ./figures/Fig. 2c(right).svg / .png      (else country_diffusion_{windows}_{base_tag}_top{pct}.svg/.png)
    ./cache/Fig. 2c(right).csv               (the combined aggregate = the data behind the
                                              figure; else aggregate_all_{windows}_{base_tag}_top{pct}.csv)
Everything else in ./cache keeps its parameterized name in every run -- the per-window
panel / selection / country-stats / aggregate caches are keyed by window and shared across
runs, and top_paper_counts_*.csv is a companion table, not the figure's data.

Usage  : edit the settings block below, then `python get_country_diffusion.py`.
"""

import math
import os
import pickle
import sys
from collections import Counter, OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator
import numpy as np
import pandas as pd
from tqdm import tqdm

# utils.py is in openalex/, one level up from this file (openalex/country_diffusion/<this>).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OPENALEX_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, OPENALEX_DIR)
from utils import META_DIRE  # noqa: E402


# ==================================================
# settings
# ==================================================
# must match get_efg.py / get_efg_s2_merge.py
use_with_review = False
meta_dire = META_DIRE + '_with_review' if use_with_review else META_DIRE
print("meta_dire", meta_dire)

year_offset = 5         # -1 = all future citations (up to max_year); must match get_efg.py
max_year = 2025         # data horizon: citations are complete through max_year - 1 (2024)
use_disp_f = False      # True -> merge of get_efg_v2_disp_f.py, False -> get_efg.py
network_mode = 'normal'  # 'normal' / 'no_self_citation' / 'no_year_filter'

file_prefix = 'gen_V3_disp_f' if use_disp_f else 'gen_V3'
NETWORK_SUFFIX_MAP = {'normal': '', 'no_self_citation': '_no_self_citation',
                      'no_year_filter': '_no_year_filter'}
assert network_mode in NETWORK_SUFFIX_MAP, network_mode
network_suffix = NETWORK_SUFFIX_MAP[network_mode]

# citation windows, each (lower bound, upper bound) and INCLUSIVE on both ends. Every window
# is processed on its own -- its own panel, its own within-cell quantiles, its own top 10%.
# Lower bounds must be >= 1 (marker size scales with log10 of them).
windows = [(10, 15), (100, 150), (1000, 1500)]

# The input is get_efg_s3_merge_with_concept.py's output, which has ALREADY done the
# top-concept join (exactly one root concept per paper, ambiguous/missing pids dropped) and
# the reference / citation / pub_year filters. These three must match that script's settings
# because they only serve to rebuild its output FILENAME -- change them there first.
reference_threshold = 1
citation_threshold = 5
s3_ybefore = max_year - year_offset        # s3's `max_year`: it kept pub_year < this

# top X% of each index that gets profiled
top_pct = 10.0

# the cell each paper's e/f/g is ranked within (on top of the citation window). 'top_concept'
# alone would pool every year of a domain together, and a 1950 paper's e/f/g is not comparable
# with a 2015 one, so pub_year is in the cell too -- drop it to normalise by domain only.
group_cols = ['pub_year', 'top_concept']

# True -> ONLY redraw the figure from the cached per-window aggregates; never computes, and
# errors out if any window has no cached aggregate yet. Use it to restyle an existing figure.
# The FIRST run must be False (a False run still reuses every cache it finds, so it is also
# the right setting for adding a window or changing top_pct).
redraw = False

print("year_offset", year_offset)
print("max_year", max_year)
print("file_prefix", file_prefix)
print("network_suffix", repr(network_suffix))
print("citation windows", windows)
print("reference_threshold", reference_threshold, "(from get_efg_s3_merge_with_concept.py)")
print("citation_threshold", citation_threshold, "(from get_efg_s3_merge_with_concept.py)")
print("s3_ybefore", s3_ybefore, "-> input holds pub_year <=", s3_ybefore - 1)
print("top_pct", top_pct)
print("group_cols", group_cols)
print("redraw", redraw)

# ---- deliverable naming ----------------------------------------------------------------
# This scatter at the paper's settings IS Fig. 2c (right panel), so the figure and the
# combined aggregate behind it are named after the panel; every other setting keeps the
# descriptive parameter stem. The gate lists every parameter that changes what the points
# show -- the input table (offset / review / disp_f / network mode / thresholds), the
# citation windows, the top-X% cut and the ranking cell -- so no other run can land on this
# name. The PER-WINDOW caches (panel / selection / country stats / per-window aggregate) and
# the top_paper_counts table keep their parameterized names in every run: they are keyed by
# window and shared across runs, so renaming them would only defeat the caching.
FIG_NAME = 'Fig. 2c(right)'
is_fig2c_right = (year_offset == 5 and (not use_with_review) and (not use_disp_f)
                  and network_mode == 'normal'
                  and reference_threshold == 1 and citation_threshold == 5
                  and windows == [(10, 15), (100, 150), (1000, 1500)]
                  and top_pct == 10.0
                  and group_cols == ['pub_year', 'top_concept'])
print("is_fig2c_right", is_fig2c_right, ("-> %s" % FIG_NAME) if is_fig2c_right else "")

# every window must sit inside what s3 already kept, else the window is silently truncated
for _lo, _hi in windows:
    assert _lo >= citation_threshold, (
        f"window {_lo}-{_hi} starts below get_efg_s3_merge_with_concept.py's "
        f"citation_threshold={citation_threshold}: that input has already dropped papers with "
        f"fewer than {citation_threshold} citations, so this window would be incomplete.")

citation_col = f'citation_count_y{year_offset}'
INDEX_COLS = OrderedDict([          # column in the merged table -> (label, colour)
    (f'f_y{year_offset}', ('Foundation', 'red')),
    (f'e_y{year_offset}', ('Extension', 'orange')),
    (f'g_y{year_offset}', ('Generalization', 'blue')),
])

# marker diameter (points) = MARKER_SCALE * log10(window lower bound)
MARKER_SCALE = 6.0
# the line joining the three indices of one citation window
CONNECT_COLOR = 'gray'
CONNECT_LW = 2.2

# get_efg_s3_merge_with_concept.py's output: gen_V3 + top_concept, already filtered
efg_load_path = os.path.join(
    meta_dire, 'efg_computation',
    f'{file_prefix}_y{year_offset}{network_suffix}_with_top_concept_'
    f'rt_{reference_threshold}_ct_{citation_threshold}_ybefore_{s3_ybefore}.csv')
print("efg_load_path", efg_load_path)

cache_dire = os.path.join(BASE_DIR, 'cache')
figure_dire = os.path.join(BASE_DIR, 'figures')
os.makedirs(cache_dire, exist_ok=True)
os.makedirs(figure_dire, exist_ok=True)

base_tag = (f'y{year_offset}{network_suffix}_{file_prefix}'
            f'_rt{reference_threshold}_ct{citation_threshold}_yb{s3_ybefore}'
            f'_by{"-".join(group_cols)}')


def window_tag(lo, hi):
    return f'c{lo}-{hi}_{base_tag}'


def window_top_tag(lo, hi):
    return f'{window_tag(lo, hi)}_top{top_pct:g}'


def panel_path(lo, hi):
    return os.path.join(cache_dire, f'panel_{window_tag(lo, hi)}.pkl')


def top_pids_path(lo, hi):
    # the selected top-X% pid list of each index, plus the window's paper count
    return os.path.join(cache_dire, f'top_pids_{window_top_tag(lo, hi)}.pkl')


def country_stats_path(lo, hi):
    return os.path.join(cache_dire, f'country_stats_{window_top_tag(lo, hi)}.pkl')


def aggregate_path(lo, hi):
    return os.path.join(cache_dire, f'aggregate_{window_top_tag(lo, hi)}.csv')


class _ProgressReader:
    """File wrapper that ticks a tqdm bar as pickle.load pulls bytes off it.

    The citation network and country map are multi-GB, so an unpickle can sit silent for
    minutes; this turns that into a byte-level bar. Every read path pickle uses (read /
    readinto / readline / peek) is forwarded, so the count stays exact.
    """

    def __init__(self, fh, bar):
        self._fh = fh
        self._bar = bar

    def read(self, size=-1):
        chunk = self._fh.read(size)
        self._bar.update(len(chunk))
        return chunk

    def readinto(self, buf):
        n = self._fh.readinto(buf)
        self._bar.update(n)
        return n

    def readline(self, *a):
        line = self._fh.readline(*a)
        self._bar.update(len(line))
        return line

    def peek(self, *a):
        return self._fh.peek(*a)          # peek does not consume, so it must not tick

    def __getattr__(self, name):
        return getattr(self._fh, name)


def load_pkl(path, desc=None):
    desc = desc or os.path.basename(path)
    with open(path, 'rb') as f:
        with tqdm(total=os.path.getsize(path), desc=f'load {desc}', unit='B',
                  unit_scale=True, unit_divisor=1024, leave=False) as bar:
            return pickle.load(_ProgressReader(f, bar))


def dump_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# ==================================================
# 1. the shared table: merged E/F/G + top concept + pub_year / reference filters
#    (built once and sliced per window -- the concept join is the expensive part)
# ==================================================
def build_shared_table():
    """Load get_efg_s3_merge_with_concept.py's output as-is.

    That table has already done the expensive part -- the top-concept join (exactly one root
    concept per paper; missing / ambiguous pids dropped) plus the reference, citation and
    pub_year filters -- so nothing is recomputed here and pid2root_concepts is never loaded.
    """
    assert os.path.exists(efg_load_path), (
        f"{efg_load_path}\nRun get_efg_s3_merge_with_concept.py first (with the same "
        f"year_offset / reference_threshold / citation_threshold).")

    load_cols = ['pid', 'pub_year', 'up_refer_count', citation_col,
                 'top_concept'] + list(INDEX_COLS)
    parts, total_rows = [], 0
    with tqdm(desc='read s3 csv', unit='row', unit_scale=True) as bar:
        for chunk in pd.read_csv(efg_load_path, usecols=load_cols, chunksize=2_000_000):
            total_rows += len(chunk)
            parts.append(chunk)
            bar.update(len(chunk))
    df = pd.concat(parts, ignore_index=True)
    print(f"s3 rows: {total_rows}")
    print(f"  pub_year range in file: {df['pub_year'].min()} - {df['pub_year'].max()}")
    print(f"  {citation_col} range in file: {df[citation_col].min()} - {df[citation_col].max()}")
    print(f"  distinct top_concept: {df['top_concept'].nunique()}")

    # s3 guarantees these; assert rather than re-filter, so a mismatched input is loud
    assert df['top_concept'].notna().all(), "s3 output should carry a top_concept for every row"
    assert (df[citation_col] >= citation_threshold).all(), \
        f"s3 output should already satisfy {citation_col} >= {citation_threshold}"
    assert (df['up_refer_count'] >= reference_threshold).all(), \
        f"s3 output should already satisfy up_refer_count >= {reference_threshold}"

    return df


# ==================================================
# 2. one window's panel: slice the citation range, rank within its own cells
# ==================================================
def build_panel(df_all, lo, hi):
    df = df_all[(df_all[citation_col] >= lo) & (df_all[citation_col] <= hi)].copy()
    tqdm.write(f"  rows inside citation window [{lo}, {hi}]: {len(df)}")

    # quantile of each index against same-year, same-domain papers *in this window only*.
    # rank(pct=True) with 'average' ties -> a paper tied at the top of a cell does not get a
    # free pass into the top X%.
    grouped = df.groupby(group_cols)
    tqdm.write(f"  ranking within {grouped.ngroups} ({' x '.join(group_cols)}) cells")
    for col in tqdm(list(INDEX_COLS), desc=f'  rank {lo}-{hi}', leave=False):
        df[f'q_{col}'] = grouped[col].rank(pct=True, method='average')

    return df.reset_index(drop=True)


# ==================================================
# 3. country spread of the selected papers' citations
# ==================================================
def build_country_stats(target_pids, pid2year, cited_pid2citing_works, pid2countries, desc):
    rows = []
    no_country_papers = 0      # selected papers left with no country-valid citation
    dropped_citations = 0      # citing papers ignored for having no valid country

    bar = tqdm(target_pids, desc=desc, unit='paper', leave=False)
    for i, pid in enumerate(bar, 1):
        # the same citation set get_efg.py counted
        if year_offset == -1:
            year_threshold = max_year
        else:
            year_threshold = min(pid2year[pid] + year_offset, max_year - 1)

        citing_papers = set()
        for citing in cited_pid2citing_works.get(pid, []):
            citing_year = pid2year.get(citing)
            if citing_year is not None and citing_year <= year_threshold:
                citing_papers.add(citing)

        # one citing paper contributes +1 to EACH distinct country it carries; papers with
        # no valid country are skipped, so they leave the denominator untouched too
        country_counter = Counter()
        valid_citations = 0
        for citing in citing_papers:
            countries = pid2countries.get(citing)
            if not countries:
                continue
            valid_citations += 1
            for country in countries:
                country_counter[country] += 1

        dropped_citations += len(citing_papers) - valid_citations
        if valid_citations == 0:
            no_country_papers += 1
            continue

        if i % 2000 == 0:
            bar.set_postfix(measured=len(rows), no_country=no_country_papers,
                            dropped_cit=dropped_citations, refresh=False)

        rows.append({
            'pid': pid,
            'citation_count': len(citing_papers),
            'valid_citation_count': valid_citations,
            'n_countries': len(country_counter),
            'max_country_count': max(country_counter.values()),
            # denominator is the citation count, NOT the sum of country counts: a US+CN
            # paper is one citation that lands on two countries
            'max_country_ratio': max(country_counter.values()) / valid_citations,
        })

    bar.close()
    tqdm.write(f"  measured {len(rows)} papers;"
               f" {no_country_papers} had no citation with a valid country;"
               f" {dropped_citations} citations ignored for having no valid country")

    cols = ['pid', 'citation_count', 'valid_citation_count',
            'n_countries', 'max_country_count', 'max_country_ratio']
    return pd.DataFrame(rows, columns=cols).set_index('pid')


# ==================================================
# 4. aggregate one window: one row per index
# ==================================================
# Each mean in the aggregate is a mean over the measured top-X% papers, so it carries a
# sampling error; the cache stores a bootstrap 95% CI of every mean next to it (percentile
# method), not just the SEM. The figure does not draw them -- they are stored for lookup.
BOOT_N = 2000              # bootstrap replicates
BOOT_SEED = 20260801       # fixed, so re-running reproduces the same CI
BOOT_EXACT_MAX = 200_000   # a group larger than this uses the normal approximation below
# the per-paper columns whose mean gets a CI (all three go into the aggregate)
BOOT_COLS = ['n_countries', 'max_country_ratio', 'valid_citation_count']


def bootstrap_means_ci(sub, cols=BOOT_COLS, n_boot=BOOT_N, seed=BOOT_SEED):
    """({column -> (ci_low, ci_high)}, how the replicates were drawn) for the mean of each
    column of `sub`.

    The PAPERS are resampled, not the columns: one set of resampled paper indices produces
    all the columns' replicate means at once, so the intervals describe the same resampled
    sets of papers (an x-CI and a y-CI then refer to one resampling).
    Resampling is exact while the group is small enough; beyond BOOT_EXACT_MAX papers it
    would cost n x n_boot draws for a bootstrap distribution that is by then indistinguishable
    from N(mean, s^2/n) (the bootstrap mean is asymptotically normal), so the replicate means
    are drawn from that normal instead -- per column, so THAT path drops the cross-column
    coupling described above.
    """
    arr = sub[cols].to_numpy(dtype=float)
    arr = arr[np.isfinite(arr).all(axis=1)]          # a paper needs every column to count
    n, k_cols = arr.shape
    if n < 2:                                        # no CI is definable from < 2 papers
        return {c: (float('nan'), float('nan')) for c in cols}, ''
    rng = np.random.default_rng(seed)
    if n > BOOT_EXACT_MAX:
        reps = rng.normal(arr.mean(axis=0), arr.std(axis=0, ddof=1) / np.sqrt(n),
                          size=(n_boot, k_cols))
        mode = 'normal'
    else:
        block = max(1, int(2e7 // max(n * k_cols, 1)))   # cap the draw at ~2e7 entries
        reps = np.empty((n_boot, k_cols), dtype=float)
        for s in range(0, n_boot, block):
            b = min(block, n_boot - s)
            reps[s:s + b] = arr[rng.integers(0, n, size=(b, n))].mean(axis=1)
        mode = 'resample'
    lo_hi = np.percentile(reps, [2.5, 97.5], axis=0)
    return {c: (float(lo_hi[0, j]), float(lo_hi[1, j])) for j, c in enumerate(cols)}, mode


def build_aggregate(df_stats, col2top_pids, window_paper_count, lo, hi):
    agg_rows = []
    for col, (label, colour) in INDEX_COLS.items():
        top_pids = col2top_pids[col]
        sub = df_stats.reindex(top_pids).dropna(subset=['n_countries'])
        ci, boot_mode = bootstrap_means_ci(sub)
        agg_rows.append({
            'cite_min': lo,
            'cite_max': hi,
            'window': f'{lo}-{hi}',
            'index': label,
            'column': col,
            'colour': colour,
            # papers in the window (after concept + reference filters) the quantiles ranked
            'window_paper_count': window_paper_count,
            # papers selected into this index's top X%
            'top_paper_count': len(top_pids),
            # of those, the ones with >= 1 country-valid citation -- what the means are over
            'measured_paper_count': len(sub),
            'mean_n_countries': sub['n_countries'].mean(),
            'sem_n_countries': sub['n_countries'].sem(),
            'ci_low_n_countries': ci['n_countries'][0],
            'ci_high_n_countries': ci['n_countries'][1],
            'mean_max_country_ratio': sub['max_country_ratio'].mean(),
            'sem_max_country_ratio': sub['max_country_ratio'].sem(),
            'ci_low_max_country_ratio': ci['max_country_ratio'][0],
            'ci_high_max_country_ratio': ci['max_country_ratio'][1],
            'mean_valid_citation_count': sub['valid_citation_count'].mean(),
            'ci_low_valid_citation_count': ci['valid_citation_count'][0],
            'ci_high_valid_citation_count': ci['valid_citation_count'][1],
            # how the CIs above were produced, so a row is self-describing
            'n_boot': BOOT_N,
            'seed': BOOT_SEED,
            'boot_mode': boot_mode,
        })
    return pd.DataFrame(agg_rows)


# an aggregate CSV written before the CIs existed has to be rebuilt -- cheap, since it only
# reindexes the cached country stats. Every other cache stays valid.
AGG_REQUIRED_COLS = ['ci_low_n_countries', 'ci_high_n_countries',
                     'ci_low_max_country_ratio', 'ci_high_max_country_ratio',
                     'ci_low_valid_citation_count', 'ci_high_valid_citation_count',
                     'n_boot', 'seed', 'boot_mode']


def aggregate_is_current(lo, hi):
    """True when this window's aggregate CSV exists AND carries the bootstrap CI columns."""
    path = aggregate_path(lo, hi)
    if not os.path.exists(path):
        return False
    have = set(pd.read_csv(path, nrows=0).columns)
    missing = [c for c in AGG_REQUIRED_COLS if c not in have]
    if missing:
        print(f"[{lo}-{hi}] cached aggregate predates the bootstrap CIs ({missing[0]} ...)"
              f" -> rebuilding it from the cached country stats")
        return False
    return True


# ==================================================
# driver: windows are processed one after another, each fully cached
# ==================================================
if redraw:
    missing = [f'{lo}-{hi}' for lo, hi in windows if not os.path.exists(aggregate_path(lo, hi))]
    if missing:
        raise SystemExit(
            f"redraw = True only re-reads ./cache and never computes, but no cached aggregate "
            f"exists for these windows: {missing}\n"
            f"Set redraw = False to compute them (that run still reuses whatever IS cached).")
    stale = [f'{lo}-{hi}' for lo, hi in windows if not aggregate_is_current(lo, hi)]
    if stale:
        print(f"WARNING: {stale} have no bootstrap CI columns. Set redraw = False once to "
              f"rebuild those aggregates (only the cached country stats are re-read).")

agg_list = []

if redraw:
    for lo, hi in windows:
        agg_list.append(pd.read_csv(aggregate_path(lo, hi)))
        print(f"[{lo}-{hi}] loaded cached aggregate")
else:
    # the windows still needing work decide which heavy inputs get loaded at all. An
    # aggregate that predates the bootstrap CI columns counts as work, but only the last
    # (cheap) step of it -- its panel / selection / country-stats caches still hold.
    todo = [(lo, hi) for lo, hi in windows if not aggregate_is_current(lo, hi)]
    need_panel = [w for w in todo
                  if not (os.path.exists(top_pids_path(*w)) or os.path.exists(panel_path(*w)))]
    need_stats = [w for w in todo if not os.path.exists(country_stats_path(*w))]

    print(f"windows needing work: {[f'{lo}-{hi}' for lo, hi in todo]}"
          f" (of {len(windows)}); panels to build: {len(need_panel)},"
          f" country passes to run: {len(need_stats)}")

    # the csv + concept join happen ONCE and are shared by every window that needs a panel
    df_all = build_shared_table() if need_panel else None

    # per-window panels + top-pct selections. The selection itself is cached (pid lists +
    # how many papers the window ranked), so a rerun never has to touch the panel again.
    window2selection = OrderedDict()
    cutoff = 1.0 - top_pct / 100.0
    for w_i, (lo, hi) in enumerate(todo, 1):
        tqdm.write(f"[window {w_i}/{len(todo)}] {lo}-{hi} top {top_pct:g}% selection")
        if os.path.exists(top_pids_path(lo, hi)):
            selection = load_pkl(top_pids_path(lo, hi), desc=f'selection {lo}-{hi}')
            tqdm.write("  loaded cached selection")
        else:
            if os.path.exists(panel_path(lo, hi)):
                df_panel = load_pkl(panel_path(lo, hi), desc=f'panel {lo}-{hi}')
                tqdm.write(f"  loaded cached panel: {df_panel.shape}")
            else:
                df_panel = build_panel(df_all, lo, hi)
                dump_pkl(df_panel, panel_path(lo, hi))
                tqdm.write(f"  wrote panel cache: {df_panel.shape}")

            col2top_pids = OrderedDict()
            for col in INDEX_COLS:
                col2top_pids[col] = df_panel.loc[df_panel[f'q_{col}'] > cutoff, 'pid'].tolist()
            selection = {'window_paper_count': len(df_panel), 'col2top_pids': col2top_pids}
            dump_pkl(selection, top_pids_path(lo, hi))
            tqdm.write(f"  wrote selection cache: {os.path.basename(top_pids_path(lo, hi))}")
            del df_panel

        tqdm.write(f"  window papers ranked: {selection['window_paper_count']}")
        for col, (label, _) in INDEX_COLS.items():
            tqdm.write(f"  top {top_pct:g}% {label}:"
                       f" {len(selection['col2top_pids'][col])} papers")
        window2selection[(lo, hi)] = selection

    del df_all

    # the heavy maps are loaded once and shared by every window still missing stats
    pid2year = cited_pid2citing_works = pid2countries = None
    if need_stats:
        pid2year = load_pkl(os.path.join(meta_dire, 'pid2pub_year.pkl'))
        print(f"pid2pub_year load finished ({len(pid2year)} pids)")
        cited_pid2citing_works = load_pkl(
            os.path.join(meta_dire, f'cited_pid2citing_pids_list{network_suffix}.pkl'))
        print(f"cited_pid2citing_pids_list load finished ({len(cited_pid2citing_works)} pids)")

        # pid -> [(author_id, [country_code, ...]), ...]; collapse to the distinct set of
        # countries per paper, dropping papers that end up with none
        pid2author_country = load_pkl(os.path.join(meta_dire, 'pid2author_country_list.pkl'))
        print(f"pid2author_country_list load finished ({len(pid2author_country)} pids)")
        pid2countries = {}
        for pid, author_list in tqdm(pid2author_country.items(), desc='collapse countries'):
            countries = {c for _author, codes in author_list for c in (codes or []) if c}
            if countries:
                pid2countries[pid] = countries
        del pid2author_country
        print(f"papers with >= 1 valid country: {len(pid2countries)}")

    for w_i, (lo, hi) in enumerate(todo, 1):
        tqdm.write(f"[window {w_i}/{len(todo)}] {lo}-{hi} country spread")
        selection = window2selection[(lo, hi)]
        col2top_pids = selection['col2top_pids']
        if os.path.exists(country_stats_path(lo, hi)):
            df_stats = load_pkl(country_stats_path(lo, hi), desc=f'country stats {lo}-{hi}')
            tqdm.write(f"  loaded cached country stats: {df_stats.shape}")
        else:
            # one pass over the union, so a paper in two top-X% groups is measured once
            union_pids = sorted({pid for pids in col2top_pids.values() for pid in pids})
            tqdm.write(f"  papers needing country stats (union): {len(union_pids)}")
            df_stats = build_country_stats(union_pids, pid2year, cited_pid2citing_works,
                                           pid2countries, desc=f'  citations {lo}-{hi}')
            dump_pkl(df_stats, country_stats_path(lo, hi))
            tqdm.write(f"  wrote country stats cache: {df_stats.shape}")

        df_agg = build_aggregate(df_stats, col2top_pids, selection['window_paper_count'], lo, hi)
        df_agg.to_csv(aggregate_path(lo, hi), index=False)
        tqdm.write(f"  wrote aggregate: {os.path.basename(aggregate_path(lo, hi))}")
        del df_stats

    del pid2year, cited_pid2citing_works, pid2countries

    # cached windows are picked up here in the order the user asked for
    for lo, hi in windows:
        agg_list.append(pd.read_csv(aggregate_path(lo, hi)))

df_agg_all = pd.concat(agg_list, ignore_index=True)

windows_tag = '_'.join(f'{lo}-{hi}' for lo, hi in windows)
# the combined aggregate IS the data behind the figure, so it takes the panel name too
combined_stem = (FIG_NAME if is_fig2c_right
                 else f'aggregate_all_{windows_tag}_{base_tag}_top{top_pct:g}')
combined_path = os.path.join(cache_dire, f'{combined_stem}.csv')
df_agg_all.to_csv(combined_path, index=False)
print("wrote combined aggregate:", combined_path)
print(df_agg_all.to_string(index=False))

# how many top-X% papers each index contributes in each citation window, printed on every run
# (the per-window selection log above only fires for windows that get recomputed, so a fully
# cached or redraw run would otherwise show none of it).
# window_paper_count is identical across the three indices of a window, so it is taken once.
print(f"\ntop {top_pct:g}% paper counts per citation window"
      f" (ranked within {' x '.join(group_cols)})")
count_rows = []
for lo, hi in windows:
    sub = df_agg_all[(df_agg_all['cite_min'] == lo) & (df_agg_all['cite_max'] == hi)]
    row = OrderedDict([('window', f'{lo}-{hi}'),
                       ('window_papers', int(sub['window_paper_count'].iloc[0]))])
    for _, r in sub.iterrows():
        row[f"{r['index']}_top"] = int(r['top_paper_count'])
        row[f"{r['index']}_measured"] = int(r['measured_paper_count'])
    count_rows.append(row)
df_counts = pd.DataFrame(count_rows)
print(df_counts.to_string(index=False))

counts_path = os.path.join(
    cache_dire, f'top_paper_counts_{windows_tag}_{base_tag}_top{top_pct:g}.csv')
df_counts.to_csv(counts_path, index=False)
print("wrote top-paper counts:", counts_path)

# ==================================================
# 5. plot: one point per (window x index)
# ==================================================
plt.rcParams['font.family'] = 'Gill Sans MT'

label_fontsize = 28
tick_fontsize = 20
title_fontsize = 16

fig, ax = plt.subplots(figsize=(7.0, 5.5))

# connect the three indices WITHIN each citation window, walking from the highest
# max-country ratio down to the lowest. Drawn on top of the markers so it visibly reaches
# each marker's centre; neutral grey keeps it from competing with the index colours.
for lo, hi in windows:
    sub = df_agg_all[(df_agg_all['cite_min'] == lo) & (df_agg_all['cite_max'] == hi)]
    sub = sub.dropna(subset=['mean_n_countries', 'mean_max_country_ratio'])
    if len(sub) < 2:                       # an empty window has nothing to connect
        continue
    sub = sub.sort_values('mean_max_country_ratio', ascending=False)
    ax.plot(sub['mean_n_countries'], sub['mean_max_country_ratio'],
            color=CONNECT_COLOR, linewidth=CONNECT_LW, linestyle='--', zorder=4)

for _, row in df_agg_all.iterrows():
    markersize = MARKER_SCALE * math.log10(row['cite_min'])
    ax.plot(row['mean_n_countries'], row['mean_max_country_ratio'],
            marker='o', markersize=markersize, color=row['colour'],
            linestyle='none', zorder=3)

# x is a count that spans orders of magnitude across the citation windows -> log scale.
# label the ticks as plain integers rather than matplotlib's 4x10^0 scientific notation.
ax.set_xscale('log')
int_fmt = FuncFormatter(lambda v, _pos: f'{v:g}')
ax.xaxis.set_major_formatter(int_fmt)
# label only the 2/3/5 subdivisions between decades -- labelling every minor tick puts
# "9" hard against "10"
ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 3.0, 5.0), numticks=12))
ax.xaxis.set_minor_formatter(int_fmt)

ax.set_xlabel("Number of citing countries", fontsize=label_fontsize)
ax.set_ylabel("Major country citation ratio", fontsize=label_fontsize)
# 'both' minor+major: the x-axis labels its 2/3/5 minor ticks too (see the minor
# formatter above), and those would otherwise keep the default size
ax.tick_params(axis='both', which='both', labelsize=tick_fontsize)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

fig_base = os.path.join(
    figure_dire,
    FIG_NAME if is_fig2c_right
    else f'country_diffusion_{windows_tag}_{base_tag}_top{top_pct:g}')
plt.savefig(f'{fig_base}.svg', dpi=300, format='svg', transparent=True)
plt.savefig(f'{fig_base}.png', dpi=300)
print("saved figure:", f'{fig_base}.svg')

print("finished")
