"""
Influence spread of top-10% foundation / extension / generalization papers.

Given one or more citation windows (e.g. 10-15, 100-150, 1000-1100), each window is
processed on its own: take every paper whose citation_count_y{offset} falls inside it,
attach its OpenAlex primary field, and rank e / f / g *within* (pub_year x primary field)
cells of that window -- so a paper is only ever compared against same-year, same-field
papers sitting in the same citation window. The top 10% of each index is then profiled by
how widely its citations spread over OpenAlex primary topics:

    n_topics        = # distinct primary topics among the paper's citing papers
    max_topic_ratio = (citations from the single most-frequent topic) / (all citations)

Both are averaged over the papers of each top-10% group. The figure draws one point per
(window x index): x = mean n_topics, y = mean max_topic_ratio, colour = index
(f red, e orange, g blue), marker size proportional to log10(window lower bound).

Papers with no primary field are dropped. Citing papers with no primary topic are
ignored (they count towards neither numerator nor denominator).

Input  : META_DIRE_2026/gen_V3_keep_disp/gen_V3_y{offset}.csv   (get_efg_2026_merge.py)
         META_DIRE_2026/pid2primary_field.pkl                   (get_meta_data_preprint_2026_s2_topic.py)
         META_DIRE_2026/pid2primary_topic.pkl                            "  (use_all_topics=False)
         META_DIRE_2026/pid2topics.pkl              (get_meta_data_preprint_2026_s3_topic_all.py; use_all_topics=True)
         META_DIRE_2026/pid2pub_year.pkl                         (get_meta_data_preprint_2026.py)
         META_DIRE_2026/cited_pid2citing_pids_list.pkl                    "
Output : ./cache/*   (every intermediate is cached per window, so a redraw never
                      recomputes, and adding a window only computes that window;
                      top_pids_* holds each index's selected top-X% pid list, and the
                      aggregate CSVs carry window_paper_count / top_paper_count /
                      measured_paper_count next to the means; top_paper_counts_*.csv is a
                      one-row-per-window table of how many top-X% papers each of e/f/g holds)
                     Every mean in the aggregate CSVs also gets a bootstrap 95% CI
                     (ci_low_* / ci_high_*, with n_boot / seed / boot_mode recording how it
                     was made). The CIs are stored only -- the figure does not draw them.
         ./figures/influence_spread_{windows}_y{offset}_{level}_top{pct}.svg/.png

At the paper's settings (use_all_topics=True, year_offset=5, max_year=2025,
min_pub_year=1945, windows=[(10,15),(100,150),(1000,1500)], top_pct=10,
group_level='field') this scatter IS the LEFT panel of Fig. 2c (is_fig2c_left) -- the
topic-spread twin of country_domain_diffusion/get_country_diffusion.py's Fig. 2c(right) --
so two outputs take the panel name:
    ./figures/Fig. 2c(left).svg / .png   (else influence_spread_{windows}_y{offset}_{level}_top{pct}{topic_source_suffix}.svg/.png)
    ./cache/Fig. 2c(left).csv            (the combined aggregate = the data behind the figure;
                                          else aggregate_all_{windows}_y{offset}_{level}_top{pct}{topic_source_suffix}.csv)
Everything else in ./cache keeps its parameterized name in every run -- the per-window
panel / selection / topic-stats / aggregate caches are keyed by window and shared across
runs, and top_paper_counts_*.csv is a companion table, not the figure's data.

Usage  : edit the settings block below (windows / top_pct / group_level / redraw),
         then `python get_influence_spread.py`.
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

# utils.py is in openalex/, two levels up from this file
# (openalex/2026_version/influence_spread/<this file>).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OPENALEX_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.insert(0, OPENALEX_DIR)
from utils import META_DIRE_2026  # noqa: E402


# ==================================================
# settings
# ==================================================
meta_dire = META_DIRE_2026
print("meta_dire", meta_dire)

# citation windows, each (lower bound, upper bound) and inclusive on both ends.
# every window is processed on its own; lower bounds must be >= 1 (marker size scales
# with log10 of them).
windows = [(10, 15), (100, 150), (1000, 1500)]

# top X% of each index that gets profiled
top_pct = 10.0

# hierarchy level the e/f/g quantiles are computed within:
# 'domain' / 'field' / 'subfield' / 'topic'
group_level = 'field'

# citing-side topic granularity for the spread stats:
#   False -> each citing paper contributes its single primary_topic (pid2primary_topic.pkl)
#   True  -> each citing paper contributes ALL of its topics (pid2topics.pkl, from
#            get_meta_data_preprint_2026_s3_topic_all.py). The topics are unioned across a
#            cited paper's citing papers; a citing paper counts once per DISTINCT topic it
#            carries, and the ratio's denominator stays the number of citing papers -- so a
#            paper with topics {A,B} and one with {A} give A in 2/2 papers -> max ratio 100%.
use_all_topics = True

# all-topics runs get their own cache / figure names so the two versions never collide;
# primary keeps the original (suffix-free) names.
topic_source_suffix = '_alltopics' if use_all_topics else ''

# True -> only redraw the figure from the cached per-window aggregates, never recompute
redraw = False

# must match get_efg_2026.py / get_efg_2026_merge.py
year_offset = 5
max_year = 2025          # data horizon: citations are complete through max_year included
min_pub_year = 1945

# a paper only gets a full year_offset-long citation window if pub_year + year_offset
# still lands inside the complete data, i.e. pub_year <= max_year - year_offset (2020).
# get_efg_2026.py only drops pub_year >= max_year, so its last year_offset cohorts carry
# truncated citation counts -- they would enter a citation window they do not belong in.
max_pub_year = max_year - year_offset + 1    # exclusive bound: keep pub_year < max_pub_year

print("year_offset", year_offset)
print("citation windows", windows)
print("top_pct", top_pct)
print("group_level", group_level)
print("use_all_topics", use_all_topics)
print("pub_year range kept", (min_pub_year, max_pub_year - 1))

# ---- deliverable naming ----------------------------------------------------------------
# This scatter at the paper's settings IS Fig. 2c (left panel) -- the topic-spread twin of
# country_domain_diffusion/get_country_diffusion.py's Fig. 2c(right) -- so the figure and the
# combined aggregate behind it are named after the panel; every other setting keeps the
# descriptive parameter stem. The gate lists every parameter that changes what the points
# show: the citation windows, the top-X% cut, the ranking level, the citing-side topic
# granularity (use_all_topics) and the year window. The PER-WINDOW caches (panel / selection
# / topic stats / per-window aggregate) and the top_paper_counts table keep their
# parameterized names in every run: they are keyed by window and shared across runs, so
# renaming them would only defeat the caching.
FIG_NAME = 'Fig. 2c(left)'
is_fig2c_left = (use_all_topics
                 and year_offset == 5 and max_year == 2025 and min_pub_year == 1945
                 and windows == [(10, 15), (100, 150), (1000, 1500)]
                 and top_pct == 10.0
                 and group_level == 'field')
print("is_fig2c_left", is_fig2c_left, ("-> %s" % FIG_NAME) if is_fig2c_left else "")

citation_col = f'citation_count_y{year_offset}'
INDEX_COLS = OrderedDict([          # column in gen_V3 -> (label, colour)
    (f'f_y{year_offset}', ('Foundation', 'red')),
    (f'e_y{year_offset}', ('Extension', 'orange')),
    (f'g_y{year_offset}', ('Generalization', 'blue')),
])

# marker diameter (points) = MARKER_SCALE * log10(window lower bound)
MARKER_SCALE = 6.0

# the line joining the three indices of one citation window
CONNECT_COLOR = 'gray'
CONNECT_LW = 2.2

gen_file_load_dire = os.path.join(meta_dire, 'gen_V3_keep_disp', f'gen_V3_y{year_offset}.csv')

cache_dire = os.path.join(BASE_DIR, 'cache')
figure_dire = os.path.join(BASE_DIR, 'figures')
os.makedirs(cache_dire, exist_ok=True)
os.makedirs(figure_dire, exist_ok=True)


def window_tag(lo, hi):
    return f'c{lo}-{hi}_y{year_offset}_{group_level}'


def window_top_tag(lo, hi):
    return f'{window_tag(lo, hi)}_top{top_pct:g}'


def panel_path(lo, hi):
    return os.path.join(cache_dire, f'panel_{window_tag(lo, hi)}.pkl')


def top_pids_path(lo, hi):
    # the selected top-X% pid list of each index, plus the window's paper count
    return os.path.join(cache_dire, f'top_pids_{window_top_tag(lo, hi)}.pkl')


def topic_stats_path(lo, hi):
    return os.path.join(cache_dire,
                        f'topic_stats_{window_top_tag(lo, hi)}{topic_source_suffix}.pkl')


def aggregate_path(lo, hi):
    return os.path.join(cache_dire,
                        f'aggregate_{window_top_tag(lo, hi)}{topic_source_suffix}.csv')


class _ProgressReader:
    """File wrapper that ticks a tqdm bar as pickle.load pulls bytes off it.

    The maps here are multi-GB, so an unpickle can sit silent for minutes; this turns
    that into a byte-level bar. Every read path pickle uses (read / readinto / readline /
    peek) is forwarded, so the count stays exact.
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


def read_gen_csv(path, load_cols, chunk_rows=2_000_000):
    """Read the gen_V3 table in chunks, filtering pub_year as it streams in.

    Chunking keeps the progress bar moving on a multi-GB CSV and drops the out-of-range
    cohorts before they ever pile up in memory.
    """
    kept, total_rows = [], 0
    with tqdm(desc='read gen_V3 csv', unit='row', unit_scale=True) as bar:
        for chunk in pd.read_csv(path, usecols=load_cols, chunksize=chunk_rows):
            total_rows += len(chunk)
            kept.append(chunk[(chunk['pub_year'] >= min_pub_year)
                              & (chunk['pub_year'] < max_pub_year)])
            bar.update(len(chunk))
            bar.set_postfix(kept=sum(len(c) for c in kept))
    df = pd.concat(kept, ignore_index=True)
    print(f"gen_V3 rows: {total_rows}; kept pub_year in "
          f"[{min_pub_year}, {max_pub_year - 1}]: {len(df)}")
    return df


# ==================================================
# 1. papers inside one citation window, with primary field + within-cell quantiles
# ==================================================
def build_panel(df_all, pid2group, lo, hi):
    df = df_all[(df_all[citation_col] >= lo) & (df_all[citation_col] <= hi)].copy()
    tqdm.write(f"  rows inside citation window [{lo}, {hi}]: {len(df)}")

    # primary field: papers without one are dropped entirely
    df[group_level] = df['cid'].map(pid2group)
    before = len(df)
    df = df[df[group_level].notna()]
    tqdm.write(f"  dropped {before - len(df)} papers with no primary {group_level};"
               f" kept {len(df)}")

    # quantile of each index against same-year, same-field papers *in this window only*.
    # rank(pct=True) with 'average' ties -> a paper tied at the top of a cell does not
    # get a free pass into the top 10%.
    grouped = df.groupby(['pub_year', group_level])
    tqdm.write(f"  ranking within {grouped.ngroups} (pub_year x {group_level}) cells")
    for col in tqdm(list(INDEX_COLS), desc=f'  rank {lo}-{hi}', leave=False):
        df[f'q_{col}'] = grouped[col].rank(pct=True, method='average')

    return df.reset_index(drop=True)


# ==================================================
# 2. topic spread of the selected papers' citations
# ==================================================
def build_topic_stats(target_pids, pid2year, cited_pid2citing_works, pid2citing_topics,
                      desc, use_all_topics):
    rows = []
    no_topic_papers = 0        # selected papers left with no topic-valid citation at all
    dropped_citations = 0      # citing papers ignored for having no topic

    bar = tqdm(target_pids, desc=desc, unit='paper', leave=False)
    for i, pid in enumerate(bar, 1):
        # same citation set get_efg_2026.py counted: citing papers published no later
        # than pub_year + year_offset
        year_threshold = pid2year[pid] + year_offset
        citing_papers = set()
        for citing in cited_pid2citing_works.get(pid, []):
            citing_year = pid2year.get(citing)
            if citing_year is not None and citing_year <= year_threshold:
                citing_papers.add(citing)

        # each citing paper contributes the DISTINCT set of topics it carries: +1 per topic,
        # but the denominator stays the number of citing papers (a paper with two topics is
        # still one citation). primary mode -> that set is the single primary topic, so this
        # reduces to one topic per paper. citing papers with no topic are ignored on both
        # sides of the ratio.
        topic_counter = Counter()
        valid_citations = 0
        for citing in citing_papers:
            entry = pid2citing_topics.get(citing)
            if not entry:
                continue
            topics = set(entry) if use_all_topics else (entry,)
            valid_citations += 1
            for topic in topics:
                topic_counter[topic] += 1

        dropped_citations += len(citing_papers) - valid_citations
        if valid_citations == 0:
            no_topic_papers += 1
            continue

        # live read on what the run is producing, refreshed every 2000 papers
        if i % 2000 == 0:
            bar.set_postfix(measured=len(rows), no_topic=no_topic_papers,
                            dropped_cit=dropped_citations, refresh=False)

        rows.append({
            'cid': pid,
            'citation_count': len(citing_papers),
            'valid_citation_count': valid_citations,
            'n_topics': len(topic_counter),
            'max_topic_count': max(topic_counter.values()),
            'max_topic_ratio': max(topic_counter.values()) / valid_citations,
        })

    bar.close()
    tqdm.write(f"  measured {len(rows)} papers;"
               f" {no_topic_papers} had no citation with a topic;"
               f" {dropped_citations} citations ignored for having no topic")

    cols = ['cid', 'citation_count', 'valid_citation_count',
            'n_topics', 'max_topic_count', 'max_topic_ratio']
    return pd.DataFrame(rows, columns=cols).set_index('cid')


# ==================================================
# 3. aggregate one window: one row per index
# ==================================================
# Each mean in the aggregate is a mean over the measured top-X% papers, so it carries a
# sampling error; the cache stores a bootstrap 95% CI of every mean next to it (percentile
# method), not just the SEM.
BOOT_N = 2000              # bootstrap replicates
BOOT_SEED = 20260801       # fixed, so re-running reproduces the same CI
BOOT_EXACT_MAX = 200_000   # a group larger than this uses the normal approximation below
# the per-paper columns whose mean gets a CI (all three go into the aggregate)
BOOT_COLS = ['n_topics', 'max_topic_ratio', 'valid_citation_count']


def bootstrap_means_ci(sub, cols=BOOT_COLS, n_boot=BOOT_N, seed=BOOT_SEED):
    """({column -> (ci_low, ci_high)}, how the replicates were drawn) for the mean of each
    column of `sub`.

    The PAPERS are resampled, not the columns: one set of resampled paper indices produces
    all the columns' replicate means at once, so the intervals describe the same resampled
    sets of papers (an x-CI and a y-CI on the figure then refer to one resampling).
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
        sub = df_stats.reindex(top_pids).dropna(subset=['n_topics'])
        ci, boot_mode = bootstrap_means_ci(sub)
        agg_rows.append({
            'cite_min': lo,
            'cite_max': hi,
            'window': f'{lo}-{hi}',
            'index': label,
            'column': col,
            'colour': colour,
            # papers in the window (after the primary-field filter) the quantiles ranked
            'window_paper_count': window_paper_count,
            # papers selected into this index's top X%
            'top_paper_count': len(top_pids),
            # of those, the ones with >= 1 topic-valid citation -- what the means are over
            'measured_paper_count': len(sub),
            'mean_n_topics': sub['n_topics'].mean(),
            'sem_n_topics': sub['n_topics'].sem(),
            'ci_low_n_topics': ci['n_topics'][0],
            'ci_high_n_topics': ci['n_topics'][1],
            'mean_max_topic_ratio': sub['max_topic_ratio'].mean(),
            'sem_max_topic_ratio': sub['max_topic_ratio'].sem(),
            'ci_low_max_topic_ratio': ci['max_topic_ratio'][0],
            'ci_high_max_topic_ratio': ci['max_topic_ratio'][1],
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
# reindexes the cached topic stats. Every other cache stays valid.
AGG_REQUIRED_COLS = ['ci_low_n_topics', 'ci_high_n_topics',
                     'ci_low_max_topic_ratio', 'ci_high_max_topic_ratio',
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
              f" -> rebuilding it from the cached topic stats")
        return False
    return True


# ==================================================
# driver: windows are processed one after another, each fully cached
# ==================================================
if redraw:
    missing = [f'{lo}-{hi}' for lo, hi in windows if not os.path.exists(aggregate_path(lo, hi))]
    if missing:
        raise SystemExit(f"redraw needs a cached aggregate for every window, missing: {missing}")
    stale = [f'{lo}-{hi}' for lo, hi in windows if not aggregate_is_current(lo, hi)]
    if stale:
        print(f"WARNING: {stale} have no bootstrap CI columns. Set redraw = False once to "
              f"rebuild those aggregates (only the cached topic stats are re-read).")

agg_list = []

if redraw:
    for lo, hi in windows:
        agg_list.append(pd.read_csv(aggregate_path(lo, hi)))
        print(f"[{lo}-{hi}] loaded cached aggregate")
else:
    # the windows still needing work decide which heavy inputs get loaded at all. An
    # aggregate that predates the bootstrap CI columns counts as work, but only the last
    # (cheap) step of it -- its panel / selection / topic-stats caches still hold.
    todo = [(lo, hi) for lo, hi in windows if not aggregate_is_current(lo, hi)]
    need_panel = [w for w in todo
                  if not (os.path.exists(top_pids_path(*w)) or os.path.exists(panel_path(*w)))]
    need_stats = [w for w in todo if not os.path.exists(topic_stats_path(*w))]

    print(f"windows needing work: {[f'{lo}-{hi}' for lo, hi in todo]}"
          f" (of {len(windows)}); panels to build: {len(need_panel)},"
          f" topic passes to run: {len(need_stats)}")

    df_all = pid2group = None
    if need_panel:
        load_cols = ['cid', 'pub_year', citation_col] + list(INDEX_COLS)
        df_all = read_gen_csv(gen_file_load_dire, load_cols)
        pid2group = load_pkl(os.path.join(meta_dire, f'pid2primary_{group_level}.pkl'))
        print(f"pid2primary_{group_level} load finished ({len(pid2group)} pids)")

    # per-window panels + top-pct selections (cheap once the CSV is in memory).
    # the selection itself is cached (pid lists + how many papers the window ranked), so
    # a rerun never has to touch the panel again.
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
                df_panel = build_panel(df_all, pid2group, lo, hi)
                dump_pkl(df_panel, panel_path(lo, hi))
                tqdm.write(f"  wrote panel cache: {df_panel.shape}")

            col2top_pids = OrderedDict()
            for col in INDEX_COLS:
                col2top_pids[col] = df_panel.loc[df_panel[f'q_{col}'] > cutoff, 'cid'].tolist()
            selection = {'window_paper_count': len(df_panel), 'col2top_pids': col2top_pids}
            dump_pkl(selection, top_pids_path(lo, hi))
            tqdm.write(f"  wrote selection cache: {os.path.basename(top_pids_path(lo, hi))}")
            del df_panel

        tqdm.write(f"  window papers ranked: {selection['window_paper_count']}")
        for col, (label, _) in INDEX_COLS.items():
            tqdm.write(f"  top {top_pct:g}% {label}:"
                       f" {len(selection['col2top_pids'][col])} papers")
        window2selection[(lo, hi)] = selection

    del df_all, pid2group

    # the three big maps are loaded once and shared by every window still missing stats.
    # the citing-side topic map is either the single-primary map or the all-topics map
    # (pid -> [topic_id, ...] from get_meta_data_preprint_2026_s3_topic_all.py).
    pid2year = cited_pid2citing_works = pid2citing_topics = None
    if need_stats:
        pid2year = load_pkl(os.path.join(meta_dire, 'pid2pub_year.pkl'))
        print(f"pid2pub_year load finished ({len(pid2year)} pids)")
        cited_pid2citing_works = load_pkl(os.path.join(meta_dire, 'cited_pid2citing_pids_list.pkl'))
        print(f"cited_pid2citing_pids_list load finished ({len(cited_pid2citing_works)} pids)")
        topic_map_file = 'pid2topics.pkl' if use_all_topics else 'pid2primary_topic.pkl'
        pid2citing_topics = load_pkl(os.path.join(meta_dire, topic_map_file))
        print(f"{topic_map_file} load finished ({len(pid2citing_topics)} pids)")

    for w_i, (lo, hi) in enumerate(todo, 1):
        tqdm.write(f"[window {w_i}/{len(todo)}] {lo}-{hi} topic spread")
        selection = window2selection[(lo, hi)]
        col2top_pids = selection['col2top_pids']
        if os.path.exists(topic_stats_path(lo, hi)):
            df_stats = load_pkl(topic_stats_path(lo, hi), desc=f'topic stats {lo}-{hi}')
            tqdm.write(f"  loaded cached topic stats: {df_stats.shape}")
        else:
            # one pass over the union, so a paper in two top-10% groups is measured once
            union_pids = sorted({pid for pids in col2top_pids.values() for pid in pids})
            tqdm.write(f"  papers needing topic stats (union): {len(union_pids)}")
            df_stats = build_topic_stats(union_pids, pid2year, cited_pid2citing_works,
                                         pid2citing_topics, desc=f'  citations {lo}-{hi}',
                                         use_all_topics=use_all_topics)
            dump_pkl(df_stats, topic_stats_path(lo, hi))
            tqdm.write(f"  wrote topic stats cache: {df_stats.shape}")

        df_agg = build_aggregate(df_stats, col2top_pids, selection['window_paper_count'], lo, hi)
        df_agg.to_csv(aggregate_path(lo, hi), index=False)
        tqdm.write(f"  wrote aggregate: {os.path.basename(aggregate_path(lo, hi))}")
        del df_stats

    del pid2year, cited_pid2citing_works, pid2citing_topics

    # cached windows are picked up here in the order the user asked for
    for lo, hi in windows:
        agg_list.append(pd.read_csv(aggregate_path(lo, hi)))

df_agg_all = pd.concat(agg_list, ignore_index=True)

windows_tag = '_'.join(f'{lo}-{hi}' for lo, hi in windows)
# the combined aggregate IS the data behind the figure, so it takes the panel name too
combined_stem = (
    FIG_NAME if is_fig2c_left
    else f'aggregate_all_{windows_tag}_y{year_offset}_{group_level}_top{top_pct:g}{topic_source_suffix}')
combined_path = os.path.join(cache_dire, f'{combined_stem}.csv')
df_agg_all.to_csv(combined_path, index=False)
print("wrote combined aggregate:", combined_path)
print(df_agg_all.to_string(index=False))

# how many top-X% papers each index contributes in each citation window, printed on every
# run (the per-window selection log above only fires for windows that get recomputed).
# window_paper_count is the same for the three indices of a window, so it is taken once.
print(f"\ntop {top_pct:g}% paper counts per citation window ({group_level}-level ranking)")
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
    cache_dire,
    f'top_paper_counts_{windows_tag}_y{year_offset}_{group_level}_top{top_pct:g}{topic_source_suffix}.csv')
df_counts.to_csv(counts_path, index=False)
print("wrote top-paper counts:", counts_path)

# ==================================================
# 4. plot: one point per (window x index)
# ==================================================
plt.rcParams['font.family'] = 'Gill Sans MT'

# the use_all_topics version is the one reproduced at full page width, so it uses
# enlarged fonts; the primary-topic version keeps the original sizes. (tweakable)
label_fontsize = 28 if use_all_topics else 19
tick_fontsize = 20 if use_all_topics else 12
title_fontsize = 16
legend_fontsize = 17 if use_all_topics else 15

fig, ax = plt.subplots(figsize=(7.0, 5.5))

# connect the three indices WITHIN each citation window, walking from the highest
# max-topic ratio down to the lowest. The line runs ON TOP of the markers (higher zorder)
# so it visibly reaches each marker's centre instead of being hidden under the dot;
# neutral grey keeps it from competing with the index colours.
for lo, hi in windows:
    sub = df_agg_all[(df_agg_all['cite_min'] == lo) & (df_agg_all['cite_max'] == hi)]
    sub = sub.dropna(subset=['mean_n_topics', 'mean_max_topic_ratio'])
    if len(sub) < 2:                       # an empty window has nothing to connect
        continue
    sub = sub.sort_values('mean_max_topic_ratio', ascending=False)
    ax.plot(sub['mean_n_topics'], sub['mean_max_topic_ratio'],
            color=CONNECT_COLOR, linewidth=CONNECT_LW, linestyle='--', zorder=4)

for _, row in df_agg_all.iterrows():
    markersize = MARKER_SCALE * math.log10(row['cite_min'])
    ax.plot(row['mean_n_topics'], row['mean_max_topic_ratio'],
            marker='o', markersize=markersize, color=row['colour'],
            linestyle='none', zorder=3)

# one legend carrying both encodings: colour = index (left column), marker size = citation
# window (right column). matplotlib fills a multi-column legend column-major, so listing the
# colour handles before the size handles puts each group in its own column.
colour_handles = [Line2D([], [], marker='o', linestyle='none', color=colour, markersize=8, label=label)
                  for label, colour in INDEX_COLS.values()]
size_handles = [Line2D([], [], marker='o', linestyle='none', color='grey',
                       markersize=MARKER_SCALE * math.log10(lo), label=f'{lo}-{hi} citations')
                for lo, hi in windows]

ax.legend(handles=colour_handles + size_handles, ncol=2, fontsize=legend_fontsize,
          frameon=False, loc='lower left', labelspacing=1.0,
          columnspacing=1.4, handletextpad=0.6)

# x is a count that spans orders of magnitude across the citation windows -> log scale.
# label the ticks as plain integers (4, 6, 10, 20) rather than matplotlib's default
# 4x10^0 / 2x10^1 scientific notation, which reads badly for a topic count.
ax.set_xscale('log')
int_fmt = FuncFormatter(lambda v, _pos: f'{v:g}')
ax.xaxis.set_major_formatter(int_fmt)
# label only the 2/3/5 subdivisions between decades -- labelling every minor tick puts
# "9" hard against "10"
ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 3.0, 5.0), numticks=12))
ax.xaxis.set_minor_formatter(int_fmt)

ax.set_xlabel("Number of citing topics", fontsize=label_fontsize)
ax.set_ylabel("Major topic citation ratio", fontsize=label_fontsize)
# 'both' minor+major: the x-axis labels its 2/3/5 minor ticks too (see the minor
# formatter above), and those would otherwise keep the default size
ax.tick_params(axis='both', which='both', labelsize=tick_fontsize)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

fig_base = os.path.join(
    figure_dire,
    FIG_NAME if is_fig2c_left
    else f'influence_spread_{windows_tag}_y{year_offset}_{group_level}_top{top_pct:g}{topic_source_suffix}')
plt.savefig(f'{fig_base}.svg', dpi=300, format='svg', transparent=True)
plt.savefig(f'{fig_base}.png', dpi=300)
print("saved figure:", f'{fig_base}.svg')

print("finished")
