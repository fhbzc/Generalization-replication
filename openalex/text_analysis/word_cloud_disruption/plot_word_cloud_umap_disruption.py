"""
UMAP word cloud of the co-appearance word2vec vocabulary, with the disruption anchor
words highlighted and labelled.

This is purely a word-embedding plot -- it does NOT touch any paper / disruption table.

Pipeline
--------
1. Load the co-appearance word2vec model (train_title_word2vec_coappearance.py).

2. Keep the tokens that are NOT stopwords and whose word2vec vocabulary frequency
   (count) > MIN_FREQ. The anchor words (below) are always kept even if they fall
   short, so they can always be labelled.

3. L2-normalise every kept word vector and reduce the whole set to 2D with UMAP
   (metric='cosine'). The 2D layout is cached (keyed by the vocab + filter + UMAP params)
   so re-runs only redraw.

4. Scatter every token as a thin light-grey dot. On top, draw each dimension: a
   dimension is the pairwise product of two word groups (A - B, C - D, E - F), and every
   (pos, neg) pair is an arrow pointing FROM the negative word TO the positive word --
   dimension 1 in red, dimension 2 in blue, dimension 3 in green. Word names are NOT drawn.

Inputs (filenames mirror train_title_word2vec_coappearance.py):
    META_DIRE/text_embedding_coappearance/word_embedding_coappearance_before_{year_cutoff}_dim_{dim}_
        min_count_{mc}_epoch_{ep}_seed_{seed}{efgd_tag}.model
Output:
    ./figures/word_cloud_umap/word_cloud_umap_before_{year_cutoff}_{efgd_tag}_
        minfreq_{MIN_FREQ}.svg/.png (+ a .csv with word, x, y, count, is_anchor)
"""

import os
import sys
import pickle
import itertools

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec

# utils.py lives three levels up (openalex/); add it to the path regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import META_DIRE

meta_dire = META_DIRE
print("meta_dire", meta_dire)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
year_cutoff = 2020                # papers/titles before this year; CLI override below
if len(sys.argv) > 1:
    year_cutoff = int(sys.argv[1])

# ---- six word groups -> three dimensions ------------------------------------------
# Each dimension is the pairwise product of two groups (positive group - negative group):
#   dimension 1 = A - B  (every a in A paired with every b in B)   -> drawn RED
#   dimension 2 = C - D                                            -> drawn BLUE
#   dimension 3 = E - F                                            -> drawn GREEN
# Every pair (pos, neg) is drawn as an arrow pointing FROM neg TO pos.
GROUP_A = ('software', 'device', 'tool', 'technique')        # dimension 1 positive end
GROUP_B = ('theory', 'hypothesis', 'model')     # dimension 1 negative end
GROUP_C = ('introduction', 'creation', 'create', 'introduce')          # dimension 2 positive end
GROUP_D = ('refinement', 'extension', 'refine', 'extend')           # dimension 2 negative end
# GROUP_E = ('novel', 'new', 'innovative', 'emergent')                      # dimension 3 positive end
# GROUP_F = ('established',)                       # dimension 3 negative end

# which tokens to include in the cloud: NOT a stopword AND with word2vec vocabulary
# frequency (count) strictly greater than MIN_FREQ.
MIN_FREQ = 800       # only plot tokens whose word2vec vocab count is > this N

# ---- 2D reduction: pick the algorithm to compare their layouts ----
REDUCTION_METHOD = 'umap'     # 'umap' | 'tsne'

# UMAP parameters
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = 'cosine'
UMAP_SEED = 42

# t-SNE parameters
TSNE_PERPLEXITY = 30
TSNE_LEARNING_RATE = 'auto'
TSNE_METRIC = 'cosine'
TSNE_SEED = 42

print("REDUCTION_METHOD", REDUCTION_METHOD)
if REDUCTION_METHOD == 'umap':
    print("UMAP_N_NEIGHBORS", UMAP_N_NEIGHBORS)
    print("UMAP_MIN_DIST", UMAP_MIN_DIST)
    print("UMAP_METRIC", UMAP_METRIC)
    print("UMAP_SEED", UMAP_SEED)
elif REDUCTION_METHOD == 'tsne':
    print("TSNE_PERPLEXITY", TSNE_PERPLEXITY)
    print("TSNE_LEARNING_RATE", TSNE_LEARNING_RATE)
    print("TSNE_METRIC", TSNE_METRIC)
    print("TSNE_SEED", TSNE_SEED)

# re-use a cached 2D layout when the vocab + filters + UMAP params match; set True to ignore it
FORCE_RECOMPUTE = False

# when drawing, drop background points that sit far out in the 2D plane (robust outliers):
# a point is removed if |x - median(x)| > LEVERAGE_K * MAD(x) OR the same in y. Anchor words
# (arrow endpoints) are ALWAYS kept so no arrow breaks. Set REMOVE_HIGH_LEVERAGE = False to
# keep every point. This only affects the plot, never the cached layout / CSVs.
REMOVE_HIGH_LEVERAGE = True
LEVERAGE_K = 2.0

# ---- locate the word2vec model (must match train_title_word2vec_coappearance.py) ----
embed_size = 128
min_count = 0
epochs = 3
seed = 100
restrict_to_efgd = True
quantile_kind = ''            # '' (year,domain) | '_by_year' | '_global'
reference_threshold = 1       # rt
citation_threshold = 5        # ct
with_review = False
data_suffix = '_with_review' if with_review else ''
efgd_tag = ('_efgd%s_%s_rt%s_ct%s'
            % (data_suffix, quantile_kind.lstrip('_') or 'year_domain',
               reference_threshold, citation_threshold)
            if restrict_to_efgd else '')

model_path = os.path.join(
    meta_dire, 'text_embedding_coappearance',
    'word_embedding_coappearance_before_%s_dim_%s_min_count_%s_epoch_%s_seed_%s%s.model'
    % (year_cutoff, embed_size, min_count, epochs, seed, efgd_tag))

# short tag for the output filenames (encodes the model params, no paper/term info)
run_tag = efgd_tag.lstrip('_') or 'plain'


# a bare string would be iterated character-by-character, so coerce every group to a tuple
def _as_word_tuple(g):
    return (g,) if isinstance(g, str) else tuple(g)


# each dimension = the pairwise product of its positive x negative groups
axis1_pairs = list(dict.fromkeys(itertools.product(_as_word_tuple(GROUP_A), _as_word_tuple(GROUP_B))))
axis2_pairs = list(dict.fromkeys(itertools.product(_as_word_tuple(GROUP_C), _as_word_tuple(GROUP_D))))

# words to keep in the vocab (so the arrow endpoints have coordinates)
anchor_words = list(dict.fromkeys(
    [w for pair in (axis1_pairs + axis2_pairs) for w in pair]))

print("year_cutoff", year_cutoff)
print("MIN_FREQ", MIN_FREQ)
print("dimension 1 pairs (red,  A x B, %d):" % len(axis1_pairs), axis1_pairs)
print("dimension 2 pairs (blue, C x D, %d):" % len(axis2_pairs), axis2_pairs)
print("anchor words (%d):" % len(anchor_words), anchor_words)
print("model_path", model_path)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
assert os.path.exists(model_path), (
    "%s\nRun train_title_word2vec_coappearance.py first." % model_path)
model = Word2Vec.load(model_path)
wv = model.wv
print("loaded model, vocab size", len(wv))


def vocab_count(word):
    """word2vec training-corpus frequency of `word` (0 if not in vocab)."""
    return int(wv.get_vecattr(word, 'count')) if word in wv else 0


def get_english_stopwords():
    try:
        return set(stopwords.words('english'))
    except LookupError:
        print("NLTK stopwords corpus not found, downloading ...")
        nltk.download('stopwords')
        return set(stopwords.words('english'))


stop_words_set = get_english_stopwords()
print("loaded stopwords:", len(stop_words_set))

# ---------------------------------------------------------------------------
# Eligible tokens: non-stopword and word2vec vocab count > MIN_FREQ.
# ---------------------------------------------------------------------------
eligible_words = []
for word in wv.index_to_key:
    if word in stop_words_set:
        continue
    if vocab_count(word) <= MIN_FREQ:             # frequency must be strictly > MIN_FREQ
        continue
    eligible_words.append(word)

print("eligible tokens (non-stopword, count > %d): %d" % (MIN_FREQ, len(eligible_words)))
assert eligible_words, "no eligible tokens (loosen MIN_FREQ)"

# anchor words are always kept so they can be labelled; warn about any not in the vocab.
anchor_in_vocab = [w for w in anchor_words if w in wv]
anchor_missing = [w for w in anchor_words if w not in wv]
for w in anchor_missing:
    print("  [WARN] anchor word %r has no embedding (not in vocab) -> cannot be plotted" % w)
below_freq = [w for w in anchor_in_vocab if vocab_count(w) <= MIN_FREQ]
for w in below_freq:
    print("  [note] anchor word %r has count=%d (<= MIN_FREQ=%d) -- force-kept for labelling"
          % (w, vocab_count(w), MIN_FREQ))

# final word list = eligible tokens UNION the anchor words present in the vocab (order-stable)
eligible_set = set(eligible_words)
words = eligible_words + [w for w in anchor_in_vocab if w not in eligible_set]
words = list(dict.fromkeys(words))
anchor_set = set(anchor_in_vocab)
print("total words for UMAP:", len(words), "| anchors among them:", len(anchor_set))

# ---------------------------------------------------------------------------
# 2D layout via UMAP (cached).
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
fig_dir = os.path.join(BASE_DIR, 'figures', 'word_cloud_umap')
os.makedirs(fig_dir, exist_ok=True)

assert REDUCTION_METHOD in ('umap', 'tsne'), "REDUCTION_METHOD must be 'umap' or 'tsne'"
# the method is baked into the filenames so the UMAP and t-SNE layouts / figures never collide
base = ('word_cloud_%s_before_%s_%s_minfreq_%s'
        % (REDUCTION_METHOD, year_cutoff, run_tag, MIN_FREQ))
cache_path = os.path.join(fig_dir, base + '_layout.pkl')

# a config the cached layout must match to be reusable (includes the method + its params)
if REDUCTION_METHOD == 'umap':
    method_params = {'n_neighbors': UMAP_N_NEIGHBORS, 'min_dist': UMAP_MIN_DIST,
                     'metric': UMAP_METRIC, 'seed': UMAP_SEED}
else:
    method_params = {'perplexity': TSNE_PERPLEXITY, 'learning_rate': TSNE_LEARNING_RATE,
                     'metric': TSNE_METRIC, 'seed': TSNE_SEED}
cache_key = {'words': tuple(words), 'method': REDUCTION_METHOD, 'params': method_params,
             'year_cutoff': year_cutoff, 'MIN_FREQ': MIN_FREQ}

emb2d = None
if not FORCE_RECOMPUTE and os.path.exists(cache_path):
    with open(cache_path, 'rb') as f:
        cached = pickle.load(f)
    if cached.get('key') == cache_key:
        emb2d = cached['emb2d']
        print("loaded %s layout cache: %s (%d words)" % (REDUCTION_METHOD.upper(), cache_path, len(emb2d)))
    else:
        print("cache config changed -> recomputing %s layout" % REDUCTION_METHOD.upper())

if emb2d is None:
    # L2-normalised (norm=1) embedding matrix for every kept word
    X = np.vstack([wv.get_vector(w, norm=True) for w in words]).astype(np.float32)

    if REDUCTION_METHOD == 'umap':
        try:
            import umap
        except ImportError:
            raise SystemExit("umap-learn is required. Install it with:\n    pip install umap-learn")
        print("running UMAP on", X.shape, "(metric=%s) ..." % UMAP_METRIC)
        reducer = umap.UMAP(n_components=2, n_neighbors=UMAP_N_NEIGHBORS,
                            min_dist=UMAP_MIN_DIST, metric=UMAP_METRIC,
                            random_state=UMAP_SEED)
        emb2d = reducer.fit_transform(X)
    else:
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            raise SystemExit("scikit-learn is required for t-SNE. Install it with:\n    pip install scikit-learn")
        print("running t-SNE on", X.shape, "(metric=%s, perplexity=%s) ..." % (TSNE_METRIC, TSNE_PERPLEXITY))
        reducer = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY,
                       learning_rate=TSNE_LEARNING_RATE, metric=TSNE_METRIC,
                       init='pca', random_state=TSNE_SEED)
        emb2d = reducer.fit_transform(X)

    emb2d = np.asarray(emb2d, dtype=float)
    with open(cache_path, 'wb') as f:
        pickle.dump({'key': cache_key, 'emb2d': emb2d}, f, pickle.HIGHEST_PROTOCOL)
    print("saved %s layout cache: %s" % (REDUCTION_METHOD.upper(), cache_path))

# assemble a tidy table
df = pd.DataFrame({
    'word': words,
    'x': emb2d[:, 0],
    'y': emb2d[:, 1],
    'count': [vocab_count(w) for w in words],
    'is_anchor': [w in anchor_set for w in words],
})

# df_plot = what actually gets drawn / bounds the view. Optionally drop robust outliers far
# out in x or y (they stretch the view and leave big empty bands); anchors are always kept.
df['high_leverage_removed'] = False
df_plot = df
n_drop = 0
if REMOVE_HIGH_LEVERAGE:
    def _robust_inlier(v):
        med = np.median(v)
        mad = np.median(np.abs(v - med))
        scale = 1.4826 * mad if mad > 0 else (v.std(ddof=0) or 1.0)   # ~sigma; fallback to std
        return np.abs(v - med) <= LEVERAGE_K * scale
    keep = (_robust_inlier(df['x'].values) & _robust_inlier(df['y'].values)) | df['is_anchor'].values
    df['high_leverage_removed'] = ~keep
    n_drop = int((~keep).sum())
    df_plot = df[keep]
    print("high-leverage removal: dropped %d / %d far points (LEVERAGE_K=%s, anchors kept)"
          % (n_drop, len(df), LEVERAGE_K))

# ---------------------------------------------------------------------------
# Plot: thin grey dots for every token; on top, draw each dimension's pairs as arrows
# (negative word -> positive word), red = dimension 1, blue = dimension 2,
# green = dimension 3. The word names are NOT labelled.
# ---------------------------------------------------------------------------
AXIS1_COLOR = '#d62728'   # red   -> dimension 1
AXIS2_COLOR = '#1f5fd6'   # blue  -> dimension 2
AXIS3_COLOR = '#2ca02c'   # green -> dimension 3

fig, ax = plt.subplots(figsize=(16, 14))

# background cloud: very fine light-grey points (outliers dropped if REMOVE_HIGH_LEVERAGE)
ax.scatter(df_plot['x'], df_plot['y'], s=3, c='#c8c8c8', alpha=0.5, linewidths=0, zorder=1)

# coordinate lookup for the arrow endpoints
word2xy = {w: (float(x), float(y)) for w, x, y in zip(df['word'], df['x'], df['y'])}


def draw_axis_arrows(pairs, color, label):
    """Draw one arrow per (first, second) pair, pointing FROM `second` TO `first`.
    Skips a pair if either word has no coordinate (not in the vocab)."""
    drawn = 0
    for first, second in pairs:
        if first not in word2xy or second not in word2xy:
            print("  [skip arrow] %r -> %r (missing coordinate)" % (second, first))
            continue
        x1, y1 = word2xy[second]           # tail = second item
        x2, y2 = word2xy[first]            # head = first item
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=1.8, alpha=0.85,
                                    shrinkA=3, shrinkB=3,
                                    mutation_scale=18),
                    zorder=3)
        drawn += 1
    # one proxy handle for the legend
    ax.plot([], [], color=color, lw=2.2, label=label)
    return drawn


draw_axis_arrows(axis1_pairs, AXIS1_COLOR, 'Non Modular -> Modular (dimension 1)')
draw_axis_arrows(axis2_pairs, AXIS2_COLOR, 'Extend -> Create (dimension 2)')

ax.legend(loc='best', fontsize=18, frameon=False)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():         # drop the box frame
    spine.set_visible(False)
# clamp the view to the plotted point cloud (+ a small pad) so there is no wide empty band
xpad = 0.02 * (df_plot['x'].max() - df_plot['x'].min())
ypad = 0.02 * (df_plot['y'].max() - df_plot['y'].min())
ax.set_xlim(df_plot['x'].min() - xpad, df_plot['x'].max() + xpad)
ax.set_ylim(df_plot['y'].min() - ypad, df_plot['y'].max() + ypad)
fig.tight_layout()

svg_path = os.path.join(fig_dir, base + '.svg')
png_path = os.path.join(fig_dir, base + '.png')
csv_path = os.path.join(fig_dir, base + '.csv')
fig.savefig(svg_path, format='svg', bbox_inches='tight')
fig.savefig(png_path, dpi=200, bbox_inches='tight')
df.sort_values('count', ascending=False).to_csv(csv_path, index=False)
plt.close(fig)
print("saved:", svg_path)
print("saved:", png_path)
print("saved:", csv_path)

# summary CSV: how many far points were removed by the high-leverage filter
leverage_csv = os.path.join(fig_dir, base + '_high_leverage_removed.csv')
pd.DataFrame([{
    'reduction_method': REDUCTION_METHOD,
    'remove_high_leverage': REMOVE_HIGH_LEVERAGE,
    'leverage_k': LEVERAGE_K,
    'n_total': len(df),
    'n_removed': n_drop,
    'n_plotted': len(df_plot),
}]).to_csv(leverage_csv, index=False)
print("saved:", leverage_csv)

# ---------------------------------------------------------------------------
# Pairwise similarity among the constituent (pos - neg) DIRECTIONS within each dimension
# (from the raw word2vec vectors, NOT the 2D layout). Each product pair (pos, neg) defines
#     dir = unit( unit_vec(pos) - unit_vec(neg) )
# exactly as plot_word_axis_projection_disruption.py. Dimension 1 (A x B) has |A|*|B| such
# directions; we report the cosine similarity between every unordered pair of them (and
# likewise for dimensions 2 and 3), so a value near 1 means the two point the same way.
# ---------------------------------------------------------------------------
def pair_direction(pair):
    """unit(unit_vec(pos) - unit_vec(neg)); None if either word is out of vocab."""
    pos, neg = pair
    if pos not in wv or neg not in wv:
        return None
    diff = wv.get_vector(pos, norm=True) - wv.get_vector(neg, norm=True)
    norm = np.linalg.norm(diff)
    return diff / norm if norm > 0 else None


cos_rows = []
for dim_name, pairs in [('dimension1', axis1_pairs), ('dimension2', axis2_pairs)]:
    # build a direction per (pos, neg) pair (labelled "pos-neg"), skipping any with a missing word
    dims = []
    for pair in pairs:
        d = pair_direction(pair)
        if d is None:
            print("  [skip pair] %s (missing word) for %s" % ('-'.join(pair), dim_name))
            continue
        dims.append(('%s-%s' % (pair[0], pair[1]), d))
    # cosine similarity between every unordered pair of directions (i < j)
    for i in range(len(dims)):
        for j in range(i + 1, len(dims)):
            name_i, di = dims[i]
            name_j, dj = dims[j]
            cos = float(np.dot(di, dj))          # both are unit vectors -> dot = cosine
            cos_rows.append({'dimension': dim_name, 'pair_i': name_i, 'pair_j': name_j,
                             'cosine_similarity': cos})

cos_csv = os.path.join(fig_dir, base + '_dimension_pairwise_cosine.csv')
pd.DataFrame(cos_rows).to_csv(cos_csv, index=False)
print("saved:", cos_csv)
print("finished")
