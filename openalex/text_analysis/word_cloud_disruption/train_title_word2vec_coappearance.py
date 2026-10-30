import os
import pickle
import sys
from tqdm import tqdm
import pandas as pd
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec


class EpochProgressLoss(CallbackAny2Vec):
    """Drives a per-epoch tqdm bar and reports the per-epoch training loss.

    gensim's get_latest_training_loss() is CUMULATIVE across epochs, so the loss
    of a single epoch is the difference from the previous epoch's cumulative value.
    Requires the model to be built with compute_loss=True."""

    def __init__(self, epochs):
        self.epoch = 0
        self.prev_cumulative = 0.0
        self.pbar = tqdm(total=epochs, desc='train epochs')

    def on_epoch_end(self, model):
        cumulative = model.get_latest_training_loss()
        epoch_loss = cumulative - self.prev_cumulative
        self.prev_cumulative = cumulative
        self.epoch += 1
        self.pbar.set_postfix(loss='%.2f' % epoch_loss)
        self.pbar.update(1)
        self.pbar.write('epoch %d/%d  loss %.4f' % (self.epoch, self.pbar.total, epoch_loss))

    def on_train_end(self, model):
        self.pbar.close()

# utils.py lives three levels up (openalex/), not in this
# text_analysis/word_claude_disruption/ directory; add the openalex directory to
# the path so the import resolves regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import META_DIRE


def get_english_stopwords():
    try:
        return set(stopwords.words('english'))
    except LookupError:
        print("NLTK stopwords corpus not found, downloading ...")
        nltk.download('stopwords')
        return set(stopwords.words('english'))


stop_words_set = get_english_stopwords()

meta_dire = META_DIRE
print("meta_dire", meta_dire)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Only titles published strictly before `year_cutoff` are used for training.
# Override on the command line, e.g.:  python train_title_word2vec_coappearance.py 2000
year_cutoff = 2020
if len(sys.argv) > 1:
    year_cutoff = int(sys.argv[1])

# `year_before` selects which parse_title_to_token.py output file to read; it is
# the upper bound of the token dump, independent of the training cutoff above.
year_before = 2025
parse_method = 'lemma'  # must match parse_title_to_token.py's method_string

embed_size = 128
seed = 100
epochs = 3
min_count = 0

# Co-appearance training: every token in a title should be a context word for
# every other token in the same title (a title is treated as a bag of
# co-appearing tokens, not an ordered sequence). A window larger than any
# realistic title length makes gensim's skip-gram consider all within-title
# pairs, which is exactly title-level co-appearance.
window_size = 100

# ---------------------------------------------------------------------------
# Restrict the training titles to the papers that pass compute_efgd_quantile.py's
# reference / citation (rt / ct) filters, i.e. the pids present in its output CSV.
# (pub_year is handled separately by `year_cutoff` above.) The CSV's `pid` column is
# used directly as a key, matching the pid type in pid2token_list / pid2pub_year.
# Set restrict_to_efgd = False to fall back to using every title before year_cutoff.
#
# The parameters below are named EXACTLY as in compute_efgd_quantile.py, so setting
# them the same way selects which quantile output (i.e. which filtered population) to
# train on. The pid SET is all that's used here, but which CSV that set comes from is
# fully determined by these knobs.
# ---------------------------------------------------------------------------
restrict_to_efgd = True
year_offset = 5              # must match compute_efgd_quantile.py
reference_threshold = 1      # rt
citation_threshold = 5       # ct
with_review = False          # compute_efgd_quantile.py with_review toggle (adds the '_with_review' suffix)
# pick ONE of the two pub_year cutoffs compute_efgd_quantile.py writes (its max_years list):
#   (2025 - year_offset, '')  complete-window table   |   (2025, '_full_cover')  all papers before 2025
max_year = 2025
name_tag = '_full_cover'
data_suffix = '_with_review' if with_review else ''

# which quantile family decides "has a quantile" (same CSV, different columns):
#   ''         -> within (pub_year, top_domain): only SINGLE-domain papers have it; no-/multi-domain
#                 papers are NaN and are therefore EXCLUDED from training
#   '_by_year' -> within pub_year : defined for ALL rt/ct papers (keeps every pid in the CSV)
#   '_global'  -> global          : defined for ALL rt/ct papers (keeps every pid in the CSV)
# Membership is judged by the disp (disruption) quantile column of this family being non-NaN; the
# (year,domain) NaN is shared across e/f/g/disp, so disp is a faithful representative.
quantile_kind = ''

print("year_cutoff", year_cutoff)
print("year_before", year_before)
print("parse_method", parse_method)
print("embed_size", embed_size)
print("seed", seed)
print("epochs", epochs)
print("min_count", min_count)
print("window_size (co-appearance)", window_size)
print("restrict_to_efgd", restrict_to_efgd)
if restrict_to_efgd:
    print("  efgd select: year_offset", year_offset, "| rt", reference_threshold,
          "| ct", citation_threshold, "| with_review", with_review,
          "| max_year", max_year, "| name_tag", name_tag,
          "| quantile_kind", quantile_kind or '(year,domain)')

# ---------------------------------------------------------------------------
# Load inputs
# ---------------------------------------------------------------------------
with open(os.path.join(meta_dire, 'pid2eng_token_list_%s_before_%s.pkl' % (parse_method, year_before)), 'rb') as f:
    pid2token_list = pickle.load(f)
print("load pid2token_list finished, size", len(pid2token_list))

with open(os.path.join(meta_dire, 'pid2pub_year.pkl'), 'rb') as f:
    pid2pub_year = pickle.load(f)
print("load pid2pub_year finished, size", len(pid2pub_year))

# ---------------------------------------------------------------------------
# (optional) load the set of pids that pass compute_efgd_quantile.py's rt/ct filters
# AND have a quantile of the chosen `quantile_kind` (non-NaN disp quantile of that family)
# ---------------------------------------------------------------------------
selected_pids = None
if restrict_to_efgd:
    efgd_csv = os.path.join(
        meta_dire, 'efg_computation',
        'efgd_within_year_domain_quantile_y%s%s_rt_%s_ct_%s_ybefore_%s%s.csv'
        % (year_offset, data_suffix, reference_threshold,
           citation_threshold, max_year, name_tag))
    assert os.path.exists(efgd_csv), (
        "%s\nRun compute_efgd_quantile.py first (or set restrict_to_efgd = False)." % efgd_csv)
    rep_col = 'disp_y%s_quantile%s' % (year_offset, quantile_kind)   # representative of the family
    df_sel = pd.read_csv(efgd_csv, usecols=['pid', rep_col]).dropna(subset=[rep_col])
    selected_pids = set(df_sel['pid'])
    print("load efgd pid set finished, size", len(selected_pids),
          "(with non-NaN %s) from" % rep_col, efgd_csv)

# ---------------------------------------------------------------------------
# Build the co-appearance corpus: one "sentence" per title, keeping only titles
# published before year_cutoff (and, when restrict_to_efgd, whose pid passed rt/ct).
# ---------------------------------------------------------------------------
list_of_token_list = []
for pid, token_list in tqdm(pid2token_list.items(), desc='build corpus'):
    pub_year = pid2pub_year.get(pid)
    if pub_year is None or pub_year >= year_cutoff:
        continue
    if selected_pids is not None and pid not in selected_pids:
        continue

    valid_token_list = [token for token in token_list
                        if token not in stop_words_set and len(token) > 1]
    # need at least 2 tokens for any co-appearance pair
    if len(valid_token_list) > 1:
        list_of_token_list.append(valid_token_list)

print("number of titles used for training", len(list_of_token_list))

# ---------------------------------------------------------------------------
# Train word2vec (skip-gram) on title-level co-appearance
# ---------------------------------------------------------------------------
print("training word2vec for titles before %s ..." % year_cutoff)
model = Word2Vec(sentences=list_of_token_list,
                 vector_size=embed_size,
                 window=window_size,
                 min_count=min_count,
                 sg=1,
                 seed=seed,
                 workers=16,
                 epochs=epochs,
                 compute_loss=True,
                 callbacks=[EpochProgressLoss(epochs)])

save_directory = os.path.join(meta_dire, 'text_embedding_coappearance')
os.makedirs(save_directory, exist_ok=True)
# carry the efgd restriction (rt/ct + quantile family) in the filename so different selections coexist
efgd_tag = ('_efgd%s_%s_rt%s_ct%s'
            % (data_suffix, quantile_kind.lstrip('_') or 'year_domain',
               reference_threshold, citation_threshold)
            if restrict_to_efgd else '')
save_file_name = os.path.join(
    save_directory,
    'word_embedding_coappearance_before_%s_dim_%s_min_count_%s_epoch_%s_seed_%s%s.model'
    % (year_cutoff, embed_size, min_count, epochs, seed, efgd_tag))

model.save(save_file_name)
print("saved model to", save_file_name)
print("vocabulary size", len(model.wv))
print("finished")
