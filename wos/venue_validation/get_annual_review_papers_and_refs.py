"""
WoS version of the OpenAlex get_annual_review_papers_and_refs.py.

Collect the set of "review" papers and the references they cite:
    1. the review papers                                        -> a list
    2. every reference cited by those papers                    -> deduplicated list

Two ways to decide which papers are reviews, selected by detection_mode:
    'venue' : the paper's source_title is one of the annual-review venues found by
              find_annual_review_venues.py (their lower-cased source_title strings).
    'title' : the paper's title, lower-cased, CONTAINS the substring REVIEW_TITLE_KEYWORD
              ('review'), so 'review', 'reviews', 'reviewed', ... all qualify.

The 'title' mode tags its outputs with mode_suffix ('_title_review') so they never
overwrite the 'venue' mode's files.

Paper -> venue comes from get_meta_data_get_venue.py
(pid2source_title.pkl : pid -> source_title, None if missing).
Paper -> title / references come from get_meta_data.py
(pid2title.pkl ; citing_pid2cited_pids_list.pkl : citing pid -> list of cited pids).

Venue matching is on the lower-cased source_title (same convention as
find_annual_review_venues.py, which stores lower-cased names).

The reference collection is a union (set) over all the papers, saved as a sorted
list (so it is deduplicated but a plain list on disk).

Set use_with_review to match the upstream get_meta_data[_with_review].py /
get_meta_data_get_venue.py / find_annual_review_venues.py runs: when True it reads the
'_with_review' inputs and tags its outputs with the same suffix.

data_suffix = '_with_review' if use_with_review else '' ; out_suffix = data_suffix + mode_suffix

Input  : META_DIRE/annual_review_source_titles{data_suffix}.pkl   (find_annual_review_venues.py; list of lower-cased names; 'venue' mode only)
         META_DIRE/pid2source_title{data_suffix}.pkl               (get_meta_data_get_venue.py)
         META_DIRE/pid2title{data_suffix}.pkl                       (get_meta_data[_with_review].py; 'title' matching + sampled titles)
         META_DIRE/citing_pid2cited_pids_list{data_suffix}.pkl      (get_meta_data[_with_review].py)
Output : META_DIRE/annual_review_papers{out_suffix}.pkl            (list of paper pids)
         META_DIRE/annual_review_paper_references{out_suffix}.pkl  (list of referenced pids, deduplicated)
         ./sampled_review{out_suffix}.csv                          (100 randomly sampled review papers: id, title, source_title)
"""

import os
import sys
import csv
import pickle
import random

# utils.py lives in the parent wos/ directory
WOS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WOS_DIR)
from utils import META_DIRE  # noqa: E402

# match the upstream get_meta_data[_with_review].py / get_meta_data_get_venue.py /
# find_annual_review_venues.py runs: when True read the '_with_review' inputs and tag
# outputs with the same suffix.
use_with_review = True
data_suffix = '_with_review' if use_with_review else ''
print("use_with_review", use_with_review)
print("data_suffix", repr(data_suffix))

# How to decide which papers are "reviews":
#   'venue' : the paper's source_title is one of the annual-review venues found by
#             find_annual_review_venues.py (original behaviour).
#   'title' : the paper's title, lower-cased, CONTAINS the substring REVIEW_TITLE_KEYWORD
#             (so 'review', 'reviews', 'reviewed', ... all match).
# The 'title' mode tags its outputs with mode_suffix so it never overwrites the 'venue'
# mode's files.
detection_mode = 'title'   # 'venue' or 'title'
REVIEW_TITLE_KEYWORD = 'review'
assert detection_mode in ('venue', 'title'), detection_mode
mode_suffix = '' if detection_mode == 'venue' else '_title_review'
print("detection_mode", detection_mode)
print("mode_suffix", repr(mode_suffix))
# combined suffix appended to every output file (data_suffix first, then mode_suffix)
out_suffix = data_suffix + mode_suffix


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# venue names to EXCLUDE from the annual-review set (false positives you spot);
# fill in LOWER-CASED source_title strings, one per line.
EXCLUDE_VENUE_TITLES = {
    'annual review of applied linguistics',
    'annual review of cybertherapy and telemedicine',
    'lloyds bank annual review',
}


# --- paper -> source_title (needed for venue matching and for the sampled CSV) ---
pid2source_title = load_pkl(os.path.join(META_DIRE, f'pid2source_title{data_suffix}.pkl'))
print("pid2source_title size:", len(pid2source_title))

# --- paper -> title (needed for title matching and for the sampled CSV) ---
pid2title = load_pkl(os.path.join(META_DIRE, f'pid2title{data_suffix}.pkl'))
print("pid2title size:", len(pid2title))

# 1. select the review papers according to detection_mode
if detection_mode == 'venue':
    # venues found by find_annual_review_venues.py: list of lower-cased names
    venue_titles = load_pkl(os.path.join(META_DIRE, f'annual_review_source_titles{data_suffix}.pkl'))
    venue_titles = {name.lower() for name in venue_titles}
    print("annual-review venues (matched):", len(venue_titles))

    # drop the manually-excluded venues
    excluded = venue_titles & EXCLUDE_VENUE_TITLES
    venue_titles -= EXCLUDE_VENUE_TITLES
    print(f"excluded {len(excluded)} venues -> {len(venue_titles)} kept")

    # every paper published in one of those venues (match on lower-cased source_title)
    papers = [pid for pid, title in pid2source_title.items()
              if title and title.lower() in venue_titles]
    print("papers in annual-review venues:", len(papers))
else:
    # title mode: a paper is a review iff its lower-cased title CONTAINS the keyword
    # (substring match, so 'review', 'reviews', 'reviewed', ... all qualify)
    papers = [pid for pid, title in pid2title.items()
              if title and REVIEW_TITLE_KEYWORD in title.lower()]
    print(f"papers whose title contains {REVIEW_TITLE_KEYWORD!r}:", len(papers))

# --- paper -> list of cited pids (references) ---
citing2cited = load_pkl(os.path.join(META_DIRE, f'citing_pid2cited_pids_list{data_suffix}.pkl'))
print("citing_pid2cited_pids_list size:", len(citing2cited))

# 2. union of all references cited by those papers
references = set()
n_with_refs = 0
for pid in papers:
    refs = citing2cited.get(pid)
    if refs:
        references.update(refs)
        n_with_refs += 1
references = sorted(references)
print(f"papers with >=1 reference: {n_with_refs} / {len(papers)}")
print("unique references:", len(references))

# --- save ---
papers_path = os.path.join(META_DIRE, f'annual_review_papers{out_suffix}.pkl')
with open(papers_path, 'wb') as f:
    pickle.dump(papers, f, pickle.HIGHEST_PROTOCOL)
print("wrote papers list:", papers_path)

refs_path = os.path.join(META_DIRE, f'annual_review_paper_references{out_suffix}.pkl')
with open(refs_path, 'wb') as f:
    pickle.dump(references, f, pickle.HIGHEST_PROTOCOL)
print("wrote references list:", refs_path)

# --- randomly sample 100 review papers and save their id + title + venue ---
random.seed(42)
sample_n = min(100, len(papers))
sampled = random.sample(papers, sample_n)   # papers = the annual-review venue papers
print(f"sampled {sample_n} review papers")

out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'sampled_review{out_suffix}.csv')
with open(out_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'title', 'source_title'])
    for pid in sampled:
        writer.writerow([pid, pid2title.get(pid), pid2source_title.get(pid)])
print("wrote sampled review papers csv:", out_csv)

print("finished")
