"""
From the venues found by find_annual_review_venues.py (their source ids), collect:
    1. every paper whose primary location is one of those venues   -> a list
    2. every reference cited by those papers                        -> deduplicated list

Set USE_REVIEW_META below to choose the meta directory (must match the value used
in find_annual_review_venues.py):
    * False -> META_DIRE                  (get_meta_data_preprint_primary_location.py;
               article/preprint only -> annual-review (type=='review') papers are MISSING)
    * True  -> META_DIRE + '_with_review' (get_meta_data_preprint_review.py, KEEP_TYPES
               includes 'review'; required to actually find the annual-review papers)
Paper -> venue comes from pid2primay_location.pkl (pid -> primary source id).
Paper -> references comes from citing_pid2cited_pids_list.pkl (citing pid -> cited pids).

The reference collection is a union (set) over all the papers, saved as a sorted
list (so it is deduplicated but a plain list on disk).

Input  : <META>/annual_review_venues.pkl            (find_annual_review_venues.py; list of (sid, name))
         <META>/pid2primay_location.pkl             (meta pipeline)
         <META>/citing_pid2cited_pids_list.pkl       (meta pipeline)
Output : <META>/annual_review_papers.pkl             (list of paper pids)
         <META>/annual_review_paper_references.pkl   (list of referenced pids, deduplicated)
"""

import os
import sys
import pickle

# utils.py lives in the parent openalex/ directory
OPENALEX_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, OPENALEX_DIR)
from utils import META_DIRE  # noqa: E402

# ---- option: read/write the review-inclusive meta (get_meta_data_preprint_review.py) ----
USE_REVIEW_META = True
META = META_DIRE + '_with_review' if USE_REVIEW_META else META_DIRE
# suffix matching find_annual_review_venues.py so the two variants never collide
NAME_SUFFIX = '_with_review' if USE_REVIEW_META else ''
print("USE_REVIEW_META:", USE_REVIEW_META)
print("META dir:", META)


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# venue ids to EXCLUDE from the annual-review set (false positives you spotted);
# fill in source ids like 'S1234567', one per line.
EXCLUDE_VENUE_IDS = {
        'S4404676542', # African scientific annual review.
        'S4210233253', # Annual Center Review
        'S2764565036', # Annual Research & Review in Biology
        'S4210224225', # Annual Review Debats Revista de Cultura Poder i Societat
        'S4210178519', # Annual Review of Cognitive Linguistics
        'S4306417681', # Annual review of Cybertherapy and Telemedicine
        'S2898415390', # Annual Review of Fish Diseases
        'S2764798477', # Annual Review of Gerontology and Geriatrics
        'S4210228017', # Annual Review of Health Social Science
        'S4306502363', # Annual Review of Hiroshima Society for Science of Arts
        'S4210199373', # Annual Review of Language Acquisition
        'S4404675196', # Annual Review of Legal Studies
        'S4210190168', # Annual Review of Museum Anthropology
        'S2764581491', # Annual review of OCLC research
        'S4306503003', # ARELE : annual review of English language education in Japan
        'S4306503004', # ARELE: Annual Review of English Language Education in Japan
        'S4306505349', # BYZANTINA . Annual Review of the "Byzantine Research Centre"
        'S4210201822', # Design Principles and Practices An International Journalâ€”Annual Review
        'S4306511543', # Global Health: Annual Review
        'S2764398765', # Health transition review/Annual report
        'S4210232927', # Implementation of the Inuvialuit Final Agreement, annual review
        'S4210195811', # Jahrbuch fÃ¼r Recht und Ethik / Annual Review of Law and Ethics
        'S4387291028', # Journal of Business and Political Economy Biannual Review of The Indonesian Economy
        'S2738759712', # Lithuanian Annual Strategic Review
        'S4210183290', # Lithuanian energy security annual review
        'S4306520362', # MUN HWA JAE - Annual Review in Cultural Heritage Studies
        'S4210222343', # RCCS Annual Review
        'S4210189534', # SHEDET ANNUAL PEER-REVIEWED JOURNAL ISSUED BY THE FACULTY OF ARCHAEOLOGY FAYOUM UNIVERSITY
        'S4210189026', # The Annual Review of Social Partnerships
        'S4210193327', # The International Journal of Diversity in Organizations Communities and Nations Annual Review
        'S2765067168', # The International Journal of Diversity in Organizations, Communities, and Nations: Annual Review
        'S2764992306', # The International Journal of Environmental Cultural Economic and Social Sustainability Annual Review
        'S4210224347', # The International Journal of Interdisciplinary Social Sciences Annual Review
        'S2764858909', # The International Journal of Interdisciplinary Social Sciences: Annual Review
        'S4210169622', # The International Journal of Knowledge Culture and Change Management Annual Review
        'S2764818234', # The International Journal of Learning Annual Review
        'S2764485552', # The International Journal of the Arts in Society Annual Review
        'S2764591829', # The International Journal of the Humanities Annual Review
        'S2764653414', # The New Zealand Annual Review of Education
        'S4306532740', # The Supreme Court Law Review: Osgoodeâ€™s Annual Constitutional Cases Conference
        'S4363605385', # The Tennessee Williams Annual Review
        'S4387283130', # The Tennessee Williams Annual Review
        'S4387286167', # World Scientific Annual Review of Islamic Finance
}


# --- venues found by find_annual_review_venues.py: list of (sid, name) ---
venues = load_pkl(os.path.join(META, f'annual_review_venues{NAME_SUFFIX}.pkl'))
venue_sids = {sid for sid, _ in venues}
print("annual-review venues (matched):", len(venue_sids))

# drop the manually-excluded venues
excluded = venue_sids & EXCLUDE_VENUE_IDS
venue_sids -= EXCLUDE_VENUE_IDS
print(f"excluded {len(excluded)} venues -> {len(venue_sids)} kept")

# --- paper -> primary venue (source) id ---
pid2primary_location = load_pkl(os.path.join(META, 'pid2primay_location.pkl'))
print("pid2primay_location size:", len(pid2primary_location))

# 1. every paper published in one of those venues
papers = [pid for pid, sid in pid2primary_location.items() if sid in venue_sids]
print("papers in annual-review venues:", len(papers))

# --- paper -> list of cited pids (references) ---
citing2cited = load_pkl(os.path.join(META, 'citing_pid2cited_pids_list.pkl'))
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
papers_path = os.path.join(META, f'annual_review_papers{NAME_SUFFIX}.pkl')
with open(papers_path, 'wb') as f:
    pickle.dump(papers, f, pickle.HIGHEST_PROTOCOL)
print("wrote papers list:", papers_path)

refs_path = os.path.join(META, f'annual_review_paper_references{NAME_SUFFIX}.pkl')
with open(refs_path, 'wb') as f:
    pickle.dump(references, f, pickle.HIGHEST_PROTOCOL)
print("wrote references list:", refs_path)

print("finished")
