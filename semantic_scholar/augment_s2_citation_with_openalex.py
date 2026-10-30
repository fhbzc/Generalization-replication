"""
Augment the Semantic Scholar (S2AG) citation network with citation edges that exist in
OpenAlex but are missing from S2AG.

Idea
----
Using the cids from get_s2ag_meta.py as the base, link each S2AG corpusid (cid) to its
OpenAlex paper id (pid) through DOI:
    cid --(cid2DOI.pkl, get_s2ag_meta.py)--> DOI <--(pid2doi.pkl, OpenAlex
                                                      get_meta_data_preprint_doi.py)-- pid
The DOI links are many-to-many: a DOI may map to MULTIPLE cids (S2AG duplicate corpusids),
and a cid/pid may carry MULTIPLE DOIs, so the link maps are list-valued (a pid resolves to
a LIST of cids). Then walk the OpenAlex citation graph (citing_pid2cited_pids_list.pkl from
OpenAlex get_meta_data_preprint.py). For every OpenAlex edge (citing_pid -> cited_pid) whose
BOTH endpoints link to >=1 S2AG cid, add every (citing_cid -> cited_cid) pair over the
cross-product of the two cid lists that is NOT already present in the S2AG network. Only
the citing->cited direction is loaded and augmented; the cited->citing direction is then
rebuilt as its exact transpose, so both written directions describe the same edge set.

Edge direction is identical in both sources ("citing cites cited"), so an OpenAlex edge
(citing_pid -> cited_pid) maps to the S2AG edge (citing_cid -> cited_cid).

We augment the '_remove_circle' network -- the temporally-consistent variant that the
downstream analysis (get_efg.py) actually consumes -- not the raw base network. To keep
it consistent, every OpenAlex edge we add must pass the SAME temporal check
get_s2ag_citation_s2_remove_circle.py applies (cited published on/before citing, by
publication date when both are known, else by year), and self-loops are skipped. So the
augmented output is itself a valid '_remove_circle' network, directly usable by get_efg.py
(point it at the *_openalex_augmented files) with no extra remove-circle pass.

Only cids in the S2AG network's universe (valid_cid_set = cid2pub_year ∩ cid2title, exactly
the node set get_s2ag_citation.py used) are eligible endpoints, so no out-of-universe node
is ever introduced.

Input  : META_DIRE/cid2DOI.pkl                          (get_s2ag_meta.py)
         META_DIRE/{cid2pub_year, cid2pub_date}.pkl     (get_s2ag_meta.py)
         META_DIRE/valid_cid_list.pkl                   (get_s2ag_citation.py; the network node set)
         META_DIRE/citing2cited_cid_list_remove_circle.pkl (get_s2ag_citation_s2_remove_circle.py; citing -> [cited]; only this direction is loaded)
         OPENALEX_META_DIRE/pid2doi.pkl                 (OpenAlex get_meta_data_preprint_doi.py)
         OPENALEX_META_DIRE/citing_pid2cited_pids_list.pkl (OpenAlex get_meta_data_preprint.py)
Output : META_DIRE/citing2cited_cid_list_remove_circle_openalex_augmented.pkl
         META_DIRE/cited2citing_cid_list_remove_circle_openalex_augmented.pkl
         META_DIRE/pid2cids_via_doi.pkl                 (OpenAlex pid -> list of S2AG cids link map)
"""

import os
import sys
import pickle
import importlib.util
from collections import defaultdict

from tqdm import tqdm

# utils.py is in this same directory; make it importable regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from utils import META_DIRE  # noqa: E402  (Semantic Scholar processed-data dir)

meta_dire = META_DIRE
print("meta_dire (S2AG)", meta_dire)

# ---- OpenAlex processed-data dir: read it from the sibling openalex/utils.py so the
#      path never drifts from that pipeline's own configuration. ----
OPENALEX_UTILS_PATH = os.path.join(os.path.dirname(BASE_DIR), 'openalex', 'utils.py')
assert os.path.exists(OPENALEX_UTILS_PATH), OPENALEX_UTILS_PATH
_spec = importlib.util.spec_from_file_location('openalex_utils', OPENALEX_UTILS_PATH)
_openalex_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_openalex_utils)
OPENALEX_META_DIRE = _openalex_utils.META_DIRE
print("OPENALEX_META_DIRE", OPENALEX_META_DIRE)

# ---- file names: augment the '_remove_circle' network (the one get_efg.py consumes);
#      outputs are suffixed so the originals are untouched. Only the citing->cited
#      direction is loaded; cited->citing is rebuilt as its transpose at the end. ----
S2_CITING2CITED_IN = 'citing2cited_cid_list_remove_circle.pkl'
S2_CITING2CITED_OUT = 'citing2cited_cid_list_remove_circle_openalex_augmented.pkl'
S2_CITED2CITING_OUT = 'cited2citing_cid_list_remove_circle_openalex_augmented.pkl'


def load_pkl(path):
    assert os.path.exists(path), path
    with open(path, 'rb') as f:
        return pickle.load(f)


def normalize_doi(doi):
    """Normalize a DOI for matching across sources: strip and lowercase (DOIs are
    case-insensitive). Returns None for empty/None."""
    if doi is None:
        return None
    doi = str(doi).strip().lower()
    return doi if doi else None


def iter_normalized_dois(doi_value):
    """Yield normalized DOIs from a value that may be a single DOI string, a list/tuple/set
    of DOIs, or None -- a cid/pid can carry more than one DOI. Empty/None entries skipped."""
    if doi_value is None:
        return
    if isinstance(doi_value, (list, tuple, set)):
        for d in doi_value:
            nd = normalize_doi(d)
            if nd is not None:
                yield nd
    else:
        nd = normalize_doi(doi_value)
        if nd is not None:
            yield nd


# ==================================================
# 1. base S2AG cid universe + publication chronology
#    (valid_cid_set is exactly the node set get_s2ag_citation.py used)
# ==================================================
# cid2pub_year / cid2pub_date are needed both for the cid universe and for the temporal
# check applied to every added edge (see is_consistent_citing_to_cited).
cid2pub_year = load_pkl(os.path.join(meta_dire, 'cid2pub_year.pkl'))
print("cid2pub_year size:", len(cid2pub_year))
cid2pub_date = load_pkl(os.path.join(meta_dire, 'cid2pub_date.pkl'))
print("cid2pub_date size:", len(cid2pub_date))

# valid_cid_list.pkl is cached by get_s2ag_citation.py (the node set behind the network
# we augment), so it always exists by the time this script runs -- just load it.
valid_cid_set = set(load_pkl(os.path.join(meta_dire, 'valid_cid_list.pkl')))
print("loaded valid_cid_set from valid_cid_list.pkl:", len(valid_cid_set))


def is_consistent_citing_to_cited(citing_cid, cited_cid):
    """Replicates get_s2ag_citation_s2_remove_circle.py's citing2cited rule for the edge
    (citing_cid -> cited_cid): keep iff the cited paper was published on/before the citing
    paper -- by publication DATE when both are known, otherwise by YEAR -- and it is not a
    self-loop. Both cids are guaranteed to be in valid_cid_set (so cid2pub_year[...] exists).
    """
    if citing_cid == cited_cid:
        return False
    citing_date = cid2pub_date.get(citing_cid)
    if citing_date is not None:
        cited_date = cid2pub_date.get(cited_cid)
        if cited_date is not None:
            return cited_date < citing_date
        return cid2pub_year[cited_cid] <= cid2pub_year[citing_cid]
    return cid2pub_year[cited_cid] <= cid2pub_year[citing_cid]

# ==================================================
# 2. DOI -> cid (restricted to the S2AG network universe)
# ==================================================
cid2DOI = load_pkl(os.path.join(meta_dire, 'cid2DOI.pkl'))
print("cid2DOI size:", len(cid2DOI))

# A single DOI can map to MULTIPLE S2AG cids (S2AG sometimes stores duplicate corpusids
# for the same paper), so keep ALL valid cids per DOI as a list rather than dropping
# collisions -- otherwise links to the duplicate records would be lost.
doi2cids = defaultdict(list)
for cid, doi in tqdm(cid2DOI.items(), desc='build doi2cids'):
    if cid not in valid_cid_set:
        continue
    # a cid's DOI value may be a single string or a list of DOIs; set() dedups within one cid
    for nd in set(iter_normalized_dois(doi)):
        doi2cids[nd].append(cid)
del cid2DOI
n_multi = sum(1 for cids in doi2cids.values() if len(cids) > 1)
print("doi2cids size (distinct DOIs w/ >=1 valid cid):", len(doi2cids),
      "| DOIs mapping to >1 cid:", n_multi)

# ==================================================
# 3. OpenAlex pid -> S2AG cid(s) (via DOI); value is a LIST (a DOI may map to several cids)
# ==================================================
pid2doi = load_pkl(os.path.join(OPENALEX_META_DIRE, 'pid2doi.pkl'))
print("OpenAlex pid2doi size:", len(pid2doi))

pid2cids = {}
for pid, doi in tqdm(pid2doi.items(), desc='build pid2cids'):
    # a pid's DOI value may also be a single string or a list of DOIs; collect the cids
    # for every DOI and dedup (different DOIs may resolve to overlapping cids)
    cids = set()
    for nd in iter_normalized_dois(doi):
        c = doi2cids.get(nd)
        if c:
            cids.update(c)
    if cids:
        pid2cids[pid] = list(cids)
del pid2doi, doi2cids
print("pid2cids size (OpenAlex pids linked to >=1 S2AG cid):", len(pid2cids))

with open(os.path.join(meta_dire, 'pid2cids_via_doi.pkl'), 'wb') as f:
    pickle.dump(pid2cids, f, pickle.HIGHEST_PROTOCOL)
print("saved pid2cids_via_doi.pkl")

# ==================================================
# 4. load ONLY the citing->cited direction, as dict-of-sets for fast membership tests
#    and de-duplicated insertion. cited->citing is rebuilt by transposing at the end.
# ==================================================
citing2cited_raw = load_pkl(os.path.join(meta_dire, S2_CITING2CITED_IN))
citing2cited = defaultdict(set)
for cid, cited_list in tqdm(citing2cited_raw.items(), desc='load citing2cited'):
    citing2cited[cid] = set(cited_list)
del citing2cited_raw

n_edges_before = sum(len(v) for v in citing2cited.values())
print("S2AG edges before augmentation:", n_edges_before)

# ==================================================
# 5. walk the OpenAlex citation graph; add edges missing from S2AG
# ==================================================
openalex_citing2cited = load_pkl(os.path.join(OPENALEX_META_DIRE, 'citing_pid2cited_pids_list.pkl'))
print("OpenAlex citing_pid2cited_pids_list size:", len(openalex_citing2cited))

n_oa_edges = 0            # OpenAlex edges seen
n_both_linked = 0         # OpenAlex edges whose BOTH endpoints link to >=1 S2AG cid
n_cid_pairs = 0           # cid-level (citing_cid, cited_cid) pairs considered (cross-product)
n_already_present = 0     # already in the S2AG network
n_skipped_inconsistent = 0  # self-loop OR temporally inconsistent in S2AG chronology
n_added = 0               # newly added (cid-level edges)

for citing_pid, cited_pids in tqdm(openalex_citing2cited.items(), desc='augment from OpenAlex'):
    n_oa_edges += len(cited_pids)
    citing_cids = pid2cids.get(citing_pid)
    if citing_cids is None:        # citing side not linked -> skip all its references
        continue
    for cited_pid in cited_pids:
        cited_cids = pid2cids.get(cited_pid)
        if cited_cids is None:
            continue
        n_both_linked += 1
        # one OpenAlex edge can map to several cid-level edges when a DOI resolves to
        # multiple duplicate cids -> add every (citing_cid -> cited_cid) pair
        for citing_cid in citing_cids:
            for cited_cid in cited_cids:
                n_cid_pairs += 1
                if cited_cid in citing2cited[citing_cid]:
                    n_already_present += 1
                    continue
                # only add edges that are temporally consistent in S2AG's own chronology,
                # so the augmented network stays a valid '_remove_circle' network (this
                # also drops self-loops, incl. distinct duplicate cids of the same paper
                # that fail the date/year test)
                if not is_consistent_citing_to_cited(citing_cid, cited_cid):
                    n_skipped_inconsistent += 1
                    continue
                # missing in S2AG and consistent -> add it (citing->cited only;
                # cited->citing is rebuilt by transposing after the loop)
                citing2cited[citing_cid].add(cited_cid)
                n_added += 1

del openalex_citing2cited, pid2cids

print("-" * 60)
print("OpenAlex edges seen                       :", n_oa_edges)
print("  both endpoints linked to >=1 S2AG cid   :", n_both_linked)
print("  cid-level pairs considered (cross-prod) :", n_cid_pairs)
print("    already present in S2AG               :", n_already_present)
print("    skipped (self-loop / temporally incons.):", n_skipped_inconsistent)
print("    NEWLY ADDED to S2AG                   :", n_added)
print("S2AG edges before -> after                :",
      n_edges_before, "->", n_edges_before + n_added)
print("-" * 60)

# ==================================================
# 6. save the augmented citing->cited (dict-of-lists, matching get_s2ag_citation.py),
#    then rebuild cited->citing as its exact transpose and save that too.
# ==================================================
citing2cited_out = {cid: list(s) for cid, s in tqdm(citing2cited.items(), desc='dump citing2cited')}
with open(os.path.join(meta_dire, S2_CITING2CITED_OUT), 'wb') as f:
    pickle.dump(citing2cited_out, f, pickle.HIGHEST_PROTOCOL)
print("saved", S2_CITING2CITED_OUT, "(keys:", len(citing2cited_out), ")")
del citing2cited_out

# transpose: cited2citing[cited] = [every citing that cites it]
cited2citing = defaultdict(list)
for citing_cid, cited_set in tqdm(citing2cited.items(), desc='transpose to cited2citing'):
    for cited_cid in cited_set:
        cited2citing[cited_cid].append(citing_cid)
del citing2cited

cited2citing = dict(cited2citing)
with open(os.path.join(meta_dire, S2_CITED2CITING_OUT), 'wb') as f:
    pickle.dump(cited2citing, f, pickle.HIGHEST_PROTOCOL)
print("saved", S2_CITED2CITING_OUT, "(keys:", len(cited2citing), ")")

print("finished")
