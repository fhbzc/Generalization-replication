"""
Semantic Scholar (S2AG) version of the OpenAlex evaluate_reshuffle_control_venue.py.

Evaluate how much the VENUE-CONTROLLED reshuffled citation network's degree
distribution differs from the input (original) network. The node id is the S2AG corpusid.

construct_reshuffled_network_control_venue.py writes, per iteration, in
    META_DIRE/random_shuffle_citation_control_venue/<iter>/constructed_network/
        shuffled_citing_pid2cited_pids_list_before_<Y>.pkl  citing -> [cited]
        shuffled_cited_pid2citing_pids_list_before_<Y>.pkl  cited  -> [citing]
    where <Y> is that script's upper_year (set-deduped, keeping only edges whose
    citing/source year is STRICTLY BEFORE <Y>).

The venue-controlled reshuffle (network_reshuffle_efficient_control_venue.py) only
touched edges whose CITING paper's publication year is within [LOWER_YEAR, UPPER_YEAR]
(= [-1, 2024]) AND additionally keeps a venue control: a swap keeps both endpoints'
(year, venue) fixed, so it is degree-preserving per node. In practice degree can still be
LOST because:
  * reshuffle_network_approx does not prevent creating parallel edges, and the final
    set() de-dup in construct collapses them;
  * (source_year, source_venue, target_year, target_venue) blocks with E == 1 get
    shuffle_count == 0 and are skipped (their single edge never reaches the output).

IMPORTANT — input network: the venue-controlled reshuffle's INPUT is the venue-filtered
network citing2cited_cid_list_remove_circle_valid_venue.pkl (every cid in it has a valid
venue). The baseline ("original") network here is therefore that SAME venue-filtered
file, so the comparison is like-for-like.

IMPORTANT — edge subset: construct's `before_<Y>` cut (SHUFFLED_BEFORE_YEAR) is
INDEPENDENT of the reshuffle's [LOWER_YEAR, UPPER_YEAR] inclusion range and may be
stricter (e.g. before 2010 while the reshuffle ran up to 2024). The input network is
restricted to the SAME effective subset the shuffled output holds: citing year in
[LOWER_YEAR, UPPER_YEAR] AND citing year < SHUFFLED_BEFORE_YEAR.

The evaluation can be further restricted to nodes that, in the ORIGINAL graph, have
citation count (in-degree) >= CITATION_THRESHOLD and reference count (out-degree)
>= REFERENCE_THRESHOLD (both must hold; 0/0 keeps every node).

For both out-degree (per citing paper) and in-degree (per cited paper) it compares:
  * summary stats (n_nodes, total_edges, mean, percentiles, max)
  * edge-loss fraction
  * per-node degree change (how many nodes changed / lost degree, total lost)
  * degree-distribution divergence: total-variation distance + KS (max CDF gap)
  * the full degree histograms (for plotting)
  * NEIGHBOUR-IDENTITY change: fraction of nodes whose neighbour SET differs from the
    input (a set inequality captures both "points to/from a different node" and
    "count changed").

Read-only: it does NOT modify the pipeline or its outputs.

Outputs (per iteration), in <iter>/constructed_network/degree_eval/ :
  degree_eval_summary_iter_<i>.csv            full metric table (both directions)
  degree_eval_detailed_summary_iter_<i>.csv   explicit headline answers to the three
                                              questions below, one row per (question,
                                              direction):
                                                1) edge loss (how many / what %)
                                                2) # nodes whose in/out-degree COUNT changed
                                                2b) of the node union, what % had degree
                                                    COUNT increased / decreased / unchanged
                                                    (in-degree and out-degree separately)
                                                3) # nodes whose in/out-degree SET changed
                                                   (neighbour identity, not just the count)
  outdegree_distribution_iter_<i>.csv
  indegree_distribution_iter_<i>.csv
"""

import os
import sys
import csv
import pickle
from collections import Counter

from tqdm import tqdm

# utils.py is in the parent (semantic_scholar/) directory; make it importable regardless of CWD
# (matches get_efg_reshuffle_control_venue.py).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PARENT_DIR)
from utils import META_DIRE  # noqa: E402

# must match network_reshuffle_efficient_control_venue.py's filter (lower_year, upper_year)
UPPER_YEAR = 2010
LOWER_YEAR = -1

# the venue-controlled reshuffle's INPUT network (venue-filtered); same file
# network_reshuffle_efficient_control_venue.py reshuffles, so it is the right baseline.
INPUT_CITING2CITED = os.path.join(META_DIRE,
                                  'citing2cited_cid_list_remove_circle_valid_venue.pkl')
PID2YEAR_PATH = os.path.join(META_DIRE, 'cid2pub_year.pkl')

# construct_reshuffled_network_control_venue.py bakes its `upper_year` into the
# shuffled output filenames as 'shuffled_*_before_{upper_year}.pkl', keeping only edges
# whose citing (source) year is STRICTLY BEFORE that year. Set this to match that
# script's upper_year. NOTE: in the control-venue pipeline the construct cut is an
# independent (often stricter) bound -- it is NOT derived as UPPER_YEAR + 1.
SHUFFLED_BEFORE_YEAR = 2010
SHUFFLED_CITING2CITED_NAME = \
    'shuffled_citing_pid2cited_pids_list_before_%d.pkl' % SHUFFLED_BEFORE_YEAR
SHUFFLED_CITED2CITING_NAME = \
    'shuffled_cited_pid2citing_pids_list_before_%d.pkl' % SHUFFLED_BEFORE_YEAR

TARGET_ITERATIONS = [0]

# Restrict the evaluation to nodes that, in the ORIGINAL (input) graph, have
#   citation count (in-degree, papers citing this node)   >= CITATION_THRESHOLD
#   reference count (out-degree, papers this node cites)   >= REFERENCE_THRESHOLD
# Both conditions must hold (AND). 0 / 0 keeps every original-graph node.
REFERENCE_THRESHOLD = 1
CITATION_THRESHOLD = 5

_EMPTY_SET = frozenset()       # shared empty-neighbour sentinel (never mutated)


# --------------------------------------------------------------------------- #
#  degree helpers
# --------------------------------------------------------------------------- #
def citing_year_in_range(pid2year, citing):
    """True iff `citing` paper's publication year is within the exact edge subset
    held by the shuffled output: within [LOWER_YEAR, UPPER_YEAR] (the reshuffle's
    inclusion range) AND strictly before SHUFFLED_BEFORE_YEAR (construct's cut)."""
    sy = pid2year.get(citing)
    if sy is None:
        return False
    if UPPER_YEAR >= 0 and sy > UPPER_YEAR:
        return False
    if LOWER_YEAR >= 0 and sy < LOWER_YEAR:
        return False
    if sy >= SHUFFLED_BEFORE_YEAR:        # construct kept only source year < this
        return False
    return True


def build_input_network(pid2year):
    """Neighbour SETS (not just counts), restricted to the exact edge subset that
    entered + survived the shuffle (citing year in range, distinct targets).

    Returns:
      out_adj : citing cid -> set(cited cids)   (distinct targets)
      in_adj  : cited  cid -> set(citing cids)  (distinct sources)
      n_edges : total distinct edges in the restricted subset
    Degrees are simply len(neighbour set), derived downstream.
    """
    print("Loading", INPUT_CITING2CITED, "...")
    with open(INPUT_CITING2CITED, 'rb') as f:
        citing2cited = pickle.load(f)

    out_adj = {}           # citing cid -> set(cited cids)
    in_adj = {}            # cited  cid -> set(citing cids)
    n_edges = 0
    for citing, cited_list in tqdm(citing2cited.items(), desc='input network'):
        if not citing_year_in_range(pid2year, citing):
            continue
        targets = set(cited_list)
        if not targets:
            continue
        out_adj[citing] = targets
        for t in targets:
            in_adj.setdefault(t, set()).add(citing)
        n_edges += len(targets)

    del citing2cited
    print("  input restricted edges:", n_edges,
          " citing nodes:", len(out_adj), " cited nodes:", len(in_adj))
    return out_adj, in_adj, n_edges


def load_shuffled_network(constructed_dir, pid2year):
    """Neighbour SETS per citing and per cited from the shuffled output, restricted
    to the SAME edge subset as the input: only edges whose CITING paper's year is in
    range are kept.
      * out_adj: keep only citing keys whose year is in range.
      * in_adj : within each cited node, keep only citing sources whose year is in
                 range; drop cited nodes left with no such source.
    """
    c2c_path = os.path.join(constructed_dir, SHUFFLED_CITING2CITED_NAME)
    c2g_path = os.path.join(constructed_dir, SHUFFLED_CITED2CITING_NAME)
    with open(c2c_path, 'rb') as f:
        c2c = pickle.load(f)
    out_adj = {k: set(v) for k, v in c2c.items()
               if citing_year_in_range(pid2year, k)}
    del c2c
    with open(c2g_path, 'rb') as f:
        c2g = pickle.load(f)
    in_adj = {}
    for cited, citings in c2g.items():
        kept = {c for c in citings if citing_year_in_range(pid2year, c)}
        if kept:
            in_adj[cited] = kept
    del c2g
    return out_adj, in_adj, sum(len(v) for v in out_adj.values())


# --------------------------------------------------------------------------- #
#  node-eligibility threshold (computed on the ORIGINAL / input graph)
# --------------------------------------------------------------------------- #
def compute_eligible_nodes(out_adj, in_adj):
    """Nodes whose ORIGINAL-graph citation/reference counts pass the thresholds.

      citation  count = in-degree  = len(in_adj[node])   (papers citing node)
      reference count = out-degree = len(out_adj[node])  (papers node cites)

    A node is eligible iff citation >= CITATION_THRESHOLD AND
    reference >= REFERENCE_THRESHOLD. With both thresholds 0 every node in the
    union is eligible.
    """
    nodes = set(out_adj) | set(in_adj)
    eligible = set()
    for n in nodes:
        ref = len(out_adj.get(n, _EMPTY_SET))   # references made (out-degree)
        cit = len(in_adj.get(n, _EMPTY_SET))    # citations received (in-degree)
        if cit >= CITATION_THRESHOLD and ref >= REFERENCE_THRESHOLD:
            eligible.add(n)
    return eligible


def filter_adj_by_nodes(adj, nodes):
    """Keep only adjacency entries whose KEY (the node being evaluated) is in
    `nodes`. Neighbour sets (the values) are left untouched."""
    return {k: v for k, v in adj.items() if k in nodes}


# --------------------------------------------------------------------------- #
#  comparison metrics
# --------------------------------------------------------------------------- #
def summary_stats(deg_map):
    vals = sorted(deg_map.values())
    n = len(vals)
    tot = sum(vals)

    def pct(p):
        if n == 0:
            return 0
        return vals[min(n - 1, int(p * n))]

    return {
        'n_nodes': n,
        'total_edges': tot,
        'mean': tot / n if n else 0.0,
        'min': vals[0] if n else 0,
        'p50': pct(0.50),
        'p90': pct(0.90),
        'p99': pct(0.99),
        'max': vals[-1] if n else 0,
    }


def degree_histogram(deg_map):
    """degree value -> number of nodes with that degree."""
    return Counter(deg_map.values())


def tv_distance(h1, h2):
    """Total-variation distance between two degree distributions (0..1)."""
    n1, n2 = sum(h1.values()), sum(h2.values())
    if n1 == 0 or n2 == 0:
        return 0.0
    keys = set(h1) | set(h2)
    return 0.5 * sum(abs(h1.get(k, 0) / n1 - h2.get(k, 0) / n2) for k in keys)


def ks_statistic(h1, h2):
    """Max absolute gap between the two empirical CDFs over degree (KS)."""
    n1, n2 = sum(h1.values()), sum(h2.values())
    if n1 == 0 or n2 == 0:
        return 0.0
    c1 = c2 = 0
    mx = 0.0
    for k in sorted(set(h1) | set(h2)):
        c1 += h1.get(k, 0)
        c2 += h2.get(k, 0)
        mx = max(mx, abs(c1 / n1 - c2 / n2))
    return mx


def node_level_change(input_deg, shuf_deg):
    """Per-node degree comparison over the union of node ids."""
    nodes = set(input_deg) | set(shuf_deg)
    only_input = changed = lost = gained = 0
    sum_abs = sum_in = sum_sh = 0
    for n in nodes:
        di = input_deg.get(n, 0)
        ds = shuf_deg.get(n, 0)
        sum_in += di
        sum_sh += ds
        sum_abs += abs(di - ds)
        if di and not ds:
            only_input += 1
        if di != ds:
            changed += 1
        if ds < di:
            lost += 1
        elif ds > di:
            gained += 1
    nn = len(nodes)
    return {
        'n_nodes_union': nn,
        'n_nodes_input': len(input_deg),
        'n_nodes_shuffled': len(shuf_deg),
        'nodes_dropped_entirely': only_input,
        'nodes_changed_degree': changed,
        'nodes_lost_degree': lost,
        'nodes_gained_degree': gained,
        'mean_abs_degree_diff': sum_abs / nn if nn else 0.0,
        'total_degree_input': sum_in,
        'total_degree_shuffled': sum_sh,
        'frac_nodes_changed': changed / nn if nn else 0.0,
        'frac_degree_lost': (sum_in - sum_sh) / sum_in if sum_in else 0.0,
    }


def neighbor_set_change(input_adj, shuf_adj):
    """Per-node comparison of the actual NEIGHBOUR SETS (not just the count).

    A node "changed" when its shuffled neighbour set differs from its input
    neighbour set -- this single condition `in_set != sh_set` covers both
    "at least one edge now points to / comes from a DIFFERENT node" and
    "the degree count changed".

    Two denominators are reported:
      * unconditional: over the union of all node ids seen in either network.
      * conditional   : over only the nodes that had >= 1 neighbour in the
                        INPUT (i.e. degree>=1 before reshuffle).
    `nodes_with_new_neighbor` additionally counts nodes that gained at least one
    genuinely new neighbour (a node not present in the input set).
    """
    union_nodes = set(input_adj) | set(shuf_adj)
    n_union = len(union_nodes)

    changed_union = 0          # in_set != sh_set, over union
    changed_cond = 0           # in_set != sh_set, over input-degree>=1 nodes
    new_neighbor_union = 0     # gained >=1 neighbour absent from input, union
    new_neighbor_cond = 0      # ... over input-degree>=1 nodes
    n_input = 0                # nodes with >=1 neighbour in the input

    for node in union_nodes:
        in_set = input_adj.get(node)
        sh_set = shuf_adj.get(node, _EMPTY_SET)
        has_input = bool(in_set)
        if in_set is None:
            in_set = _EMPTY_SET
        differs = in_set != sh_set
        gained_new = len(sh_set - in_set) > 0
        if differs:
            changed_union += 1
        if gained_new:
            new_neighbor_union += 1
        if has_input:
            n_input += 1
            if differs:
                changed_cond += 1
            if gained_new:
                new_neighbor_cond += 1

    return {
        # unconditional (denominator = union of all nodes)
        'n_nodes_union': n_union,
        'nodes_neighbor_set_changed': changed_union,
        'frac_nodes_neighbor_set_changed': changed_union / n_union if n_union else 0.0,
        'nodes_with_new_neighbor': new_neighbor_union,
        'frac_nodes_with_new_neighbor': new_neighbor_union / n_union if n_union else 0.0,
        # conditional (denominator = nodes with >=1 neighbour in the input)
        'n_nodes_input_degree_ge1': n_input,
        'cond_nodes_neighbor_set_changed': changed_cond,
        'cond_frac_nodes_neighbor_set_changed': changed_cond / n_input if n_input else 0.0,
        'cond_nodes_with_new_neighbor': new_neighbor_cond,
        'cond_frac_nodes_with_new_neighbor': new_neighbor_cond / n_input if n_input else 0.0,
    }


def write_distribution_csv(path, h_in, h_sh):
    n_in, n_sh = sum(h_in.values()), sum(h_sh.values())
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['degree', 'input_node_count', 'shuffled_node_count',
                    'input_fraction', 'shuffled_fraction'])
        for d in sorted(set(h_in) | set(h_sh)):
            ci, cs = h_in.get(d, 0), h_sh.get(d, 0)
            w.writerow([d, ci, cs,
                        '%.8f' % (ci / n_in if n_in else 0.0),
                        '%.8f' % (cs / n_sh if n_sh else 0.0)])


def evaluate_direction(name, input_adj, shuf_adj, out_csv):
    # degree maps derived from the neighbour sets
    input_deg = {k: len(v) for k, v in input_adj.items()}
    shuf_deg = {k: len(v) for k, v in shuf_adj.items()}

    h_in = degree_histogram(input_deg)
    h_sh = degree_histogram(shuf_deg)
    write_distribution_csv(out_csv, h_in, h_sh)

    s_in = summary_stats(input_deg)
    s_sh = summary_stats(shuf_deg)
    chg = node_level_change(input_deg, shuf_deg)
    nbr = neighbor_set_change(input_adj, shuf_adj)
    tv = tv_distance(h_in, h_sh)
    ks = ks_statistic(h_in, h_sh)

    rows = []
    for k in ('n_nodes', 'total_edges', 'mean', 'min', 'p50', 'p90', 'p99', 'max'):
        rows.append([name, k, s_in[k], s_sh[k]])
    rows.append([name, 'tv_distance', '', '%.6g' % tv])
    rows.append([name, 'ks_statistic', '', '%.6g' % ks])
    for k, v in chg.items():
        rows.append([name, k, '', v])
    for k, v in nbr.items():
        rows.append([name, k, '', v])

    # concise console headline
    print("\n  [%s]" % name)
    print("    input  : nodes=%d edges=%d mean=%.3f p99=%d max=%d"
          % (s_in['n_nodes'], s_in['total_edges'], s_in['mean'],
             s_in['p99'], s_in['max']))
    print("    shuffle: nodes=%d edges=%d mean=%.3f p99=%d max=%d"
          % (s_sh['n_nodes'], s_sh['total_edges'], s_sh['mean'],
             s_sh['p99'], s_sh['max']))
    print("    edge-loss=%.4f%%  nodes_changed=%.4f%%  nodes_dropped=%d  "
          "mean|Δdeg|=%.4f  TV=%.4g  KS=%.4g"
          % (100 * chg['frac_degree_lost'],
             100 * chg['frac_nodes_changed'],
             chg['nodes_dropped_entirely'],
             chg['mean_abs_degree_diff'], tv, ks))
    print("    neighbour-set changed (>=1 different/changed edge):")
    print("        unconditional: %.4f%% (%d / %d nodes)"
          % (100 * nbr['frac_nodes_neighbor_set_changed'],
             nbr['nodes_neighbor_set_changed'],
             nbr['n_nodes_union']))
    print("        conditional on input degree>=1: %.4f%% (%d / %d nodes)"
          % (100 * nbr['cond_frac_nodes_neighbor_set_changed'],
             nbr['cond_nodes_neighbor_set_changed'],
             nbr['n_nodes_input_degree_ge1']))

    # headline numbers for the consolidated detailed summary (see main):
    #   edge loss, # nodes whose degree COUNT changed, # nodes whose neighbour SET changed
    headline = {
        'direction': name,
        'input_edges': s_in['total_edges'],
        'shuffled_edges': s_sh['total_edges'],
        'edges_lost': s_in['total_edges'] - s_sh['total_edges'],
        'edge_loss_frac': chg['frac_degree_lost'],
        'n_nodes_union': chg['n_nodes_union'],
        'n_nodes_input_degree_ge1': nbr['n_nodes_input_degree_ge1'],
        # degree COUNT change (the in/out degree NUMBER differs)
        'nodes_changed_degree_count': chg['nodes_changed_degree'],
        'frac_nodes_changed_degree_count': chg['frac_nodes_changed'],
        'nodes_dropped_entirely': chg['nodes_dropped_entirely'],
        # direction of the degree-COUNT change (over the node union):
        #   increased = shuffled degree > input degree
        #   decreased = shuffled degree < input degree
        #   unchanged = same degree count (may still have a changed SET, see below)
        'nodes_degree_increased': chg['nodes_gained_degree'],
        'nodes_degree_decreased': chg['nodes_lost_degree'],
        'nodes_degree_unchanged': chg['n_nodes_union'] - chg['nodes_changed_degree'],
        'frac_nodes_degree_increased':
            chg['nodes_gained_degree'] / chg['n_nodes_union'] if chg['n_nodes_union'] else 0.0,
        'frac_nodes_degree_decreased':
            chg['nodes_lost_degree'] / chg['n_nodes_union'] if chg['n_nodes_union'] else 0.0,
        'frac_nodes_degree_unchanged':
            (chg['n_nodes_union'] - chg['nodes_changed_degree']) / chg['n_nodes_union']
            if chg['n_nodes_union'] else 0.0,
        # degree SET change (the in/out neighbour identity differs -- count and/or who)
        'nodes_changed_degree_set': nbr['nodes_neighbor_set_changed'],
        'frac_nodes_changed_degree_set': nbr['frac_nodes_neighbor_set_changed'],
        'cond_nodes_changed_degree_set': nbr['cond_nodes_neighbor_set_changed'],
        'cond_frac_nodes_changed_degree_set': nbr['cond_frac_nodes_neighbor_set_changed'],
    }
    return rows, headline


# --------------------------------------------------------------------------- #
def main():
    print("Loading", PID2YEAR_PATH, "...")
    with open(PID2YEAR_PATH, 'rb') as f:
        pid2year = pickle.load(f)

    # The observed (input) network is iteration-independent: build it lazily the
    # first time it is needed, AFTER the random (shuffled) network is loaded.
    input_out_adj = input_in_adj = None
    eligible = None
    n_all_nodes = input_edges = 0

    for it in TARGET_ITERATIONS:
        constructed_dir = os.path.join(META_DIRE, 'random_shuffle_citation_control_venue',
                                       str(it), 'constructed_network')
        c2c = os.path.join(constructed_dir, SHUFFLED_CITING2CITED_NAME)
        c2g = os.path.join(constructed_dir, SHUFFLED_CITED2CITING_NAME)
        if not (os.path.exists(c2c) and os.path.exists(c2g)):
            print("\n=== iteration %s: shuffled output not found, skip ===" % it)
            continue

        print("\n" + "=" * 70)
        print("ITERATION %s" % it)
        print("=" * 70)

        # 1) load the RANDOM (shuffled) network first
        shuf_out_adj, shuf_in_adj, _ = load_shuffled_network(constructed_dir,
                                                             pid2year)

        # 2) then load the OBSERVED (input) network -- once, reused across iters
        if input_out_adj is None:
            input_out_adj, input_in_adj, _ = build_input_network(pid2year)
            # original-graph nodes passing the citation/reference thresholds
            eligible = compute_eligible_nodes(input_out_adj, input_in_adj)
            n_all_nodes = len(set(input_out_adj) | set(input_in_adj))
            print("  thresholds: citation>=%d (in-degree)  reference>=%d (out-degree)"
                  % (CITATION_THRESHOLD, REFERENCE_THRESHOLD))
            print("  eligible nodes: %d / %d (%.4f%%)"
                  % (len(eligible), n_all_nodes,
                     100 * len(eligible) / n_all_nodes if n_all_nodes else 0.0))
            input_out_adj = filter_adj_by_nodes(input_out_adj, eligible)
            input_in_adj = filter_adj_by_nodes(input_in_adj, eligible)
            # edges restricted to the evaluated (eligible) citing nodes
            input_edges = sum(len(v) for v in input_out_adj.values())

        # evaluate the SAME eligible node set in the shuffled network
        shuf_out_adj = filter_adj_by_nodes(shuf_out_adj, eligible)
        shuf_in_adj = filter_adj_by_nodes(shuf_in_adj, eligible)
        shuf_edges = sum(len(v) for v in shuf_out_adj.values())
        print("  input edges=%d  shuffled edges=%d  overall edge-loss=%.4f%%"
              % (input_edges, shuf_edges,
                 100 * (input_edges - shuf_edges) / input_edges
                 if input_edges else 0.0))

        eval_dir = os.path.join(constructed_dir, 'degree_eval')
        os.makedirs(eval_dir, exist_ok=True)

        out_rows, out_head = evaluate_direction(
            'out_degree (citing)', input_out_adj, shuf_out_adj,
            os.path.join(eval_dir, 'outdegree_distribution_iter_%s.csv' % it))
        in_rows, in_head = evaluate_direction(
            'in_degree (cited)', input_in_adj, shuf_in_adj,
            os.path.join(eval_dir, 'indegree_distribution_iter_%s.csv' % it))

        summary_path = os.path.join(eval_dir,
                                    'degree_eval_summary_iter_%s.csv' % it)
        with open(summary_path, 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow(['direction', 'metric', 'input', 'shuffled'])
            w.writerow(['overall', 'citation_threshold', CITATION_THRESHOLD, ''])
            w.writerow(['overall', 'reference_threshold', REFERENCE_THRESHOLD, ''])
            w.writerow(['overall', 'eligible_nodes', len(eligible), ''])
            w.writerow(['overall', 'all_nodes', n_all_nodes, ''])
            w.writerow(['overall', 'input_edges', input_edges, ''])
            w.writerow(['overall', 'shuffled_edges', '', shuf_edges])
            w.writerows(out_rows)
            w.writerows(in_rows)
        print("  wrote", summary_path)

        # ------------------------------------------------------------------ #
        #  DETAILED, EXPLICIT summary of the three headline questions:
        #    1) how much EDGE LOSS  (out-direction and in-direction)
        #    2) how many nodes have a CHANGED in/out-degree COUNT (the number)
        #    3) how many nodes have a CHANGED in/out-degree SET (which neighbours)
        #  Note: "set changed" >= "count changed", because a node can keep the same
        #  degree count while pointing to/from a different set of neighbours.
        # ------------------------------------------------------------------ #
        def _pct(part, whole):
            return 100.0 * part / whole if whole else 0.0

        print("\n  " + "-" * 66)
        print("  DETAILED SUMMARY (iteration %s)" % it)
        print("  " + "-" * 66)
        print("  1) EDGE LOSS")
        for h in (out_head, in_head):
            print("       %-20s input=%d  shuffled=%d  lost=%d  (%.4f%%)"
                  % (h['direction'], h['input_edges'], h['shuffled_edges'],
                     h['edges_lost'], 100.0 * h['edge_loss_frac']))
        print("  2) NODES WITH CHANGED DEGREE COUNT (in/out-degree NUMBER differs)")
        for h in (out_head, in_head):
            print("       %-20s %d / %d nodes  (%.4f%%)   [dropped entirely: %d]"
                  % (h['direction'], h['nodes_changed_degree_count'],
                     h['n_nodes_union'],
                     100.0 * h['frac_nodes_changed_degree_count'],
                     h['nodes_dropped_entirely']))
        print("  2b) DEGREE-COUNT DIRECTION (increased / decreased / unchanged, "
              "%% of node union)")
        for h in (out_head, in_head):
            print("       %-20s increased %.4f%% (%d) | decreased %.4f%% (%d) | "
                  "unchanged %.4f%% (%d)"
                  % (h['direction'],
                     100.0 * h['frac_nodes_degree_increased'], h['nodes_degree_increased'],
                     100.0 * h['frac_nodes_degree_decreased'], h['nodes_degree_decreased'],
                     100.0 * h['frac_nodes_degree_unchanged'], h['nodes_degree_unchanged']))
        print("  3) NODES WITH CHANGED DEGREE SET (in/out neighbour identity differs)")
        for h in (out_head, in_head):
            print("       %-20s %d / %d nodes  (%.4f%%)   "
                  "[cond. on input deg>=1: %d / %d = %.4f%%]"
                  % (h['direction'], h['nodes_changed_degree_set'],
                     h['n_nodes_union'],
                     100.0 * h['frac_nodes_changed_degree_set'],
                     h['cond_nodes_changed_degree_set'],
                     h['n_nodes_input_degree_ge1'],
                     100.0 * h['cond_frac_nodes_changed_degree_set']))

        # write the same three blocks to a dedicated, easy-to-read CSV
        detailed_path = os.path.join(
            eval_dir, 'degree_eval_detailed_summary_iter_%s.csv' % it)
        with open(detailed_path, 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow(['question', 'direction', 'count', 'total', 'percent', 'note'])
            # 1) edge loss
            for h in (out_head, in_head):
                w.writerow(['edge_loss', h['direction'], h['edges_lost'],
                            h['input_edges'],
                            '%.6f' % (100.0 * h['edge_loss_frac']),
                            'shuffled_edges=%d' % h['shuffled_edges']])
            # 2) changed degree COUNT
            for h in (out_head, in_head):
                w.writerow(['changed_degree_count', h['direction'],
                            h['nodes_changed_degree_count'], h['n_nodes_union'],
                            '%.6f' % (100.0 * h['frac_nodes_changed_degree_count']),
                            'nodes_dropped_entirely=%d' % h['nodes_dropped_entirely']])
            # 2b) direction of the degree-COUNT change: increased / decreased / unchanged
            for h in (out_head, in_head):
                w.writerow(['degree_count_increased', h['direction'],
                            h['nodes_degree_increased'], h['n_nodes_union'],
                            '%.6f' % (100.0 * h['frac_nodes_degree_increased']), ''])
            for h in (out_head, in_head):
                w.writerow(['degree_count_decreased', h['direction'],
                            h['nodes_degree_decreased'], h['n_nodes_union'],
                            '%.6f' % (100.0 * h['frac_nodes_degree_decreased']), ''])
            for h in (out_head, in_head):
                w.writerow(['degree_count_unchanged', h['direction'],
                            h['nodes_degree_unchanged'], h['n_nodes_union'],
                            '%.6f' % (100.0 * h['frac_nodes_degree_unchanged']),
                            'unchanged COUNT may still have a changed SET'])
            # 3) changed degree SET (neighbour identity)
            for h in (out_head, in_head):
                w.writerow(['changed_degree_set', h['direction'],
                            h['nodes_changed_degree_set'], h['n_nodes_union'],
                            '%.6f' % (100.0 * h['frac_nodes_changed_degree_set']),
                            'cond_input_deg>=1: %d/%d=%.6f%%'
                            % (h['cond_nodes_changed_degree_set'],
                               h['n_nodes_input_degree_ge1'],
                               100.0 * h['cond_frac_nodes_changed_degree_set'])])
        print("  wrote", detailed_path)

    print("\nfinished")


if __name__ == "__main__":
    main()
