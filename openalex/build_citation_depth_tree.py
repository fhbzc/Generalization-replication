"""
For a given pid_list, build a "citation-depth tree" per focal paper F.

For each focal pid F:
  1. Collect its citations within `year_offset` years:
        C = { c that cites F : pid2year[c] <= pid2year[F] + year_offset }
     (the same year-window rule as get_disruption.py's citing set; F itself excluded.)
  2. Restrict to citation edges AMONG C (a -> b iff a cites b, both in C). F is the root;
     every paper in C cites F directly, so that direct edge is deliberately ignored here.
  3. Each citing paper's depth = the LONGEST chain of intra-C citations beneath it:
        depth(c) = 0                                if c cites no other paper in C
                 = 1 + max(depth(x) for x in C that c cites)   otherwise
     i.e. depth = number of intermediate C-papers on the longest path from c down to F
     once its own direct edge to F is removed (so "cannot reach after removal" -> depth_0).
  4. Draw a radial tree with the focal paper as the ROOT AT THE CENTRE (each paper sits one
     depth-ring out from its longest-chain parent, fanning around the circle) and dump data.

Cycles (rare; the citation graph is essentially time-ordered) are broken by ignoring
back-edges during the longest-path DFS, so depth is always well-defined.

The data cache (citation_depth_tree_data.pkl) stores each focal paper's citation/depth
structure (pid_depth + the intra-cohort edges). On a later run, any pid already in the cache
is RE-DRAWN straight from that stored info -- the big input pkls are NOT loaded for it. The
big pkls are loaded only for pids whose info is not yet cached.

The focal titles are supplied DIRECTLY in the `pid2title` dict below (we do NOT load
pid2title.pkl). A title is only used to slug the first 20 chars into the svg filename; it is
not drawn on the figure, which carries no title, legend or text -- just the radial tree.

Input  : META_DIRE/pid2pub_year.pkl
         META_DIRE/citing_pid2cited_pids_list.pkl   (c -> [refs])
         META_DIRE/cited_pid2citing_pids_list.pkl   (F -> [citers])
         pid2title dict (inline, below)             (focal pid -> title)
Output : ./figures/citation_depth_tree/citation_depth_tree_data.pkl   (focal -> depth structure)
         ./figures/citation_depth_tree/citation_depth_tree_data.json   (same, json-friendly)
         ./figures/citation_depth_tree/{color_scheme}/tree_{focal}_{title20}.svg
                                                        (one tree per focal pid, per colour scheme)

The branch lines are drawn either black or mid-gray ('0.55'), picked by `color_scheme`
below; the node dots stay on the viridis depth colormap and the focal marker stays red in
both schemes. Each scheme keeps its svgs in its own subdirectory while sharing the one
data cache.
"""

import os
import re
import sys
import json
import shutil
import pickle
from collections import defaultdict

import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from utils import META_DIRE  # noqa: E402

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
meta_dire = META_DIRE
print("meta_dire", meta_dire)

year_offset = 5   # citation window: keep citers published within F_year + year_offset
print("year_offset", year_offset)

# HARD upper bound on the CITING paper's publication year: a paper published in or after
# this year is NEVER counted as a citer, even if it falls inside the F_year + year_offset
# window. The snapshot is incomplete past this point (2025), so those citers would carry
# partial citation/reference data and must be excluded. None = no cap.
citing_max_year_exclusive = 2025   # keep only citers with pub_year < 2025
print("citing_max_year_exclusive", citing_max_year_exclusive)

# >>> PUT YOUR FOCAL PIDS (and their titles) HERE <<<
# pid -> title, supplied DIRECTLY here (we do NOT load pid2title.pkl). The first 20 chars of
# the title are slugged into the saved svg filename; the title is not drawn on the figure.
pid2title = {
    'W1255676896': 'Genomic and Epigenomic Landscapes of Adult De Novo Acute Myeloid Leukemia',
    'W2161817637': 'Cancer statistics, 2013',
    'W2045435533': 'A Programmable Dual-RNA–Guided DNA Endonuclease in Adaptive Bacterial Immunity',
    'W2099540110': 'NIH Image to ImageJ: 25 years of image analysis',
    'W2163605009': 'ImageNet Classification with Deep Convolutional Neural Networks',
    'W2136072582': 'A new data set of educational attainment in the world, 1950–2010',
    'W3123933399': 'Econometric measures of connectedness and systemic risk in the finance and insurance sectors',
    'W2019655958': 'A general and simple method for obtaining R2 from generalized linear mixed‐effects models',
    'W3124617746': 'The Convex Geometry of Linear Inverse Problems',
    'W4294214983': 'Interrater reliability: the kappa statistic',
    'W2102605133': 'Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation',
    'W2586637063': 'In situ click chemistry generation of cyclooxygenase-2 inhibitors',
    'W2615203216': 'Molecular Optimization Enables over 13% Efficiency in Organic Solar Cells',
    'W2788213189': 'Nitrogen fixation and reduction at boron',
    'W2103004421': 'Decoding tumour phenotype by noninvasive imaging using a quantitative radiomics approach',
    'W2626446274': 'Canagliflozin and Cardiovascular and Renal Events in Type 2 Diabetes',
    'W2765081049': 'GW170817: Observation of Gravitational Waves from a Binary Neutron Star Inspiral',
    'W2420913016': 'Multiple Dirac cones at the surface of the topological metal LaBi',
    'W1858542512': 'Observation of a new particle in the search for the Standard Model Higgs boson with the ATLAS detector at the LHC',
    'W1846653373': 'On the nonexistence of elements of Kervaire invariant one',
    'W2119160928': 'Fast Stable Restricted Maximum Likelihood and Marginal Likelihood Estimation of Semiparametric Generalized Linear Models',
    'W2150462872': 'Bounded gaps between primes',
    'W2104165247': 'When Is the Government Spending Multiplier Large?',
    'W2169150420': 'Now I see it, now I don’t: researcher’s position and reflexivity in qualitative research',
    'W2656044294': 'Emotion shapes the diffusion of moralized content in social networks',
    'W1984383581': 'Eviction and the Reproduction of Urban Poverty',
    'W1982855075': 'Pilon: An Integrated Tool for Comprehensive Microbial Variant Detection and Genome Assembly Improvement',
    'W2132632499': 'MEGA5: Molecular Evolutionary Genetics Analysis Using Maximum Likelihood, Evolutionary Distance, and Maximum Parsimony Methods',
    'W1999574084': 'RSEM: accurate transcript quantification from RNA-Seq data with or without a reference genome',
    'W1904106279': 'Beyond Homophily: A Decade of Advances in Understanding Peer Influence Processes',
    'W2593962681': 'The relation between 21st-century skills and digital skills: A systematic literature review',
    'W2003220763': 'Emotion Regulation: Current Status and Future Prospects',
    'W2113929081': 'Toward a General Theory of Strategic Action Fields',
    'W2080458794': 'Transcending the metabolic rift: a theory of crises in the capitalist world-ecology',
    'W2013622618': 'A New Theory for Public Service Management? Toward a (Public) Service-Dominant Approach',
    'W2301490063': 'Ballistic Majorana nanowire devices',
    'W1994870116': 'Advanced Virgo: a second-generation interferometric gravitational wave detector',
    'W2069740276': 'An updated version of wannier90: A tool for obtaining maximally-localised Wannier functions',
    'W1922304770': 'A Stupidity‐Based Theory of Organizations',
    'W2024405748': 'PROVEAN web server: a tool to predict the functional effect of amino acid substitutions and indels',
    'W2028666120': 'Sensory expectations based on product-extrinsic food cues: An interdisciplinary review of the empirical evidence and theoretical accounts',
    'W2612870289': 'A Systematic Literature Review of Students as Partners in Higher Education',
    'W2999044305': 'Improved protein structure prediction using potentials from deep learning', # AlphaFold1
    'W3177828909': 'Highly accurate protein structure prediction with AlphaFold', # AlphaFold2
    'W2187089797': 'Visualizing Data using t-SNE',
    
}
# pids supplied by hand above (kept separate from the Nobel set loaded below). A pid that is
# BOTH manual and Nobel gets its tree drawn TWICE -- once in the manual output dir and once in
# the Nobel dir -- so each collection is self-contained.
manual_pids = set(pid2title)

# ---------------------------------------------------------------------------
# optionally ALSO output every Nobel-prize paper, read (pid + title) straight from
# compute_efgd_quantile_s2.1_extract_nobel.py's CSV. Their trees go in a SEPARATE Nobel
# subdirectory (see save_dire_nobel / save_dire_layout_nobel below).
# ---------------------------------------------------------------------------
output_nobel = True
print("output_nobel", output_nobel)
# must match compute_efgd_quantile_s2.1_extract_nobel.py's citation_threshold (the qct in the filename)
nobel_extract_citation_threshold = 100
nobel_csv_path = os.path.join(
    meta_dire, 'efgd_quantile_top_extract',
    f'efgd_nobel_qct_{nobel_extract_citation_threshold}_extract.csv')

nobel_pids = set()
if output_nobel:
    assert os.path.exists(nobel_csv_path), (
        f"{nobel_csv_path}\nRun compute_efgd_quantile_s2.1_extract_nobel.py first "
        f"(or set output_nobel = False).")
    df_nobel_in = pd.read_csv(nobel_csv_path)
    for pid, title in zip(df_nobel_in['pid'], df_nobel_in['title']):
        nobel_pids.add(pid)
        # keep the manual title if the pid is also in the manual list; else use the CSV title
        pid2title.setdefault(pid, title if isinstance(title, str) else '')
    print(f"loaded {len(nobel_pids)} Nobel pids from {nobel_csv_path}")

# process the UNION (manual first, then Nobel-only); dict insertion order dedups overlaps
pid_list = list(pid2title)
print("pid_list", len(pid_list), "pids ("
      f"{len(manual_pids)} manual, {len(nobel_pids)} Nobel, "
      f"{len(manual_pids & nobel_pids)} in both)")


def svg_path(focal):
    """Output svg path for a focal pid, with the first 20 chars of its title slugged in."""
    slug = re.sub(r'[^0-9A-Za-z]+', '_', (pid2title.get(focal) or '')[:20]).strip('_')
    name = f'{slug}_{focal}_tree.svg' if slug else f'{focal}_tree.svg'
    return os.path.join(save_dire, name)
# if a focal paper has more citing papers than this, skip drawing its (unreadable) tree
# but still keep its data. None = always draw.
max_plot_nodes = None
print("max_plot_nodes",max_plot_nodes)

# if a focal paper's svg already exists on disk, skip re-drawing it (its data is still
# kept/saved). Set redraw = True to force every tree to be re-drawn (overwriting the svg).
redraw = True
print("redraw", redraw)

# LAYOUT TOGGLE.  False -> the original RADIAL tree (focal at the centre). True -> a top-down
# HIERARCHICAL tree (focal/root at the TOP, each depth on its own horizontal level descending).
# In the top-down layout every output figure has a DIFFERENT size: its HEIGHT is proportional to
# the tree's depth (a fixed length per depth interval) and its WIDTH is proportional to the
# widest level (the level holding the most papers). Both layouts share the SAME data cache
# (citation_depth_tree_data.pkl) but the top-down version writes its svgs under a SEPARATE
# subdirectory with a DIFFERENT filename prefix, so the two never collide.
tree_layout = True
print("tree_layout", tree_layout)

# EDGE COLOUR TOGGLE. The ONLY thing this switches is the colour of the branch lines:
#   'black' -> solid black
#   'gray'  -> mid-gray '0.55' (matplotlib reads the string float as a gray level,
#              '0'=black .. '1'=white; it renders as #8c8c8c)
# The 'gray' scheme reproduces the earlier figures exactly (e.g.
# A_new_method_for_seq_W1992595091_tree_layout.svg: #8c8c8c branches, viridis dots, red focal).
# Everything else is identical in both schemes: the citing-paper dots stay on the viridis
# colormap keyed by depth, and the focal paper stays the big red triangle.
# Each scheme writes its svgs into its OWN subdirectory (…/citation_depth_tree/black/,
# …/citation_depth_tree/gray/, and likewise for the layout / Nobel dirs), so the two
# versions never overwrite each other and can be compared side by side. The data cache
# sits ABOVE those subdirectories and is shared: switching scheme only redraws, it never
# recomputes and never reloads the big input pkls.
color_scheme = 'gray'          # 'black' or 'gray'
EDGE_COLORS = {'black': 'black', 'gray': '0.55'}
assert color_scheme in EDGE_COLORS, \
    f"color_scheme must be one of {sorted(EDGE_COLORS)}, got {color_scheme!r}"
edge_color = EDGE_COLORS[color_scheme]      # branch lines only
node_cmap_name = 'viridis'                  # node dots: colormap keyed by depth, both schemes
focal_color = 'red'                         # focal root marker, both schemes
print("color_scheme", color_scheme)
print("edge_color", edge_color)
print("node_cmap_name", node_cmap_name)
print("focal_color", focal_color)

# the shared data cache (citation_depth_tree_data.pkl/.json) lives here, ABOVE the per-scheme
# svg subdirectories, so both colour schemes read and write the same cache
save_dire_root = os.path.join(BASE_DIR, 'figures', 'citation_depth_tree')
os.makedirs(save_dire_root, exist_ok=True)

save_dire = os.path.join(save_dire_root, color_scheme)
os.makedirs(save_dire, exist_ok=True)

# top-down (hierarchical) trees go in their own subdirectory with their own filename prefix
save_dire_layout = os.path.join(BASE_DIR, 'figures', 'citation_depth_tree_layout',
                                color_scheme)
os.makedirs(save_dire_layout, exist_ok=True)

# Nobel papers get their OWN parallel subdirectories (one per layout), so the Nobel collection
# is self-contained. A paper that is both manual and Nobel is written to BOTH its manual dir
# and the matching Nobel dir (two identical svgs).
save_dire_nobel = os.path.join(BASE_DIR, 'figures', 'citation_depth_tree_nobel',
                               color_scheme)
save_dire_layout_nobel = os.path.join(BASE_DIR, 'figures', 'citation_depth_tree_layout_nobel',
                                      color_scheme)
if output_nobel:
    os.makedirs(save_dire_nobel, exist_ok=True)
    os.makedirs(save_dire_layout_nobel, exist_ok=True)

# which directories THIS run writes into (they carry the color_scheme subdir)
print("save_dire", save_dire)
print("save_dire_layout", save_dire_layout)
if output_nobel:
    print("save_dire_nobel", save_dire_nobel)
    print("save_dire_layout_nobel", save_dire_layout_nobel)


def svg_path_layout(focal):
    """Output svg path for the top-down layout of a focal pid (separate dir + filename prefix)."""
    slug = re.sub(r'[^0-9A-Za-z]+', '_', (pid2title.get(focal) or '')[:20]).strip('_')
    name = f'{slug}_{focal}_tree_layout.svg' if slug else f'{focal}_tree_layout.svg'
    return os.path.join(save_dire_layout, name)


def svg_path_nobel(focal):
    """Radial-layout svg path for a focal pid inside the Nobel subdirectory."""
    slug = re.sub(r'[^0-9A-Za-z]+', '_', (pid2title.get(focal) or '')[:20]).strip('_')
    name = f'{slug}_{focal}_tree.svg' if slug else f'{focal}_tree.svg'
    return os.path.join(save_dire_nobel, name)


def svg_path_layout_nobel(focal):
    """Top-down-layout svg path for a focal pid inside the Nobel subdirectory."""
    slug = re.sub(r'[^0-9A-Za-z]+', '_', (pid2title.get(focal) or '')[:20]).strip('_')
    name = f'{slug}_{focal}_tree_layout.svg' if slug else f'{focal}_tree_layout.svg'
    return os.path.join(save_dire_layout_nobel, name)


# which cache field / svg-path / plot-routine to use, picked by the `tree_layout` toggle above
svg_key = 'svg_layout' if tree_layout else 'svg'
# parallel cache field for the Nobel copy of the svg path
nobel_svg_key = svg_key + '_nobel'


def out_svg_path(focal):
    return svg_path_layout(focal) if tree_layout else svg_path(focal)


def nobel_out_svg_path(focal):
    return svg_path_layout_nobel(focal) if tree_layout else svg_path_nobel(focal)


def target_svg_paths(focal):
    """The svg paths this focal paper must be written to: its manual-dir path if it is a manual
    pid, AND its Nobel-dir path if Nobel output is on and it is a Nobel pid. A pid in both lists
    yields two paths (two identical figures)."""
    paths = []
    if focal in manual_pids:
        paths.append(('manual', out_svg_path(focal)))
    if output_nobel and focal in nobel_pids:
        paths.append(('nobel', nobel_out_svg_path(focal)))
    return paths


def draw_tree_all(focal, C, children, depth):
    """Draw the tree ONCE and copy the identical svg into every target dir. Returns
    {svg_key: manual_path_or_None, nobel_svg_key: nobel_path_or_None} for the cache."""
    targets = target_svg_paths(focal)
    out = {svg_key: None, nobel_svg_key: None}
    if not targets:
        return out
    kinds = {kind: path for kind, path in targets}
    out[svg_key] = kinds.get('manual')
    out[nobel_svg_key] = kinds.get('nobel')
    paths = [path for _, path in targets]
    # honour `redraw`: draw only when forced or some target svg is missing
    if redraw or any(not os.path.exists(p) for p in paths):
        draw_tree(focal, C, children, depth, paths[0])
        for p in paths[1:]:
            shutil.copyfile(paths[0], p)   # guaranteed byte-identical copy
    return out


def load_pkl(name):
    path = os.path.join(meta_dire, name)
    with open(path, 'rb') as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# core
# ---------------------------------------------------------------------------
def get_citing_within_window(focal):
    """The citers of `focal` published within the year window (focal itself excluded)."""
    f_year = pid2year[focal]
    max_year = f_year + year_offset
    C = set()
    for c in cited_pid2citing.get(focal, ()):
        if c == focal:
            continue
        cy = pid2year.get(c)
        if cy is None:
            continue
        if cy > max_year:
            continue
        # hard snapshot ceiling: drop citers published in/after the upper-bound year
        if citing_max_year_exclusive is not None and cy >= citing_max_year_exclusive:
            continue
        C.add(c)
    return C


def build_children(C):
    """children[c] = papers in C that c cites (the intra-cohort out-edges, c & F excluded)."""
    children = {}
    for c in C:
        refs = citing_pid2cited.get(c, ())
        children[c] = [x for x in refs if x in C and x != c]
    return children


def compute_depth(C, children):
    """Longest-chain depth within C, computed with an iterative 3-colour DFS so deep
    chains never overflow the recursion stack and cycles are broken at back-edges.
        depth(c) = 0                         if c has no (forward) child in C
                 = 1 + max(depth(child))     otherwise
    """
    depth = {}
    WHITE, GRAY, BLACK = 0, 1, 2
    state = {c: WHITE for c in C}
    for root in C:
        if state[root] != WHITE:
            continue
        stack = [(root, 0)]
        while stack:
            node, idx = stack[-1]
            if idx == 0:
                state[node] = GRAY
            ch = children[node]
            if idx < len(ch):
                stack[-1] = (node, idx + 1)
                x = ch[idx]
                if state[x] == WHITE:
                    stack.append((x, 0))
                # GRAY child -> back-edge (cycle), skip; BLACK child -> already done
            else:
                best = -1
                for x in ch:
                    dx = depth.get(x)        # only BLACK (finished) children are in depth
                    if dx is not None and dx > best:
                        best = dx
                depth[node] = 0 if best < 0 else best + 1
                state[node] = BLACK
                stack.pop()
    return depth


def build_tree_and_leaforder(focal, C, children, depth):
    """Turn the depth structure into a real tree rooted at `focal` and assign each node a
    horizontal "leaf index".

    Each citing paper gets a single tree-parent one depth nearer the root: a depth-0 paper
    hangs directly under the focal, and a depth-d paper hangs under one of the C-papers it
    cites whose depth is d-1 (the edge that realises its longest chain). Leaves are then laid
    out one per slot (left to right) and every parent is centred over its children, computed
    with an iterative post-order walk so a long chain can never overflow the stack.

    Returns (order, leaf_idx, n_leaves): the per-parent ordered child lists, the float leaf
    index of every node, and the number of leaf slots. Both the radial and the top-down plot
    routines share this layout.
    """
    # --- give every node exactly one parent -> a real tree rooted at `focal` ---
    tree_children = defaultdict(list)
    for c in C:
        d = depth[c]
        if d == 0:
            parent = focal                       # only route to F is the direct citation
        else:
            # among the papers c cites (in C), pick one a depth shallower (closer to F)
            cands = [x for x in children[c] if depth[x] == d - 1]
            parent = min(cands)                  # deterministic choice
        tree_children[parent].append(c)
    order = {n: sorted(kids) for n, kids in tree_children.items()}

    leaf_idx = {}
    counter = [0]
    stack = [(focal, False)]
    while stack:
        node, processed = stack.pop()
        kids = order.get(node, [])
        if processed:
            if kids:
                leaf_idx[node] = sum(leaf_idx[k] for k in kids) / len(kids)
            else:
                leaf_idx[node] = counter[0]
                counter[0] += 1
        else:
            stack.append((node, True))
            for k in reversed(kids):
                stack.append((k, False))
    n_leaves = max(1, counter[0])
    return order, leaf_idx, n_leaves


def plot_tree(focal, C, children, depth, out_path):
    """Draw a RADIAL tree with the focal paper as the ROOT AT THE CENTRE.

    Each citing paper is given a single tree-parent (one depth nearer the root): a depth-0
    paper hangs directly under the focal, and a depth-d paper hangs under one of the C-papers
    it cites whose depth is d-1 (the edge that realises its longest chain). Leaves are spread
    evenly around the circle (by angle) and every parent is centred over its children, so the
    tree fans out radially from the central focal paper.
    """
    import math

    max_d = max(depth.values()) if depth else 0

    # --- give every node exactly one parent + leaf ordering (shared with the top-down plot) ---
    order, leaf_idx, n_leaves = build_tree_and_leaforder(focal, C, children, depth)

    # radius = depth from the centre: focal (root) at r=0, depth-d papers at r=d+1
    # angle = leaf-ordering mapped onto the full circle -> (x, y) in the plane
    def polar(node):
        r = 0.0 if node == focal else (depth[node] + 1)
        theta = 2.0 * math.pi * leaf_idx[node] / n_leaves
        return (r * math.cos(theta), r * math.sin(theta))

    pos = {focal: polar(focal)}
    for c in C:
        pos[c] = polar(c)

    # --- figure: every depth-ring is a CONSTANT physical distance apart, no matter how many
    #     layers a tree has. Radius runs 0..(max_d+1); we size the canvas in proportion to
    #     that radius and pin the axis limits so one depth-unit is always `inch_per_layer`
    #     inches -- deeper trees get a bigger canvas instead of squeezed rings. ---
    R = max_d + 1                        # outermost radius (focal sits at r=0)
    inch_per_layer = 0.7                 # fixed gap between consecutive layers (inches)
    margin = 1.05
    side = min(60, max(7, 2 * R * inch_per_layer * margin))
    fig, ax = plt.subplots(figsize=(side, side))

    # edges: parent -> child, thin lines in the scheme colour (the radial branches)
    for parent, kids in order.items():
        x0, y0 = pos[parent]
        for k in kids:
            x1, y1 = pos[k]
            ax.plot([x0, x1], [y0, y1], color=edge_color, linewidth=1.0, zorder=1)

    # nodes: small dots coloured by depth
    cmap = plt.get_cmap(node_cmap_name)
    for d in range(0, max_d + 1):
        nodes = [c for c in C if depth[c] == d]
        if not nodes:
            continue
        xs = [pos[c][0] for c in nodes]
        ys = [pos[c][1] for c in nodes]
        color = cmap(d / max_d) if max_d > 0 else cmap(0.0)
        ax.scatter(xs, ys, s=10, color=color, zorder=2, linewidths=0)
    # focal root marker at the centre (big triangle)
    fx, fy = pos[focal]
    ax.scatter([fx], [fy], s=900, marker='^', color=focal_color, zorder=3)

    # no title, no legend, no text -- just the radial tree
    # pin the limits so one depth-unit is exactly `inch_per_layer` inches everywhere
    ax.set_xlim(-R * margin, R * margin)
    ax.set_ylim(-R * margin, R * margin)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_path, format='svg', transparent=True)
    plt.close(fig)


def plot_tree_layout(focal, C, children, depth, out_path):
    """Draw a TOP-DOWN hierarchical tree: the focal paper (root) sits at the TOP and each depth
    is its own horizontal level descending below it (depth-0 papers on the first level under the
    focal, depth-d papers d+1 levels down).

    Only the figure HEIGHT varies between focal papers; the WIDTH is the SAME for every tree:
      * HEIGHT is proportional to the tree's depth -- every depth interval occupies a FIXED
        length, so a deeper tree is simply a taller image (not a squashed one);
      * WIDTH is a FIXED constant for all trees -- a level with more papers therefore just packs
        its points more DENSELY into that same width (denser = more papers in the level).
    Node x-positions come from the same leaf-ordering as the radial layout (parents centred over
    their children); y is the level, negated so the root is on top.
    """
    max_d = max(depth.values()) if depth else 0

    # --- give every node exactly one parent + leaf ordering (shared with the radial plot) ---
    order, leaf_idx, n_leaves = build_tree_and_leaforder(focal, C, children, depth)

    # level: focal (root) = 0; a depth-d paper sits on level d+1 (one below its parent)
    max_level = max_d + 1                       # deepest level (root is level 0)

    # position: x from the leaf-ordering, y = -level so the root is at the TOP
    # children x from the leaf-ordering; y = -level so the root is at the TOP
    pos = {}
    for c in C:
        pos[c] = (leaf_idx[c], -(depth[c] + 1))
    # the ROOT is pinned to the exact horizontal CENTRE of the drawing (the midpoint of all the
    # other nodes' x-range), NOT to the mean of its immediate children -- a lopsided subtree would
    # otherwise pull that mean off-centre. With the x-limits centred on this midpoint below, the
    # root then sits dead-centre on the figure regardless of how unbalanced the tree is.
    xs_children = [p[0] for p in pos.values()] or [0.0]
    center_x = (min(xs_children) + max(xs_children)) / 2.0
    pos[focal] = (center_x, 0.0)

    # --- canvas: HEIGHT grows with depth (fixed inches per depth interval); WIDTH is a constant
    #     shared by every tree. Because all leaves are mapped into that same fixed width, a level
    #     holding more papers simply shows up as a DENSER row of points -- the image never widens. ---
    inch_per_layer = 0.9                 # fixed vertical length each depth interval occupies
    pad = 0.6                            # small inch margin around the drawing
    fixed_width = 8.0                    # SAME width for every tree (denser layer == more points)
    height = min(60, max(2.5, max_level * inch_per_layer + 2 * pad))
    width = fixed_width
    fig, ax = plt.subplots(figsize=(width, height))

    # edges: parent -> child, thin lines in the scheme colour (the branches)
    for parent, kids in order.items():
        x0, y0 = pos[parent]
        for k in kids:
            x1, y1 = pos[k]
            ax.plot([x0, x1], [y0, y1], color=edge_color, linewidth=1.0, zorder=1)

    # nodes: small dots coloured by depth
    cmap = plt.get_cmap(node_cmap_name)
    for d in range(0, max_d + 1):
        nodes = [c for c in C if depth[c] == d]
        if not nodes:
            continue
        xs = [pos[c][0] for c in nodes]
        ys = [pos[c][1] for c in nodes]
        color = cmap(d / max_d) if max_d > 0 else cmap(0.0)
        ax.scatter(xs, ys, s=10, color=color, zorder=2, linewidths=0)
    # focal root marker at the top (big triangle)
    fx, fy = pos[focal]
    ax.scatter([fx], [fy], s=900, marker='^', color=focal_color, zorder=3)

    # no title, no legend, no text -- just the tree. x-limits are SYMMETRIC about the root (which
    # sits at the x-range midpoint `center_x`), so the root lands on the figure's horizontal centre.
    xs_all = [p[0] for p in pos.values()]
    xpad = 0.5
    half = max(center_x - min(xs_all), max(xs_all) - center_x) + xpad
    ax.set_xlim(center_x - half, center_x + half)
    ax.set_ylim(-max_level - 0.5, 0.5)
    ax.axis('off')
    # make the axes fill the WHOLE figure (no asymmetric default margins from tight_layout), so
    # that the symmetric x-range above actually puts the root on the figure's horizontal centre.
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    fig.savefig(out_path, format='svg', transparent=True)
    plt.close(fig)


def draw_tree(focal, C, children, depth, out_path):
    """Dispatch to the top-down or the radial plot routine per the `tree_layout` toggle."""
    if tree_layout:
        plot_tree_layout(focal, C, children, depth, out_path)
    else:
        plot_tree(focal, C, children, depth, out_path)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
assert pid_list, "pid_list is empty -- put your focal pids in pid_list at the top of the script"

pkl_path = os.path.join(save_dire_root, 'citation_depth_tree_data.pkl')

# load whatever was computed before (the cache)
result = {}
if os.path.exists(pkl_path):
    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)
    print("loaded data cache:", pkl_path, "(%d pids)" % len(result))


# the fields a cache entry must carry to be re-drawable WITHOUT touching the big input pkls
# (the focal title is NOT cached -- it is supplied inline via pid2title and only affects the
#  output filename, never the drawing itself)
_REDRAW_KEYS = ('pid_depth', 'children')


def has_cached_data(pid):
    """True iff `result` already holds everything needed to redraw this pid's tree from cache
    alone (depth structure + intra-cohort edges). Entries written by an older version of this
    script lack some of these keys -> they are recomputed once.

    An entry is also treated as stale (recomputed) if it was built under a DIFFERENT citing
    year cap: the cap changes which citers make up the cohort, so the cached depth structure
    is no longer valid and must be rebuilt."""
    info = result.get(pid)
    if not (bool(info) and all(k in info for k in _REDRAW_KEYS)):
        return False
    # cap mismatch (incl. old entries with no cap recorded) -> recompute under the new cap
    return info.get('citing_max_year_exclusive') == citing_max_year_exclusive


def draw_from_info(focal, info):
    """Redraw a focal paper's tree purely from its cached info; returns
    {svg_key: manual_path_or_None, nobel_svg_key: nobel_path_or_None}. No big input pkls are
    loaded here. The tree is drawn once and copied into every target dir (manual and/or Nobel)."""
    depth = info['pid_depth']
    children = info['children']
    C = set(depth)
    if not C:
        print(f"{focal}: cached, no citing papers -> no tree")
        return {svg_key: None, nobel_svg_key: None}
    if max_plot_nodes is not None and len(C) > max_plot_nodes:
        print(f"{focal}: cached, {len(C)} > max_plot_nodes={max_plot_nodes} -> tree skipped")
        return {svg_key: None, nobel_svg_key: None}
    svgs = draw_tree_all(focal, C, children, depth)
    print(f"{focal}: redrawn from cache ->", [p for p in svgs.values() if p])
    return svgs


pids_cached     = [p for p in pid_list if has_cached_data(p)]
pids_to_compute = [p for p in pid_list if not has_cached_data(p)]
print("requested pids:", len(pid_list),
      "| from cache:", len(pids_cached),
      "| to compute:", len(pids_to_compute), pids_to_compute)

# track whether `result` actually changed; a pure cache-load run leaves it untouched and we
# then skip rewriting the (identical) pkl/json at the end.
dirty = False

# ---- pids whose citation info is already cached: just redraw, no big-pkl load ----
for focal in pids_cached:
    info = result[focal]
    svgs = draw_from_info(focal, info)
    for k, v in svgs.items():
        if info.get(k) != v or k not in info:
            info[k] = v
            dirty = True

# ---- pids with no cached info: load the big input pkls, compute, store, draw ----
if pids_to_compute:
    pid2year = load_pkl('pid2pub_year.pkl')
    print("pid2year load finished:", len(pid2year))
    citing_pid2cited = load_pkl('citing_pid2cited_pids_list.pkl')   # c -> [papers c cites]
    print("citing_pid2cited load finished:", len(citing_pid2cited))
    cited_pid2citing = load_pkl('cited_pid2citing_pids_list.pkl')   # F -> [papers citing F]
    print("cited_pid2citing load finished:", len(cited_pid2citing))

    for focal in pids_to_compute:
        if focal not in pid2year:
            print("skip (no pub_year):", focal)
            continue
        C = get_citing_within_window(focal)
        children = build_children(C)
        depth = compute_depth(C, children)

        depth_to_pids = defaultdict(list)
        for c, d in depth.items():
            depth_to_pids[d].append(c)
        depth_to_pids = {d: sorted(depth_to_pids[d]) for d in sorted(depth_to_pids)}

        dist = {d: len(p) for d, p in depth_to_pids.items()}
        print(f"{focal}: {len(C)} citing papers, depth distribution {dist}")

        # stash everything needed to redraw this tree from cache on future runs. The svg path is
        # stored under `svg_key` (svg / svg_layout) so the two layouts never overwrite each other.
        entry = result.get(focal, {})
        entry.update({
            'pub_year': pid2year.get(focal),
            'n_citing': len(C),
            'max_depth': (max(depth.values()) if depth else None),
            'depth_to_pids': depth_to_pids,
            'pid_depth': depth,
            'children': children,
            # record the cap this cohort was built under so a later run with a different
            # cap treats the entry as stale (see has_cached_data)
            'citing_max_year_exclusive': citing_max_year_exclusive,
        })
        entry.setdefault(svg_key, None)
        entry.setdefault(nobel_svg_key, None)
        result[focal] = entry
        dirty = True       # a freshly computed pid -> result changed, must be saved

        # tree figure (drawn once, copied into every target dir: manual and/or Nobel)
        if not C:
            print("  (no citing papers -> no tree drawn)")
            result[focal][svg_key] = None
            result[focal][nobel_svg_key] = None
        elif max_plot_nodes is not None and len(C) > max_plot_nodes:
            print(f"  ({len(C)} > max_plot_nodes={max_plot_nodes} -> data saved, tree skipped)")
            result[focal][svg_key] = None
            result[focal][nobel_svg_key] = None
        else:
            svgs = draw_tree_all(focal, C, children, depth)
            result[focal].update(svgs)
            print("  tree saved:", [p for p in svgs.values() if p])
else:
    print("no pids needed computing -> skipped loading pid2year / networks")

# ---------------------------------------------------------------------------
# save data (both pickle and json) -- ONLY if `result` actually changed. A pure cache-load
# run (every pid already cached, no svg path updated) leaves `result` identical to what is
# already on disk, so there is nothing to rewrite.
# ---------------------------------------------------------------------------
if not dirty:
    print("nothing changed (all served from cache) -> data pkl/json not re-saved")
else:
    with open(pkl_path, 'wb') as f:
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    print("saved data pkl:", pkl_path)

    # json: stringify the int depth keys so the structure round-trips cleanly
    json_result = {}
    for focal, info in result.items():
        json_result[focal] = {
            'pub_year': info['pub_year'],
            'n_citing': info['n_citing'],
            'max_depth': info['max_depth'],
            'depth_to_pids': {str(d): pids for d, pids in info['depth_to_pids'].items()},
            'pid_depth': info['pid_depth'],
            'svg': info.get('svg'),
            'svg_layout': info.get('svg_layout'),
            'svg_nobel': info.get('svg_nobel'),
            'svg_layout_nobel': info.get('svg_layout_nobel'),
        }
    json_path = os.path.join(save_dire_root, 'citation_depth_tree_data.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_result, f, indent=2)
    print("saved data json:", json_path)

print("finished")
