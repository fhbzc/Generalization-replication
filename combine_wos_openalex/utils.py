"""
Shared metadata roots for the WoS + OpenAlex joint plots (plot_longitudional_change.py).

OPENALEX_META / WOS_META must point at the metadata roots that the two merge
scripts actually write to, so the joint plot reads exactly what:
  - openalex/2026_version/e_f_g_index/get_efg_2026_merge.py   (-> openalex utils.META_DIRE_2026)
  - wos/get_efg_s2_merge.py                                    (-> wos utils.META_DIRE)
produced.

We DERIVE them from those projects' own utils.py (the single source of truth)
rather than hardcoding, so the paths can never drift out of sync with the merge
scripts. Both sibling utils.py only define path constants (no side effects), and
each is named "utils", so we load them by file path under distinct module names
to avoid the import-name collision.
"""

import os
import importlib.util

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_REBUILD_DIR = os.path.dirname(_BASE_DIR)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_oa_utils = _load_module('openalex_utils', os.path.join(_REBUILD_DIR, 'openalex', 'utils.py'))
_wos_utils = _load_module('wos_utils', os.path.join(_REBUILD_DIR, 'wos', 'utils.py'))

# OpenAlex 2026 metadata root (get_efg_2026_merge.py writes under here)
OPENALEX_META = _oa_utils.META_DIRE_2026
# WoS metadata root (wos/get_efg_s2_merge.py writes under here)
WOS_META = _wos_utils.META_DIRE
