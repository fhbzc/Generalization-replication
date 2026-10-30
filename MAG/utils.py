
import os

# anchor relative paths to this file's directory so scripts run regardless of CWD
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# may be written as either a relative path (anchored to this file's directory) or an absolute path

# the directory where the raw Microsoft Academic Graph (MAG) snapshot is saved.
# This is the `mag/` sub-folder of the snapshot and holds Papers.txt, PaperReferences.txt, ...
MAG_DATA_DIRE = '/path/to/mag_snapshot/mag'  # the directory where the MAG dataset is saved

# some tables (PaperFieldsOfStudy.txt, FieldsOfStudy.txt, ...) live in the sibling
# `advanced/` folder of the same snapshot rather than in `mag/`
MAG_ADVANCED_DIRE = os.path.join(os.path.dirname(MAG_DATA_DIRE), 'advanced')  # the directory where the MAG advanced tables are saved

META_DIRE = os.path.join(_BASE_DIR, 'data')  # the directory where you want to save the processed files
