#!/usr/bin/env python3
"""
Convenience entry point for Stage 1: Preprocessing Layer
"""
import os
import sys

# Add code/business_entity_resolution to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.join(script_dir, "code", "business_entity_resolution")
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from src.preprocess import (
    detect_country,
    normalize_name,
    normalize_address,
    preprocess_record,
    preprocess_dataframe,
    preprocess_tsv_file
)

if __name__ == '__main__':
    import subprocess
    target = os.path.join(pkg_dir, "src", "preprocess.py")
    subprocess.run([sys.executable, target] + sys.argv[1:])
