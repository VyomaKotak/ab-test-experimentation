"""
Loaders for the prepared Criteo data.

Three levels of access, and the notebook should reach for the smallest one
that answers the question:

  load_counts()  exact aggregates over all 14 million rows, instant
  load_sample()  a ten percent random sample, for plots and exploration
  load_full()    the whole parquet, only when a per-row operation is genuinely needed
"""

import json
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("../data/processed")


def load_counts(path=None):
    """
    The aggregates the statistical tests actually consume.
    A two proportion z test needs four numbers, not fourteen million rows.
    """
    path = Path(path) if path else PROCESSED_DIR / "criteo_counts.json"
    with open(path) as f:
        return json.load(f)


def load_sample(path=None):
    """Ten percent stratified sample. Use for distributions, plots, segment work."""
    path = Path(path) if path else PROCESSED_DIR / "criteo_sample.parquet"
    return pd.read_parquet(path)


def load_full(columns=None, path=None):
    """
    Full parquet. Pass a column list, because reading twelve float columns you
    do not need is the fastest way to run out of memory.
    """
    path = Path(path) if path else PROCESSED_DIR / "criteo_uplift.parquet"
    return pd.read_parquet(path, columns=columns)