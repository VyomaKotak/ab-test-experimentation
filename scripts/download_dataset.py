"""
Download the Criteo Uplift dataset from Hugging Face and prepare working copies.

The raw file is 311 MB compressed and roughly 3 GB in memory as a pandas frame,
which is more than most laptops want to hold while also running a notebook kernel.
So this script does three things in one pass:

1. Downloads the gzipped CSV from the Hugging Face hub
2. Converts it to Parquet, which is columnar and lets us read only the columns
   an analysis actually needs
3. Writes a stratified sample and a full-file aggregate summary, so every
   downstream notebook can run in seconds without touching the whole file

Run once from the project root:
    python scripts/download_dataset.py

No Hugging Face account or token is needed. The dataset is public.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "criteo/criteo-uplift"
FILENAME = "criteo-research-uplift-v2.1.csv.gz"

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Read the CSV in chunks so we never hold the whole thing at once.
CHUNK_SIZE = 1_000_000
SAMPLE_FRACTION = 0.10
SEED = 42

DTYPES = {
    "f0": "float32", "f1": "float32", "f2": "float32", "f3": "float32",
    "f4": "float32", "f5": "float32", "f6": "float32", "f7": "float32",
    "f8": "float32", "f9": "float32", "f10": "float32", "f11": "float32",
    "treatment": "int8",
    "exposure": "int8",
    "conversion": "int8",
    "visit": "int8",
}


def download():
    """Pull the file from the Hugging Face hub into data/raw."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {FILENAME} from {REPO_ID}")
    print("This is a 311 MB file. It will be cached, so a repeat run is instant.")

    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        repo_type="dataset",
        local_dir=str(RAW_DIR),
    )
    print(f"Downloaded to {local_path}")
    return Path(local_path)


def convert_and_summarise(csv_path):
    """
    Stream through the file once. On the way we:
      - write the whole thing to Parquet
      - accumulate exact counts for the SRM check and the proportion tests
      - collect a random sample for exploratory work

    Everything the statistical analysis needs is a count. We never need the
    full frame in memory, and building the pipeline this way is the difference
    between a notebook that works on a laptop and one that does not.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    # Exact counts, accumulated across chunks.
    counts = {
        "n_control": 0,
        "n_treatment": 0,
        "visit_control": 0,
        "visit_treatment": 0,
        "conversion_control": 0,
        "conversion_treatment": 0,
        "exposure_control": 0,
        "exposure_treatment": 0,
        "n_total": 0,
    }

    sample_chunks = []
    parquet_writer = None
    parquet_path = PROCESSED_DIR / "criteo_uplift.parquet"

    import pyarrow as pa
    import pyarrow.parquet as pq

    reader = pd.read_csv(
        csv_path,
        compression="gzip",
        dtype=DTYPES,
        chunksize=CHUNK_SIZE,
    )

    for i, chunk in enumerate(reader):
        counts["n_total"] += len(chunk)

        is_t = chunk["treatment"] == 1
        is_c = chunk["treatment"] == 0

        counts["n_treatment"] += int(is_t.sum())
        counts["n_control"] += int(is_c.sum())

        counts["visit_treatment"] += int(chunk.loc[is_t, "visit"].sum())
        counts["visit_control"] += int(chunk.loc[is_c, "visit"].sum())

        counts["conversion_treatment"] += int(chunk.loc[is_t, "conversion"].sum())
        counts["conversion_control"] += int(chunk.loc[is_c, "conversion"].sum())

        counts["exposure_treatment"] += int(chunk.loc[is_t, "exposure"].sum())
        counts["exposure_control"] += int(chunk.loc[is_c, "exposure"].sum())

        # Write this chunk to the parquet file.
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(parquet_path, table.schema, compression="snappy")
        parquet_writer.write_table(table)

        # Keep a random slice for the sample.
        mask = rng.random(len(chunk)) < SAMPLE_FRACTION
        sample_chunks.append(chunk[mask])

        print(f"  processed chunk {i + 1}, running total {counts['n_total']:,} rows")

    if parquet_writer is not None:
        parquet_writer.close()

    sample = pd.concat(sample_chunks, ignore_index=True)
    sample_path = PROCESSED_DIR / "criteo_sample.parquet"
    sample.to_parquet(sample_path, index=False)

    # Derived rates, for the record.
    counts["control_share"] = counts["n_control"] / counts["n_total"]
    counts["treatment_share"] = counts["n_treatment"] / counts["n_total"]
    counts["visit_rate_control"] = counts["visit_control"] / counts["n_control"]
    counts["visit_rate_treatment"] = counts["visit_treatment"] / counts["n_treatment"]
    counts["conversion_rate_control"] = counts["conversion_control"] / counts["n_control"]
    counts["conversion_rate_treatment"] = counts["conversion_treatment"] / counts["n_treatment"]

    summary_path = PROCESSED_DIR / "criteo_counts.json"
    pd.Series(counts).to_json(summary_path)

    print()
    print(f"Full parquet   : {parquet_path}")
    print(f"Sample parquet : {sample_path}  ({len(sample):,} rows)")
    print(f"Exact counts   : {summary_path}")
    print()
    print("Headline numbers from the full file:")
    print(f"  Total users            {counts['n_total']:,}")
    print(f"  Control / treatment    {counts['control_share']:.4f} / {counts['treatment_share']:.4f}")
    print(f"  Conversion, control    {counts['conversion_rate_control']:.5f}")
    print(f"  Conversion, treatment  {counts['conversion_rate_treatment']:.5f}")
    print(f"  Visit, control         {counts['visit_rate_control']:.5f}")
    print(f"  Visit, treatment       {counts['visit_rate_treatment']:.5f}")

    return counts


if __name__ == "__main__":
    csv_path = download()
    convert_and_summarise(csv_path)
    print()
    print("Done. Nothing in data/ is committed to git, by design.")