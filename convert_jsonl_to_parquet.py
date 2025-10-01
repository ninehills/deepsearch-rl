#!/usr/bin/env python3
"""
Convert JSONL files to Parquet format.

Usage:
    python convert_jsonl_to_parquet.py <input_jsonl> <output_parquet>

Example:
    python convert_jsonl_to_parquet.py data/train.jsonl data/train.parquet
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def convert_jsonl_to_parquet(input_path: str, output_path: str) -> None:
    """
    Convert a JSONL file to Parquet format.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output Parquet file
    """
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        print(f"Error: Input file '{input_path}' does not exist", file=sys.stderr)
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Read JSONL file
    print(f"Reading {input_path}...")
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON on line {line_num}: {e}", file=sys.stderr)

    if not data:
        print("Error: No valid data found in input file", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(data)} records")

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Write to Parquet
    print(f"Writing to {output_path}...")
    df.to_parquet(output_file, engine='pyarrow', compression='snappy', index=False)

    print(f"✓ Successfully converted {len(data)} records to {output_path}")
    print(f"  File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL files to Parquet format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_jsonl_to_parquet.py data/train.jsonl data/train.parquet
  python convert_jsonl_to_parquet.py data/val_mini.jsonl data/val_mini.parquet
        """
    )

    parser.add_argument(
        'input_jsonl',
        help='Path to input JSONL file'
    )

    parser.add_argument(
        'output_parquet',
        help='Path to output Parquet file'
    )

    args = parser.parse_args()

    convert_jsonl_to_parquet(args.input_jsonl, args.output_parquet)


if __name__ == '__main__':
    main()