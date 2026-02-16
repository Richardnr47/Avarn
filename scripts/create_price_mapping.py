"""
Utility script to create a price mapping file from existing training data.
This helps migrate from CSV-based training data to JSON config-based training.

Usage:
    python create_price_mapping.py --csv data/training_data.csv --output data/price_mapping.csv
    python create_price_mapping.py --config-dir data/configs --output data/price_mapping.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from app.data.config_loader import load_configs_from_directory
except ImportError:
    load_configs_from_directory = None


def create_mapping_from_csv(csv_path: str, output_path: str):
    """
    Create price mapping from CSV file.
    
    Assumes CSV has a site_id column (or similar) and a price column.
    """
    df = pd.read_csv(csv_path)
    
    # Try to find site_id column (case-insensitive)
    site_col = None
    for col in df.columns:
        if "site" in col.lower() and "id" in col.lower():
            site_col = col
            break
    
    if site_col is None:
        # If no site_id, use index or first column
        print(f"Warning: No site_id column found. Using index as site_id.")
        df["site_id"] = df.index.astype(str)
        site_col = "site_id"
    
    # Find price column
    price_col = None
    for col in df.columns:
        if col.lower() == "price":
            price_col = col
            break
    
    if price_col is None:
        raise ValueError(f"No 'price' column found in CSV. Columns: {df.columns.tolist()}")
    
    # Create mapping
    mapping_df = df[[site_col, price_col]].copy()
    mapping_df.columns = ["site_id", "price"]
    
    # Save
    mapping_df.to_csv(output_path, index=False)
    print(f"Created price mapping with {len(mapping_df)} entries")
    print(f"Saved to: {output_path}")
    print(f"\nFirst few entries:")
    print(mapping_df.head())
    
    return mapping_df


def create_mapping_from_configs(config_dir: str, output_path: str, price_file: str = None):
    """
    Create a template price mapping file from JSON configs.
    
    If price_file is provided, it will be used as a base and updated with
    any missing entries from the configs.
    """
    if load_configs_from_directory is None:
        raise ImportError("Cannot load JSON configs: app.data.config_loader not available")
    
    # Load configs to get site_ids
    df = load_configs_from_directory(config_dir=config_dir)
    site_ids = df["site_id"].tolist()
    
    # Load existing price mapping if provided
    existing_mapping = {}
    if price_file and Path(price_file).exists():
        existing_df = pd.read_csv(price_file)
        if "site_id" in existing_df.columns and "price" in existing_df.columns:
            existing_mapping = dict(zip(existing_df["site_id"], existing_df["price"]))
            print(f"Loaded {len(existing_mapping)} existing price mappings")
    
    # Create mapping (use existing prices or NaN for missing)
    mapping_data = []
    for site_id in site_ids:
        price = existing_mapping.get(site_id, None)
        mapping_data.append({"site_id": site_id, "price": price})
    
    mapping_df = pd.DataFrame(mapping_data)
    
    # Save
    mapping_df.to_csv(output_path, index=False)
    
    missing_count = mapping_df["price"].isna().sum()
    print(f"Created price mapping template with {len(mapping_df)} entries")
    if missing_count > 0:
        print(f"Warning: {missing_count} entries have no price (NaN)")
        print("Please fill in the missing prices manually in the CSV file.")
    print(f"Saved to: {output_path}")
    
    return mapping_df


def main():
    parser = argparse.ArgumentParser(
        description="Create price mapping file for JSON config-based training"
    )
    
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to CSV file with site_id and price columns",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Path to directory containing JSON config files",
    )
    parser.add_argument(
        "--price-file",
        type=str,
        default=None,
        help="Optional existing price mapping file to merge with",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/price_mapping.csv",
        help="Output path for price mapping CSV",
    )
    
    args = parser.parse_args()
    
    if args.csv:
        create_mapping_from_csv(args.csv, args.output)
    elif args.config_dir:
        create_mapping_from_configs(args.config_dir, args.output, args.price_file)
    else:
        parser.error("Either --csv or --config-dir must be provided")


if __name__ == "__main__":
    main()
