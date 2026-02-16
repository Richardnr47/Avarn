"""
Generate a price mapping file with mock prices based on JSON config complexity.
Prices are generated based on extracted features to make them realistic.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.data.config_loader import load_configs_from_directory


def generate_mock_prices(config_dir: str, output_path: str, base_price: float = 5000.0):
    """
    Generate mock prices based on configuration complexity.
    
    Args:
        config_dir: Path to directory containing JSON configs
        output_path: Path to save the price mapping CSV
        base_price: Base price in SEK
    """
    # Load configs to get features
    print(f"Loading JSON configs from {config_dir}...")
    df = load_configs_from_directory(config_dir=config_dir)
    
    print(f"Loaded {len(df)} configurations")
    
    # Generate prices based on complexity
    # More complex systems = higher price
    prices = []
    
    for idx, row in df.iterrows():
        # Base price
        price = base_price
        
        # Add price based on number of addresses (main driver)
        price += row.get('addresses_total', 0) * 150  # ~150 SEK per address
        
        # Add price based on loop controllers
        price += row.get('n_loop_controllers', 0) * 2000  # ~2000 SEK per controller
        
        # Add price based on number of loops
        price += row.get('n_loops_total', 0) * 500  # ~500 SEK per loop
        
        # Add price for manual call points
        price += row.get('manual_call_points', 0) * 300  # ~300 SEK per call point
        
        # Add price for monitored outputs
        price += row.get('monitored_outputs_true', 0) * 400  # ~400 SEK per monitored output
        
        # Add price for complexity index
        complexity = row.get('complexity_index_v1', 0)
        price += complexity * 10  # ~10 SEK per complexity unit
        
        # Add some randomness (±10%)
        random_factor = np.random.uniform(0.9, 1.1)
        price = price * random_factor
        
        # Round to nearest integer
        price = int(round(price))
        
        prices.append(price)
    
    # Create mapping DataFrame
    mapping_df = pd.DataFrame({
        'site_id': df['site_id'].values,
        'price': prices
    })
    
    # Save to CSV
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(output_path, index=False)
    
    print(f"\nGenerated price mapping:")
    print(f"  Total entries: {len(mapping_df)}")
    print(f"  Price range: {mapping_df['price'].min():.0f} - {mapping_df['price'].max():.0f} SEK")
    print(f"  Average price: {mapping_df['price'].mean():.0f} SEK")
    print(f"\nFirst few entries:")
    print(mapping_df.head(10).to_string(index=False))
    print(f"\nSaved to: {output_path}")
    
    return mapping_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate mock price mapping from JSON configs")
    parser.add_argument(
        "--config-dir",
        type=str,
        default="data/configs",
        help="Path to directory containing JSON config files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/price_mapping.csv",
        help="Output path for price mapping CSV",
    )
    parser.add_argument(
        "--base-price",
        type=float,
        default=5000.0,
        help="Base price in SEK (default: 5000)",
    )
    
    args = parser.parse_args()
    
    generate_mock_prices(
        config_dir=args.config_dir,
        output_path=args.output,
        base_price=args.base_price,
    )
