"""
Script to generate dummy training data for fire alarm testing price prediction.
Creates a larger, more diverse dataset with realistic patterns.
"""

import random

import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)


def generate_dummy_data(n_samples=200):
    """
    Generate dummy data for fire alarm testing price prediction.

    Args:
        n_samples: Number of samples to generate

    Returns:
        DataFrame with generated data
    """
    data = []

    stader = [
        "Stockholm",
        "Göteborg",
        "Malmö",
        "Uppsala",
        "Linköping",
        "Örebro",
        "Västerås",
        "Helsingborg",
    ]

    # Location multipliers (Stockholm and Göteborg are typically more expensive)
    stad_multipliers = {
        "Stockholm": 1.15,
        "Göteborg": 1.10,
        "Malmö": 1.05,
        "Uppsala": 1.00,
        "Linköping": 0.95,
        "Örebro": 0.90,
        "Västerås": 0.92,
        "Helsingborg": 0.98,
    }

    # Frequency multipliers (more frequent = more expensive)
    frequency_multipliers = {
        "månadsvis": 1.5,  # Monthly is most expensive
        "kvartalsvis": 1.2,  # Quarterly
        "årsvis": 1.0,  # Yearly is base
    }

    for i in range(n_samples):
        # Antal sektioner (number of fire alarm sections)
        # Typically 1-20 sections depending on building complexity
        antal_sektioner = np.random.choice(
            [
                np.random.randint(1, 3),  # Small buildings: 1-2 sections
                np.random.randint(3, 6),  # Medium buildings: 3-5 sections
                np.random.randint(6, 10),  # Large buildings: 6-9 sections
                np.random.randint(10, 15),  # Very large: 10-14 sections
                np.random.randint(15, 21),  # Extra large: 15-20 sections
            ],
            p=[0.20, 0.30, 0.30, 0.15, 0.05],
        )

        # Antal detektorer (number of detectors)
        # Correlated with sections, typically 2-5 detectors per section
        detectors_per_section = np.random.uniform(2.5, 5.5)
        antal_detektorer = int(
            antal_sektioner * detectors_per_section
        ) + np.random.randint(-2, 3)
        antal_detektorer = max(2, antal_detektorer)  # Minimum 2 detectors

        # Antal larmdon (number of alarm devices)
        # Typically 1-3 per section (alarm bells, sirens, etc.)
        larmdon_per_section = np.random.uniform(1.5, 3.5)
        antal_larmdon = int(antal_sektioner * larmdon_per_section) + np.random.randint(
            -1, 2
        )
        antal_larmdon = max(1, antal_larmdon)  # Minimum 1 alarm device

        # Stad (city)
        stad = np.random.choice(stader)

        # Frequency (kvartalsvis, månadsvis, årsvis) - mutually exclusive
        # Larger systems more likely to have more frequent testing
        if antal_sektioner >= 10:
            frequency = np.random.choice(
                ["månadsvis", "kvartalsvis", "årsvis"], p=[0.4, 0.4, 0.2]
            )
        elif antal_sektioner >= 5:
            frequency = np.random.choice(
                ["månadsvis", "kvartalsvis", "årsvis"], p=[0.2, 0.5, 0.3]
            )
        else:
            frequency = np.random.choice(
                ["månadsvis", "kvartalsvis", "årsvis"], p=[0.1, 0.3, 0.6]
            )

        # Set frequency flags (mutually exclusive)
        kvartalsvis = 1 if frequency == "kvartalsvis" else 0
        månadsvis = 1 if frequency == "månadsvis" else 0
        årsvis = 1 if frequency == "årsvis" else 0

        # Dörrhållarmagneter (door holder magnets)
        # Typically 0-10 per building, more in larger buildings
        if antal_sektioner >= 10:
            dörrhållarmagneter = np.random.randint(5, 15)
        elif antal_sektioner >= 5:
            dörrhållarmagneter = np.random.randint(2, 8)
        else:
            dörrhållarmagneter = np.random.choice([0, 1, 2, 3], p=[0.3, 0.3, 0.3, 0.1])

        # Ventilation (boolean: 0 or 1, or could be count)
        # Larger buildings more likely to have ventilation systems
        if antal_sektioner >= 8:
            ventilation = np.random.choice([0, 1], p=[0.1, 0.9])  # 90% have ventilation
        elif antal_sektioner >= 4:
            ventilation = np.random.choice([0, 1], p=[0.3, 0.7])  # 70% have ventilation
        else:
            ventilation = np.random.choice([0, 1], p=[0.6, 0.4])  # 40% have ventilation

        # Calculate base price with realistic relationships
        # Base price depends on sections, detectors, and alarm devices
        base_price = (
            (antal_sektioner * 2000) + (antal_detektorer * 300) + (antal_larmdon * 500)
        )

        # Add cost for door holder magnets
        base_price += dörrhållarmagneter * 400

        # Add cost for ventilation (if present, adds complexity)
        if ventilation:
            base_price += 1500

        # Apply multipliers
        stad_mult = stad_multipliers[stad]
        frequency_mult = frequency_multipliers[frequency]

        # Calculate final price
        price = base_price * stad_mult * frequency_mult

        # Add some random noise (±8%)
        noise = np.random.uniform(0.92, 1.08)
        price = int(price * noise)

        # Ensure minimum price
        price = max(3000, price)

        data.append(
            {
                "antal_sektioner": antal_sektioner,
                "antal_detektorer": antal_detektorer,
                "kvartalsvis": kvartalsvis,
                "månadsvis": månadsvis,
                "årsvis": årsvis,
                "stad": stad,
                "dörrhållarmagneter": dörrhållarmagneter,
                "ventilation": ventilation,
                "antal_larmdon": antal_larmdon,
                "price": price,
            }
        )

    df = pd.DataFrame(data)
    return df


def main():
    """Generate and save dummy data."""
    print("Generating dummy training data...")

    # Generate 250 samples
    df = generate_dummy_data(n_samples=500)

    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save to CSV
    output_path = "../data/training_data.csv"
    df.to_csv(output_path, index=False)

    print(f"\nGenerated {len(df)} samples")
    print(f"Data saved to {output_path}")

    # Display statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"\nAntal Sektioner:")
    print(f"  Min: {df['antal_sektioner'].min()}")
    print(f"  Max: {df['antal_sektioner'].max()}")
    print(f"  Mean: {df['antal_sektioner'].mean():.1f}")
    print(f"\nAntal Detektorer:")
    print(f"  Min: {df['antal_detektorer'].min()}")
    print(f"  Max: {df['antal_detektorer'].max()}")
    print(f"  Mean: {df['antal_detektorer'].mean():.1f}")
    print(f"\nAntal Larmdon:")
    print(f"  Min: {df['antal_larmdon'].min()}")
    print(f"  Max: {df['antal_larmdon'].max()}")
    print(f"  Mean: {df['antal_larmdon'].mean():.1f}")
    print(f"\nDörrhållarmagneter:")
    print(f"  Min: {df['dörrhållarmagneter'].min()}")
    print(f"  Max: {df['dörrhållarmagneter'].max()}")
    print(f"  Mean: {df['dörrhållarmagneter'].mean():.1f}")
    print(f"\nVentilation:")
    print(df["ventilation"].value_counts().to_string())
    print(f"\nFrequency:")
    print(f"  Månadsvis: {df['månadsvis'].sum()}")
    print(f"  Kvartalsvis: {df['kvartalsvis'].sum()}")
    print(f"  Årsvis: {df['årsvis'].sum()}")
    print(f"\nPrice:")
    print(f"  Min: {df['price'].min():,} SEK")
    print(f"  Max: {df['price'].max():,} SEK")
    print(f"  Mean: {df['price'].mean():,.0f} SEK")
    print(f"\nStad:")
    print(df["stad"].value_counts().to_string())
    print("=" * 60)

    # Show first few rows
    print("\nFirst 10 rows:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
