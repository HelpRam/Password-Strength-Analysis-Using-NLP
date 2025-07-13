# score_entropy.py

import os
import sys
# Add the project root to sys.path so 'src' can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.entropy_scoring import score_passwords, save_labeled_data
import pandas as pd

if __name__ == "__main__":
    input_path = "data/processed/rockyou_cleaned.csv"
    output_path = "data/processed/rockyou_labeled.csv"

    df = pd.read_csv(input_path)
    print(f"Scoring {len(df)} passwords...")

    scored_df = score_passwords(df)
    save_labeled_data(scored_df, output_path)

    print(f" Labeled data saved to {output_path}")
    print(scored_df.head())
