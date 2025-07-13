#entropy_scoring.py

import pandas as pd
from zxcvbn import zxcvbn
import tqdm  # for progress bar visualization

def score_passwords(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds zxcvbn strength score (0–4) and corresponding strength label to each password.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing a 'password' column.

    Returns:
        pd.DataFrame: Original DataFrame with added 'zxcvbn_score' and 'strength_label' columns.
    """
    df = df.copy()  # Avoid modifying the original DataFrame

    scores = []  # List to store numeric scores

    # Loop through each password with a progress bar
    for pw in tqdm.tqdm(df["password"], desc="Scoring passwords"):
        try:
            # Use zxcvbn to analyze password strength
            result = zxcvbn(pw)
            scores.append(result["score"])  # Score ranges from 0 (weak) to 4 (strong)
        except Exception as e:
            # If any error occurs (e.g., bad input), print it and assign weakest score
            print(f"Error scoring password: {pw} — {e}")
            scores.append(0)

    # Add numeric scores to DataFrame
    df["zxcvbn_score"] = scores

    # Function to convert numeric score to strength category
    def label_strength(score):
        if score <= 1:
            return "weak"
        elif score == 2:
            return "medium"
        else:
            return "strong"

    # Apply labeling function to each score
    df["strength_label"] = df["zxcvbn_score"].apply(label_strength)

    return df


def save_labeled_data(df: pd.DataFrame, output_path: str):
    """
    Save the scored and labeled DataFrame to a CSV file.

    Parameters:
        df (pd.DataFrame): DataFrame with 'password', 'zxcvbn_score', and 'strength_label'.
        output_path (str): File path to save the output CSV.
    """
    df.to_csv(output_path, index=False)  # Save without row indices
