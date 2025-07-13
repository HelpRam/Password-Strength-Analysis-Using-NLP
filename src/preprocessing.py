# preprocessing.py

import pandas as pd

def load_rockyou(file_path: str) -> pd.DataFrame:
    """
    Load RockYou dataset into a DataFrame.
    Assumes one password per line (may include non-ASCII).
    """
    with open(file_path, 'r', encoding='latin-1') as file:
        passwords = file.read().splitlines()
    return pd.DataFrame(passwords, columns=["password"])


def clean_passwords(df: pd.DataFrame, min_len=6, max_len=20) -> pd.DataFrame:
    """
    Clean password list:
    - Remove nulls
    - Convert to string (in case of any weird types)
    - Filter by length
    - Keep only ASCII passwords (filters out emoji, accents, etc.)
    - Drop duplicates
    """
    # Step 1: Drop nulls
    df = df[df['password'].notnull()]

    # Step 2: Ensure everything is string
    df['password'] = df['password'].astype(str)

    # Step 3: Filter by password length
    df = df[df['password'].str.len().between(min_len, max_len)]

    # Step 4: Keep only ASCII passwords
    df = df[df['password'].apply(lambda x: x.isascii())]

    # Step 5: Remove duplicates
    df = df.drop_duplicates()

    return df.reset_index(drop=True)


def save_cleaned_data(df: pd.DataFrame, output_path: str):
    """
    Save cleaned passwords to CSV.
    """
    df.to_csv(output_path, index=False)
