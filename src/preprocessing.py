# preprocessing.py

import pandas as pd

def load_rockyou(file_path: str) -> pd.DataFrame:
    """
    Load RockYou dataset into a DataFrame.
    """
    with open(file_path, 'r', encoding='latin-1') as file:
        passwords = file.read().splitlines()
    return pd.DataFrame(passwords, columns=["password"])

def clean_passwords(df: pd.DataFrame, min_len=6, max_len=20) -> pd.DataFrame:
    """
    Clean password list: remove nulls, short/long, non-ASCII, duplicates.
    """
    df = df[df['password'].notnull()]
    df = df[df['password'].str.len().between(min_len, max_len)]
    df = df[df['password'].str.isascii()]
    df = df.drop_duplicates()
    return df.reset_index(drop=True)

def save_cleaned_data(df: pd.DataFrame, output_path: str):
    """
    Save cleaned passwords to CSV.
    """
    df.to_csv(output_path, index=False)
