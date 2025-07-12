#clean_data.py

from src.preprocessing import load_rockyou, clean_passwords, save_cleaned_data

if __name__ == "__main__":
    input_path = "data/raw/rockyou.txt"
    output_path = "data/processed/rockyou_cleaned.csv"

    df = load_rockyou(input_path)
    print(f"Loaded {len(df)} passwords.")

    cleaned_df = clean_passwords(df)
    print(f"Cleaned: {len(cleaned_df)} passwords remain.")

    save_cleaned_data(cleaned_df, output_path)
    print(f"Saved cleaned data to {output_path}")
