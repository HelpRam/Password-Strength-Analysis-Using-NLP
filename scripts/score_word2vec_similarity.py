import os
import sys

# Add the project root to sys.path so 'src' can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.word2vec_model import compute_similarity_to_weak, load_model

if __name__ == "__main__":
    model_path = "models/word2vec.model"
    input_path = "data/processed/rockyou_cleaned.csv"
    output_path = "data/processed/rockyou_with_similarity.csv"

    weak_passwords = ["password", "123456", "iloveyou", "football", "angel", "monkey", "princess", "baby", "love"]

    print("🔁 Loading Word2Vec model...")
    model = load_model(model_path)

    df = pd.read_csv(input_path)
    print(f"🔍 Scoring {len(df)} passwords for similarity to weak patterns...")

    df["w2v_similarity_score"] = df["password"].apply(
        lambda pw: compute_similarity_to_weak(model, pw, weak_passwords)
    )

    df.to_csv(output_path, index=False)
    print(f"✅ Saved: {output_path}")
    print(df.head())
