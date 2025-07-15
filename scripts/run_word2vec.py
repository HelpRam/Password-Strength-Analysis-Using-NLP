import os
import sys

# Add the project root to sys.path so 'src' can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.word2vec_model import train_word2vec, save_model
import pandas as pd

if __name__ == "__main__":
    input_path = "data/processed/rockyou_cleaned.csv"
    output_model_path = "models/word2vec.model"

    df = pd.read_csv(input_path)
    model = train_word2vec(df)
    save_model(model, output_model_path)

    print(f"✅ Word2Vec model saved to {output_model_path}")
