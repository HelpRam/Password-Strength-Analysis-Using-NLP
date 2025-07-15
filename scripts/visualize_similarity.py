import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys

# Add the project root to sys.path so 'src' can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.word2vec_model import compute_similarity_to_weak, load_model


def plot_similarity_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.histplot(df["w2v_similarity_score"], bins=50, kde=True, color='purple')
    plt.title("Word2Vec Semantic Similarity to Weak Passwords")
    plt.xlabel("Cosine Similarity Score")
    plt.ylabel("Password Count")
    plt.tight_layout()
    plt.savefig("visualizations/w2v_similarity_distribution.png")
    plt.show()

def plot_similarity_by_zxcvbn(df):
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="strength_label", y="w2v_similarity_score", data=df, palette="Set2")
    plt.title("Word2Vec Similarity Score by zxcvbn Label")
    plt.xlabel("zxcvbn Strength Label")
    plt.ylabel("Semantic Similarity")
    plt.tight_layout()
    plt.savefig("visualizations/w2v_similarity_by_label.png")
    plt.show()

if __name__ == "__main__":
    input_path = "data/processed/rockyou_with_similarity.csv"
    df = pd.read_csv(input_path)

    # Always plot this
    plot_similarity_distribution(df)

    # Only if label column exists
    if "strength_label" in df.columns:
        plot_similarity_by_zxcvbn(df)

    print("✅ Visualizations saved in 'visualizations/' folder.")
