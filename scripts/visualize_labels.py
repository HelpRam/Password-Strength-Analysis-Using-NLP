# scripts/visualize_labels.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_score_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(x='zxcvbn_score', data=df, palette='Blues_r')
    plt.title('Zxcvbn Score Distribution (0 = Weak, 4 = Strong)')
    plt.xlabel('Zxcvbn Score')
    plt.ylabel('Password Count')
    plt.savefig("visualizations/zxcvbn_score_distribution.png")
    plt.show()

def plot_label_distribution(df):
    plt.figure(figsize=(6, 5))
    sns.countplot(x='strength_label', data=df, palette='Set2')
    plt.title('Password Strength Labels')
    plt.xlabel('Strength Category')
    plt.ylabel('Password Count')
    plt.savefig("visualizations/strength_label_distribution.png")
    plt.show()

if __name__ == "__main__":
    input_path = "data/processed/rockyou_labeled.csv"
    df = pd.read_csv(input_path)

    plot_score_distribution(df)
    plot_label_distribution(df)

    print("✅ Visualizations saved in 'visualizations/' folder.")
