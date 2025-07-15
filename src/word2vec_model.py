# src/word2vec_model.py

from gensim.models import Word2Vec
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def tokenize_password(password: str) -> list:
    """
    Tokenize password into character n-grams or simple characters
    """
    return list(password.lower())  # Character-level tokens

def train_word2vec(df: pd.DataFrame, vector_size=100, window=5, min_count=1, sg=1) -> Word2Vec:
    """
    Train Word2Vec model on password data (character-based tokens)
    """
    corpus = df["password"].apply(tokenize_password).tolist()
    model = Word2Vec(sentences=corpus, vector_size=vector_size, window=window, min_count=min_count, sg=sg)
    return model

def save_model(model: Word2Vec, path: str):
    model.save(path)

def load_model(path: str) -> Word2Vec:
    return Word2Vec.load(path)


def password_to_vector(model: Word2Vec, password: str) -> np.ndarray:
    """
    Converts a password to its average embedding vector using the trained model.
    """
    tokens = list(password.lower())
    valid_tokens = [t for t in tokens if t in model.wv]
    
    if not valid_tokens:
        return np.zeros(model.vector_size)
    
    vectors = [model.wv[t] for t in valid_tokens]
    return np.mean(vectors, axis=0)

def compute_similarity_to_weak(model: Word2Vec, password: str, weak_passwords: list) -> float:
    """
    Compute max cosine similarity between password and known weak passwords.
    """
    pw_vec = password_to_vector(model, password)
    weak_vecs = [password_to_vector(model, w) for w in weak_passwords]
    
    sims = cosine_similarity([pw_vec], weak_vecs)
    return float(np.max(sims))
