
# 🔐 Password Strength Analysis Using NLP & Machine Learning

![Project Status](https://img.shields.io/badge/status-complete-brightgreen)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## 📌 Overview

This project presents a **comprehensive, research-driven pipeline** for analyzing and predicting **password strength** using a combination of:

- Traditional rule-based systems (`zxcvbn`)
- Statistical NLP models (`Word2Vec`)
- Deep Learning Transformers (`BERT`)



---

## 🎯 Goals

- Build a robust, generalizable password strength classifier
- Compare multiple approaches: **heuristics vs embeddings vs transformers**
- Analyze model performance and failure cases
- Create a user-facing **interactive app**

---

## 🗂️ Project Structure

```
password-strength-nlp/
├── data/                  # Raw & processed password datasets
├── src/                   # All modeling + NLP modules
├── models/                # Saved Word2Vec and BERT models
├── visualizations/        # Output plots and confusion matrices
├── scripts/               # Modular scripts for each pipeline stage
├── app/                   # Streamlit web app
├── reports/               # Results summary and markdown notes
├── notebooks/             # Exploratory Jupyter notebooks
├── README.md              # 🔐 You're here
└── requirements.txt       # Dependencies
```

---

## 🧠 Dataset

- **Source:** RockYou Password Leak
- **Size:** Over 32M raw passwords (sampled ~900k)
- **Processing:**
  - Removed short/duplicate/empty passwords
  - Applied `zxcvbn` library to assign strength scores

---

## 🧠 Modeling Approaches

| Model       | Description                                      | Trained? | Notes                                     |
|-------------|--------------------------------------------------|----------|-------------------------------------------|
| `zxcvbn`    | Rule-based password strength estimator           | ❌       | Used to **generate labels** only — not for evaluation |
| `Word2Vec`  | Measures semantic similarity to weak passwords   | ✅       | Statistical embedding model               |
| `BERT`      | Transformer-based classifier (fine-tuned)        | ✅       | Main learning model for password strength |

---

## 📊 Visual Results

### 🔷 BERT Loss Curve  
![BERT Loss Curve](https://i.postimg.cc/qNvMxzLQ/Screenshot-2025-07-22-201408.png)

### 🔶 BERT Confusion Matrix  
![BERT Confusion Matrix](https://i.postimg.cc/LnT8Pzzn/Screenshot-2025-07-22-201255.png)

---

## 🧪 Key Insights

- **zxcvbn** was used as a **labeling tool**, not evaluated directly.
- **Word2Vec** provides useful semantic features, but weak classification.
- **BERT** generalizes well, achieving ~88% test accuracy.
- BERT is capable of learning complex strength signals from data, unlike rule-based systems.

---

## 💻 Streamlit App

The app provides an interactive interface where users can input passwords and get predicted strength using the **fine-tuned BERT model**.

```bash
cd app/
streamlit run password_strength_ui.py
```

---

## 🚀 How to Run

```bash
# Setup environment
pip install -r requirements.txt

# Train BERT (optional)
python scripts/train_bert.py

# Run evaluation
python scripts/test_bert.py

# Launch app
streamlit run app/password_strength_ui.py
```

---

## 📄 License

MIT License. Feel free to use, cite, or extend the work with credit.

---

## 👨‍🎓 Author

**A.A.**  
Master’s in Computer Science, University of London  
Supervised by: Prof. [Name]
