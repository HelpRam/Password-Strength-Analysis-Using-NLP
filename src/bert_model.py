# bert_model.py

# Import PyTorch library for tensor operations and deep learning
import torch

# Import necessary modules from Hugging Face Transformers library
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Import train-test split utility from scikit-learn
from sklearn.model_selection import train_test_split

# Import Dataset class to define a custom PyTorch dataset
from torch.utils.data import Dataset

# Import pandas for data handling (e.g., reading CSVs)
import pandas as pd

# Define a custom dataset class for password classification
class PasswordDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=16):
        # Store the input text samples
        self.texts = texts
        # Store the corresponding labels
        self.labels = labels
        # Store the tokenizer for encoding text
        self.tokenizer = tokenizer
        # Define the maximum token length
        self.max_len = max_len

    # Return the number of samples in the dataset
    def __len__(self):
        return len(self.texts)

    # Return a single data point (features and label) at a given index
    def __getitem__(self, idx):
        # Convert the selected text to string
        text = str(self.texts[idx])
        # Tokenize the text with padding and truncation, and convert to tensor format
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors="pt")
        # Return a dictionary containing input IDs, attention mask, and the label tensor
        return {
            'input_ids': inputs['input_ids'].squeeze(),           # Tensor of token IDs
            'attention_mask': inputs['attention_mask'].squeeze(), # Tensor indicating which tokens are padding
            'labels': torch.tensor(self.labels[idx], dtype=torch.long) # Label tensor as long integer
        }

# Function to define a mapping from string labels to integer classes
def get_label_mapping():
    return {"weak": 0, "medium": 1, "strong": 2}
