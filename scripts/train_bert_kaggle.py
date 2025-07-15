# Import necessary libraries
import pandas as pd                     # For loading and handling the dataset
import torch                            # For tensor operations and PyTorch support
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split  # For splitting the dataset
from torch.utils.data import Dataset    # Base class for custom datasets

# 1. Load Data
df = pd.read_csv("/kaggle/input/rockyou-labeled/rockyou_labeled.csv")  # Load the labeled password dataset
label_map = {"weak": 0, "medium": 1, "strong": 2}                       # Map text labels to numeric classes
df = df[df["strength_label"].isin(label_map)]                          # Filter only valid labels
df["label_id"] = df["strength_label"].map(label_map)                   # Add numeric label column

# 2. Split into Train / Val / Test sets (70% / 15% / 15%)
train_df, temp_df = train_test_split(
    df, test_size=0.3, stratify=df["label_id"], random_state=42        # Split 70% train / 30% temp
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label_id"], random_state=42  # Split temp into 15% val / 15% test
)

# 3. Tokenizer and Dataset Preparation
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")         # Load pre-trained BERT tokenizer

# Define custom PyTorch Dataset to work with Hugging Face Trainer
class PasswordDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=16):
        self.texts = texts                      # List of password strings
        self.labels = labels                    # Corresponding label integers
        self.tokenizer = tokenizer              # Tokenizer for text encoding
        self.max_len = max_len                  # Max token length for BERT

    def __len__(self):
        return len(self.texts)                  # Total number of samples

    def __getitem__(self, idx):
        # Tokenize the text with padding and truncation
        inputs = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"                # Return PyTorch tensors
        )
        # Return input IDs, attention mask, and label (squeezed to remove batch dimension)
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Convert DataFrames to Datasets
train_dataset = PasswordDataset(train_df["password"].tolist(), train_df["label_id"].tolist(), tokenizer)
val_dataset = PasswordDataset(val_df["password"].tolist(), val_df["label_id"].tolist(), tokenizer)
test_dataset = PasswordDataset(test_df["password"].tolist(), test_df["label_id"].tolist(), tokenizer)

# 4. Load BERT Model and Define Training Arguments
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)  # Load BERT with 3 output labels

# Set up training configuration
training_args = TrainingArguments(
    output_dir="./results",                    # Directory to save model output
    evaluation_strategy="epoch",              # Evaluate at the end of each epoch
    num_train_epochs=3,                       # Train for 3 epochs
    per_device_train_batch_size=32,           # Training batch size
    per_device_eval_batch_size=32,            # Evaluation batch size
    save_strategy="no",                       # Do not save checkpoints
    logging_dir="./logs",                     # Directory to store logs
    load_best_model_at_end=False              # Do not load best model after training
)

# Initialize Hugging Face Trainer
trainer = Trainer(
    model=model,                              # Model to train
    args=training_args,                       # Training configurations
    train_dataset=train_dataset,             # Training data
    eval_dataset=val_dataset                 # Validation data
)

# Start training
trainer.train()

# 5. Save model and test set for future evaluation
model.save_pretrained("/kaggle/working/bert_model")         # Save the fine-tuned model
tokenizer.save_pretrained("/kaggle/working/bert_model")     # Save tokenizer used in training
test_df.to_csv("/kaggle/working/bert_test_set.csv", index=False)  # Save test set to CSV

# Final output confirmation
print("✅ Model and tokenizer saved in /kaggle/working/bert_model")
print("📁 Test set saved as /kaggle/working/bert_test_set.csv")
