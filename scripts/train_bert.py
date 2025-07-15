# train_bert.py
# Import pandas for data loading and manipulation
import pandas as pd
import os
import sys

# Add the project root to sys.path so 'src' can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import PyTorch (used for tensor operations and model training)
import torch

# Import BERT tokenizer, model, and training utilities from Hugging Face
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments


# Import function to split dataset into training/validation/test sets
from sklearn.model_selection import train_test_split

# Import custom dataset class and label mapping from local source module
from src.bert_model import PasswordDataset, get_label_mapping

# Main execution block to prevent running code on import
if __name__ == "__main__":
    # Load the labeled password dataset
    df = pd.read_csv("data/processed/rockyou_labeled.csv")

    # Get mapping from string labels ("weak", "medium", "strong") to integer IDs
    label_map = get_label_mapping()

    # Filter out rows with invalid labels and map string labels to integers
    df = df[df["strength_label"].isin(label_map.keys())]
    df["label_id"] = df["strength_label"].map(label_map)

    # Split data: 70% train, 15% validation, 15% test using stratified sampling for balance
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df["label_id"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["label_id"]
    )

    # Print sizes of each split
    print(f"📊 Split sizes → Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Load BERT tokenizer (uncased version: lowercase input)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Create dataset objects for training, validation, and test
    train_dataset = PasswordDataset(train_df["password"].tolist(), train_df["label_id"].tolist(), tokenizer)
    val_dataset = PasswordDataset(val_df["password"].tolist(), val_df["label_id"].tolist(), tokenizer)
    test_dataset = PasswordDataset(test_df["password"].tolist(), test_df["label_id"].tolist(), tokenizer)

    # Load pre-trained BERT model with classification head (3 output classes)
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

    # Define training arguments/configuration
    training_args = TrainingArguments(
        output_dir="./models/bert",                # Where to save model files
        # evaluation_strategy="epoch",              # Evaluate at the end of each epoch
        save_strategy="no",                       # Don’t save checkpoints during training
        per_device_train_batch_size=32,           # Training batch size per GPU/CPU
        per_device_eval_batch_size=32,            # Evaluation batch size
        num_train_epochs=3,                       # Total number of training epochs
        logging_dir="./logs",                     # Directory to store logs
        logging_steps=10,                         # Log every 10 steps
        load_best_model_at_end=True,              # Load the best model after training
        metric_for_best_model="eval_loss",        # Use eval loss to determine the best model
        greater_is_better=False                   # Lower loss is better
    )

    # Create a Trainer object that handles training and evaluation
    trainer = Trainer(
        model=model,                      # BERT model for training
        args=training_args,               # Training arguments
        train_dataset=train_dataset,     # Training data
        eval_dataset=val_dataset         # Validation data
    )

    # Train the model
    trainer.train()

    # Save the final model and tokenizer to disk
    model.save_pretrained("models/bert")
    tokenizer.save_pretrained("models/bert")

    # Save test dataset for future evaluation
    test_df.to_csv("data/processed/bert_test_set.csv", index=False)

    # Print confirmation messages
    print("✅ BERT model saved to models/bert")
    print("📁 Test set saved to data/processed/bert_test_set.csv")
