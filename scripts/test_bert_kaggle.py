import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 1. Load model + tokenizer
model = BertForSequenceClassification.from_pretrained("/kaggle/input/bert_model/transformers/default/1")
tokenizer = BertTokenizer.from_pretrained("/kaggle/input/bert_model/transformers/default/1")
model.eval().cuda()  # Use GPU

# 2. Load test set
df = pd.read_csv("/kaggle/input/test-rockyou-label/bert_test_set.csv")
label_map = {"weak": 0, "medium": 1, "strong": 2}
inv_label_map = {v: k for k, v in label_map.items()}
df = df[df["strength_label"].isin(label_map)]
df["label_id"] = df["strength_label"].map(label_map)

# 3. Dataset class
class PasswordDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=16):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# 4. Create dataloader
dataset = PasswordDataset(df["password"].tolist(), df["label_id"].tolist(), tokenizer)
dataloader = DataLoader(dataset, batch_size=64)

# 5. Run predictions
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        labels = batch["labels"].cuda()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

# 6. Print & Save classification report
print("📊 Classification Report:")
report_text = classification_report(all_labels, all_preds, target_names=label_map.keys())
print(report_text)

# Save as .txt
with open("/kaggle/working/bert_classification_report.txt", "w") as f:
    f.write("📊 Classification Report\n")
    f.write(report_text)

# Save as CSV
report_dict = classification_report(all_labels, all_preds, target_names=label_map.keys(), output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv("/kaggle/working/bert_classification_report.csv", index=True)

# 7. Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_map.keys()))
disp.plot(cmap="Blues", values_format="d")
plt.title("BERT Test Set Confusion Matrix")
plt.tight_layout()
plt.savefig("/kaggle/working/bert_confusion_matrix.png")
plt.show()
