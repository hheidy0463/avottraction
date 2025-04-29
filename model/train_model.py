import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing import TextProcessor, preprocess_text
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import f1_score, accuracy_score
from torch import sigmoid
from sklearn.metrics import roc_auc_score


# Set device to MPS if available, otherwise use CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

"""
1. Load Data
Assumes CSV file with columns: ["comment_text", "sexual", "flirty", "wordplay", "funny"]
"""
df = pd.read_csv('data/avottraction_dataset.csv')

# Use all four label columns (make sure the column names match exactly)
X = df["Text"].fillna("").tolist()
y = df[["Sexual", "Flirty", "Wordplay", "Funny"]].values  # shape: (num_samples, 4)

"""
2. Train/Temp Split
Split data into training + temp (test + validation) sets 
"""
X_train_temp, X_test, y_train_temp, y_test = train_test_split(
    X, 
    y, 
    test_size=0.15, # 15% for final test set
    random_state=42
)

"""
3. Split train_temp into train + validation
We want ~15% of total for validation, so we split train_temp accordingly.
"""
X_train, X_val, y_train, y_val = train_test_split(
    X_train_temp,
    y_train_temp,
    test_size=0.176647,  # ~15% of original data
    random_state=42
)


# Checkpoint: Should give you around 70%, 15%, 15%
print(len(X_train), len(X_val), len(X_test))

"""
4. Apply Preprocessing 
Use TextProcessor class to clean the text
"""
text_processor = TextProcessor(preprocess_func=preprocess_text)
X_train_clean = text_processor.transform(pd.Series(X_train)).tolist()
X_val_clean = text_processor.transform(pd.Series(X_val)).tolist()

"""
5. Tokenize with DistilBERT
"""
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(
    X_train_clean, truncation=True, padding=True, max_length=128
)
val_encodings = tokenizer(
    X_val_clean, truncation=True, padding=True, max_length=128
)

"""
6. Create PyTorch Datasets
Make sure labels are cast to float (as BCEWithLogitsLoss expects float values)
"""
class AvottractionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        # Convert labels to float for multi-label BCE loss.
        self.labels = labels.astype(float)  

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = AvottractionDataset(train_encodings, y_train)
val_dataset = AvottractionDataset(val_encodings, y_val)

"""
7. Load DistilBERT for Multi-Label Classification
Set num_labels to 4 and specify the problem type.
"""
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=4,
    problem_type="multi_label_classification"
)
model.to(device) # Move the model to the correct device here



"""
8. Fine-tune with Hugging Face Trainer
"""
training_args = TrainingArguments(
    output_dir='model/results',
    num_train_epochs=5,
    per_device_train_batch_size=16,  # adjust if needed
    per_device_eval_batch_size=64,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',  # you might later adjust metrics for multi-label
    learning_rate=2e-5,
    weight_decay=0.01,
)



# Optimize threshold for each label
best_thresholds = [0.5, 0.5, 0.5, 0.5]  # default thresholds for 4 labels

def compute_metrics(pred):
    logits = torch.tensor(pred.predictions)
    probs = sigmoid(logits)
    # Ensure conversion to NumPy for consistent processing
    probs_np = probs.cpu().numpy() if probs.is_cuda else probs.numpy()
    
    # Apply the best thresholds for each label
    y_preds = np.array([ (probs_np[:, i] >= best_thresholds[i]).astype(int) for i in range(probs_np.shape[1]) ]).T
    y_true = pred.label_ids
    
    # Compute metrics
    per_label_f1 = f1_score(y_true, y_preds, average=None)
    overall_f1 = f1_score(y_true, y_preds, average="macro")
    per_label_acc = (y_preds == y_true).mean(axis=0)
    acc = np.mean(np.all(y_preds == y_true, axis=1))
    overall_auc = roc_auc_score(y_true, probs_np)
    
    return {
        "accuracy": acc, 
        "f1_macro": overall_f1,
        "per_label_f1": per_label_f1.tolist(),
        "per_label_acc": per_label_acc.tolist(),
        "roc_auc": overall_auc
    }



early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3  # adjust as needed
)

# uses best_thresholds during training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping]
)

trainer.train()

# After training, gets validation outputs and update best_thresholds
# Get model outputs on the validation set
val_logits = trainer.predict(val_dataset).predictions
val_probs = sigmoid(torch.tensor(val_logits)).numpy()  

best_thresholds = []
for i in range(y_train.shape[1]):
    best_f1 = 0
    best_t = 0.1
    for t in np.arange(0.1, 0.7, 0.05):
        preds = (val_probs[:, i] >= t).astype(int)
        f1 = f1_score(y_val[:, i], preds)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    best_thresholds.append(best_t)

print("Best thresholds per label:", best_thresholds)


"""
9. Evaluate on Test Set
"""
X_test_clean = text_processor.transform(pd.Series(X_test)).tolist()
test_encodings = tokenizer(X_test_clean, truncation=True, padding=True, max_length=128)
test_dataset = AvottractionDataset(test_encodings, y_test)

metrics = trainer.evaluate(test_dataset)
print("Test Set Performance: ", metrics)

# """
# 10. Predictions on New Text (without UI)
# For multi-label, the predictions are a vector of probabilities per label.
# """
new_samples = [
"You are cute. Wanna go on a date?"
]
sample_clean_texts = [preprocess_text(x) for x in new_samples]
# When tokenizing new samples, move the tensors to the device:
sample_encodings = tokenizer(sample_clean_texts, truncation=True, padding=True, return_tensors="pt")
sample_encodings = {k: v.to(device) for k, v in sample_encodings.items()}

with torch.no_grad():
    outputs = model(**sample_encodings)
    logits = outputs.logits
# Apply sigmoid and threshold at 0.35 for multi-label predictions
probs = sigmoid(logits)
print("Raw probabilities:", probs.cpu().numpy())

threshold = 0.1  # Try a lower threshold
predictions = (probs >= threshold).int().cpu().numpy()
print("Predictions:", predictions)

