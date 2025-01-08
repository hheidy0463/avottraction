# train_model.py

import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing import TextProcessor, preprocess_text
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from transformers import EarlyStoppingCallback


"""
1. Load Data

Single CSV file with text and labels
["comment_text", "label"]
"""
df = pd.read_csv(_______)


X = df["comment_text"].fillna("").tolist() # List of comments
y = df["label"].tolist() # List of binary labels

"""
2. Train/Temp Split

Split data into training + temp (test + val) sets 
"""
X_train_temp, X_test, y_train_temp, y_test = train_test_split(
    X, 
    y, 
    test_size=0.15, # 15% for final test set
    random_state=42
    )

"""
3. Split train_temp into train + val

Want 15% of the total, but because we already have 15% for the test,
we split 15/85 -> ~ 17.6 of the current X_train_temp for val
"""
X_train, X_val, y_train, y_val = train_test_split(
    X_train_temp,
    y_train_temp,
    test_size=0.176647, # ~15% of original data
    random_state=42
)

# Checkpoint: Should give you around 70%, 15%, 15%
print(len(X_train), len(X_val), len(X_test))

"""
4. Apply Preprocessing 

Use TextProcessor class
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
"""
class AvottractionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Binary Classification - Change as needed
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = AvottractionDataset(train_encodings, y_train)
val_dataset = AvottractionDataset(val_encodings, y_val)

"""
7. Load DistilBERT for Classification

Change num_labels paramater as needed (binary/multi-class/multi-label)
"""
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2
)

"""
8. Fine-tune with Hugging Face Trainer
"""
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=16, # lower if out-of-memory errors pop up
    per_device_eval_batch_size=64, # lwower if out-of-memory errors pop up
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    learning_rate=2e-5,
    weight_decay=0.01,
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return {'accuracy': acc, 'f1': f1}

# If training stops to quickly -> increase early_stopping_patience
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=1 # Number of times validation can fail to improve
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping]
)

trainer.train()

X_test_clean = text_processor.transform(pd.Series(X_test)).tolist()
test_encodings = tokenizer(X_test_clean, truncation=True, padding=True, max_length=128)
test_dataset = AvottractionDataset(test_encodings, y_test)

metrics = trainer.evaluate(test_dataset)
print("Test Set Performance: ", metrics)

"""
9. Predictions on New Text w/out UI
"""
new_samples = ["She came in and she was like a shot of espresso. She's like being bathed in sunlight. She's incredibly energetic and enthusiastic and she had this sense of play and fun which was incredibly exciting", "Not really impressed"]
sample_clean_texts = [preprocess_text(x) for x in new_samples]
sample_encodings = tokenizer(sample_clean_texts, truncation=True, padding=True, return_tensors="pt")

with torch.no_grad():
    outputs = model(**sample_encodings)
    logits = outputs.logits
predictions = torch.argmax(logits, dim=-1).numpy()

print("Predictions (0=not attractive, 1=attractive):", predictions)