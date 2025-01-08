import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from preprocessing import TextProcessor, preprocess_text
from transformers import DistilBertTokenizerFast

"""
1. Load Data

Single CSV file with text and labels
["comment_text", "label"]
"""
df = pd.read_csv(_______)


X = df["comment_text"].fillna("").tolist() # List of comments
y = df["label"].tolist() # List of binary labels

"""
2. Train/Test Split

Split data into training and testing sets (80/20)
"""
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


"""
3. Apply Preprocessing 

Use TextProcessor class
"""
text_processor = TextProcessor(preprocess_func=preprocess_text)
X_train_clean = text_processor.transform(pd.Series(X_train_raw)).tolist()
X_test_clean = text_processor.transform(pd.Series(X_test_raw)).tolist()

"""
4. Tokenize with DistilBERT
"""
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(
    X_train_clean, truncation=True, padding=True, max_length=128
)

test_encodings = tokenizer(
    X_test_clean, truncation=True, padding=True, max_length=128
)

"""
5. Create PyTorch Datasets
"""
class AvottractionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __get_items__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Binary Classification - Change as needed
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)