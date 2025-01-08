import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from preprocessing import TextProcessor, preprocess_text
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


"""
3. Apply Preprocessing 

Use TextProcessor class
"""
text_processor = TextProcessor(preprocess_func=preprocess_text)