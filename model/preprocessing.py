# preprocessing.oy

import re
import emoji
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions

nltk.download('stopwords')
nltk.download('wordnet')

emoticon_map = {
    ":-)": ":smiley_face:",
    ":p": ":playful_tongue:",
    # etc.
}

def preprocess_text(text):
    text = text.lower()
    # Expand contractions
    text = contractions.fix(text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Convert emojis to text
    text = emoji.demojize(text, language='en')

    for x in emoticon_map:
        if x in text:
            text = text.replace(x, emoticon_map[x])

    # Tokenize text
    words = text.split()
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    # Join words to a single string
    return text.strip()

class TextProcessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn-compatible preprocessor
    """
    def __init__(self, preprocess_func=None):
        self.preprocess_func = preprocess_func if preprocess_func else preprocess_text

    def fit(self, X, y=None):
        """
        Update if needed
        """
        return self
    
    def transform(self, X):
        return X.apply(self.preprocess_func)
