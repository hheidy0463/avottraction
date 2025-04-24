"""
from flask import Flask, request, jsonify, render_template
import joblib
import re
import contractions
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load the trained model and vectorizer
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('multi_output_model.pkl')

app = Flask(__name__)

# Preprocessing function (same as in your training process)
def preprocess_text(text):
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s.,]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Route to serve the index.html page
@app.route('/')
def index():
    return render_template('index.html')  # Flask will look for index.html in the templates folder

# Define the API route to check toxicity
@app.route('/check_toxicity', methods=['POST'])
def check_toxicity():
    data = request.get_json()  # Receive the input from the front end
    comment = data.get('comment')

    # Preprocess the comment
    preprocessed_comment = preprocess_text(comment)

    # Vectorize the comment
    comment_vector = vectorizer.transform([preprocessed_comment])

    # Predict the toxicity levels using the trained model
    prediction = model.predict(comment_vector)

    # Create a response dictionary with the results (convert numpy.int64 to int)
    response = {
        'toxicity': int(prediction[0][0]),
        'severe_toxic': int(prediction[0][1]),
        'obscene': int(prediction[0][2]),
        'threat': int(prediction[0][3]),
        'insult': int(prediction[0][4]),
        'identity_hate': int(prediction[0][5])
    }

    return jsonify(response)  # Send the results back to the front end

if __name__ == '__main__':
    app.run(debug=True)
"""

from flask import Flask, request, jsonify, render_template
import joblib
import re
import contractions
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import googleapiclient.discovery
import googleapiclient.errors



# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load the trained model and vectorizer
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('multi_output_model.pkl')



app = Flask(__name__)

# Preprocessing function (same as in your training process)
def preprocess_text(text):
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s.,]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

import httplib2
from googleapiclient.discovery import build

api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "AIzaSyC8q0AQcPeEDEXIf6-o_zDrh-VM80feJ3Y"

# Create an HTTP client with SSL verification disabled
http = httplib2.Http(disable_ssl_certificate_validation=True)

# Create the YouTube API client using the custom HTTP object
youtube = build(api_service_name, api_version, http=http, developerKey=DEVELOPER_KEY)

def get_all_comments(videoID):
    all_comments = []
    next_page_token = None

    while True:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=videoID,
            maxResults=100,
            pageToken=next_page_token
        )
        try:
            response = request.execute()
        except Exception as e:
            print(f"An error occurred: {e}")
            break

        for item in response['items']:
            top_comment = item['snippet']['topLevelComment']['snippet']
            all_comments.append({
                'Username': top_comment['authorDisplayName'],
                'Comment': top_comment['textDisplay'],
                'Date': top_comment.get('updatedAt', top_comment['publishedAt']),
            })

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    return all_comments


# Route to serve the index.html page
@app.route('/')
def index():
    return render_template('index.html')

# Define the API route to check toxicity
@app.route('/check_toxicity', methods=['POST'])
def check_toxicity():
    data = request.get_json()
    comment = data.get('comment')

    preprocessed_comment = preprocess_text(comment)
    comment_vector = vectorizer.transform([preprocessed_comment])
    prediction = model.predict(comment_vector)

    response = {
        'toxicity': int(prediction[0][0]),
        'severe_toxic': int(prediction[0][1]),
        'obscene': int(prediction[0][2]),
        'threat': int(prediction[0][3]),
        'insult': int(prediction[0][4]),
        'identity_hate': int(prediction[0][5])
    }

    return jsonify(response)

# Define a new route to get YouTube comments and classify them
@app.route('/youtube_comments', methods=['POST'])
def youtube_comments():
    data = request.get_json()
    video_id = data.get('video_id')

    # Retrieve comments from YouTube
    comments = get_all_comments(video_id)

    # Preprocess comments
    comments_text = [preprocess_text(comment['Comment']) for comment in comments]

    # Vectorize comments
    comments_vector = vectorizer.transform(comments_text)

    # Predict toxicity for each comment
    predictions = model.predict(comments_vector)

    # Prepare the response
    response = []
    for i, comment in enumerate(comments):
        prediction = predictions[i]
        response.append({
            'Username': comment['Username'],
            'Comment': comment['Comment'],
            'Date': comment['Date'],
            'toxicity': int(prediction[0]),
            'severe_toxic': int(prediction[1]),
            'obscene': int(prediction[2]),
            'threat': int(prediction[3]),
            'insult': int(prediction[4]),
            'identity_hate': int(prediction[5])
        })

    return jsonify(response)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing")


if __name__ == '__main__':
    app.run(debug=True)
