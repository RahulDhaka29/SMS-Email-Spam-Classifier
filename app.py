# --- Spam Classifier Backend (NLTK Fix 3 - Runtime Download) ---
from flask import Flask, render_template, request
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
import numpy as np
import os # Import os to manage paths

# --- Define NLTK Data Path and Ensure it Exists ---
# Create a specific folder within your project for NLTK data
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
# Tell NLTK to look for data here
nltk.data.path.append(nltk_data_path)

# --- Function to download NLTK data to the specified path ---
def download_nltk_data():
    """Checks for and downloads NLTK data to the specified path if not found."""
    try:
        nltk.data.find('corpora/stopwords', paths=[nltk_data_path])
        print("NLTK stopwords found.")
    except LookupError:
        print(f"Downloading NLTK stopwords to {nltk_data_path}...")
        nltk.download('stopwords', download_dir=nltk_data_path)
    try:
        nltk.data.find('tokenizers/punkt', paths=[nltk_data_path])
        print("NLTK punkt found.")
    except LookupError:
        print(f"Downloading NLTK punkt to {nltk_data_path}...")
        nltk.download('punkt', download_dir=nltk_data_path)

# --- Run the download check when the script starts ---
download_nltk_data()

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Load Model Files ---
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    print("Vectorizer and model loaded successfully.")
except FileNotFoundError:
    print("ERROR: 'vectorizer.pkl' or 'model.pkl' not found.")
    exit()
except Exception as e:
    print(f"Error loading pickle files: {e}")
    exit()

# --- NLTK Setup ---
ps = PorterStemmer()
# Load stopwords after ensuring they are downloaded
try:
    stop_words = set(stopwords.words('english'))
except Exception as e:
     print(f"Could not load stopwords even after download attempt: {e}")
     stop_words = set() # Fallback

def transform_text(text):
    text = text.lower()
    # Tokenization should work now after download check
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]
    text = [i for i in y if i not in stop_words and i not in string.punctuation]
    y = [ps.stem(i) for i in text]
    return " ".join(y)

# --- Flask Routes ---
@app.route('/')
def home():
    """Renders the main page (classifier)."""
    return render_template('index.html')

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request."""
    prediction_text = ""
    is_spam_flag = None
    message = request.form.get('message', '')

    if request.method == 'POST':
        if not message.strip():
            prediction_text = "Please enter a message to classify."
        else:
            try:
                # Add extra check here for safety
                if 'punkt' not in nltk.data.path and nltk_data_path not in nltk.data.path:
                     nltk.data.path.append(nltk_data_path) # Ensure path is added if missed

                transformed_sms = transform_text(message)

                if not transformed_sms and message.strip():
                     prediction_text = "Message contains only stopwords or punctuation."
                     is_spam_flag = False # Treat as not spam
                     print(f"Note: Preprocessing resulted in empty string for message: {message}")
                else:
                    vector_input = tfidf.transform([transformed_sms])
                    result = model.predict(vector_input)[0]

                    if result == 1:
                        prediction_text = "This looks like Spam."
                        is_spam_flag = True
                    else:
                        prediction_text = "This looks like a legitimate message."
                        is_spam_flag = False

            except LookupError as le:
                 # Catching the specific error again just in case
                 prediction_text = f"NLTK data missing: {le}. Please ensure deployment downloads data."
                 print(f"LookupError during prediction: {le}")
            except Exception as e:
                prediction_text = f"An error occurred during prediction: {e}"
                print(f"Prediction Error: {e}")

    return render_template('index.html',
                           prediction_text=prediction_text,
                           is_spam=is_spam_flag,
                           message_text=message)

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)