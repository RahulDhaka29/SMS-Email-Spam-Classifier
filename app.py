from flask import Flask, render_template, request
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd # Needed if you load data structures
import numpy as np # Needed if you load data structures
# --- Pre-computation and Model Loading ---

app = Flask(__name__)

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

# Make sure stopwords are accessible after build command downloads them
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    print("NLTK stopwords not found. Make sure build command includes nltk.download('stopwords')")
    stop_words = set() # Use empty set as fallback

def transform_text(text):
    text = text.lower()
    # Ensure punkt is available for tokenization
    try:
        text = nltk.word_tokenize(text)
    except LookupError:
        print("NLTK punkt tokenizer not found. Make sure build command includes nltk.download('punkt')")
        return "" # Return empty string or handle error appropriately

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
    message = request.form.get('message', '') # Use .get for safety

    if request.method == 'POST':
        if not message.strip():
            prediction_text = "Please enter a message to classify."
        else:
            try:
                transformed_sms = transform_text(message)
                # Check if transformation returned empty due to missing NLTK data
                if not transformed_sms and message.strip():
                    raise ValueError("Text preprocessing failed, likely due to missing NLTK data.")

                vector_input = tfidf.transform([transformed_sms])
                result = model.predict(vector_input)[0]

                if result == 1:
                    prediction_text = "This looks like Spam."
                    is_spam_flag = True
                else:
                    prediction_text = "This looks like a legitimate message."
                    is_spam_flag = False
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