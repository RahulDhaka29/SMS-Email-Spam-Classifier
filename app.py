from flask import Flask, render_template, request
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# --- Pre-computation and Model Loading ---

# It's good practice to handle potential NLTK download issues.
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading NLTK punkt...")
    nltk.download('punkt')


ps = PorterStemmer()

def transform_text(text):
    """
    Performs text preprocessing:
    1. Lowercasing
    2. Tokenization
    3. Removing special characters
    4. Removing stop words and punctuation
    5. Stemming
    """
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    text = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in text]
    return " ".join(y)

# Load the vectorizer and model
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    print("FATAL ERROR: 'vectorizer.pkl' or 'model.pkl' not found.")
    print("Please run your Jupyter notebook to generate these files.")
    exit()


# --- Flask App Definition ---

app = Flask(__name__)

@app.route('/')
def home():
    """Renders the main page (classifier)."""
    return render_template('index.html')

# --- NEW: Route for the About Page ---
@app.route('/about')
def about():
    """Renders the about page."""
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request."""
    if request.method == 'POST':
        message = request.form.get('message', '')

        if not message.strip():
            return render_template('index.html', 
                                   prediction_text="Please enter a message to classify.", 
                                   is_spam=None, 
                                   message_text=message)

        transformed_sms = transform_text(message)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            prediction = "This looks like Spam."
            is_spam_flag = True
        else:
            prediction = "This looks like a legitimate message."
            is_spam_flag = False
            
        return render_template('index.html', 
                               prediction_text=prediction, 
                               is_spam=is_spam_flag, 
                               message_text=message)

if __name__ == '__main__':
    app.run(debug=True)

