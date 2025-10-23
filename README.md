# 📧 Email/SMS Spam Classifier

**Email/SMS Spam Classifier** is a machine learning project built with **Python, Flask, Scikit-learn**, and **NLTK**.  
It classifies text messages (SMS or email) as **"Spam"** or **"Ham"** (Not Spam).

---

## ✨ Features

- 📨 **Spam Detection:**  
  Uses a **Multinomial Naive Bayes** model to predict whether a message is spam or not.

- 📝 **NLP Preprocessing:**  
  Performs lowercasing, tokenization, removing punctuation and special characters, stop word removal, and stemming using **NLTK**.

- 🔢 **Feature Extraction:**  
  Converts text messages into numerical vectors using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

- 💻 **Web Interface:**  
  Clean and responsive UI built with **Flask** and **Tailwind CSS**, allowing users to input messages and get instant predictions.

---

## 📁 Project Structure

```
/spam-classifier/
|-- app.py # Main Flask application
|-- vectorizer.pkl # Saved TF-IDF vectorizer
|-- model.pkl # Saved trained Naive Bayes model
|-- sms-spam-detection.ipynb # Jupyter notebook for model building & EDA
|-- spam.csv # Original dataset
|-- README.md # This file
|-- requirements.txt # Python dependencies
|-- /templates/
| |-- index.html # Main UI page
| |-- about.html # About page template
```

---

## Dataset 📊

This project uses the SMS Spam Collection dataset.

**Source:** [Kaggle SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

**Required Files:**  
- `spam.csv` – dataset file containing messages and labels.

---

## ⚙️ Setup and Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/spam-classifier.git
cd spam-classifier

> Replace 'your-username' with your GitHub username.

---

###2️⃣ Create and Activate a Virtual Environment
```bash
python -m venv venv
```


#### On Windows:
```bash
.env\Scriptsctivate
```

#### On macOS/Linux:
```bash
source venv/bin/activate
```
---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Download NLTK Data

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## 🚀 How to Run

1. Ensure vectorizer.pkl and model.pkl are present in the root directory.
If not, run the sms-spam-detection.ipynb notebook to generate them.

2. Activate your virtual environment.

3. Run the Flask app:
   ```bash
   python app.py
   ```
4. Open your browser and visit:  
   👉 [http://127.0.0.1:5000](http://127.0.0.1:5000)
   
---

## 🧠 Model Building Process (Summary)

The **sms-spam-detection.ipynb** notebook details the model creation:

1. **Data Loading & Cleaning:**
    Loaded the SMS dataset and cleaned text messages.

2. **Text Preprocessing:**
    Applied NLP techniques using NLTK: lowercasing, tokenization, removing punctuation and stopwords, and stemming.

3. **Feature Extraction:**
    Converted text into numerical vectors using TF-IDF Vectorizer.

4. **Model Training:**
    Trained a Multinomial Naive Bayes classifier on the training data.

5. **Evaluation:**
    Evaluated model using accuracy, precision, recall, and confusion matrix.

6. **Model Export:**
    Saved trained TF-IDF Vectorizer and Naive Bayes model as .pkl files.

---

## 💻 Technologies Used

| Technology | Purpose |
|-------------|----------|
| **Python** | Core programming language |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computations |
| **Scikit-learn** | Feature scaling (`MinMaxScaler`), similarity calculation (`cosine_similarity`) |
| **Flask** | Backend web framework |
| **Spotipy** | Spotify API integration |
| **HTML** | Frontend structure |
| **Tailwind CSS** | Modern and responsive UI styling |
| **Jupyter Notebook** | Model development, EDA, and experimentation |
| **Git & GitHub** | Version control and hosting |

---

## 👨‍💻 Author

Project created by **[Rahul Dhaka]**  
[LinkedIn](https://www.linkedin.com/in/rahul-dhaka-56b975289/),  [GitHub](https://github.com/RahulDhaka29)

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).