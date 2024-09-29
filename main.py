from flask import Flask, render_template, request
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the model and TF-IDF vectorizer
with open('model (1).pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

# Preprocessing function
def preprocess_review(review):
    review = review.lower()
    review = re.sub(r'\W', ' ', review)
    tokens = review.split()
    stop_words = set([
        'the', 'and', 'is', 'in', 'it', 'of', 'to', 'a', 'was', 'with', 'as', 'for', 'on', 'that', 'by', 'this', 'at', 'an', 
        'be', 'are', 'or', 'not', 'from', 'but', 'if', 'you', 'they', 'can', 'we', 'will', 'all', 'would', 'there', 'their'
    ])
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling the prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the review from the form
        user_review = request.form['review']

        # Preprocess the review
        cleaned_review = preprocess_review(user_review)

        # Transform the review into TF-IDF features
        review_tfidf = tfidf.transform([cleaned_review])

        # Make a prediction
        predicted_rating = model.predict(review_tfidf)

        # Return the result
        return render_template('index.html', prediction=f"Predicted Rating: {predicted_rating[0]}")

if __name__ == '__main__':
    app.run(debug=True)
