from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Global variables to store the pipeline and the accuracy
pipeline = None
accuracy = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    global pipeline, accuracy

    # Mock dataset
    data = {
        'symptoms': [
            'chest pain, difficulty breathing', 
            'persistent cough, coughing up blood', 
            'frequent urination, excessive thirst', 
            'rash, itchy skin', 
            'headache, dizziness', 
            'stomach pain, diarrhea',
            'joint pain, muscle weakness'
        ],
        'department': [
            'Cardiology', 
            'Pulmonology', 
            'Endocrinology', 
            'Dermatology', 
            'Neurology', 
            'Gastroenterology',
            'Rheumatology'
        ]
    }

    df = pd.DataFrame(data)

    # Text preprocessing and model training pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('classifier', LogisticRegression())
    ])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(df['symptoms'], df['department'], test_size=0.3, random_state=42)

    # Train the model
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    
    return jsonify({'accuracy': accuracy * 100})

@app.route('/predict', methods=['POST'])
def predict():
    global pipeline
    if pipeline is None:
        return jsonify({'error': 'Model is not trained yet.'})

    symptoms = request.json.get('symptoms')
    if not symptoms:
        return jsonify({'error': 'No symptoms provided.'})

    department = pipeline.predict([symptoms])[0]
    return jsonify({'department': department})

if __name__ == '__main__':
    app.run(debug=True)
