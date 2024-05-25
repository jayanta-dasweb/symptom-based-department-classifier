from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Global variables to store the pipeline and the accuracy
pipeline = None
accuracy = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    global pipeline, accuracy
    try:
        # Expanded dataset for better results
        data = {
            'symptoms': [
                'chest pain, difficulty breathing', 
                'persistent cough, coughing up blood', 
                'frequent urination, excessive thirst', 
                'rash, itchy skin', 
                'headache, dizziness', 
                'stomach pain, diarrhea',
                'joint pain, muscle weakness',
                'fever, chills, body ache',
                'sore throat, swollen glands',
                'nausea, vomiting, loss of appetite',
                'weight gain, fatigue, depression',
                'shortness of breath, chest tightness',
                'skin rash, itchy eyes',
                'heart palpitations, sweating',
                'abdominal pain, bloating',
                'blurred vision, eye pain',
                'hearing loss, ear pain',
                'back pain, leg pain',
                'burning sensation during urination',
                'chest pain, nausea, dizziness',
                'fatigue, muscle aches',
                'coughing, shortness of breath',
                'skin lesions, bruising',
                'frequent headaches, blurred vision',
                'abdominal cramps, diarrhea',
                'swollen joints, morning stiffness',
                'fever, night sweats',
                'difficulty swallowing, hoarse voice',
                'vomiting, dehydration',
                'rapid weight loss, fatigue',
                'chronic cough, chest infections'
            ],
            'department': [
                'Cardiology', 
                'Pulmonology', 
                'Endocrinology', 
                'Dermatology', 
                'Neurology', 
                'Gastroenterology',
                'Rheumatology',
                'Infectious Diseases',
                'Otolaryngology',
                'Gastroenterology',
                'Endocrinology',
                'Pulmonology',
                'Dermatology',
                'Cardiology',
                'Gastroenterology',
                'Ophthalmology',
                'Otolaryngology',
                'Orthopedics',
                'Urology',
                'Cardiology',
                'Internal Medicine',
                'Pulmonology',
                'Dermatology',
                'Neurology',
                'Gastroenterology',
                'Rheumatology',
                'Infectious Diseases',
                'Otolaryngology',
                'Gastroenterology',
                'Endocrinology',
                'Pulmonology'
            ]
        }

        df = pd.DataFrame(data)

        # Text preprocessing and model training pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(df['symptoms'], df['department'], test_size=0.3, random_state=42)

        # Train the model
        pipeline.fit(X_train, y_train)

        # Predict on test data and calculate accuracy
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred) * 100

        logging.info(f'Model trained with accuracy: {accuracy}%')
        return jsonify({'accuracy': accuracy})

    except Exception as e:
        logging.error(f'Error during training: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    global pipeline
    if pipeline is None:
        return jsonify({'error': 'Model is not trained yet.'})

    symptoms = request.json.get('symptoms')
    if not symptoms:
        return jsonify({'error': 'No symptoms provided.'})

    try:
        department = pipeline.predict([symptoms])[0]
        return jsonify({'department': department})
    except Exception as e:
        logging.error(f'Error during prediction: {str(e)}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
