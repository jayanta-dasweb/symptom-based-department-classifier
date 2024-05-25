from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import MySQLdb
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# MySQL configurations
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'
app.config['MYSQL_DB'] = 'medical_data'

# Initialize MySQL
db = MySQLdb.connect(
    host=app.config['MYSQL_HOST'],
    user=app.config['MYSQL_USER'],
    passwd=app.config['MYSQL_PASSWORD'],
    db=app.config['MYSQL_DB']
)

pipeline = None
accuracy = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Read the uploaded file into a DataFrame
    if file and (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        # Clean the data (example: only keep 'symptoms' and 'department' columns)
        if 'symptoms' in df.columns and 'department' in df.columns:
            df = df[['symptoms', 'department']]
        else:
            return jsonify({'error': 'Required columns not found in the file'})

        # Insert the cleaned data into the MySQL table
        cursor = db.cursor()
        for index, row in df.iterrows():
            cursor.execute("INSERT INTO symptoms_data (symptoms, department) VALUES (%s, %s)", (row['symptoms'], row['department']))
        db.commit()
        cursor.close()

        return jsonify({'success': 'File uploaded and data inserted successfully'})
    else:
        return jsonify({'error': 'Invalid file format. Please upload a .csv or .xlsx file'})

@app.route('/train', methods=['POST'])
def train():
    global pipeline, accuracy
    try:
        # Fetch data from the database
        cursor = db.cursor()
        cursor.execute("SELECT symptoms, department FROM symptoms_data")
        data = cursor.fetchall()
        cursor.close()

        if len(data) == 0:
            return jsonify({'error': 'No data available to train the model'})

        # Create DataFrame from the fetched data
        df = pd.DataFrame(data, columns=['symptoms', 'department'])

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
