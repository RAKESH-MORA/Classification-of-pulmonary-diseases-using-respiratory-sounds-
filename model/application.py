import os
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

app = Flask(__name__)
app.secret_key = 'your_secret_key'


# Define your class-to-numeric mapping
class_to_numeric = {'AK-47': 0, 'IMI Desert Eagle': 1, 'AK-12': 2, 'M16': 3, 'M249': 4, 'MG-42': 5, 'MP5': 6, 'Zastava M92': 7}
numeric_to_class = {v: k for k, v in class_to_numeric.items()}

# Define file paths
MODEL_FILE = 'model.pkl'
SCALER_FILE = 'scaler.pkl'

# Function to extract features from audio file
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled
users = {}

@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users and users[username] == password:
            session['username'] = username  # Set the session variable
            return redirect(url_for('index'))
        else:
            return "Login failed. Please check your username and password."

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username not in users:
            users[username] = password
            return redirect(url_for('login'))
        else:
            return "Registration failed. User already exists."

    return render_template('register.html')

@app.route('/index')
def index():
    return render_template('index.html')

# Endpoint to train the model
@app.route('/train', methods=['POST'])
def train():
    x = []
    y = []
    for file_path, class_name in request.json['data']:
        features = extract_features(file_path)
        x.append(features)
        y.append(class_to_numeric[class_name])

    x = np.array(x)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid_re = {
        'C': [1, 10, 100],
        'gamma': [0.1, 0.01, 0.001],
        'decision_function_shape': ['ovo', 'ovr'],
        'kernel': ['rbf', 'linear', 'poly']
    }

    stratified_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_re = GridSearchCV(SVC(), param_grid_re, cv=stratified_k_fold, refit=True, verbose=1)
    grid_re.fit(X_train_scaled, y_train)

    best_params = grid_re.best_params_
    svc = SVC(**best_params, probability=True, verbose=1)
    svc.fit(X_train_scaled, y_train)

    with open(MODEL_FILE, 'wb') as file:
        pickle.dump(svc, file)
    with open(SCALER_FILE, 'wb') as file:
        pickle.dump(scaler, file)

    y_pred = svc.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    return jsonify({'accuracy': accuracy, 'best_params': best_params})

# Endpoint to make predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    features = extract_features(file_path)

    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        return jsonify({'error': 'Model or scaler not found. Please train the model first.'}), 400

    with open(MODEL_FILE, 'rb') as file:
        model = pickle.load(file)
    with open(SCALER_FILE, 'rb') as file:
        scaler = pickle.load(file)

    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    predicted_class = numeric_to_class[prediction[0]]

    return render_template('result.html', prediction=predicted_class)



@app.route('/chart')
def chart():
    return render_template('chart.html')

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
