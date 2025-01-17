from flask import Flask, jsonify, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the Breast Cancer Dataset
breast_cancer_dataset = load_breast_cancer()
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
data_frame['label'] = breast_cancer_dataset.target

@app.route('/')
def home():
    return render_template('index.html')  # Ensure index.html is in the 'templates' folder

# Endpoint to handle file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read the uploaded CSV file into a DataFrame
        df = pd.read_csv(file)
        shape = df.shape
        missing_values = df.isnull().sum().to_dict()
        head = df.head().to_dict(orient='records')

        return jsonify({
            "shape": shape,
            "missing_values": missing_values,
            "head": head
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint to train the model
@app.route('/train', methods=['POST'])
def train_model():
    X = data_frame.drop(columns='label', axis=1)
    Y = data_frame['label']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    return jsonify({
        "message": "Model trained successfully",
        "accuracy": accuracy
    })

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
