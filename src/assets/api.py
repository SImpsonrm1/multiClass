import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, jsonify, request, send_from_directory
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score
from flask_cors import CORS
import os
import io
from contextlib import redirect_stdout

app = Flask(__name__)
CORS(app)

# Global variables or state
model = None

def prob_to_predictions(array):
    max_indices = [np.argmax(row) for row in array]
    predictions = np.array(max_indices)
    return predictions

def initialize_state():
    global model
    model = load_model('nn_model.h5', compile=True)  # Re-initialize the model

@app.route('/run-model', methods=['POST'])
def run_model():
    global model
    file_path = 'FinalData.csv'
    data = pd.read_csv(file_path)

    # Data processing
    x = data.iloc[:, 1:-3]  # Features
    y = data.iloc[:, -3:]   # Target
    y = y.idxmax(axis=1)
    map = {"Classification_Alex": 0, "Classification_Jordan": 1, "Classification_Ryan": 2}
    y = [map.get(item, item) for item in y]
    y = pd.DataFrame(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Predict and calculate accuracy
    x_valid = np.array(x_test)
    y_valid = np.array(y_test).flatten()

    prob_array = [model.predict(np.array([x]), verbose=0) for x in x_valid]  # Set verbose to 0
    y_pred = prob_to_predictions(prob_array)
    accuracy = accuracy_score(y_valid, y_pred)

    # Generate confusion matrix plot
    plt.clf()
    graph = confusion_matrix(y_valid, y_pred)
    sns.heatmap(graph, annot=True, fmt='d', cmap='Greens')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    # Capture formatted console output
    f = io.StringIO()
    with redirect_stdout(f):
        print("Predictions:\n", y_pred)
        print("Accuracy: {:.2f}%".format(accuracy * 100))

    # Save console output and graph
    console_output_filename = 'console_output.txt'
    console_output_path = os.path.join('static', 'generated', console_output_filename)
    with open(console_output_path, 'w') as text_file:
        text_file.write(f.getvalue())

    graph_filename = 'graph_image.png'
    graph_path = os.path.join('static', 'generated', graph_filename)
    plt.savefig(graph_path)

    return jsonify({'console_output_file': console_output_filename, 'graph_file': graph_filename})

@app.route('/reset', methods=['GET'])
def reset_model():
    initialize_state()
    return jsonify({'message': 'API state reset successfully'})

@app.route('/generated/<path:filename>', methods=['GET'])
def generated_files(filename):
    return send_from_directory('static/generated', filename)

# Initialize state when starting the server
initialize_state()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
