from flask import Flask, request, jsonify
import cv2
import numpy as np
from skimage.feature import hog
import joblib

app = Flask(__name__)

clf = joblib.load('models/paddy_disease_svm_model.pkl')

class_labels = ['Bacterial_Leaf_Blight', 'Brown_Spot', 'Leaf_Smut', 'Healthy']

def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False)
    return features

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (128, 128))  
        features = extract_hog_features(image)
        features = features.reshape(1, -1) 

        predicted_class = clf.predict(features)[0]
        confidence = np.max(clf.predict_proba(features))

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': float(confidence)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)