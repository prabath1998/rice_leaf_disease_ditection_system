import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def load_dataset(data_dir):
    images = []
    labels = []
    class_names = os.listdir(data_dir)
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (128, 128))  
            images.append(image)
            labels.append(class_name)
    return np.array(images), np.array(labels)

def extract_hog_features(images):
    hog_features = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False)
        hog_features.append(features)
    return np.array(hog_features)

data_dir = 'data/train' 
images, labels = load_dataset(data_dir)

X = extract_hog_features(images)
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("Classification Report:")
print(classification_report(y_test, y_pred))

os.makedirs('models', exist_ok=True)
joblib.dump(clf, 'models/paddy_disease_svm_model.pkl')
print("Model saved as 'models/paddy_disease_svm_model.pkl'")