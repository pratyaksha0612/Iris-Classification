import joblib
import numpy as np

# Load saved model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Sample flower measurements
sample = np.array([[5.1, 3.5, 1.4, 0.2]])

# Scale input
sample_scaled = scaler.transform(sample)

# Predict
prediction = model.predict(sample_scaled)

# Class names
classes = ["Setosa", "Versicolor", "Virginica"]

print("Predicted Flower:", classes[prediction[0]])