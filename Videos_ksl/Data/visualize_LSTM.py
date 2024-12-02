import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Load your CSV data
data = pd.read_csv('ksl_keypoints.csv')

# Separate features and labels
X = data.drop(columns=['label']).values
y = data['label'].values

# Encode string labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# Reshape features for LSTM
timesteps = 1  # Static gestures (single frame per sample)
X_train_reshaped = X_train.reshape((X_train.shape[0], timesteps, X_train.shape[1]))
X_test_reshaped = X_test.reshape((X_test.shape[0], timesteps, X_test.shape[1]))

# One-hot encode labels
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

# Print shapes
print(f"Training Features Shape: {X_train_reshaped.shape}")
print(f"Training Labels Shape: {y_train_one_hot.shape}")
print(f"Test Features Shape: {X_test_reshaped.shape}")
print(f"Test Labels Shape: {y_test_one_hot.shape}")
