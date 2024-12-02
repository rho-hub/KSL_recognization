import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import joblib

# Load your CSV data
data = pd.read_csv('ksl_keypoints.csv')

# Separate features and labels
X = data.drop(columns=['label']).values
y = data['label'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Reshape features for LSTM
timesteps = 1  # Static gestures (single frame per sample)
X_train_reshaped = X_train.reshape((X_train.shape[0], timesteps, X_train.shape[1]))
X_test_reshaped = X_test.reshape((X_test.shape[0], timesteps, X_test.shape[1]))

# One-hot encode labels
unique_labels = np.unique(y_train)
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
y_train_encoded = np.array([label_mapping[label] for label in y_train])
y_test_encoded = np.array([label_mapping[label] for label in y_test])

y_train_one_hot = to_categorical(y_train_encoded)
y_test_one_hot = to_categorical(y_test_encoded)

# Save preprocessed data
joblib.dump((X_train_reshaped, X_test_reshaped, y_train_one_hot, y_test_one_hot), 'preprocessed_data.pkl')

# Print shapes
print(f"Training Features Shape: {X_train_reshaped.shape}")
print(f"Training Labels Shape: {y_train_one_hot.shape}")
print(f"Test Features Shape: {X_test_reshaped.shape}")
print(f"Test Labels Shape: {y_test_one_hot.shape}")