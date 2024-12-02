import pandas as pd
import joblib

# Load your CSV data
data = pd.read_csv('ksl_keypoints.csv')

# Step 1: Generate Label Mapping
unique_labels = data['label'].unique()
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
print("Label Mapping:", label_mapping)

# Step 2: Encode Labels
data['encoded_label'] = data['label'].map(label_mapping)
print(data[['label', 'encoded_label']].head())

# Step 3: Create Reverse Mapping
reverse_mapping = {idx: label for label, idx in label_mapping.items()}
print("Reverse Mapping:", reverse_mapping)

# Step 4: Save the Mapping (Optional)
joblib.dump(label_mapping, 'label_mapping.pkl')
print("Label mapping saved as 'label_mapping.pkl'.")

# Example: Load the Mapping Later
loaded_mapping = joblib.load('label_mapping.pkl')
print("Loaded Mapping:", loaded_mapping)

# Example Usage for Predictions
def get_predicted_label(predicted_idx):
    # Convert numeric prediction back to the label
    return reverse_mapping.get(predicted_idx, "Unknown Label")

# Dummy Prediction Example
predicted_label_idx = 2  # Example numeric prediction from your model
predicted_label = get_predicted_label(predicted_label_idx)
print("Predicted Gesture:", predicted_label)

# Save the preprocessed CSV with encoded labels
data.to_csv('ksl_keypoints_encoded.csv', index=False)
print("Preprocessed data saved to 'ksl_keypoints_encoded.csv'.")
