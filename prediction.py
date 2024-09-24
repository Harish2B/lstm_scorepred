"""this model is the prediction of the training done. first it finds the failure as 0 or 1 and creates a new column failure.
then using the failure marked dataset, the score is predicted using a custom loss function which is mse+mae"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import load_model
import tensorflow as tf
import keras

# Failure Prediction
# Load the saved model
model_failure = load_model('my_model_failure.keras')

# Load the new dataset to predict
new_df = pd.read_excel('C:/Users/haris/OneDrive/Desktop/NCV/project code/ml model/lstm_scorepred/timestamp_data.xlsx')

# Select only the numerical columns from the new dataset
numerical_cols = new_df.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
new_data = scaler.fit_transform(new_df[numerical_cols])

# Reshape the new data to have a 3D shape
new_data = new_data.reshape(-1, 1, new_data.shape[1])

# Make predictions using the trained model
reconstructions = model_failure.predict(new_data)

# Calculate the reconstruction error (MSE)
mse = np.mean((reconstructions - new_data) ** 2, axis=(1, 2))

# Identify anomalies based on a threshold (e.g., 3 standard deviations)
threshold = 3
anomaly_scores = np.where(mse > (np.mean(mse) + threshold * np.std(mse)), 1, 0)

# Add the anomaly scores to the new dataset
new_df['failure'] = anomaly_scores  # Create a new column 'anomaly' with anomaly scores

# Score Prediction
# Define the scaler object
scaler = MinMaxScaler()

# Select only the numerical columns from the dataset
numerical_cols = new_df.select_dtypes(include=[np.number]).columns
new_data_scaled = scaler.fit_transform(new_df[numerical_cols])

# Reshape the data to have a 3D shape
new_data_reshaped = new_data_scaled.reshape(-1, 1, new_data_scaled.shape[1])

# Load the saved LSTM model with custom loss function
def custom_loss(y_true, y_pred):
    y_pred = tf.expand_dims(y_pred, axis=-1)  # Add a new axis to the y_pred tensor
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    mae_loss = tf.reduce_mean(tf.abs(y_pred[:, :, -1] - y_true[:, :, -1]))
    return mse_loss + mae_loss

try:
    # Load the trained model with custom loss function
    with keras.utils.custom_object_scope({'custom_loss': custom_loss}):
        model_score = load_model('C:/Users/haris/OneDrive/Desktop/NCV/project code/ml model/lstm_scorepred/my_model.keras')

    # Make predictions using the trained model
    predictions = model_score.predict(new_data_reshaped)

    # Rank the items based on their predicted failure likelihood scores
    rankings = np.argsort(predictions[:, 0])

    # Calculate the percentage score for each item
    percentages = []
    for i in range(len(predictions)):
        score = np.abs(predictions[i, 0])  # Take the absolute value of the predicted score
        percentage = score / np.max(np.abs(predictions[:, 0])) * 100  # percent to failure
        percentages.append(percentage)

    # Add the percentage scores to the dataset
    new_df['percentage_close_to_failure'] = percentages
    new_df['date_time'] = pd.to_datetime(new_df['date_time'])
    new_df.to_excel('output_data.xlsx', index=False)

    # Print the updated dataset
    print(new_df.head())

except Exception as e:
    print(f"Error: {e}")
