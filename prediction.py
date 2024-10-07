"""This model is the prediction for the new dataset. the two model trained are failure and scores.
the model first predicts the failure and uses the updated dataset to score the failure with the custom loss formula"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import load_model
import tensorflow as tf
import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

# Failure Prediction
# Load the saved model
model_failure = load_model('my_model_failure.keras')

# Load the new dataset to predict
new_df = pd.read_excel('C:/Users/haris/OneDrive/Desktop/NCV/project code/ml model/output.xlsx')

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

# Identify anomalies using Local Outlier Factor (LOF)
lof = LocalOutlierFactor(n_neighbors=10, contamination=0.013)
anomaly_scores = lof.fit_predict(mse.reshape(-1, 1))

# Convert anomaly scores to 0/1 labels
anomaly_labels = np.where(anomaly_scores == -1, 1, 0)

# Add the anomaly labels to the new dataset
new_df['failure'] = anomaly_labels  # Create a new column 'anomaly' with anomaly labels

# Evaluate the failure prediction model
y_true = new_df['failure']
y_pred = anomaly_labels
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
print("Failure Prediction Model Evaluation:")
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-score: {f1:.3f}")

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
        model_score = load_model(
            'C:/Users/haris/OneDrive/Desktop/NCV/project code/ml model/lstm_scorepred/my_model.keras')

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
    #
    new_df.to_excel('output_data.xlsx', index=False)

    # Evaluate the score prediction model
    y_true = new_df['percentage_close_to_failure']
    y_pred = percentages
    mae = mean_absolute_error(y_true, y_pred)
    print("Score Prediction Model Evaluation:")
    print(f"Mean Absolute Error (MAE): {mae:.3f}")

    # Visualize the results
    plt.figure(figsize=(12, 8))
    plt.plot(new_df['percentage_close_to_failure'], new_df['Time stamp'])
    plt.xlabel('Date Time')
    plt.ylabel('Failure')
    plt.title('Predicted Failure Likelihood Scores')
    plt.show()

except Exception as e:
    print(f"Error: {e}")
