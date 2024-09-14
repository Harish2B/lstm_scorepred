import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import tensorflow as tf
import keras
# Load the new dataset
new_data = pd.read_excel('output.xlsx')

# Define the scaler object
scaler = MinMaxScaler()
# Select only the numerical columns from the dataset
numerical_cols = new_data.select_dtypes(include=[np.number]).columns
new_data_scaled = scaler.fit_transform(new_data[numerical_cols])

# Reshape the data to have a 3D shape
new_data_reshaped = new_data_scaled.reshape(-1, 1, new_data_scaled.shape[1])


# Load the saved LSTM model
def custom_loss(y_true, y_pred):
    y_pred = tf.expand_dims(y_pred, axis=-1)  # Add a new axis to the y_pred tensor
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    mae_loss = tf.reduce_mean(tf.abs(y_pred[:, :, -1] - y_true[:, :, -1]))
    return mse_loss + mae_loss


try:
    # Load the trained model with custom loss function
    with keras.utils.custom_object_scope({'custom_loss': custom_loss}):
        model = load_model('C:/Users/haris/OneDrive/Desktop/NCV/project code/ml model/my_model.keras')

    # Make predictions using the trained model
    predictions = model.predict(new_data_reshaped)

    # Rank the items based on their predicted failure likelihood scores
    rankings = np.argsort(predictions[:, 0])

    # Calculate the percentage score for each item
    percentages = []
    for i in range(len(predictions)):
        score = np.abs(predictions[i, 0])  # Take the absolute value of the predicted score
        percentage = score / np.max(np.abs(predictions[:, 0])) * 100  # percent to failure
        percentages.append(percentage)

    # Add the percentage scores to the dataset
    new_data['percentage_close_to_failure'] = percentages
    new_data['date_time'] = pd.to_datetime(new_data['date_time'], unit='s')
    new_data.to_excel('output.xlsx', index=False)

    # Print the updated dataset
    print(new_data.head())

except Exception as e:
    print(f"Error: {e}")