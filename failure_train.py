"""this is the failure training model, this lstm model see if the row is failure or not """
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

def train_lstm_autoencoder(dataframes):
    # Concatenate the DataFrames
    concat_df = pd.concat(dataframes, ignore_index=True)

    # Define the scaler object
    scaler = StandardScaler()

    # Select only the numerical columns from the dataset
    numerical_cols = concat_df.select_dtypes(include=[np.number]).columns
    data = scaler.fit_transform(concat_df[numerical_cols])

    # Reshape the data to have a 3D shape
    data = data.reshape(-1, 1, data.shape[1])

    # Create the LSTM autoencoder model
    model = Sequential()
    model.add(LSTM(units=32, return_sequences=True, input_shape=(1, data.shape[2])))
    model.add(LSTM(units=32, dropout=0.2))
    model.add(Dense(units=data.shape[2]))  # Output layer

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    model.fit(data, data, epochs=25, batch_size=16)

    # Make predictions using the trained model
    reconstructions = model.predict(data)

    # Calculate the reconstruction error (MSE)
    mse = np.mean((reconstructions - data) ** 2, axis=(1, 2))

    # Identify anomalies using Local Outlier Factor (LOF)
    from sklearn.neighbors import LocalOutlierFactor
    lof = LocalOutlierFactor(n_neighbors=10, contamination=0.013)
    anomaly_scores = lof.fit_predict(mse.reshape(-1, 1))

    # Convert anomaly scores to 0/1 labels
    anomaly_labels = np.where(anomaly_scores == -1, 1, 0)

    # Add the anomaly labels to the concatenated dataset
    concat_df['failure'] = anomaly_labels

    # Split the data into training and validation sets
    from sklearn.model_selection import train_test_split
    train_data, val_data, train_labels, val_labels = train_test_split(data, anomaly_labels, test_size=0.2,
                                                                      random_state=42)

    # Create a new model to classify failures
    failure_model = Sequential()
    failure_model.add(LSTM(units=32, return_sequences=True, input_shape=(1, data.shape[2])))
    failure_model.add(LSTM(units=32, dropout=0.2))
    failure_model.add(Dense(units=1, activation='sigmoid'))  # Output layer

    # Compile the model
    failure_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    failure_model.fit(train_data, train_labels, epochs=25, batch_size=16, validation_data=(val_data, val_labels))

    # Save the trained model
    failure_model.save('my_model_failure_classifier.keras')

    return failure_model


# Example usage:
dataframes = [
    pd.read_excel('C:/Users/haris/OneDrive/Desktop/NCV/datasets/Alert prediction data_frem.xlsx'),
    # Add more DataFrames to the list as needed
]

model = train_lstm_autoencoder(dataframes)
