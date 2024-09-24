"""this is the failure training model, this lstm model see if the row is failure or not """
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
# Load the dataset
df = pd.read_excel('C:/Users/haris/OneDrive/Desktop/NCV/datasets/Alert prediction data_frem.xlsx')

# Define the scaler object
scaler = StandardScaler()

# Select only the numerical columns from the dataset
numerical_cols = df.select_dtypes(include=[np.number]).columns
data = scaler.fit_transform(df[numerical_cols])

# Reshape the data to have a 3D shape
data = data.reshape(-1, 1, data.shape[1])

# Create the LSTM autoencoder model
model = Sequential()
model.add(LSTM(units=32, return_sequences=True, input_shape=(1, data.shape[2])))
model.add(LSTM(units=32, dropout=0.2))
model.add(Dense(units=data.shape[2]))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(data, data, epochs=25, batch_size=16)

# Save the trained model
model.save('my_model_failure.keras')
