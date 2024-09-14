import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping

# Load the dataset
df = pd.read_excel('C:/Users/haris/OneDrive/Desktop/NCV/datasets/Alert prediction data.xlsx')

# Define the scaler object
scaler = MinMaxScaler()

# Select only the numerical columns from the dataset
numerical_cols = df.select_dtypes(include=[np.number]).columns
data = scaler.fit_transform(df[numerical_cols])

# Split the data into training, validation, and testing sets
train_size = int(0.8 * len(data))
val_size = int(0.1 * len(data))
train_data, val_data, test_data = data[0:train_size], data[train_size:train_size+val_size], data[train_size+val_size:]

# Reshape the data to have a 3D shape
train_data = train_data.reshape(-1, 1, data.shape[1])
val_data = val_data.reshape(-1, 1, data.shape[1])
test_data = test_data.reshape(-1, 1, data.shape[1])

# Define the target data ( failures )
train_targets = df['failure'].values[:train_size]
val_targets = df['failure'].values[train_size:train_size+val_size]

# Create the LSTM model
model = Sequential()
model.add(LSTM(units=32, return_sequences=True, input_shape=(1, data.shape[1])))
model.add(LSTM(units=32, dropout=0.2))
model.add(Dense(units=1, activation='sigmoid'))  # Output layer for failure prediction (0 or 1)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam')

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, min_delta=0.001)

# Train the model
model.fit(train_data, train_targets, epochs=25, batch_size=16, validation_data=(val_data, val_targets), callbacks=[early_stopping])

# Predict the failure probabilities for the entire dataset
data = data.reshape(-1, 1, data.shape[1])
pred_probs = model.predict(data)

# Predict the failure probabilities for the test data
test_pred = model.predict(data)

# Convert the probabilities to binary values (0 or 1)
test_pred_class = np.where(test_pred > 0.5, 1, 0)

model.save('my_model_failure.keras')
