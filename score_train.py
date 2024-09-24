"""
this LSTM model uses the alert prediction dataset.
this model is the preliminary framework for the intelligent agent
it uses date-time as the indexed data
it has 2 layers, dropout and regularization
the model 3D shapes the samples,timestamps and features
this model is trained with early stopping if minimum delta is 0.001
this model has 80-20 training-testing split.
MSE is printed as output for this model
C:/Users/haris/OneDrive/Desktop/NCV/datasets/Alert prediction data.xlsx
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
import tensorflow as tf

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

# Define the target data
train_targets = train_data
val_targets = val_data

# Create the LSTM model
model = Sequential()
model.add(LSTM(units=32, return_sequences=True, input_shape=(1, data.shape[1])))
model.add(LSTM(units=32, dropout=0.2))
model.add(Dense(units=6, activation='linear'))  # Output layer for failure likelihood score

# Define the custom loss function
def custom_loss(y_true, y_pred):
    y_pred = tf.expand_dims(y_pred, axis=-1)  # Add a new axis to the y_pred tensor
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    mae_loss = tf.reduce_mean(tf.abs(y_pred[:, :, -1] - y_true[:, :, -1]))
    return mse_loss + mae_loss

# Compile the model with the custom loss function
model.compile(loss=custom_loss, optimizer='RMSprop')    # RMS prop works properly

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, min_delta=0.001)

# Train the model
model.fit(train_data, train_targets, epochs=25, batch_size=16, validation_data=(val_data, val_targets), callbacks=[early_stopping])

# Predict the failure likelihood scores for the test data
data = data.reshape(-1, 1, data.shape[1] // 1)
test_pred = model.predict(data)

# Rank the items based on their predicted failure likelihood scores
rankings = np.argsort(test_pred[:, 0])

# Calculate the percentage score for each item
percentages = []
for i in range(len(test_pred)):
    score = np.abs(test_pred[i, 0])  # Take the absolute value of the predicted score
    percentage = score / np.max(np.abs(test_pred[:, 0])) *100   # percent to failure
    percentages.append(percentage)

model.save('my_model.keras')
