import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense
import matplotlib.pyplot as plt

# Load data from CSV into Pandas DataFrame
data = pd.read_csv('csv_wholesale_X_X.csv')

# Extract features
data['Date'] = pd.to_datetime(data['Date'])
data['Day'] = data['Date'].dt.day
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
data['DayOfYear'] = data['Date'].dt.dayofyear
data['Time'] = pd.to_datetime(data['Time'], format='%H:%M').dt.time
data['Hour'] = data['Time'].apply(lambda x: x.hour)
data['Minute'] = data['Time'].apply(lambda x: x.minute)

# Find the most recent date in the DataFrame, this is the day we will simulate
most_recent_date = data['Date'].max()
day_to_predict = most_recent_date.day
month_to_predict = most_recent_date.month
year_to_predict = most_recent_date.year

# Create training set using data up to the most recent date
X = data[data['Date'] < most_recent_date][['Day', 'Month', 'Year', 'DayOfYear', 'Hour', 'Minute', 'Price']].values

# Normalize the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Prepare sequences for time series prediction
sequence_length = 24  # Number of time steps to consider for prediction
X_sequences, y = [], []
for i in range(len(X_scaled) - sequence_length):
    X_sequences.append(X_scaled[i:i + sequence_length, :-1])  # Exclude the last column (Price) from features
    y.append(X_scaled[i + sequence_length, -1])  # Use the last column (Price) as the target variable

X_sequences, y = np.array(X_sequences), np.array(y)

# Build the Bi-LSTM model
model = Sequential()
model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(X_sequences.shape[1], X_sequences.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_sequences, y, epochs=5, batch_size=32, validation_split=0.2)

# Initialize an empty DataFrame to store predictions
predictions_df = pd.DataFrame(columns=['Hour', 'Minute', 'Predicted_Price'])

# Simulate the arrival of new prices throughout the day
for hour in range(22):
    for minute in [0, 30]:
        current_data = data.loc[(data['DayOfYear'] == most_recent_date.dayofyear) & (data['Day'] == day_to_predict) &
                                (data['Month'] == month_to_predict) & (data['Year'] == year_to_predict) &
                                (data['Hour'] == hour) & (data['Minute'] == minute)]

        current_data_features = current_data[['Day', 'Month', 'Year', 'DayOfYear', 'Hour', 'Minute']].values.reshape(1,
                                                                                                                     sequence_length,
                                                                                                                     -1)
        current_data_features_scaled = scaler.transform(current_data_features[0])

        # Predict the next price
        predicted_price_scaled = model.predict(current_data_features_scaled)

        # Inverse transform the scaled prediction
        predicted_price = scaler.inverse_transform(
            np.concatenate((current_data_features_scaled[:, :, :-1], predicted_price_scaled), axis=2))[0, -1]

        # Append the prediction to the DataFrame
        predictions_df = pd.concat(
            [predictions_df, pd.DataFrame({'Hour': hour, 'Minute': minute, 'Predicted_Price': predicted_price})],
            ignore_index=True)

# Display predicted prices for each timestamp
print(f'Online predictions:\n {predictions_df}')

# Get the batch predictions using prediction_parameters
batch_predicted_prices = model.predict(
    prediction_parameters[['Day', 'Month', 'Year', 'DayOfYear', 'Hour', 'Minute']].values.reshape(1, sequence_length,
                                                                                                  -1))
batch_predicted_prices_scaled = np.concatenate((prediction_parameters[['Day', 'Month', 'Year', 'DayOfYear', 'Hour',
                                                                       'Minute']].values.reshape(1, sequence_length,
                                                                                                 -1),
                                                batch_predicted_prices), axis=2)
batch_predicted_prices_df = pd.DataFrame(scaler.inverse_transform(batch_predicted_prices_scaled)[0, :, -1],
                                         columns=['Predicted_Price'])

print(f'Batch Predicted Prices:\n {batch_predicted_prices_df}')

# Get the actual prices for comparison
current_data_prices = data.loc[(data['DayOfYear'] == most_recent_date.dayofyear) & (data['Day'] == day_to_predict) &
                               (data['Month'] == month_to_predict) & (data['Year'] == year_to_predict), ['Day', 'Month',
                                                                                                         'Year', 'Hour',
                                                                                                         'Minute',
                                                                                                         'Price']]

print(f'Actual prices:\n {current_data_prices}')

# Plot the predicted prices
plt.figure(figsize=(10, 6))
plt.plot(predictions_df['Hour'] + predictions_df['Minute'] / 60, predictions_df['Predicted_Price'], marker='o',
         linestyle='-', color='b', label='Online-Prediction Prices')
plt.plot(current_data_prices['Hour'] + current_data_prices['Minute'] / 60, current_data_prices['Price'], marker='o',
         linestyle='-', color='g', label='Actual Prices')
plt.plot(batch_predicted_prices_df.index, batch_predicted_prices_df['Predicted_Price'], marker='o', linestyle='-',
         color='r', label='Batch Prices')
plt.title(f'Electricity Price Prediction - {day_to_predict}/{month_to_predict}/{year_to_predict}')
plt.xlabel('Hour')
plt.ylabel('Predicted Price')
plt.xticks(np.arange(0, 24, 1))
plt.legend()
plt.grid(True)
plt.show()
