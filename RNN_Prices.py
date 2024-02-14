from queue import Queue
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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

# Find and remove the most recent date in the DataFrame; this is the day we will simulate
most_recent_date = data['Date'].max() - timedelta(days=1)
day_to_predict = most_recent_date.day
month_to_predict = most_recent_date.month
year_to_predict = most_recent_date.year

# Create training and test sets
# Features are Day, Month, Year, Hour and Minute
# Targets are historical price data
raw_np_features = data[data['Date'] < most_recent_date][['Day', 'Month', 'Year', 'DayOfYear', 'Hour', 'Minute']].to_numpy()
raw_np_prices = data[data['Date'] < most_recent_date][['Price']].to_numpy()

# Remove unnecessary columns from the dataframe
data = data.drop(['Date', 'Time'], axis=1)

# Normalize the data to a scale between 0 and 1 (good for neural networks)
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

scaled_features_np = feature_scaler.fit_transform(raw_np_features)
scaled_prices_np = target_scaler.fit_transform(raw_np_prices)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(scaled_features_np, scaled_prices_np, test_size=0.2, random_state=42)

# Split time-series data into individual day segments
len_time_series = 8  # Number of time steps to consider for prediction

# Create empty arrays for sequenced features and prices
sequenced_features_train = []
sequenced_prices_train = []

for i in range(len(X_train) - len_time_series + 1):
    seq_X_train = X_train[i:i+len_time_series]
    seq_y_train = y_train[i:i+len_time_series]

    sequenced_features_train.append(seq_X_train)
    sequenced_prices_train.append(seq_y_train)

sequenced_features_train = np.array(sequenced_features_train)
sequenced_prices_train = np.array(sequenced_prices_train)

# Build the Bi-LSTM model
model = Sequential()
model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(sequenced_features_train.shape[1], sequenced_features_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics='accuracy')

# Train the model
model.fit(sequenced_features_train, sequenced_prices_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
sequenced_features_test = []
sequenced_prices_test = []

for i in range(len(X_test) - len_time_series + 1):
    seq_X_test = X_test[i:i+len_time_series]
    seq_y_test = y_test[i:i+len_time_series]

    sequenced_features_test.append(seq_X_test)
    sequenced_prices_test.append(seq_y_test)

sequenced_features_test = np.array(sequenced_features_test)
sequenced_prices_test = np.array(sequenced_prices_test)

# Evaluate the model on the test set
test_loss = model.evaluate(sequenced_features_test, sequenced_prices_test, batch_size=32)

print(f'Test Loss: {test_loss}')

# Initialize an empty DataFrame to store predictions
predictions_df = pd.DataFrame(columns=['Hour', 'Minute', 'Predicted_Price'])

current_price_sequence = Queue()

# Revisit the previous len_time_series*30 mins and enqueue those prices
for half_hour in range(len_time_series, 0, -1):
    historic_time = most_recent_date - timedelta(minutes=30 * half_hour)

    historic_data = data.loc[
        (data['DayOfYear'] == historic_time.dayofyear) & (data['Day'] == historic_time.day) &
        (data['Month'] == historic_time.month) & (data['Year'] == historic_time.year) &
        (data['Hour'] == historic_time.hour) & (data['Minute'] == historic_time.minute)]

    current_price_sequence.put(historic_data[['Day', 'Month', 'Year', 'DayOfYear', 'Hour', 'Minute']].to_numpy().flatten())

# Simulate the arrival of new prices throughout the day
for hour in range(23):
    for minute in [0, 30]:
        sim_time = most_recent_date + timedelta(hours=hour, minutes=minute)

        current_data = data.loc[
            (data['DayOfYear'] == sim_time.dayofyear) & (data['Day'] == sim_time.day) &
            (data['Month'] == sim_time.month) & (data['Year'] == sim_time.year) &
            (data['Hour'] == sim_time.hour) & (data['Minute'] == sim_time.minute)]

        # Remove oldest price
        current_price_sequence.get()

        # Add the most recent price to the price sequence
        current_price_sequence.put(
            current_data[['Day', 'Month', 'Year', 'DayOfYear', 'Hour', 'Minute']].to_numpy().flatten())

        # Initialize an empty list to store previous prices
        historic_features = []

        # Iterate through the prices queue and append prices to the list
        for item in current_price_sequence.queue:
            historic_features.append(item)

        # Use the list to create a NumPy array
        historic_features_np = np.array(historic_features)

        # Scale prices
        historic_features_scaled_np = feature_scaler.transform(historic_features_np)

        # Predict the next price
        predicted_price_scaled = model.predict(historic_features_scaled_np.reshape(1, len_time_series, -1))

        # Inverse transform the scaled prediction
        predicted_price = target_scaler.inverse_transform(predicted_price_scaled)

        # Append the prediction to the DataFrame
        predictions_df = pd.concat([predictions_df, pd.DataFrame(
            {'Hour': hour, 'Minute': minute, 'Predicted_Price': predicted_price[0]}, index=[0])], ignore_index=True)

# Display predicted prices for each timestamp
print(f'Online predictions:\n {predictions_df}')

# Get the actual prices for comparison
current_data_prices = data.loc[
    (data['DayOfYear'] == most_recent_date.dayofyear) & (data['Day'] == day_to_predict) &
    (data['Month'] == month_to_predict) & (data['Year'] == year_to_predict), ['Day', 'Month', 'Year', 'Hour',
                                                                             'Minute', 'Price']]

print(f'Actual prices:\n {current_data_prices}')

# Plot the predicted prices
plt.figure(figsize=(10, 6))
plt.plot(predictions_df['Hour'] + predictions_df['Minute'] / 60, predictions_df['Predicted_Price'], marker='o',
         linestyle='-', color='b', label='Online-Prediction Prices')
plt.plot(current_data_prices['Hour'] + current_data_prices['Minute'] / 60, current_data_prices['Price'], marker='o',
         linestyle='-', color='g', label='Actual Prices')
plt.title(f'Electricity Price Prediction - {day_to_predict}/{month_to_predict}/{year_to_predict}')
plt.xlabel('Hour')
plt.ylabel('Predicted Price')
plt.xticks(np.arange(0, 24, 1))
plt.legend()
plt.grid(True)
plt.show()
