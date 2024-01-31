import numpy as np
np.float = float
import pandas as pd
import joblib
from skmultiflow.trees import HoeffdingTreeRegressor
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
X = data[data['Date'] < most_recent_date][['Day', 'Month', 'Year', 'DayOfYear', 'Hour', 'Minute']].values
y = data[data['Date'] < most_recent_date]['Price'].values

# Drop unnecessary date and time columns
data = data.drop(['Date', 'Time'], axis=1)

# Display the first few rows of the DataFrame
print(data.head())

# Initialize the Gradient Boosting Regressor with your desired parameters
model = HoeffdingTreeRegressor()
y = np.array(y)

# Train the model on the initial training set
model.fit(X, y)

# Initialize an empty DataFrame to store predictions
predictions_df = pd.DataFrame(columns=['Hour', 'Minute', 'Predicted_Price'])

prediction_parameters = pd.DataFrame({
    'Day': [day_to_predict] * 48,
    'Month': [month_to_predict] * 48,
    'Year': [year_to_predict] * 48,
    'DayOfYear': [most_recent_date.dayofyear] * 48,
    'Hour': [i for i in range(24) for _ in range(2)],
    'Minute': [0, 30] * 24
})

batch_predicted_prices = model.predict(prediction_parameters[['Day', 'Month', 'Year', 'DayOfYear', 'Hour', 'Minute']].to_numpy())
batch_predicted_prices_df = pd.DataFrame(prediction_parameters.copy())
batch_predicted_prices_df['Predicted_Price'] = batch_predicted_prices

print(f'Batch Predicted Prices:\n {batch_predicted_prices}')
# Simulate the arrival of new prices throughout the day

for hour in range(22):
    for minute in [0, 30]:
        current_price = data.loc[(data['DayOfYear'] == most_recent_date.dayofyear) & (data['Day'] == day_to_predict) & (
                    data['Month'] == month_to_predict) &
                                 (data['Year'] == year_to_predict) & (data['Hour'] == hour) & (
                                             data['Minute'] == minute), 'Price'].values[0]

        model.partial_fit(
            np.array([day_to_predict,
                      month_to_predict,
                      year_to_predict,
                      most_recent_date.dayofyear,
                      hour,
                      minute]
                    ).reshape(1,6),
            np.array(current_price).reshape(1,1)
        )

        # Predict the next price
        predicted_price = model.predict([[day_to_predict, month_to_predict, year_to_predict, most_recent_date.timetuple().tm_yday, hour, minute]])[0]

        # Append the prediction to the DataFrame
        predictions_df = pd.concat([predictions_df, pd.DataFrame({'Hour': hour, 'Minute': minute, 'Predicted_Price': predicted_price})], ignore_index=True)

# Display predicted prices for each timestamp
print(f'Online predictions:\n {predictions_df}')

current_data_prices = data.loc[(data['DayOfYear'] == most_recent_date.dayofyear) & (data['Day'] == day_to_predict) & (
                    data['Month'] == month_to_predict) &
                                 (data['Year'] == year_to_predict), ['Day', 'Month', 'Year', 'Hour', 'Minute', 'Price']]

print(f'Actual prices:\n {current_data_prices}')
# Plot the predicted prices
plt.figure(figsize=(10, 6))
plt.plot(predictions_df['Hour'] + predictions_df['Minute'] / 60, predictions_df['Predicted_Price'], marker='o', linestyle='-', color='b', label='Predicted Prices')
plt.plot(current_data_prices['Hour'] + current_data_prices['Minute'] / 60, current_data_prices['Price'], marker='o', linestyle='-', color='g', label='Actual Prices')
plt.plot(batch_predicted_prices_df['Hour'] + batch_predicted_prices_df['Minute'] / 60, batch_predicted_prices_df['Predicted_Price'], marker='o', linestyle='-', color='r', label='Batch Prices')
plt.title(f'Electricity Price Prediction - {day_to_predict}/{month_to_predict}/{year_to_predict}')
plt.xlabel('Hour')
plt.ylabel('Predicted Price')
plt.xticks(np.arange(0, 24, 1))
plt.legend()
plt.grid(True)
plt.show()
