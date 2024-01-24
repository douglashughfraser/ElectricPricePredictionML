import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np

# Load data from CSV into Pandas DataFrame
data = pd.read_csv('csv_wholesale_X_X.csv')

# Extract features
data['Date'] = pd.to_datetime(data['Date'])
data['Day'] = data['Date'].dt.day
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
data['Time'] = pd.to_datetime(data['Time'], format='%H:%M').dt.time
data['Hour'] = data['Time'].apply(lambda x: x.hour)
data['Minute'] = data['Time'].apply(lambda x: x.minute)

# Drop unnecessary date and time columns
data = data.drop(['Date', 'Time'], axis=1)

# Display the first few rows of the DataFrame
print(data.head())

# Create training set
X = data[['Day', 'Month', 'Year', 'Hour', 'Minute']]
y = data['Price']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Create a pipeline with Polynomial Regression
degree = 8  # You can adjust the degree of the polynomial
alpha = 0.4  # Regularization strength, you can adjust this
model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=alpha))

# Train the model on the training set
model.fit(X_train, y_train)

# Choose a specific day for prediction (e.g., 22/01)
day_to_predict = 23
month_to_predict = 6
year_to_predict = 2024

# Generate new data for predictions on the chosen day
new_data = pd.DataFrame({
    'Day': [day_to_predict] * 48,
    'Month': [month_to_predict] * 48,
    'Year': [year_to_predict] * 48,
    'Hour': [i for i in range(24) for _ in range(2)],
    'Minute': [0, 30] * 24
})

predictions = model.predict(X_test)

# Evaluate model using mean squared error and R-squared on the test set
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Predict prices for the chosen day
day_predictions = model.predict(new_data[['Day', 'Month', 'Year', 'Hour', 'Minute']])

# Display predicted prices for each row
# for index, prediction in enumerate(predictions_new_data):
#    print(f'Predicted Price for Hour {new_data["Hour"].iloc[index]}:{new_data["Minute"].iloc[index]}: {prediction}')

# Add the predictions as a new column in new_data
new_data['Predicted_Price'] = day_predictions

# Filter the actual data for the chosen day
actual_data_for_chosen_day = data[(data['Day'] == day_to_predict) & (data['Month'] == month_to_predict)]
actual_prices_for_chosen_day = actual_data_for_chosen_day['Price']

# Plot the predicted prices against the actual prices for the chosen day
plt.figure(figsize=(10, 6))

# Group the actual data by year and plot a line for each year
for year, group in actual_data_for_chosen_day.groupby('Year'):
    #print(f'Group for Year {year}:\n{group}')
    group = group.sort_values(['Hour', 'Minute'])  # Sort by Hour and Minute to ensure chronological order
    plt.plot(group['Hour'] + group['Minute'] / 60, group['Price'], linestyle='-', marker='o', label=f'Actual Prices - {year}')

# Plot the predicted prices for the chosen day
plt.plot(new_data['Hour'] + new_data['Minute'] / 60, day_predictions, linestyle='-', marker='o', color='b', label='Predicted Prices')

plt.title(f'Electricity Price Prediction - {day_to_predict}/{month_to_predict}/{year_to_predict}')
plt.xlabel('Hour')
plt.ylabel('Price')
plt.xticks(np.arange(0, 24, 1))
plt.legend()
plt.grid(True)
plt.show()
