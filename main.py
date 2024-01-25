import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import LinearSVR

def compute_mse(model, X, y_true, name):
    y_pred = model.predict(X)
    mse = mean_squared_error(y_true, y_pred)
    print(f'Mean Squared Error for {name}: {mse}')
def build_evaluate_fn(X_train, y_train, X_test, y_test):
    def evaluate(model):
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        print("Train Score:", train_score)
        print("Test Score:", test_score)
        print()

        compute_mse(model, X_train, y_train, 'training set')
        compute_mse(model, X_test, y_test, 'test set')

    return evaluate

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
degree = 4  # You can adjust the degree of the polynomial
alpha = 0.5  # Regularization strength, you can adjust this
linear_poly = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=alpha))
svm = LinearSVR()
gbr = LGBMRegressor(n_estimators=1000)


# Train the model on the training set
linear_poly.fit(X_train, y_train)
svm.fit(X_train, y_train)
gbr.fit(X_train, y_train)

evaluate = build_evaluate_fn(X_train, y_train, X_test, y_test)
evaluate(linear_poly)
evaluate(svm)
evaluate(gbr)

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

lp_predictions = linear_poly.predict(X_test)
svm_predictions = svm.predict(X_test)
gbr_predictions = gbr.predict(X_test)

# Evaluate model using mean squared error and R-squared on the test set
# mse = mean_squared_error(y_test, predictions)
lp_r2 = r2_score(y_test, lp_predictions)
svm_r2 = r2_score(y_test, svm_predictions)
gbr_r2 = r2_score(y_test, gbr_predictions)

#print(f'Mean squared error: {mse}')
print(f'LP R squared error: {lp_r2}')
print(f'SVM R squared error: {svm_r2}')
print(f'GBR R squared error: {gbr_r2}')

# Predict prices for the chosen day
lp_day_predictions = linear_poly.predict(new_data[['Day', 'Month', 'Year', 'Hour', 'Minute']])
svm_day_predictions = svm.predict(new_data[['Day', 'Month', 'Year', 'Hour', 'Minute']])
gbr_day_predictions = gbr.predict(new_data)

# Display predicted prices for each row
# for index, prediction in enumerate(predictions_new_data):
#    print(f'Predicted Price for Hour {new_data["Hour"].iloc[index]}:{new_data["Minute"].iloc[index]}: {prediction}')

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
plt.plot(new_data['Hour'] + new_data['Minute'] / 60, lp_day_predictions, linestyle='-', marker='o', color='b', label='Linear Poly Predicted Prices')
plt.plot(new_data['Hour'] + new_data['Minute'] / 60, svm_day_predictions, linestyle='-', marker='o', color='m', label='SVM Predicted Prices')
plt.plot(new_data['Hour'] + new_data['Minute'] / 60, gbr_day_predictions, linestyle='-', marker='o', color='k', label='GBR Predicted Prices')



plt.title(f'Electricity Price Prediction - {day_to_predict}/{month_to_predict}/{year_to_predict}')
plt.xlabel('Hour')
plt.ylabel('Price')
plt.xticks(np.arange(0, 24, 1))
plt.legend()
plt.grid(True)
plt.show()
