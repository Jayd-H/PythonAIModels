import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
xtraindataset = pd.read_csv('X_train.csv')
ytraindataset = pd.read_csv('Y_train.csv')
xtestdataset = pd.read_csv('X_test.csv')
ytestdataset = pd.read_csv('Y_test.csv')

# Linear Regression Model
regressor = LinearRegression()
regressor.fit(xtraindataset, ytraindataset)
print("Intercept")
print(regressor.intercept_)
print()
print("Coefficients")
print(regressor.coef_)

# Predicting on the test set
y_pred = regressor.predict(xtestdataset)

# Set a realistic maximum value for predictions
max_realistic_value = 1e9
realistic_prediction_mask = np.abs(y_pred) < max_realistic_value

# Filter out unrealistic predictions
realistic_y_pred = y_pred[realistic_prediction_mask]
realistic_y_test = ytestdataset[realistic_prediction_mask.ravel()]

# Evaluate the model on the realistic predictions
r2 = r2_score(realistic_y_test, realistic_y_pred)
mse = mean_squared_error(realistic_y_test, realistic_y_pred)
print(f'\nR² Score: {r2}')
print(f'Mean Squared Error: {mse}')

# Scatter plot of actual vs predicted values
plt.scatter(ytestdataset, y_pred)
plt.xlabel('Actual Sale Prices')
plt.ylabel('Predicted Sale Prices')
plt.title('Actual vs Predicted Sale Prices')
plt.plot([ytestdataset.min(), ytestdataset.max()], [ytestdataset.min(), ytestdataset.max()], 'k--', lw=4)
plt.show()

# Scatter plot of actual vs predicted values, excluding extreme predictions
plt.scatter(realistic_y_test, realistic_y_pred)
plt.xlabel('Actual Sale Prices')
plt.ylabel('Predicted Sale Prices')
plt.title('Actual vs Predicted Sale Prices (Excluding Extremes)')
plt.plot([realistic_y_test.min(), realistic_y_test.max()], [realistic_y_test.min(), realistic_y_test.max()], 'k--', lw=4)
plt.show()