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

# Evaluate the model
r2 = r2_score(ytestdataset, y_pred)
mse = mean_squared_error(ytestdataset, y_pred)
print(f'\nR² Score: {r2}')
print(f'Mean Squared Error: {mse}')

# Scatter plot of actual vs predicted values
plt.scatter(ytestdataset, y_pred)
plt.xlabel('Actual Sale Prices')
plt.ylabel('Predicted Sale Prices')
plt.title('Actual vs Predicted Sale Prices')
plt.plot([ytestdataset.min(), ytestdataset.max()], [ytestdataset.min(), ytestdataset.max()], 'k--', lw=4)
plt.show()
