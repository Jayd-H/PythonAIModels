import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
x_train = pd.read_csv('Splits/X_train.csv')
y_train = pd.read_csv('Splits/y_train.csv')
x_test = pd.read_csv('Splits/X_test.csv')
y_test = pd.read_csv('Splits/y_test.csv')

# Generate polynomial and interaction features
poly = PolynomialFeatures(degree=2, include_bias=False)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

# Fit the Polynomial Regression Model
poly_regressor = LinearRegression()
poly_regressor.fit(x_train_poly, y_train)

# Predicting on the test set
y_pred = poly_regressor.predict(x_test_poly)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Get feature importance (coefficients) from the polynomial regressor
feature_importances = np.abs(poly_regressor.coef_[0])

plt.figure(figsize=(14, 6))

# Scatter plot of actual vs predicted values
ax1 = plt.subplot(1, 2, 1) 
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel('Actual Sale Prices')
plt.ylabel('Predicted Sale Prices')
plt.title('Actual vs Predicted Sale Prices (Polynomial Regression)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)

# Annotating the R2 and MSE on the scatter plot
textstr = f'R² Score: {r2:.2f}\nMSE: {mse:.2e}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# Place the text box in upper left in axes coords
ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=12,
         verticalalignment='top', bbox=props)

# Feature Importance Plot
ax2 = plt.subplot(1, 2, 2) 
sorted_idx = np.argsort(feature_importances)[-10:]  
plt.barh(range(10), feature_importances[sorted_idx], color='blue')
plt.xlabel('Feature Importance (Coefficient Magnitude)')
plt.title('Top 10 Most Important Features (Polynomial Regression)')
plt.yticks(range(10), poly.get_feature_names_out()[sorted_idx])

plt.tight_layout() 
plt.show()