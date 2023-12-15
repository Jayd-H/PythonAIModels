import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the split datasets
X_train = pd.read_csv('Splits/X_train.csv')
y_train = pd.read_csv('Splits/y_train.csv')
X_dev = pd.read_csv('Splits/X_dev.csv')
y_dev = pd.read_csv('Splits/y_dev.csv')
X_test = pd.read_csv('Splits/X_test.csv')
y_test = pd.read_csv('Splits/y_test.csv')

# Define the Random Forest Regressor
rfregressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
rfregressor.fit(X_train, y_train)

# Evaluate the model on the development set
y_dev_pred = rfregressor.predict(X_dev)
r2_dev = r2_score(y_dev, y_dev_pred)
mse_dev = mean_squared_error(y_dev, y_dev_pred)
print(f'Development Set - R²: {r2_dev}, MSE: {mse_dev}')

# Evaluate the model on the test data
y_test_pred = rfregressor.predict(X_test)
r2_test = r2_score(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
print(f'Test Set - R²: {r2_test}, MSE: {mse_test}')

# GRAPHSHSHS!!1
plt.figure(figsize=(15, 5))

# Plot for Development Set
plt.subplot(1, 2, 1)
plt.scatter(y_dev, y_dev_pred, alpha=0.3)
plt.plot([y_dev.min(), y_dev.max()], [y_dev.min(), y_dev.max()], 'r--', lw=2)
plt.title('Development Set: Actual vs Predicted Sale Prices')
plt.xlabel('Actual Sale Prices')
plt.ylabel('Predicted Sale Prices')
plt.text(0.05, 0.95, f'R² Score: {r2_dev:.2f}\nMSE: {mse_dev:.2e}', transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot for Test Set
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Test Set: Actual vs Predicted Sale Prices')
plt.xlabel('Actual Sale Prices')
plt.ylabel('Predicted Sale Prices')
plt.text(0.05, 0.95, f'R² Score: {r2_test:.2f}\nMSE: {mse_test:.2e}', transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.show()
