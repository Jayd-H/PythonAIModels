import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load the split datasets
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')
X_dev = pd.read_csv('X_dev.csv')
y_dev = pd.read_csv('y_dev.csv')
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')

# Define the neural network architecture
model = Sequential([
    Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(10, activation='relu'),
    Dense(10, activation='relu'),
    Dense(1, activation='linear'),
])

# Compile the model with Adam optimizer and mean squared error loss
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='mean_squared_error',
              metrics=['mean_squared_error'])

# Train the model on the training data
hist = model.fit(X_train, y_train,
                 batch_size=128, epochs=50,
                 validation_data=(X_dev, y_dev))

# Evaluate the model on the test data
mse_test = model.evaluate(X_test, y_test)
print(f'Test MSE: {mse_test}')

# Predicting on the test set
y_pred = model.predict(X_test)

# Plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(hist.history['loss'], label='Train Loss')
plt.plot(hist.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot of actual vs predicted sale prices
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs Predicted Sale Prices')
plt.xlabel('Actual Sale Prices')
plt.ylabel('Predicted Sale Prices')
plt.show()
