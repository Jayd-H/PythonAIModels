import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Load the split datasets
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('Y_train.csv')
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('Y_test.csv')

# Define the neural network architecture
model = Sequential()
model.add(Dense(64, input_shape=(X_train.shape[1],), activation='relu'))  # Input layer
model.add(Dense(32, activation='relu'))  # Hidden layer
model.add(Dense(1, activation='linear'))  # Output layer for regression

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model on the training data
model.fit(X_train, y_train, validation_split=0.1, epochs=100, batch_size=32)

# Evaluate the model on the test data
mse_test = model.evaluate(X_test, y_test)
print(f'Test MSE: {mse_test}')

# Predicting on the test set
y_pred = model.predict(X_test)

