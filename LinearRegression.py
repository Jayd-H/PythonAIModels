import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


processed_dataset_path = 'ProcessedDataset.csv'
processed_ames_dataset = pd.read_csv(processed_dataset_path)

print(processed_ames_dataset.head())

missing_values = processed_ames_dataset.isnull().sum()
print(missing_values)

processed_ames_dataset.info()

print(processed_ames_dataset.shape)

