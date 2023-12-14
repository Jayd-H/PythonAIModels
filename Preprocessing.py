import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# Load the dataset
ames_dataset = pd.read_csv('UnprocessedDataset.csv')

# Calculate the percentage of missing values for each column
missing_percentage = ames_dataset.isnull().mean() * 100

# Identify columns where more than 50% of the data is missing
columns_to_drop = missing_percentage[missing_percentage > 50].index

# Drop these columns
ames_dataset.drop(columns=columns_to_drop, inplace=True)

y = ames_dataset['SalePrice']
ames_dataset = ames_dataset.drop(columns='SalePrice')

# Imputing missing values for numerical columns
numerical_cols = ames_dataset.select_dtypes(include=['int64', 'float64']).columns
imputer_num = SimpleImputer(strategy='mean')
ames_dataset[numerical_cols] = imputer_num.fit_transform(ames_dataset[numerical_cols])

# Imputing missing values for categorical columns
categorical_cols = ames_dataset.select_dtypes(include=['object']).columns
imputer_cat = SimpleImputer(strategy='most_frequent')
ames_dataset[categorical_cols] = imputer_cat.fit_transform(ames_dataset[categorical_cols])

# Encoding categorical data
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ames_dataset_encoded = pd.DataFrame(encoder.fit_transform(ames_dataset[categorical_cols]))
ames_dataset_encoded.columns = encoder.get_feature_names_out(categorical_cols)

ames_dataset_preprocessed = pd.concat([ames_dataset.drop(columns=categorical_cols), ames_dataset_encoded], axis=1)

# Feature Scaling for numerical columns
scaler = MinMaxScaler()
ames_dataset_preprocessed[numerical_cols] = scaler.fit_transform(ames_dataset_preprocessed[numerical_cols])


X_temp, X_test, y_temp, y_test = train_test_split(ames_dataset_preprocessed, y, test_size=0.2, random_state=42)
X_train, X_dev, y_train, y_dev = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)


X_train.to_csv('X_train.csv', index=False)
X_dev.to_csv('X_dev.csv', index=False)   
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_dev.to_csv('y_dev.csv', index=False)   
y_test.to_csv('y_test.csv', index=False)

print("Split Files Saved")

ames_dataset_preprocessed.to_csv('ProcessedDataset.csv', index=False)
print("Saved CSV")

