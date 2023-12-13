import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
ames_dataset = pd.read_csv('UnprocessedDataset.csv')

# Dropping columns with a high percentage of missing values
columns_to_drop = ['Alley', 'Pool QC', 'Fence', 'Misc Feature']
ames_dataset.drop(columns=columns_to_drop, inplace=True)

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


ames_dataset.reset_index(drop=True, inplace=True)
ames_dataset_encoded.reset_index(drop=True, inplace=True)


ames_dataset_preprocessed = pd.concat([ames_dataset.drop(columns=categorical_cols), ames_dataset_encoded], axis=1)

# Feature Scaling
scaler = StandardScaler()
numerical_cols = ames_dataset_preprocessed.select_dtypes(include=['int64', 'float64']).columns
ames_dataset_preprocessed[numerical_cols] = scaler.fit_transform(ames_dataset_preprocessed[numerical_cols])

# Data Splitting
X = ames_dataset_preprocessed.drop('SalePrice', axis=1)  # Features
y = ames_dataset_preprocessed['SalePrice']  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the preprocessed dataset
ames_dataset_preprocessed.to_csv('ProcessedDataset.csv', index=False)