import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load the dataset
data = pd.read_csv('expected_ctc.csv')

# Remove unnecessary columns
data = data.drop(columns=['IDX', 'Applicant_ID'])

# Fill missing values for text columns with 'Unknown'
text_cols_to_fill = [
    'Department', 'Role', 'Industry', 'Organization', 'Designation',
    'Graduation_Specialization', 'University_Grad', 'PG_Specialization',
    'University_PG', 'PHD_Specialization', 'University_PHD',
    'Last_Appraisal_Rating'
]
for col in text_cols_to_fill:
    data[col] = data[col].fillna('Unknown')

# Fill missing values for year columns with 0 and convert to integer
year_cols_to_fill = [
    'Passing_Year_Of_Graduation', 'Passing_Year_Of_PG', 'Passing_Year_Of_PHD'
]
for col in year_cols_to_fill:
    data[col] = data[col].fillna(0).astype(int)

# Convert 'Inhand_Offer' to numbers
data['Inhand_Offer'] = data['Inhand_Offer'].map({'Y': 1, 'N': 0})

# Identify all text columns for encoding
text_columns = data.select_dtypes(include='object').columns

# Convert text columns to numbers using one-hot encoding
data_encoded = pd.get_dummies(data, columns=text_columns, drop_first=True)

# Separate input features (X) and target (y)
features = data_encoded.drop('Expected_CTC', axis=1)
target = data_encoded['Expected_CTC']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create and train the Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Make predictions on the test data
preds = model.predict(X_test)

# Evaluate the model's performance
mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, preds)

print("Model Performance:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R-squared: {r2:.4f}")