import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# 1. Load the dataset 
df = pd.read_csv('car_data.csv')

# 2. Data Preprocessing (Milestone 1 & 2 foundations)
df['Current_Year'] = 2026
df['Car_Age'] = df['Current_Year'] - df['Year']
df.drop(['Year', 'Current_Year', 'Car_Name'], axis=1, inplace=True)

# Convert text tracking fields into numbers
df = pd.get_dummies(df, drop_first=True)

# 3. Split features and target price
X = df.drop(['Selling_Price'], axis=1)
y = df['Selling_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Machine Learning Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate the model performance
predictions = model.predict(X_test)
print(f"Model R2 Score (Accuracy): {r2_score(y_test, predictions):.2f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, predictions):.2f} Lakhs")
