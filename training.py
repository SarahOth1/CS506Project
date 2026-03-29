#!/usr/bin/env python3

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# get our cleaned data
df = pd.read_csv('movies_cleaned.csv')

# Features X and target Y
X = df.drop(columns=['averageRating', 'tconst', 'primaryTitle', 'directors', 'writers'])
y = df['averageRating']

# Replace missing values with 0
X = X.fillna(0)

# Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# predict rating for testing data
y_pred = model.predict(X_test)

#evaluate predictions
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("RMSE:", rmse)
print("Sample predictions:", y_pred[:5])
print("Actual values:", y_test[:5].values)


# print("RMSE:", round(rmse, 4))

# Step 8: Compare some predictions vs actual
comparison = pd.DataFrame({
    'Actual Rating': y_test.values[:10],
    'Predicted Rating': y_pred[:10].round(2)
})
print(comparison)