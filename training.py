#!/usr/bin/env python3

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load cleaned data
df = pd.read_csv('movies_cleaned.csv')

# Remove missing values
df = df.dropna(subset=['startYear', 'runtimeMinutes', 'averageRating', 'numVotes'])

# Model 1: Baseline model

# These are the basic features before adding principals
drop_cols = ['averageRating', 'tconst', 'primaryTitle']

# Principal columns created later
principal_cols = [
    'num_principals',
    'actor',
    'actress',
    'archive_footage',
    'archive_sound',
    'cinematographer',
    'composer',
    'director',
    'editor',
    'producer',
    'production_designer',
    'self',
    'writer'
]

# Keeping principal columns that exist in downloaded file
principal_cols = [col for col in principal_cols if col in df.columns]

# Baseline drops principal columns
X_baseline = df.drop(columns=drop_cols + principal_cols)
y = df['averageRating']

# confirming everything is numeric
X_baseline = X_baseline.apply(pd.to_numeric, errors='coerce')
X_baseline = X_baseline.fillna(0)

# Splitting baseline data
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_baseline,
    y,
    test_size=0.2,
    random_state=42
)

# Training on linear regression
baseline_model = LinearRegression()
baseline_model.fit(X_train_b, y_train_b)

# Predicting
y_pred_b = baseline_model.predict(X_test_b)

# Evaluating using rmse
baseline_rmse = np.sqrt(mean_squared_error(y_test_b, y_pred_b))
baseline_mae = mean_absolute_error(y_test_b, y_pred_b)
baseline_r2 = r2_score(y_test_b, y_pred_b)

print("BASELINE MODEL RESULTS")
print("RMSE:", round(baseline_rmse, 4))
print("MAE:", round(baseline_mae, 4))
print("R2:", round(baseline_r2, 4))


# Model 2: adding principals to features


X_full = df.drop(columns=drop_cols)
y = df['averageRating']

# confirming  everything is numeric
X_full = X_full.apply(pd.to_numeric, errors='coerce')
X_full = X_full.fillna(0)

# Splitting data
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_full,
    y,
    test_size=0.2,
    random_state=42
)

# Training on linear regression
full_model = LinearRegression()
full_model.fit(X_train_f, y_train_f)

# Predictting
y_pred_f = full_model.predict(X_test_f)

# Evaluating
full_rmse = np.sqrt(mean_squared_error(y_test_f, y_pred_f))
full_mae = mean_absolute_error(y_test_f, y_pred_f)
full_r2 = r2_score(y_test_f, y_pred_f)

print("\nIMPROVED MODEL WITH PRINCIPALS RESULTS")
print("RMSE:", round(full_rmse, 4))
print("MAE:", round(full_mae, 4))
print("R2:", round(full_r2, 4))


# Comparing results

results = pd.DataFrame({
    'Model': ['Baseline', 'With Principals'],
    'RMSE': [baseline_rmse, full_rmse],
    'MAE': [baseline_mae, full_mae],
    'R2': [baseline_r2, full_r2]
})

print("\nMODEL COMPARISON")
print(results.round(4))


# Show subset of predictions


comparison = pd.DataFrame({
    'Actual Rating': y_test_f.values[:10],
    'Predicted Rating': y_pred_f[:10].round(2)
})

print("\nSAMPLE PREDICTIONS FROM IMPROVED MODEL")
print(comparison)


