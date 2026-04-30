#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Load cleaned movie data
df = pd.read_csv("movies_cleaned.csv")
# drop rows missing important data
df = df.dropna(subset=['startYear', 'runtimeMinutes', 'averageRating', 'numVotes'])
# Load columns we will use from principals file
princs = pd.read_csv(
    "title.principals.tsv.gz",
    sep="\t",
    compression="gzip",
    usecols=["tconst", "nconst", "category"]
)

# Keep only useful people categories
princs = princs[princs["category"].isin(["actor", "actress", "director", "writer"])]
# Keep movies that are in the movie dataset
princs = princs[princs["tconst"].isin(df["tconst"])]

# Split data into training and testing
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

global_train_mean = train_df["averageRating"].mean()

# Function to add person average features

def add_person_avg_features(train_df, test_df, princs, category):
    feature_name = category + "_avg_rating"

    # Get ratings from training movies
    train_ratings = train_df[["tconst", "averageRating"]]

    # Merge principals with training ratings
    train_people = princs.merge(train_ratings, on="tconst", how="inner")

    # Keep only this category
    train_people_cat = train_people[train_people["category"] == category]

    # Average rating for each person based on training data
    person_avg = train_people_cat.groupby("nconst")["averageRating"].mean()

    # For all movies, map each person to their average
    all_people_cat = princs[princs["category"] == category].copy()
    all_people_cat["person_avg"] = all_people_cat["nconst"].map(person_avg)

    # Average the people averages per movie
    movie_person_avg = all_people_cat.groupby("tconst")["person_avg"].mean()

    # Add new feature to train and test
    train_df[feature_name] = train_df["tconst"].map(movie_person_avg)
    test_df[feature_name] = test_df["tconst"].map(movie_person_avg)

    # Fill unknown people with global training average
    train_df[feature_name] = train_df[feature_name].fillna(global_train_mean)
    test_df[feature_name] = test_df[feature_name].fillna(global_train_mean)

    return train_df, test_df


# Add new rating features for each person category

for category in ["actor", "actress", "director", "writer"]:
    train_df, test_df = add_person_avg_features(train_df, test_df, princs, category)

print("New columns added:")
print(["actor_avg_rating", "actress_avg_rating", "director_avg_rating", "writer_avg_rating"])


# Adding more features 
# To sclae down numVotes:
train_df["log_numVotes"] = np.log1p(train_df["numVotes"])
test_df["log_numVotes"] = np.log1p(test_df["numVotes"])
# To explore bigger relationship between runtimeMinutes and avgRating
train_df["runtime_sq"] = train_df["runtimeMinutes"] ** 2
test_df["runtime_sq"] = test_df["runtimeMinutes"] ** 2

#dropping original unedited cols
drop_cols = ["averageRating", "tconst", "primaryTitle", "numVotes"]

X_train = train_df.drop(columns=drop_cols)
y_train = train_df["averageRating"]

X_test = test_df.drop(columns=drop_cols)
y_test = test_df["averageRating"]

# confirming features are numeric
X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0)
X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0)


# Linear Regression Model

linear_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

# function for evaluation

def evaluate_model(model, name):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("\n" + name)
    print("RMSE:", round(rmse, 4))
    print("MAE:", round(mae, 4))
    print("R2:", round(r2, 4))

    return preds, rmse, mae, r2

linear_preds, linear_rmse, linear_mae, linear_r2 = evaluate_model(
    linear_model,
    "Linear Regression with Person Features"
)


plot_df=X_test.copy()
plot_df["actual_rating"]=y_test.values
plot_df["predicted_rating"]=linear_preds
plot_df.to_csv("plotting_data.csv", index=False)


import matplotlib.pyplot as plt
import pandas as pd

# Predict IMDb ratings for the test set
y_pred = linear_model.predict(X_test)

# Create dataframe with true and predicted values
predictions_df = pd.DataFrame({
    "true_rating": y_test,
    "predicted_rating": y_pred
})

# Saving the values 
predictions_df.to_csv("predicted_vs_true.csv", index=False)

# Plotting predicted values against true values
plt.figure(figsize=(8, 6))
plt.scatter(predictions_df["true_rating"], predictions_df["predicted_rating"], alpha=0.5)


min_rating = min(predictions_df["true_rating"].min(), predictions_df["predicted_rating"].min())
max_rating = max(predictions_df["true_rating"].max(), predictions_df["predicted_rating"].max())
plt.plot([min_rating, max_rating], [min_rating, max_rating], linestyle="--")

plt.xlabel("True IMDb Rating")
plt.ylabel("Predicted IMDb Rating")
plt.title("Predicted vs True IMDb Ratings on Test Set")

plt.tight_layout()
plt.savefig("predicted_vs_true.png")
plt.show()