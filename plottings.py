#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

# Load plotting data created from training.py
plot_df = pd.read_csv("plotting_data.csv")

print("Columns available in plotting_data.csv:")
print(plot_df.columns.tolist())


# Historical person average features and actual rating


person_avg_features = [
    "actor_avg_rating",
    "actress_avg_rating",
    "director_avg_rating",
    "writer_avg_rating"
]

person_avg_features = [
    feature for feature in person_avg_features
    if feature in plot_df.columns
]

for feature in person_avg_features:
    plt.figure(figsize=(7, 5))
    plt.scatter(plot_df[feature], plot_df["actual_rating"], alpha=0.3)
    plt.xlabel(feature)
    plt.ylabel("Actual IMDb Rating")
    plt.title(f"Actual Rating vs {feature}")
    plt.tight_layout()
    plt.show()