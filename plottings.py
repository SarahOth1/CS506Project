#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

# Get cleaned dataset
df = pd.read_csv('movies_cleaned.csv')

# Plot 1: average rating and genre relationship
genre_cols = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy',
              'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy',
              'Film-Noir', 'History', 'Horror', 'Music', 'Musical',
              'Mystery', 'Romance', 'Sci-Fi', 'Sport', 'Thriller',
              'War', 'Western']

# Get average rating for each genre
genre_ratings = {}
for genre in genre_cols:
    if genre in df.columns:
        avg = df[df[genre] == 1]['averageRating'].mean()
        genre_ratings[genre] = avg

# Convert to series, since genre_ratings is a dictionary and we cant plot that
genre_ratings = pd.Series(genre_ratings)

# Plot 1
plt.figure(figsize=(12, 6))
plt.scatter(genre_ratings.index, genre_ratings.values, color='blue', s=100)
plt.title('Average IMDb Rating by Genre')
plt.xlabel('Genre')
plt.ylabel('Average Rating')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('rating_by_genre.png')



# Plot 2: average rating and release year relationship

# Convert year to int
df['startYear'] = pd.to_numeric(df['startYear'], errors='coerce')

# Filter year range
df_year = df[(df['startYear'] >= 1920) & (df['startYear'] <= 2026)]

# Get average rating per year
year_ratings = df_year.groupby('startYear')['averageRating'].mean()

# Plot
plt.figure(figsize=(14, 6))
plt.scatter(year_ratings.index, year_ratings.values, color='blue', s=10)
plt.title('Average IMDb Rating by Release Year')
plt.xlabel('Release Year')
plt.ylabel('Average Rating')
plt.tight_layout()
plt.savefig('rating_by_year.png')
plt.show()
