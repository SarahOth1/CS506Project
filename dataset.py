#!/usr/bin/env python3

import pandas as pd
import numpy as np


# Loading datasets from downloaded files
basics = pd.read_csv('title.basics.tsv.gz',
                     sep='\t',                  
                     compression='gzip')          

ratings = pd.read_csv('title.ratings.tsv.gz',
                     sep='\t',                  
                     compression='gzip')      

princs = pd.read_csv('title.principals.tsv.gz',
                     sep='\t',                  
                     compression='gzip')      


# See shapes
print('SHAPES OF INDIVIDUAL FILES')
print(basics.shape)
print(ratings.shape)
print(princs.shape)



# Create new features from principals file:

# Count total number of main credited people per movie
num_principals = princs.groupby('tconst').size().reset_index(name='num_principals')

# Count each type of principal per movie (actor, actress, director, writer)
category_counts = pd.crosstab(princs['tconst'], princs['category']).reset_index()

# Merge principal features together
principal_features = pd.merge(
    num_principals,
    category_counts,
    on='tconst',
    how='left'
)

print("Principal feature columns:")
print(principal_features.columns.tolist())

# Merge basics and ratings
df = pd.merge(basics, ratings, on='tconst', how='inner')

# Merge summarized principal_features
df = pd.merge(df, principal_features, on='tconst', how='left')


print("Shape after merge:", df.shape)
print('columns:')
print(df.columns.tolist())


# Filter to movies only
df = df[df['titleType'] == 'movie']

# Replace missing vaules 
df = df.replace('\\N', np.nan)

# Convert columns to correct types
df['startYear'] = pd.to_numeric(df['startYear'], errors='coerce')
df['runtimeMinutes'] = pd.to_numeric(df['runtimeMinutes'], errors='coerce')
df['averageRating'] = pd.to_numeric(df['averageRating'], errors='coerce')
df['numVotes'] = pd.to_numeric(df['numVotes'], errors='coerce')

print("\nData types after conversion:")
print(df.dtypes)

# Convert data types
print("\nData types before conversion:")
print(df.dtypes)

# Filter out movies with not enough votes to be relevant
print("Shape before vote filter:", df.shape)
df = df[df['numVotes'].astype(float) >= 1000]
print("Shape after vote filter:", df.shape)



# Remove duplicate movie names
print("Duplicates before:", df.duplicated().sum())
print("Title duplicates:", df.duplicated(subset=['primaryTitle']).sum())

# keeping duplicate with highest vote
df = df.sort_values('numVotes', ascending=False)
df = df.drop_duplicates(subset=['primaryTitle'], keep='first')

print("Shape after removing title duplicates:", df.shape)

# Remove irrelevant columns
df = df.drop(columns=['titleType', 'originalTitle', 'endYear', 'isAdult'])
print('REMOVED IRRELEVANT COLUMNS. Updated columns:')

print(df.columns.tolist())

#cleaning principal_cols 
principal_cols = principal_features.columns.drop('tconst')
df[principal_cols] = df[principal_cols].fillna(0)


# Converting genre column into numeric representations (by adding each genre as a column)
print(df['genres'].head(10))

genres_dummies = df['genres'].str.get_dummies(sep=',')
print("Genre columns created:", genres_dummies.columns.tolist())
df = pd.concat([df, genres_dummies], axis=1)
df = df.drop(columns=['genres'])
print("Shape after encoding:", df.shape)
print(df.columns.tolist())

# Save
df.to_csv('movies_cleaned.csv', index=False)
print("Saved!")


#reomving rows with missing model values
df = df.dropna(subset=['startYear', 'runtimeMinutes', 'averageRating', 'numVotes'])


X = df.drop(columns=['averageRating', 'tconst', 'primaryTitle'])
y = df['averageRating']

print(X.select_dtypes(include="object").columns)
