#!/usr/bin/env python3

#important necessary packages
import pandas as pd
import numpy as np


# Step 1: Loading datasets from downloaded files
basics = pd.read_csv('title.basics.tsv.gz', 
                     sep='\t',                   
                     compression='gzip')           

ratings = pd.read_csv('title.ratings.tsv.gz', 
                     sep='\t',                   
                     compression='gzip')       

crew = pd.read_csv('title.crew.tsv.gz', 
                     sep='\t',                   
                     compression='gzip')       


# See shape 
print('SHAPES OF INDIVIDUAL FILES')
print(basics.shape)
print(ratings.shape)
print(crew.shape)

# Step 2: Merging files into one file

# Merge basics + ratings on tconst
df = pd.merge(basics, ratings, on='tconst', how='inner')

# Merge result + crew on tconst
df = pd.merge(df, crew, on='tconst', how='inner')

print("Shape after merge:", df.shape)
print('columns:')
print(df.columns.tolist())

# Step 3: Replace missing vaules with NaN
df = df.replace('\\N', np.nan)

# Step 4: Filter to movies only
df = df[df['titleType'] == 'movie']

# Step 5: Remove duplicate movie names
print("Duplicates before:", df.duplicated().sum())
print("Title duplicates:", df.duplicated(subset=['primaryTitle']).sum())

#keeping duplicate with highest vote
df = df.sort_values('numVotes', ascending=False)
df = df.drop_duplicates(subset=['primaryTitle'], keep='first')

print("Shape after removing title duplicates:", df.shape)

# Step 6: Remove irrelevant columns
df = df.drop(columns=['titleType', 'originalTitle', 'endYear', 'isAdult'])
print('REMOVED IRRELEVANT COLUMNS. Updated columns:')
# Confirm they are gone
print(df.columns.tolist())

# Save cleaned dataset
df.to_csv('movies_cleaned.csv', index=False)
print("Saved!")

# Step 7: Filter out movies with not enough votes to be relevant
print("Shape before vote filter:", df.shape)
df = df[df['numVotes'] >= 1000]
print("Shape after vote filter:", df.shape)

# Step 8: Convert data types
print("\nData types before conversion:")
print(df.dtypes)

# Convert columns to correct types
df['startYear'] = pd.to_numeric(df['startYear'], errors='coerce')
df['runtimeMinutes'] = pd.to_numeric(df['runtimeMinutes'], errors='coerce')
df['averageRating'] = pd.to_numeric(df['averageRating'], errors='coerce')
df['numVotes'] = pd.to_numeric(df['numVotes'], errors='coerce')

print("\nData types after conversion:")
print(df.dtypes)


# Save
df.to_csv('movies_cleaned.csv', index=False)
print("\nSaved!")

# Step 10: Converting genre column into numeric representations (by adding each genre as a column)
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