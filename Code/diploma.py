import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from isodate import parse_duration, ISO8601Error

recipes_df = pd.read_csv("../Data/Food_com/recipes.csv")
recipes_df.isnull().sum()
recipes_df = recipes_df.drop(columns='RecipeYield')

# Drop rows with NaN values in place
recipes_df.dropna(inplace=True)
recipes_df.isnull().sum()

# Convert time values to numeric
# Function to convert duration string to seconds
def convert_to_seconds(duration_str):
    try:
        if pd.notna(duration_str) and duration_str.strip() != '':
            duration = parse_duration(duration_str)
            return duration.total_seconds()
        else:
            return None  # or any other appropriate value for empty strings
    except ISO8601Error:
        return None  # or any other appropriate value for unexpected formats

time_collumns = ['CookTime', 'PrepTime', 'TotalTime']
for col in time_collumns:
    recipes_df[col] = recipes_df[col].apply(convert_to_seconds)

recipes_df.head()
recipes_df.dtypes

recipes_df_for_corr = recipes_df[
                                 ["CookTime",
                                  "PrepTime", 
                                  'TotalTime', 
                                  'AggregatedRating', 
                                  'ReviewCount', 
                                  'Calories', 
                                  'FatContent',
                                  'SaturatedFatContent', 
                                  'CholesterolContent', 
                                  'SodiumContent', 
                                  'CarbohydrateContent',
                                  'FiberContent',
                                  'SugarContent',
                                  'ProteinContent',
                                  'RecipeServings'
                                  ]
                                  ].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(recipes_df_for_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()

# Split combined keywords into separate rows
recipes_df['Keywords'] = recipes_df['Keywords'].str.extractall(r'"([^"]+)"').groupby(level=0).agg(','.join)
df_expanded = recipes_df['Keywords'].str.split(',', expand=True).stack().reset_index(level=1, drop=True).to_frame('Keyword')
sample_df = recipes_df.drop('Keywords', axis=1).join(df_expanded)

keyword_counts = sample_df['Keyword'].value_counts().nlargest(50)

# Plot the bar chart
plt.figure(figsize=(15, 6))
keyword_counts.plot(kind='bar', color='skyblue')
plt.title('Keyword Counts')
plt.xlabel('Keywords')
plt.ylabel('Count')
plt.show()

# Count the occurrences of each rating
rating_counts = recipes_df['AggregatedRating'].value_counts()

# Plot the pie chart without percentage labels
plt.figure(figsize=(8, 8))
wedges, texts, _ = plt.pie(rating_counts, labels=None, autopct='', startangle=90, colors=plt.cm.Paired.colors)

# Create percentage labels
percentage_labels = [f'{p:.1f}%' for p in rating_counts / rating_counts.sum() * 100]

# Create legend with ratings and percentage labels
legend_labels = [f'{rating} ({percentage})' for rating, percentage in zip(rating_counts.index, percentage_labels)]
plt.legend(wedges, legend_labels, title='Ratings', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.title('Distribution of Aggregated Ratings')
plt.show()

# Count the number of recipes per author
author_counts = recipes_df['AuthorId'].value_counts()

# Plot the top authors by the number of recipes
top_authors = author_counts.nlargest(50)  # Adjust the number as needed

plt.figure(figsize=(10, 6))
top_authors.plot(kind='bar', color='skyblue')
plt.title('Top Authors by Number of Recipes')
plt.xlabel('Author (ID_Name)')
plt.ylabel('Number of Recipes')
plt.show()

# Convert duration columns from seconds to minutes
recipes_df[['CookTime', 'PrepTime', 'TotalTime']] = recipes_df[['CookTime', 'PrepTime', 'TotalTime']].apply(lambda x: x / 60)

# Plot the count of unique values in each column
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot for 'CookTime'
recipes_df['CookTime'].value_counts().nlargest(15).plot(kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title('CookTime Counts')
axes[0].set_xlabel('CookTime (minutes)')
axes[0].set_ylabel('Count')

# Plot for 'PrepTime'
recipes_df['PrepTime'].value_counts().nlargest(15).plot(kind='bar', ax=axes[1], color='salmon')
axes[1].set_title('PrepTime Counts')
axes[1].set_xlabel('PrepTime (minutes)')
axes[1].set_ylabel('Count')

# Plot for 'TotalTime'
recipes_df['TotalTime'].value_counts().nlargest(15).plot(kind='bar', ax=axes[2], color='lightgreen')
axes[2].set_title('TotalTime Counts')
axes[2].set_xlabel('TotalTime (minutes)')
axes[2].set_ylabel('Count')

# Adjust layout
plt.tight_layout()
plt.show()


df = pd.read_csv("../Data/Food_com/reviews.csv")
df.head()
df.isnull().sum()
df.dropna(inplace=True)
df.isnull().sum()
# Count the occurrences of each rating
review_counts = df['Rating'].value_counts()

# Plot the pie chart without percentage labels
plt.figure(figsize=(8, 8))
wedges, texts, _ = plt.pie(review_counts, labels=None, autopct='', startangle=90, colors=plt.cm.Paired.colors)

# Create percentage labels
percentage_labels = [f'{p:.1f}%' for p in review_counts / review_counts.sum() * 100]

# Create legend with ratings and percentage labels
legend_labels = [f'{rating} ({percentage})' for rating, percentage in zip(review_counts.index, percentage_labels)]
plt.legend(wedges, legend_labels, title='Ratings', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.title('Distribution of Ratings')
plt.show()

# Count the number of recipes per author
author_counts = df['AuthorId'].value_counts()

# Plot the top authors by the number of recipes
top_authors = author_counts.nlargest(50)  # Adjust the number as needed

plt.figure(figsize=(10, 6))
top_authors.plot(kind='bar', color='skyblue')
plt.title('Top Authors by Number of Reviews')
plt.xlabel('Author (ID_Name)')
plt.ylabel('Number of Reviews')
plt.show()

len(df)