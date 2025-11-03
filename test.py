# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
import matplotlib 
import seaborn as sns
matplotlib.rcParams["figure.figsize"] = (20,10)
import re

# %%

# Load the data
df = pd.read_csv('../data/cleaned_data.csv')

# Basic exploration
print(df.head())  # View first few rows
print(df.info())  # Data types and non-null counts
print(df.describe())  # Summary statistics for numerical features

# Check for missing values
print(df.isnull().sum())

# Visualize distributions
sns.histplot(df['Price'], kde=True)
plt.title('Price Distribution')
plt.show()

# Correlation heatmap for numerical features
numerical_cols = ['Latitude', 'Longitude', 'Rooms', 'Bathrooms', 'Car Parks', 'Size', 'Storeys']
corr = df[numerical_cols + ['Price']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Boxplots for categorical vs. Price
categorical_cols = ['Location', 'Type', 'Furnishing', 'Position']
for col in categorical_cols:
    sns.boxplot(x=col, y='Price', data=df)
    plt.title(f'{col} vs. Price')
    plt.xticks(rotation=45)
    plt.show()


