import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # this is needed for plots to appear in the terminal,
# must be defined before pyplot  matplotlib.pyplot imported
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import missingno as msno
from pathlib import Path

###### Data Cleaning ######
current_dir = Path.cwd()
data = pd.read_csv(current_dir / 'chocolate-sales' / 'Chocolate Sales.csv')
print(data.head())
print(data.info())
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data['amount_clean'] = data['Amount'].str.replace(r'[\$,€£]', '', regex=True)
data['amount_clean'] = data['amount_clean'].str.replace(',', '', regex=False)
data['amount_clean'] = data['amount_clean'].astype(float)
data['Boxes Shipped'] = pd.to_numeric(data['Boxes Shipped'], errors='coerce')
print(data.head())
print("\n\n")


###### missing values validation ######
plot_missing = False
# plot_missing = True

if plot_missing:
    # Set the style for better visualization
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Matrix plot
    msno.matrix(data[['amount_clean', 'Date', 'Boxes Shipped']], 
                ax=ax1,
                sparkline=False,
                color=(0.2, 0.2, 0.2))
    ax1.set_title('Missing Values Matrix Plot', fontsize=14, pad=20)
    ax1.set_ylabel('Row Index', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)

    # Bar plot
    msno.bar(data[['amount_clean', 'Date', 'Boxes Shipped']], 
            ax=ax2,
            color='#2ecc71')
    ax2.set_title('Missing Values Distribution', fontsize=14, pad=20)
    ax2.set_ylabel('Number of Non-Missing Values', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()


# Print summary of missing values with better formatting
print("\nMissing Values Summary:")
missing_summary = data[['amount_clean', 'Date', 'Boxes Shipped']].isnull().sum()
total_rows = len(data)
missing_percentage = (missing_summary / total_rows * 100).round(2)

summary_data = pd.DataFrame({
    'Missing Count': missing_summary,
    'Missing Percentage': missing_percentage,
    'Total Rows': total_rows
})
print(summary_data)


###### EDA ######

numeric_cols = ['amount_clean', 'Boxes Shipped']
categorical_cols = ['Sales Person', 'Country', 'Product']

# Plot histograms for numeric variables
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(data[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# Plot count plots for categorical variables
for col in categorical_cols:
    plt.figure(figsize=(10, 4))
    sns.countplot(data=data, x=col, order=data[col].value_counts().index)
    plt.title(f'Count Plot of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Boxen and violin plots for numeric columns to inspect for outliers
for col in numeric_cols:
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxenplot(y=data[col], ax=ax[0])
    ax[0].set_title(f'Boxen Plot of {col}')
    sns.violinplot(y=data[col], ax=ax[1])
    ax[1].set_title(f'Violin Plot of {col}')
    plt.tight_layout()
    plt.show()

# Pairplot to inspect relationships. Since there are only 2 numeric columns, this is a simple visualization.
sns.pairplot(data[numeric_cols])
plt.show()


###### Correlation Matrix ######

# Calculate the correlation matrix
correlation_matrix = data[numeric_cols].corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numeric Variables')
plt.tight_layout()
plt.show()

print(correlation_matrix)

