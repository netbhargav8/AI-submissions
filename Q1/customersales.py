# %% [markdown]
# #### Importing libraries
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
# %% [markdown]
# ####  1: Load the Data
# %%
dataset = pd.read_csv('retail_sales_dataset.csv')
dataset.columns = dataset.columns.str.strip()

print("--- First 5 Rows of the Dataset ---")
print(dataset.head(), "\n")
# %% [markdown]
# 
# #### 2. Taking care of missing data
# %%
dataset = dataset.dropna(subset=['Customer ID'])
# Initialize the Scikit-Learn Imputers
num_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
cat_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

if 'Price' in dataset.columns:
    dataset[['Price']] = num_imputer.fit_transform(dataset[['Price']])
if 'Quantity' in dataset.columns:
    dataset[['Quantity']] = num_imputer.fit_transform(dataset[['Quantity']])

# Impute missing categories (categorical column)
if 'Category' in dataset.columns:
    dataset[['Category']] = cat_imputer.fit_transform(dataset[['Category']])

print("Missing values remaining:", dataset.isnull().sum().sum(), "\n")
# %% [markdown]
# 
# #### 3. Feature Engineering & Calculations
# %%
# Calculate Total Revenue per transaction
dataset['Revenue'] = dataset['Quantity'] * dataset['Price per Unit']

# 3a. Compute total revenue by product category
# FIX: Change 'Category' to match your actual column name (e.g., 'Product Category')
revenue_by_category = dataset.groupby('Product Category')['Revenue'].sum().reset_index()
print("--- Total Revenue by Product Category ---")
print(revenue_by_category, "\n")

# 3b. Identify the top 10 customers
top_10_customers = dataset.groupby('Customer ID')['Revenue'].sum().nlargest(10).reset_index()
print("--- Top 10 Customers ---")
print(top_10_customers, "\n")
# %% [markdown]
# 
# #### 4. Visualizing Monthly Sales Trend
# 
# %%
# Convert Date to datetime format and group by month
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset['Month'] = dataset['Date'].dt.to_period('M')

monthly_sales = dataset.groupby('Month')['Revenue'].sum().reset_index()
monthly_sales['Month'] = monthly_sales['Month'].astype(str) # Convert to string for plotting compatibility

# Plotting the Line Chart
plt.figure(figsize=(10, 5))
plt.plot(monthly_sales['Month'], monthly_sales['Revenue'], marker='o', color='g', linestyle='-', linewidth=2)

# Chart styling
plt.title('Monthly Sales Revenue Trend', fontsize=14, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Total Revenue ($)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45)

# Render the plot
plt.tight_layout()
plt.show()