import pandas as pd

# Load the cleaned data
df = pd.read_csv('data/cleaned_online_retail.csv')

# Show basic statistics for numeric columns
print(df.describe())

# Show unique countries and customers
print("\nUnique countries:", df['Country'].nunique())
print("Unique customers:", df['CustomerID'].nunique())

# Total revenue per country

df['Revenue'] = df['Quantity'] * df['UnitPrice']
country_revenue = df.groupby('Country')['Revenue'].sum().sort_values(ascending=False)
print(country_revenue)

# Total orders per customer

orders_per_customer = df.groupby('CustomerID')['InvoiceNo'].nunique()
print(orders_per_customer.describe())

# Average quantity per product

avg_quantity = df.groupby('Description')['Quantity'].mean().sort_values(ascending=False)
print(avg_quantity.head(10))
