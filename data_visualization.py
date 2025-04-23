import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv('data/cleaned_online_retail.csv')
df['Revenue'] = df['Quantity'] * df['UnitPrice']
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Month'] = df['InvoiceDate'].dt.to_period('M').astype(str)

# Create output folder
os.makedirs('plots', exist_ok=True)

# === 1. Pie Chart: Top 5 countries by revenue ===
top_countries = df.groupby('Country')['Revenue'].sum().sort_values(ascending=False).head(5)
plt.figure(figsize=(8, 6))
top_countries.plot.pie(autopct='%1.1f%%', startangle=90)
plt.title('Top 5 Countries by Revenue')
plt.ylabel('')
plt.tight_layout()
plt.savefig('plots/pie_top5_countries.png')
plt.close()

# === 2 & 3: Line Plots with For Loop ===
metrics = {
    'Revenue': df.groupby('Month')['Revenue'].sum(),
    'Orders': df.groupby('Month')['InvoiceNo'].nunique()
}

colors = ['#4caf50', '#f44336']

for i, (label, data) in enumerate(metrics.items()):
    plt.figure(figsize=(12, 6))
    data.plot(marker='o', color=colors[i])
    plt.title(f'Monthly {label}')
    plt.xlabel('Month')
    plt.ylabel(label)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'plots/line_monthly_{label.lower()}.png')
    plt.close()

# === 4. Bar Chart: Top 10 most sold products ===
top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_products.values, y=top_products.index, palette='viridis')
plt.title('Top 10 Best-Selling Products')
plt.xlabel('Total Quantity Sold')
plt.tight_layout()
plt.savefig('plots/bar_top_products.png')
plt.close()

# === 5. Bar Chart: Average spending per customer by country (top 10) ===
avg_spending = df.groupby('Country')['Revenue'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=avg_spending.values, y=avg_spending.index, palette='coolwarm')
plt.title('Average Revenue per Transaction by Country')
plt.xlabel('Average Revenue')
plt.tight_layout()
plt.savefig('plots/bar_avg_revenue_country.png')
plt.close()

print("✅ Updated visualizations saved to 'plots' folder.")
