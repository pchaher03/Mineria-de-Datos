import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv('data/cleaned_online_retail.csv')

# Create Revenue column
df['Revenue'] = df['Quantity'] * df['UnitPrice']

# === Filter Outliers ===
df_filtered = df[(df['Quantity'] > 0) & (df['Quantity'] < 10000)]

# === Linear Model: Revenue vs Quantity ===
X = df_filtered[['Quantity']]  # independent variable
y = df_filtered['Revenue']     # dependent variable

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

r2 = r2_score(y, y_pred)
print(f"R² Score (Revenue vs Quantity, filtered): {r2:.4f}")

# === Plot ===
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Quantity', y='Revenue', data=df_filtered, alpha=0.4)
plt.plot(df_filtered['Quantity'], y_pred, color='red')
plt.title('Linear Regression (Filtered): Revenue vs Quantity')
plt.xlabel('Quantity')
plt.ylabel('Revenue')
plt.tight_layout()
plt.savefig('plots/linear_regression_quantity_revenue_filtered.png')
plt.close()
