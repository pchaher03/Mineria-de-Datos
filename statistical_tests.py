import pandas as pd
from scipy.stats import f_oneway, ttest_ind, kruskal

# Load cleaned dataset
df = pd.read_csv('data/cleaned_online_retail.csv')

# Prepare the data
df['Revenue'] = df['Quantity'] * df['UnitPrice']

# Choose 3 countries with highest number of transactions
top_countries = df['Country'].value_counts().head(3).index.tolist()
df_top = df[df['Country'].isin(top_countries)]

# Group revenue data by country
grouped_revenue = [group['Revenue'].values for name, group in df_top.groupby('Country')]

# === 1. ANOVA Test ===
anova_result = f_oneway(*grouped_revenue)
print("\n📊 ANOVA test:")
print(f"F-statistic: {anova_result.statistic:.4f}")
print(f"P-value: {anova_result.pvalue:.4f}")

# === 2. Kruskal-Wallis Test ===
kruskal_result = kruskal(*grouped_revenue)
print("\n📊 Kruskal-Wallis test:")
print(f"H-statistic: {kruskal_result.statistic:.4f}")
print(f"P-value: {kruskal_result.pvalue:.4f}")

# === 3. T-test (example between first two countries only) ===
country_1, country_2 = top_countries[:2]
rev1 = df_top[df_top['Country'] == country_1]['Revenue']
rev2 = df_top[df_top['Country'] == country_2]['Revenue']
ttest_result = ttest_ind(rev1, rev2, equal_var=False)

print(f"\n📊 T-test between {country_1} and {country_2}:")
print(f"T-statistic: {ttest_result.statistic:.4f}")
print(f"P-value: {ttest_result.pvalue:.4f}")
