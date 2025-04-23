import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# === Load dataset ===
df = pd.read_csv('data/cleaned_online_retail.csv')

# === Create Revenue column ===
df['Revenue'] = df['Quantity'] * df['UnitPrice']

# === Create Binary Target: HighRevenue ===
median_revenue = df['Revenue'].median()
df['HighRevenue'] = (df['Revenue'] > median_revenue).astype(int)

# === Select Features and Target ===
X = df[['Quantity', 'UnitPrice']]
y = df['HighRevenue']

# === Scale Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# === Train KNN Classifier ===
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# === Predict and Evaluate ===
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# === Visualization 1: KNN Predicted Labels ===
plt.figure(figsize=(10, 6))
palette = {0: "dodgerblue", 1: "darkorange"}

sns.scatterplot(
    x=X_test[:, 0], y=X_test[:, 1],
    hue=y_pred, palette=palette, alpha=0.6, edgecolor='k'
)
plt.title('KNN Predicted Classification: High vs Low Revenue')
plt.xlabel('Quantity (scaled)')
plt.ylabel('UnitPrice (scaled)')
plt.grid(True, linestyle='--', alpha=0.3)
plt.xlim(-1, 3)   # Zoom into main cluster
plt.ylim(-1, 3)
plt.legend(title="Predicted HighRevenue", loc='upper right')
plt.tight_layout()
plt.savefig('plots/knn_highrevenue_classification.png')
plt.close()

# === Visualization 2: Actual vs Predicted Labels ===
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=X_test[:, 0], y=X_test[:, 1],
    hue=y_test, style=y_pred, palette=palette,
    alpha=0.6, edgecolor='k'
)
plt.title('KNN Classification: Actual vs Predicted High Revenue')
plt.xlabel('Quantity (scaled)')
plt.ylabel('UnitPrice (scaled)')
plt.grid(True, linestyle='--', alpha=0.3)
plt.xlim(-1, 3)
plt.ylim(-1, 3)
plt.legend(title="Actual HighRevenue / Marker=Predicted", loc='upper right')
plt.tight_layout()
plt.savefig('plots/knn_actual_vs_predicted.png')
plt.close()
