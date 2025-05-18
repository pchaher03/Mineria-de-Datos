import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# Asegurar que la carpeta de gráficos existe
os.makedirs("plots", exist_ok=True)

def main():
    # 1. Cargar datos
    df = pd.read_csv('./data/cleaned_online_retail.csv')

    # 2. Preprocesamiento básico
    df = df.dropna(subset=['CustomerID'])  # eliminar filas sin cliente
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]  # valores válidos

    # 3. Agregación por cliente
    customer_data = df.groupby('CustomerID').agg({
        'Quantity': 'sum',
        'UnitPrice': 'mean'
    }).reset_index()

    # 4. Escalado de características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(customer_data[['Quantity', 'UnitPrice']])

    # 5. Método del codo
    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    # Guardar gráfico del codo
    plt.figure(figsize=(8,4))
    plt.plot(k_range, inertia, 'bo-')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inercia')
    plt.title('Método del Codo')
    plt.grid(True)
    plt.savefig('./plots/elbow_method.png')
    plt.close()

    # 6. Elegir k (puedes ajustar esto según la gráfica)
    k_optimo = 4
    kmeans = KMeans(n_clusters=k_optimo, random_state=42)
    customer_data['cluster'] = kmeans.fit_predict(X_scaled)

    # 7. Mostrar primeros resultados
    print(customer_data.head())

    # 8. Guardar gráfico de clustering
    plt.figure(figsize=(8,6))
    plt.scatter(customer_data['Quantity'], customer_data['UnitPrice'], c=customer_data['cluster'], cmap='viridis')
    plt.xlabel('Cantidad Total')
    plt.ylabel('Precio Promedio Unitario')
    plt.title(f'Clusters de Clientes (k={k_optimo})')
    plt.grid(True)
    plt.savefig('./plots/customer_clusters.png')
    plt.close()

    print("Clustering completado. Gráficos guardados en la carpeta 'plots'.")

if __name__ == '__main__':
    main()
