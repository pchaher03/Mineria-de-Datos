import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os
from datetime import timedelta

# Asegurar que la carpeta de gráficas existe
os.makedirs("plots", exist_ok=True)

def main():
    # Cargar los datos
    df = pd.read_csv('./data/cleaned_online_retail.csv', parse_dates=['InvoiceDate'])

    # Preprocesamiento
    df = df.dropna(subset=['Quantity', 'InvoiceDate'])
    df = df[df['Quantity'] > 0]

    # Agrupar por mes y sumar cantidades vendidas
    df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M')
    monthly_sales = df.groupby('InvoiceMonth')['Quantity'].sum().reset_index()
    monthly_sales['InvoiceMonth'] = monthly_sales['InvoiceMonth'].dt.to_timestamp()

    # Crear variable numérica de tiempo (X) y la cantidad (y)
    monthly_sales['MonthIndex'] = np.arange(len(monthly_sales))
    X = monthly_sales[['MonthIndex']]
    y = monthly_sales['Quantity']

    # Entrenar modelo de regresión lineal
    model = LinearRegression()
    model.fit(X, y)

    # Predecir valores futuros
    future_months = 6
    future_index = np.arange(len(monthly_sales), len(monthly_sales) + future_months).reshape(-1, 1)
    future_predictions = model.predict(future_index)

    # Construir DataFrame con fechas futuras y predicciones
    last_date = monthly_sales['InvoiceMonth'].iloc[-1]
    future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(future_months)]
    forecast_df = pd.DataFrame({
        'InvoiceMonth': future_dates,
        'PredictedQuantity': future_predictions
    })

    # Graficar ventas reales + predicciones
    plt.figure(figsize=(10,6))
    plt.plot(monthly_sales['InvoiceMonth'], monthly_sales['Quantity'], label='Ventas reales')
    plt.plot(forecast_df['InvoiceMonth'], forecast_df['PredictedQuantity'], label='Predicción (Linear Regression)', linestyle='--')
    plt.xlabel('Fecha')
    plt.ylabel('Cantidad vendida')
    plt.title('Pronóstico de Ventas (por mes)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./plots/sales_forecast.png')
    plt.close()

    # Mostrar datos
    print("Predicción de los próximos 6 meses:")
    print(forecast_df)

if __name__ == '__main__':
    main()
