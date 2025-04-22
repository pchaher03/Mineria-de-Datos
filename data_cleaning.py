import pandas as pd
import numpy as np
import os

# === Cargar el dataset ===
df = pd.read_csv('data/online_retail.csv', encoding='ISO-8859-1')

# === Información básica ===
print("Forma original:", df.shape)
print("Valores nulos por columna:\n", df.isnull().sum())

# === Eliminar duplicados ===
df.drop_duplicates(inplace=True)

# === Eliminar filas sin CustomerID ===
df.dropna(subset=['CustomerID'], inplace=True)

# === Convertir InvoiceDate a formato fecha ===
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# === Filtrar Quantity y UnitPrice mayores a 0 ===
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# === Guardar dataset limpio ===
output_path = 'data/cleaned_online_retail.csv'
df.to_csv(output_path, index=False)
print(f"\n✅ Dataset limpio guardado en: {output_path}")
print("Nueva forma:", df.shape)
