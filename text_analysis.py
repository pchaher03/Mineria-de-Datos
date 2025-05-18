import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_csv('./data/cleaned_online_retail.csv')

# Eliminar valores nulos en la columna Description
df = df.dropna(subset=['Description'])

# Combinar todos los textos en un solo string
text = ' '.join(df['Description'].astype(str).tolist()).lower()

# Opcional: eliminar palabras comunes (stopwords)
stopwords = set(STOPWORDS)
stopwords.update(["http", "https", "www", "please", "colour"])  # puedes agregar más si gustas

# Crear WordCloud
wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color='white',
    stopwords=stopwords,
    colormap='viridis'
).generate(text)

# Mostrar y guardar la imagen
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.savefig('./plots/wordcloud_description.png')
plt.close()

print("Word cloud creada y guardada en 'plots/wordcloud_description.png'")
