import pandas as pd
import matplotlib.pyplot as plt

# Suponiendo que tu DataFrame se llama df y la columna es 'species'
df = pd.read_csv("dataframeTotal.csv")
# Contar las ocurrencias de cada especie
species_counts = df['species'].value_counts()

# Crear un gráfico de barras
species_counts.plot(kind='bar')

num_unique_species = df['species'].nunique()

print(f'Número de especies diferentes: {num_unique_species}')

# Añadir etiquetas y título
plt.xlabel('Especies')
plt.ylabel('Número de datos')
plt.title('Número de datos por especie')

# Mostrar el gráfico
plt.show()