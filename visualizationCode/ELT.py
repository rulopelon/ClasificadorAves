import os
import librosa
import pandas as pd
import matplotlib.pyplot as plt



# Definir la ruta al directorio donde se encuentran los archivos de audio y al archivo CSV
audio_dir = 'dataset'
csv_file_path_1 = 'train_extended.csv'
csv_file_path_2 = 'train_extended2.csv'

# Función para extraer características de un archivo de audio
def extract_features(file_path):
    # Cargar el archivo de audio
    y, sr = librosa.load(file_path)
    
    # Calcular características básicas
    duration = librosa.get_duration(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    
    # Calculando el tempo
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]

    # Calculando la energía del espectro
    S, phase = librosa.magphase(librosa.stft(y))
    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S).mean()

    return {
        'filename': os.path.basename(file_path),
        'folder': os.path.basename(os.path.dirname(file_path)),
        'duration': duration,
        'sampling_rate': sr,
        'spectral_centroid': spectral_centroid,
        'zero_crossing_rate': zero_crossing_rate,
        'spectral_rolloff': spectral_rolloff,
        'tempo': tempo,        'spectral_bandwidth': spectral_bandwidth
    }



# Extraer y transformar los datos
data = []
for dirpath, dirnames, filenames in os.walk(audio_dir):
    for file_name in filenames:
        if file_name.lower().endswith('.mp3'):
            file_path = os.path.join(dirpath, file_name)
            try:
                features = extract_features(file_path)
                data.append(features)
            except:
                print("No se ha podido cargar")

# Cargar los datos extraídos en un DataFrame de pandas

audio_df = pd.DataFrame(data)

# Cargar el CSV
csv_df_1 = pd.read_csv(csv_file_path_1)
csv_df_2 = pd.read_csv(csv_file_path_2)

csv_df_final = pd.concat([csv_df_1, csv_df_2], ignore_index=True)


print("Información")
print(csv_df_final.info())

print("Descripción:")
print(csv_df_final.info())

# Unir los DataFrames
combined_df = pd.merge(csv_df_final, audio_df, on='filename', how='inner')

combined_df.to_csv('dataframeTotal.csv', index=False)

# Mostrar el DataFrame resultante
print(combined_df.head())

"""
# Histograma de la duración de los audios
plt.figure(figsize=(10, 6))
plt.hist(combined_df['duration_x'], bins=30, color='blue', alpha=0.7)
plt.title('Histograma de la Duración de los Audios')
plt.xlabel('Duración (segundos)')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()

# Diagrama de dispersión entre la duración y el centroide espectral
plt.figure(figsize=(10, 6))
plt.scatter(combined_df['duration_x'], combined_df['spectral_centroid'], color='red')
plt.title('Relación entre Duración y Centroide Espectral')
plt.xlabel('Duración (segundos)')
plt.ylabel('Centroide Espectral (frecuencia media)')
plt.grid(True)
plt.show()

# Gráfico de caja para la frecuencia de muestreo
sampling_rates = audio_df['sampling_rate'].value_counts()

# Generar el gráfico de pastel
plt.figure(figsize=(8, 8))
plt.pie(sampling_rates, labels=sampling_rates.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired(range(len(sampling_rates))))
plt.title('Distribución de la Frecuencia de Muestreo')
plt.show()

# Histograma del Tempo
plt.figure(figsize=(10, 6))
plt.hist(audio_df['tempo'], bins=30, color='green', alpha=0.7)
plt.title('Histograma del Tempo')
plt.xlabel('Tempo (BPM)')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()

# Histograma de la Tasa de Cruce por Cero
plt.figure(figsize=(10, 6))
plt.hist(audio_df['zero_crossing_rate'], bins=30, color='purple', alpha=0.7)
plt.title('Histograma de Tasa de Cruce por Cero')
plt.xlabel('Tasa de Cruce por Cero')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()

# Diagrama de dispersión entre el Tempo y el Centroide Espectral
plt.figure(figsize=(10, 6))
plt.scatter(audio_df['tempo'], audio_df['spectral_centroid'], color='blue')
plt.title('Relación entre Tempo y Centroide Espectral')
plt.xlabel('Tempo (BPM)')
plt.ylabel('Centroide Espectral (frecuencia media)')
plt.grid(True)
plt.show()

# Gráfico de caja para el Rolloff Espectral
plt.figure(figsize=(10, 6))
plt.boxplot(audio_df['spectral_rolloff'])
plt.title('Distribución del Rolloff Espectral')
plt.ylabel('Rolloff Espectral (Hz)')
plt.grid(True)
plt.show()

# Gráfico de caja para el Ancho de Banda Espectral
plt.figure(figsize=(10, 6))
plt.boxplot(audio_df['spectral_bandwidth'])
plt.title('Distribución del Ancho de Banda Espectral')
plt.ylabel('Ancho de Banda Espectral (Hz)')
plt.grid(True)
plt.show()

"""