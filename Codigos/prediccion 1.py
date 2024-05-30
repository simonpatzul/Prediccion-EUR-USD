import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Definir la ruta relativa al archivo CSV y al modelo entrenado
ruta_csv = os.path.join('..', 'Datos', 'combined_data1.csv')
ruta_modelo = os.path.join('..', 'Modelos', 'modelo_entrenado.h5')  # Asegúrate de que esta ruta sea correcta

# Imprime la ruta actual para depuración
print(f"Buscando el archivo en: {os.path.abspath(ruta_csv)}")
print(f"Buscando el modelo en: {os.path.abspath(ruta_modelo)}")

# Verifica si el archivo CSV y el modelo existen en las ubicaciones esperadas
if not os.path.isfile(ruta_csv):
    raise FileNotFoundError(f"El archivo {ruta_csv} no se encuentra en el directorio 'Datos'.")
if not os.path.isfile(ruta_modelo):
    raise FileNotFoundError(f"El archivo del modelo {ruta_modelo} no se encuentra en el directorio 'Modelos'.")

# Carga el archivo CSV
data = pd.read_csv(ruta_csv)

# Muestra las primeras filas del DataFrame para confirmar que se cargó correctamente
print(data.head())

# Preprocesa tus datos
tamaño_entrenamiento = int(0.8 * len(data))
datos_prueba = data[tamaño_entrenamiento:]

escalador = MinMaxScaler(feature_range=(0, 1))
datos_prueba_escalados = escalador.fit_transform(datos_prueba[['Open', 'Close']].values)

# Crea tu conjunto de datos
def crear_conjunto_de_datos(conjunto_de_datos, paso_temporal=150):
    dataX, dataY = [], []
    for i in range(len(conjunto_de_datos) - paso_temporal - 1):
        a = conjunto_de_datos[i:(i + paso_temporal), :]
        dataX.append(a)
        dataY.append(conjunto_de_datos[i + paso_temporal, :])  # Guarda ambos precios de apertura y cierre
    return np.array(dataX), np.array(dataY)

# Prepara los conjuntos de datos de prueba
X_prueba, y_prueba = crear_conjunto_de_datos(datos_prueba_escalados)
X_prueba = X_prueba.reshape(X_prueba.shape[0], X_prueba.shape[1], 2)

# Carga el modelo previamente entrenado
modelo = load_model(ruta_modelo)

# Haz predicciones para las últimas ventanas de datos
num_predicciones = 20  # Ajusta según la cantidad de predicciones que desees realizar
predicciones = []
real_open = []
real_close = []

for i in range(num_predicciones):
    entrada_prueba = datos_prueba_escalados[-(150 + i):-i if i != 0 else None]
    entrada_prueba = entrada_prueba.reshape(1, 150, 2)
    prediccion = modelo.predict(entrada_prueba)
    prediccion = escalador.inverse_transform(prediccion)
    predicciones.append(prediccion[0])
    real_open.append(datos_prueba['Open'].iloc[-(i + 1)])
    real_close.append(datos_prueba['Close'].iloc[-(i + 1)])

# Convertir las predicciones en un DataFrame
predicciones = np.array(predicciones)
data = {
    'Pred Open': predicciones[:, 0],
    'Pred Close': predicciones[:, 1],
    'Real Open': real_open,
    'Real Close': real_close
}
df = pd.DataFrame(data)

# Mostrar las últimas predicciones
print("\nÚltimas predicciones:")
print(df)

# Graficar las predicciones vs los valores reales
plt.plot(df.index, df['Real Close'], label='Real Close')
plt.plot(df.index, df['Pred Close'], label='Pred Close')
plt.title('Real vs Predicted Close Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Predecir el próximo valor de Open y Close utilizando la última fila de datos
ultima_ventana = datos_prueba_escalados[-150:].reshape(1, 150, 2)
proxima_prediccion = modelo.predict(ultima_ventana)
proxima_prediccion_inversa = escalador.inverse_transform(proxima_prediccion)

print("\nPróxima predicción:")
print(f"Open: {proxima_prediccion_inversa[0][0]}, Close: {proxima_prediccion_inversa[0][1]}")
# Calcular desfase usando una media móvil
def calcular_desfase_media_movil(predicciones, valores_reales, ventana=8):
    desfases = predicciones - valores_reales
    desfase_movil = np.convolve(desfases, np.ones(ventana) / ventana, mode='valid')
    return desfase_movil

# Aplicar la función de media móvil a las últimas N predicciones y valores reales
ventana = 1
desfase_movil = calcular_desfase_media_movil(predicciones[:, 1], real_close[:len(predicciones)], ventana=ventana)

# Ajustar las predicciones restando el desfase calculado
predicciones_ajustadas = predicciones[ventana - 1:] - desfase_movil[:, np.newaxis]

# Mostrar las últimas 20 predicciones ajustadas
print("\nÚltimas 20 predicciones ajustadas:")
for i in range(len(predicciones_ajustadas)):
    print(f"Predicción: {predicciones_ajustadas[i]}, Real: {real_close[ventana - 1 + i]}")

# Graficar las últimas 20 predicciones ajustadas
plt.plot(range(len(real_close[ventana - 1:])), real_close[ventana - 1:], label='Real Close')
plt.plot(range(len(predicciones_ajustadas)), predicciones_ajustadas[:, 1], label='Pred Close Ajustado')
plt.title('Real vs Predicted Close Price (Ajustado)')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
