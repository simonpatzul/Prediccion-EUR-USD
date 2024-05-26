import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import L2
import tensorflow as tf
import random

import pandas as pd
import os

# Define la ruta relativa al archivo CSV desde el script actual
filename = os.path.join('..', 'Datos', 'combined_data1.csv')

# Imprime la ruta actual para depuración
print(f"Buscando el archivo en: {os.path.abspath(filename)}")

# Verifica si el archivo existe en la ubicación esperada
if not os.path.isfile(filename):
    raise FileNotFoundError(f"El archivo {filename} no se encuentra en el directorio 'Datos'.")

# Carga el archivo CSV
data = pd.read_csv(filename)

# Muestra las primeras filas del DataFrame para confirmar que se cargó correctamente
print(data.head())
# Divide tus datos en conjuntos de entrenamiento y prueba
tamaño_entrenamiento = int(0.8 * len(data))
datos_entrenamiento, datos_prueba = data[:tamaño_entrenamiento], data[tamaño_entrenamiento:]

# Preprocesa tus datos
escalador = MinMaxScaler(feature_range=(0, 1))
datos_entrenamiento_escalados = escalador.fit_transform(datos_entrenamiento[['Open', 'Close']].values)
datos_prueba_escalados = escalador.transform(datos_prueba[['Open', 'Close']].values)

# Crea tu conjunto de datos
def crear_conjunto_de_datos(conjunto_de_datos, paso_temporal=300):
    dataX, dataY = [], []
    for i in range(len(conjunto_de_datos)-paso_temporal-1):
        a = conjunto_de_datos[i:(i+paso_temporal), :]
        dataX.append(a)
        dataY.append(conjunto_de_datos[i + paso_temporal, :])  # Guarda ambos precios de apertura y cierre
    return np.array(dataX), np.array(dataY)

# Prepara los conjuntos de datos de entrenamiento y prueba
X_entrenamiento, y_entrenamiento = crear_conjunto_de_datos(datos_entrenamiento_escalados)
X_prueba, y_prueba = crear_conjunto_de_datos(datos_prueba_escalados)

# Reforma tus datos
X_entrenamiento = X_entrenamiento.reshape(X_entrenamiento.shape[0], X_entrenamiento.shape[1], 2)
X_prueba = X_prueba.reshape(X_prueba.shape[0], X_prueba.shape[1], 2)

# Crea tu modelo
modelo = Sequential()
modelo.add(LSTM(units=21, return_sequences=True, input_shape=(300, 2), kernel_regularizer=L2(0.001)))
modelo.add(LSTM(units=13
                , kernel_regularizer=L2(0.001)))

modelo.add(Dense(units=2))  # Predice tanto el precio de apertura como el de cierre
modelo.compile(optimizer='adam', loss='mean_squared_error')

# Definir la parada temprana
early_stopping = EarlyStopping(monitor='val_loss', patience=5
                               , restore_best_weights=True)

# Entrena el modelo con parada temprana
historial_entrenamiento = modelo.fit(X_entrenamiento, y_entrenamiento, epochs=150, batch_size=22,
                                     validation_data=(X_prueba, y_prueba), callbacks=[early_stopping])



# Evalúa el modelo en los datos de entrenamiento
predicciones_entrenamiento = modelo.predict(X_entrenamiento)
mse_entrenamiento = mean_squared_error(y_entrenamiento, predicciones_entrenamiento)
mae_entrenamiento = mean_absolute_error(y_entrenamiento, predicciones_entrenamiento)
r2_entrenamiento = r2_score(y_entrenamiento, predicciones_entrenamiento)

print("Métricas de entrenamiento:")
print("Error Cuadrático Medio (MSE): ", mse_entrenamiento)
print("Error Absoluto Medio (MAE): ", mae_entrenamiento)
print("Puntuación R2: ", r2_entrenamiento)

# Evalúa el modelo en los datos de prueba
predicciones_prueba = modelo.predict(X_prueba)
mse_prueba = mean_squared_error(y_prueba, predicciones_prueba)
mae_prueba = mean_absolute_error(y_prueba, predicciones_prueba)
r2_prueba = r2_score(y_prueba, predicciones_prueba)

print("\nMétricas de prueba:")
print("Error Cuadrático Medio (MSE): ", mse_prueba)
print("Error Absoluto Medio (MAE): ", mae_prueba)
print("Puntuación R2: ", r2_prueba)

# Haz predicciones
entrada_prueba = datos_prueba_escalados[-300:]
entrada_prueba = entrada_prueba.reshape(1, 300, 2)
prediccion = modelo.predict(entrada_prueba)

# Ajusta la forma de la predicción para que coincida con el escalador
prediccion = prediccion.reshape(-1, 2)

# Aplica la transformación inversa
prediccion = escalador.inverse_transform(prediccion)
print("Última predicción de Close: ", prediccion[0][1])
print("Última predicción de Open: ", prediccion[0][0])

# Curva de Aprendizaje
plt.plot(historial_entrenamiento.history['loss'], label='Pérdida en Entrenamiento')
plt.plot(historial_entrenamiento.history['val_loss'], label='Pérdida en Validación')
plt.title('Curva de Aprendizaje del Modelo')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

# Haz predicciones para las últimas 5 ventanas de datos
num_predicciones = 5
predicciones = []
real_close = []

for i in range(num_predicciones):
    entrada_prueba = datos_prueba_escalados[-(300 + i):-i if i != 0 else None]
    entrada_prueba = entrada_prueba.reshape(1, 300, 2)
    prediccion = modelo.predict(entrada_prueba)
    prediccion = escalador.inverse_transform(prediccion)
    predicciones.append(prediccion[0])
    real_close.append(datos_prueba['Close'].iloc[-(i + 1)])

# Convertir las predicciones en un DataFrame
predicciones = np.array(predicciones)
data = {
    'Open': predicciones[:, 0],
    'Close': predicciones[:, 1],
    'Real Close': real_close
}
df = pd.DataFrame(data)

# Mostrar las últimas 5 predicciones
print("\nÚltimas 5 predicciones:")
print(df)

