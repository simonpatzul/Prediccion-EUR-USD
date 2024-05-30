import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import LSTM, Conv1D, Dense, Flatten, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import L2
import pingouin as pg
import tensorflow as tf
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

# Divide tus datos en conjuntos de entrenamiento y prueba
tamaño_entrenamiento = int(0.8 * len(data))
datos_entrenamiento, datos_prueba = data[:tamaño_entrenamiento], data[tamaño_entrenamiento:]

# Preprocesa tus datos
escalador = MinMaxScaler(feature_range=(0, 1))
datos_entrenamiento_escalados = escalador.fit_transform(datos_entrenamiento[['Open', 'Close']].values)
datos_prueba_escalados = escalador.transform(datos_prueba[['Open', 'Close']].values)

# Crear conjunto de datos para LSTM
def crear_conjunto_de_datos_LSTM(conjunto_de_datos, paso_temporal=150):
    dataX, dataY = [], []
    for i in range(len(conjunto_de_datos)-paso_temporal-1):
        a = conjunto_de_datos[i:(i+paso_temporal), :]
        dataX.append(a)
        dataY.append(conjunto_de_datos[i + paso_temporal, :])  # Guarda ambos precios de apertura y cierre
    return np.array(dataX), np.array(dataY)

# Preparar los conjuntos de datos de entrenamiento y prueba para LSTM
X_entrenamiento_lstm, y_entrenamiento_lstm = crear_conjunto_de_datos_LSTM(datos_entrenamiento_escalados)
X_prueba_lstm, y_prueba_lstm = crear_conjunto_de_datos_LSTM(datos_prueba_escalados)

# Reformar los datos para LSTM
X_entrenamiento_lstm = X_entrenamiento_lstm.reshape(X_entrenamiento_lstm.shape[0], X_entrenamiento_lstm.shape[1], 2)
X_prueba_lstm = X_prueba_lstm.reshape(X_prueba_lstm.shape[0], X_prueba_lstm.shape[1], 2)

# Crear conjunto de datos para CNN
def crear_conjunto_de_datos_CNN(conjunto_de_datos, paso_temporal=8):
    dataX, dataY = [], []
    for i in range(len(conjunto_de_datos) - paso_temporal - 1):
        a = conjunto_de_datos[i:(i + paso_temporal), :]
        dataX.append(a)
        dataY.append(conjunto_de_datos[i + paso_temporal, :])  # Guarda ambos precios de apertura y cierre
    return np.array(dataX), np.array(dataY)

# Preparar los conjuntos de datos de entrenamiento y prueba para CNN
X_entrenamiento_cnn, y_entrenamiento_cnn = crear_conjunto_de_datos_CNN(datos_entrenamiento_escalados)
X_prueba_cnn, y_prueba_cnn = crear_conjunto_de_datos_CNN(datos_prueba_escalados)

# Reformar los datos para CNN
X_entrenamiento_cnn = X_entrenamiento_cnn.reshape(X_entrenamiento_cnn.shape[0], X_entrenamiento_cnn.shape[1], 2)
X_prueba_cnn = X_prueba_cnn.reshape(X_prueba_cnn.shape[0], X_prueba_cnn.shape[1], 2)

# Crear modelo LSTM
modelo_lstm = Sequential()
modelo_lstm.add(LSTM(units=8, return_sequences=True, input_shape=(150, 2), kernel_regularizer=L2(0.001)))
modelo_lstm.add(LSTM(units=5, kernel_regularizer=L2(0.001)))
modelo_lstm.add(Dense(units=2))
modelo_lstm.compile(optimizer='adam', loss='mean_squared_error')

# Definir la parada temprana para LSTM
early_stopping_lstm = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entrenar el modelo LSTM
historial_entrenamiento_lstm = modelo_lstm.fit(X_entrenamiento_lstm, y_entrenamiento_lstm, epochs=150, batch_size=22,
                                               validation_data=(X_prueba_lstm, y_prueba_lstm), callbacks=[early_stopping_lstm])

# Evaluar el modelo LSTM en los datos de entrenamiento
predicciones_entrenamiento_lstm = modelo_lstm.predict(X_entrenamiento_lstm)
mse_entrenamiento_lstm = mean_squared_error(y_entrenamiento_lstm, predicciones_entrenamiento_lstm)
mae_entrenamiento_lstm = mean_absolute_error(y_entrenamiento_lstm, predicciones_entrenamiento_lstm)
r2_entrenamiento_lstm = r2_score(y_entrenamiento_lstm, predicciones_entrenamiento_lstm)

print("Métricas de entrenamiento LSTM:")
print("Error Cuadrático Medio (MSE): ", mse_entrenamiento_lstm)
print("Error Absoluto Medio (MAE): ", mae_entrenamiento_lstm)
print("Puntuación R2: ", r2_entrenamiento_lstm)

# Evaluar el modelo LSTM en los datos de prueba
predicciones_prueba_lstm = modelo_lstm.predict(X_prueba_lstm)
mse_prueba_lstm = mean_squared_error(y_prueba_lstm, predicciones_prueba_lstm)
mae_prueba_lstm = mean_absolute_error(y_prueba_lstm, predicciones_prueba_lstm)
r2_prueba_lstm = r2_score(y_prueba_lstm, predicciones_prueba_lstm)

print("\nMétricas de prueba LSTM:")
print("Error Cuadrático Medio (MSE): ", mse_prueba_lstm)
print("Error Absoluto Medio (MAE): ", mae_prueba_lstm)
print("Puntuación R2: ", r2_prueba_lstm)

# Curva de Aprendizaje para LSTM
plt.plot(historial_entrenamiento_lstm.history['loss'], label='Pérdida en Entrenamiento LSTM')
plt.plot(historial_entrenamiento_lstm.history['val_loss'], label='Pérdida en Validación LSTM')

# Crear modelo CNN
modelo_cnn = Sequential()
modelo_cnn.add(Conv1D(filters=120, kernel_size=3, activation='relu', input_shape=(8, 2), kernel_regularizer=L2(0.01)))
modelo_cnn.add(MaxPooling1D(pool_size=2))
modelo_cnn.add(Flatten())
modelo_cnn.add(Dense(80, activation='relu'))
modelo_cnn.add(Dense(2))
modelo_cnn.compile(optimizer='adam', loss='mean_squared_error')

# Definir la parada temprana para CNN
early_stopping_cnn = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# Entrenar el modelo CNN
historial_entrenamiento_cnn = modelo_cnn.fit(X_entrenamiento_cnn, y_entrenamiento_cnn, epochs=200, batch_size=21,
                                             validation_data=(X_prueba_cnn, y_prueba_cnn), callbacks=[early_stopping_cnn])

# Evaluar el modelo CNN en los datos de entrenamiento
predicciones_entrenamiento_cnn = modelo_cnn.predict(X_entrenamiento_cnn)
mse_entrenamiento_cnn = mean_squared_error(y_entrenamiento_cnn, predicciones_entrenamiento_cnn)
mae_entrenamiento_cnn = mean_absolute_error(y_entrenamiento_cnn, predicciones_entrenamiento_cnn)
r2_entrenamiento_cnn = r2_score(y_entrenamiento_cnn, predicciones_entrenamiento_cnn)

print("\nMétricas de entrenamiento CNN:")
print("Error Cuadrático Medio (MSE): ", mse_entrenamiento_cnn)
print("Error Absoluto Medio (MAE): ", mae_entrenamiento_cnn)
print("Puntuación R2: ", r2_entrenamiento_cnn)

# Evaluar el modelo CNN en los datos de prueba
predicciones_prueba_cnn = modelo_cnn.predict(X_prueba_cnn)
mse_prueba_cnn = mean_squared_error(y_prueba_cnn, predicciones_prueba_cnn)
mae_prueba_cnn = mean_absolute_error(y_prueba_cnn, predicciones_prueba_cnn)
r2_prueba_cnn = r2_score(y_prueba_cnn, predicciones_prueba_cnn)

print("\nMétricas de prueba CNN:")
print("Error Cuadrático Medio (MSE): ", mse_prueba_cnn)
print("Error Absoluto Medio (MAE): ", mae_prueba_cnn)
print("Puntuación R2: ", r2_prueba_cnn)

# Curva de Aprendizaje para CNN
plt.plot(historial_entrenamiento_cnn.history['loss'], label='Pérdida en Entrenamiento CNN')
plt.plot(historial_entrenamiento_cnn.history['val_loss'], label='Pérdida en Validación CNN')
plt.legend()
plt.show()

# Función para calcular ICC
def calculate_icc(y_true, y_pred):
    data = pd.DataFrame({
        'targets': np.concatenate([y_true, y_true]),
        'ratings': np.concatenate([y_true, y_pred]),
        'raters': ['true'] * len(y_true) + ['pred'] * len(y_true)
    })
    icc = pg.intraclass_corr(data=data, targets='targets', raters='raters', ratings='ratings')
    return icc['ICC'][2]  # ICC[2] is the ICC value for "twoway" and "consistency"

# ICC para LSTM
icc_lstm_open = calculate_icc(y_prueba_lstm[:, 0], predicciones_prueba_lstm[:, 0])
icc_lstm_close = calculate_icc(y_prueba_lstm[:, 1], predicciones_prueba_lstm[:, 1])
print("\nConfiabilidad entre evaluadores (ICC) LSTM:")
print(f"ICC Open: {icc_lstm_open}")
print(f"ICC Close: {icc_lstm_close}")

# ICC para CNN
icc_cnn_open = calculate_icc(y_prueba_cnn[:, 0], predicciones_prueba_cnn[:, 0])
icc_cnn_close = calculate_icc(y_prueba_cnn[:, 1], predicciones_prueba_cnn[:, 1])
print("\nConfiabilidad entre evaluadores (ICC) CNN:")
print(f"ICC Open: {icc_cnn_open}")
print(f"ICC Close: {icc_cnn_close}")

# Obtener las dos últimas predicciones del modelo LSTM
ultimas_predicciones_lstm = predicciones_prueba_lstm[-2:]
print("\nÚltimas dos predicciones LSTM (Open, Close):")
for i, pred in enumerate(ultimas_predicciones_lstm):
    print(f"Predicción {i+1}: Open = {pred[0]}, Close = {pred[1]}")

# Obtener las dos últimas predicciones del modelo CNN
ultimas_predicciones_cnn = predicciones_prueba_cnn[-2:]
print("\nÚltimas dos predicciones CNN (Open, Close):")
for i, pred in enumerate(ultimas_predicciones_cnn):
    print(f"Predicción {i+1}: Open = {pred[0]}, Close = {pred[1]}")

# Implementación del coeficiente de correlación de concordancia (CCC)
def ccc(y_true, y_pred):
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    ccc_value = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
    return ccc_value

# Calcular el CCC para los conjuntos de prueba de ambos modelos
# CCC para LSTM
ccc_lstm_open = ccc(y_prueba_lstm[:, 0], predicciones_prueba_lstm[:, 0])
ccc_lstm_close = ccc(y_prueba_lstm[:, 1], predicciones_prueba_lstm[:, 1])
print("\nConcordancia LSTM:")
print(f"CCC Open: {ccc_lstm_open}")
print(f"CCC Close: {ccc_lstm_close}")

# CCC para CNN
ccc_cnn_open = ccc(y_prueba_cnn[:, 0], predicciones_prueba_cnn[:, 0])
ccc_cnn_close = ccc(y_prueba_cnn[:, 1], predicciones_prueba_cnn[:, 1])
print("\nConcordancia CNN:")
print(f"CCC Open: {ccc_cnn_open}")
print(f"CCC Close: {ccc_cnn_close}")
