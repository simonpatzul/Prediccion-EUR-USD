import pandas as pd
import os

# Cargar el archivo CSV en un DataFrame
file_path = os.path.join('..', 'Datos', 'united-states.gross-domestic-product-qq.csv')
db1 = pd.read_csv(file_path, sep='\t')

# Separa la columna en múltiples columnas utilizando el separador '\t'
db1[['Date', 'ActualValue1', 'ForecastValue1', 'PreviousValue1']] = db1['Date\tActualValue1\tForecastValue1\tPreviousValue1'].str.split('\t', expand=True)

# Formatea la columna 'Date' como una fecha en el formato deseado
db1['Date'] = pd.to_datetime(db1['Date'].str.replace('.', ''), format='%Y%m%d')

# Elimina la columna original que contiene los datos combinados
db1.drop(columns=['Date\tActualValue1\tForecastValue1\tPreviousValue1'], inplace=True)

# Carga el segundo archivo
file_path2 = os.path.join('..', 'Datos', 'EURUSD-D1.csv')
db2 = pd.read_csv(file_path2)

# Convierte la columna 'Date' del segundo DataFrame al mismo tipo de datos que la columna 'Date' del primer DataFrame
db2['Date'] = pd.to_datetime(db2['Date'], format='%Y%m%d')

# Carga archivo
file_path3 = os.path.join('..', 'Datos', 'united-states.fed-interest-rate-decision.csv')
db3 = pd.read_csv(file_path3, sep='\t')

# Separa la columna en múltiples columnas utilizando el separador '\t'
db3[['Date', 'ActualValue2', 'ForecastValue2', 'PreviousValue2']] = db3['Date\tActualValue2\tForecastValue2\tPreviousValue2'].str.split('\t', expand=True)

# Formatea la columna 'Date' como una fecha en el formato deseado
db3['Date'] = pd.to_datetime(db3['Date'].str.replace('.', ''), format='%Y%m%d')

# Elimina la columna original que contiene los datos combinados
db3.drop(columns=['Date\tActualValue2\tForecastValue2\tPreviousValue2'], inplace=True)

# Carga  archivo
file_path4 = os.path.join('..', 'Datos', 'united-states.ism-manufacturing-pmi.csv')
db4 = pd.read_csv(file_path4, sep='\t')

# Separa la columna en múltiples columnas utilizando el separador '\t'
db4[['Date', 'ActualValue3', 'ForecastValue3', 'PreviousValue3']] = db4['Date\tActualValue3\tForecastValue3\tPreviousValue3'].str.split('\t', expand=True)

# Formatea la columna 'Date' como una fecha en el formato deseado
db4['Date'] = pd.to_datetime(db4['Date'].str.replace('.', ''), format='%Y%m%d')

# Elimina la columna original que contiene los datos combinados
db4.drop(columns=['Date\tActualValue3\tForecastValue3\tPreviousValue3'], inplace=True)

# Carga  archivo
file_path5 = os.path.join('..', 'Datos', 'united-states.eia-crude-oil-stocks-change.csv')
db5 = pd.read_csv(file_path5, sep='\t')

# Separa la columna en múltiples columnas utilizando el separador '\t'
db5[['Date', 'ActualValue5', 'ForecastValue5', 'PreviousValue5']] = db5['Date\tActualValue5\tForecastValue5\tPreviousValue5'].str.split('\t', expand=True)

# Formatea la columna 'Date' como una fecha en el formato deseado
db5['Date'] = pd.to_datetime(db5['Date'].str.replace('.', ''), format='%Y%m%d')

# Elimina la columna original que contiene los datos combinados
db5.drop(columns=['Date\tActualValue5\tForecastValue5\tPreviousValue5'], inplace=True)

# Carga  archivo
file_path6 = os.path.join('..', 'Datos', 'united-states.adp-nonfarm-employment-change.csv')
db6 = pd.read_csv(file_path6, sep='\t')

# Separa la columna en múltiples columnas utilizando el separador '\t'
db6[['Date', 'ActualValue6', 'ForecastValue6', 'PreviousValue6']] = db6['Date\tActualValue6\tForecastValue6\tPreviousValue6'].str.split('\t', expand=True)

# Formatea la columna 'Date' como una fecha en el formato deseado
db6['Date'] = pd.to_datetime(db6['Date'].str.replace('.', ''), format='%Y%m%d')

# Elimina la columna original que contiene los datos combinados
db6.drop(columns=['Date\tActualValue6\tForecastValue6\tPreviousValue6'], inplace=True)

# Carga  archivo
file_path7 = os.path.join('..', 'Datos', 'united-states.durable-goods-orders-ex-transportation.csv')
db7 = pd.read_csv(file_path7, sep='\t')

# Separa la columna en múltiples columnas utilizando el separador '\t'
db7[['Date', 'ActualValue7', 'ForecastValue7', 'PreviousValue7']] = db7['Date\tActualValue7\tForecastValue7\tPreviousValue7'].str.split('\t', expand=True)

# Formatea la columna 'Date' como una fecha en el formato deseado
db7['Date'] = pd.to_datetime(db7['Date'].str.replace('.', ''), format='%Y%m%d')

# Elimina la columna original que contiene los datos combinados
db7.drop(columns=['Date\tActualValue7\tForecastValue7\tPreviousValue7'], inplace=True)

# Carga  archivo
file_path8 = os.path.join('..', 'Datos', 'united-states.consumer-confidence-index.csv')
db8 = pd.read_csv(file_path8, sep='\t')

# Separa la columna en múltiples columnas utilizando el separador '\t'
db8[['Date', 'ActualValue8', 'ForecastValue8', 'PreviousValue8']] = db8['Date\tActualValue8\tForecastValue8\tPreviousValue8'].str.split('\t', expand=True)

# Formatea la columna 'Date' como una fecha en el formato deseado
db8['Date'] = pd.to_datetime(db8['Date'].str.replace('.', ''), format='%Y%m%d')

# Elimina la columna original que contiene los datos combinados
db8.drop(columns=['Date\tActualValue8\tForecastValue8\tPreviousValue8'], inplace=True)

# Carga  archivo
file_path9 = os.path.join('..', 'Datos', 'united-states.building-permits.csv')
db9 = pd.read_csv(file_path9, sep='\t')

# Separa la columna en múltiples columnas utilizando el separador '\t'
db9[['Date', 'ActualValue9', 'ForecastValue9', 'PreviousValue9']] = db9['Date\tActualValue9\tForecastValue9\tPreviousValue9'].str.split('\t', expand=True)

# Formatea la columna 'Date' como una fecha en el formato deseado
db9['Date'] = pd.to_datetime(db9['Date'].str.replace('.', ''), format='%Y%m%d')

# Elimina la columna original que contiene los datos combinados
db9.drop(columns=['Date\tActualValue9\tForecastValue9\tPreviousValue9'], inplace=True)

# Carga  archivo
file_path11 = os.path.join('..', 'Datos', 'european-union.employment-change-qq.csv')
db11 = pd.read_csv(file_path11, sep='\t')

# Separa la columna en múltiples columnas utilizando el separador '\t'
db11[['Date', 'ActualValue11', 'ForecastValue11', 'PreviousValue11']] = db11['Date\tActualValue11\tForecastValue11\tPreviousValue11'].str.split('\t', expand=True)

# Formatea la columna 'Date' como una fecha en el formato deseado
db11['Date'] = pd.to_datetime(db11['Date'].str.replace('.', ''), format='%Y%m%d')

# Elimina la columna original que contiene los datos combinados
db11.drop(columns=['Date\tActualValue11\tForecastValue11\tPreviousValue11'], inplace=True)

# Carga  archivo
file_path12 = os.path.join('..', 'Datos', 'germany.ppi-mm.csv')
db12 = pd.read_csv(file_path12, sep='\t')

# Separa la columna en múltiples columnas utilizando el separador '\t'
db12[['Date', 'ActualValue12', 'ForecastValue12', 'PreviousValue12']] = db12['Date\tActualValue12\tForecastValue12\tPreviousValue12'].str.split('\t', expand=True)

# Formatea la columna 'Date' como una fecha en el formato deseado
db12['Date'] = pd.to_datetime(db12['Date'].str.replace('.', ''), format='%Y%m%d')

# Elimina la columna original que contiene los datos combinados
db12.drop(columns=['Date\tActualValue12\tForecastValue12\tPreviousValue12'], inplace=True)

# Carga  archivo
file_path13 = os.path.join('..', 'Datos', 'european-union.gross-domestic-product-qq.csv')
db13 = pd.read_csv(file_path13, sep='\t')

# Separa la columna en múltiples columnas utilizando el separador '\t'
db13[['Date', 'ActualValue13', 'ForecastValue13', 'PreviousValue13']] = db13['Date\tActualValue13\tForecastValue13\tPreviousValue13'].str.split('\t', expand=True)

# Formatea la columna 'Date' como una fecha en el formato deseado
db13['Date'] = pd.to_datetime(db13['Date'].str.replace('.', ''), format='%Y%m%d')

# Elimina la columna original que contiene los datos combinados
db13.drop(columns=['Date\tActualValue13\tForecastValue13\tPreviousValue13'], inplace=True)

# Carga  archivo
file_path14 = os.path.join('..', 'Datos', 'germany.zew-economic-sentiment-indicator.csv')
db14 = pd.read_csv(file_path14, sep='\t')

# Separa la columna en múltiples columnas utilizando el separador '\t'
db14[['Date', 'ActualValue14', 'ForecastValue14', 'PreviousValue14']] = db14['Date\tActualValue14\tForecastValue14\tPreviousValue14'].str.split('\t', expand=True)

# Formatea la columna 'Date' como una fecha en el formato deseado
db14['Date'] = pd.to_datetime(db14['Date'].str.replace('.', ''), format='%Y%m%d')

# Elimina la columna original que contiene los datos combinados
db14.drop(columns=['Date\tActualValue14\tForecastValue14\tPreviousValue14'], inplace=True)

# Carga  archivo
file_path15 = os.path.join('..', 'Datos', 'germany.gdp-qq.csv')
db15 = pd.read_csv(file_path15, sep='\t')

# Separa la columna en múltiples columnas utilizando el separador '\t'
db15[['Date', 'ActualValue15', 'ForecastValue15', 'PreviousValue15']] = db15['Date\tActualValue15\tForecastValue15\tPreviousValue15'].str.split('\t', expand=True)

# Formatea la columna 'Date' como una fecha en el formato deseado
db15['Date'] = pd.to_datetime(db15['Date'].str.replace('.', ''), format='%Y%m%d')

# Elimina la columna original que contiene los datos combinados
db15.drop(columns=['Date\tActualValue15\tForecastValue15\tPreviousValue15'], inplace=True)

# Carga  archivo
file_path16 = os.path.join('..', 'Datos', 'spain.trade-balance.csv')
db16 = pd.read_csv(file_path16, sep='\t')

# Separa la columna en múltiples columnas utilizando el separador '\t'
db16[['Date', 'ActualValue16', 'ForecastValue16', 'PreviousValue16']] = db16['Date\tActualValue16\tForecastValue16\tPreviousValue16'].str.split('\t', expand=True)

# Formatea la columna 'Date' como una fecha en el formato deseado
db16['Date'] = pd.to_datetime(db16['Date'].str.replace('.', ''), format='%Y%m%d')

# Elimina la columna original que contiene los datos combinados
db16.drop(columns=['Date\tActualValue16\tForecastValue16\tPreviousValue16'], inplace=True)

# Carga  archivo
file_path17 = os.path.join('..', 'Datos', 'united-states.existing-home-sales.csv')
db17 = pd.read_csv(file_path17, sep='\t')

# Separa la columna en múltiples columnas utilizando el separador '\t'
db17[['Date', 'ActualValue17', 'ForecastValue17', 'PreviousValue17']] = db17['Date\tActualValue17\tForecastValue17\tPreviousValue17'].str.split('\t', expand=True)

# Formatea la columna 'Date' como una fecha en el formato deseado
db17['Date'] = pd.to_datetime(db17['Date'].str.replace('.', ''), format='%Y%m%d')

# Elimina la columna original que contiene los datos combinados
db17.drop(columns=['Date\tActualValue17\tForecastValue17\tPreviousValue17'], inplace=True)

# Carga  archivo
file_path18 = os.path.join('..', 'Datos', 'germany.10-year-bond-auction.csv')
db18 = pd.read_csv(file_path18, sep='\t')

# Separa la columna en múltiples columnas utilizando el separador '\t'
db18[['Date', 'ActualValue18', 'ForecastValue18', 'PreviousValue18']] = db18['Date\tActualValue18\tForecastValue18\tPreviousValue18'].str.split('\t', expand=True)

# Formatea la columna 'Date' como una fecha en el formato deseado
db18['Date'] = pd.to_datetime(db18['Date'].str.replace('.', ''), format='%Y%m%d')

# Elimina la columna original que contiene los datos combinados
db18.drop(columns=['Date\tActualValue18\tForecastValue18\tPreviousValue18'], inplace=True)

# Carga  archivo
file_path19 = os.path.join('..', 'Datos', 'european-union.trade-balance.csv')
db19 = pd.read_csv(file_path19, sep='\t')

# Separa la columna en múltiples columnas utilizando el separador '\t'
db19[['Date', 'ActualValue19', 'ForecastValue19', 'PreviousValue19']] = db19['Date\tActualValue19\tForecastValue19\tPreviousValue19'].str.split('\t', expand=True)

# Formatea la columna 'Date' como una fecha en el formato deseado
db19['Date'] = pd.to_datetime(db19['Date'].str.replace('.', ''), format='%Y%m%d')

# Elimina la columna original que contiene los datos combinados
db19.drop(columns=['Date\tActualValue19\tForecastValue19\tPreviousValue19'], inplace=True)

# Carga  archivo
file_path20 = os.path.join('..', 'Datos', 'united-states.new-home-sales.csv')
db20 = pd.read_csv(file_path20, sep='\t')

# Separa la columna en múltiples columnas utilizando el separador '\t'
db20[['Date', 'ActualValue20', 'ForecastValue20', 'PreviousValue20']] = db20['Date\tActualValue20\tForecastValue20\tPreviousValue20'].str.split('\t', expand=True)

# Formatea la columna 'Date' como una fecha en el formato deseado
db20['Date'] = pd.to_datetime(db20['Date'].str.replace('.', ''), format='%Y%m%d')

# Elimina la columna original que contiene los datos combinados
db20.drop(columns=['Date\tActualValue20\tForecastValue20\tPreviousValue20'], inplace=True)

# Une los tres conjuntos de datos en función de la columna 'Date', manteniendo todas las filas del segundo y tercer conjunto de datos
merged_data = pd.merge(db2, db1, on='Date', how='left')
merged_data = pd.merge(merged_data, db3, on='Date', how='left')
merged_data = pd.merge(merged_data, db4, on='Date', how='left')
merged_data = pd.merge(merged_data, db5, on='Date', how='left')
merged_data = pd.merge(merged_data, db6, on='Date', how='left')
merged_data = pd.merge(merged_data, db7, on='Date', how='left')
merged_data = pd.merge(merged_data, db8, on='Date', how='left')
merged_data = pd.merge(merged_data, db9, on='Date', how='left')
merged_data = pd.merge(merged_data, db11, on='Date', how='left')
merged_data = pd.merge(merged_data, db12, on='Date', how='left')
merged_data = pd.merge(merged_data, db13, on='Date', how='left')
merged_data = pd.merge(merged_data, db14, on='Date', how='left')
merged_data = pd.merge(merged_data, db15, on='Date', how='left')
merged_data = pd.merge(merged_data, db16, on='Date', how='left')
merged_data = pd.merge(merged_data, db17, on='Date', how='left')
merged_data = pd.merge(merged_data, db18, on='Date', how='left')
merged_data = pd.merge(merged_data, db19, on='Date', how='left')
merged_data = pd.merge(merged_data, db20, on='Date', how='left')

# Rellena los valores NaN con los valores de las filas anteriores que tengan datos
merged_data.fillna(method='ffill', inplace=True)

# Elimina las filas iniciales consecutivas donde hay datos NaN
merged_data.dropna(subset=['ActualValue1', 'ForecastValue1', 'PreviousValue1'
                                                             ''], inplace=True)
# Guarda el archivo combinado en el escritorio
merged_data.to_csv('C:/Users/oscar/Documents/pythonProject3/EUR-USD Github/Datos/combined_data1.csv', index=False)

print("Archivo combinado guardado con éxito.")
print(merged_data)