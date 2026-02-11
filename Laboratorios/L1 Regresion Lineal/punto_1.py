import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from importlib.metadata import version

print(f"Version pandas: {version('pandas')}")
print(f"Version matplotlib: {version('matplotlib')}")
print(f"Version seaborn: {version('seaborn')}")

data = pd.read_csv('./data/Datos Lab 1.csv')
data_test = pd.read_csv('./data/Datos Test Lab 1.csv', sep=';')

data = data.copy()

data.head()
data.sample(5)

data.shape

data.info()

data.describe()

nulos = ((data.isnull().sum()/data.shape[0])*100).sort_values(ascending=False)
print(nulos[nulos > 0])

duplicados = data.duplicated().sum()
print(f"Duplicados: {duplicados}")

fig, ejes = plt.subplots(2, 2, figsize=(14, 10))

ejes[0, 0].hist(data['CVD Risk Score'].dropna(), bins=30, edgecolor='black')
ejes[0, 0].set_title('Distribución de Riesgo Cardiovascular')
ejes[0, 0].set_xlabel('Puntuación de Riesgo')
ejes[0, 0].set_ylabel('Frecuencia')

ejes[0, 1].hist(data['Age'].dropna(), bins=30, edgecolor='black')
ejes[0, 1].set_title('Distribución de Edad')
ejes[0, 1].set_xlabel('Edad')
ejes[0, 1].set_ylabel('Frecuencia')

ejes[1, 0].hist(data['BMI'].dropna(), bins=30, edgecolor='black')
ejes[1, 0].set_title('Distribución de IMC')
ejes[1, 0].set_xlabel('IMC')
ejes[1, 0].set_ylabel('Frecuencia')

ejes[1, 1].hist(data['Total Cholesterol (mg/dL)'].dropna(), bins=30, edgecolor='black')
ejes[1, 1].set_title('Distribución de Colesterol Total')
ejes[1, 1].set_xlabel('Colesterol Total (mg/dL)')
ejes[1, 1].set_ylabel('Frecuencia')

plt.tight_layout()
plt.show()

print("Correlacion con riesgo cardiovascular")
correlaciones = data.corr(numeric_only=True)['CVD Risk Score'].sort_values(ascending=False)
print(correlaciones)

print("Matriz de correlaciones - Variables numéricas")
plt.figure(figsize=(16, 12))
sns.heatmap(data.corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Matriz de correlación entre variables')
plt.tight_layout()
plt.show()

print("Estadísticas descriptivas")
print(data.describe())

print("RESUMEN DEL DATASET")
print(f"Cantidad de filas: {data.shape[0]}")
print(f"Cantidad de columnas: {data.shape[1]}")
variables_numericas = data.select_dtypes(include=['float64', 'int64']).shape[1]
variables_categoricas = data.select_dtypes(include=['object']).shape[1]
print(f"Cantidad de variables numéricas: {variables_numericas}")
print(f"Cantidad de variables categóricas: {variables_categoricas}")
