import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

data = pd.read_csv('./data/Datos Lab 1.csv')
data = data.copy()

print("Registros iniciales:", data.shape[0])

duplicados = data.duplicated().sum()
print(f"Duplicados: {duplicados}")

data = data.drop_duplicates()
print(f"Registros después de eliminar duplicados: {data.shape[0]}")

nulos = ((data.isnull().sum()/data.shape[0])*100).sort_values(ascending=False)
nulos = nulos[nulos > 0]
print(f"Valores nulos: {nulos}")

data = data.dropna(subset=['CVD Risk Score'])
print(f"Registros después de eliminar nulos en CVD Risk Score: {data.shape[0]}")

print("Sección de variables:")
variables_objetivo = 'CVD Risk Score'
variables_excluir = ['Patient ID', 'Date of Service', 'Blood Pressure (mmHg)', 'Height (m)', 'Height (cm)', 'CVD Risk Level']
print(f"Variables excluidas: {variables_excluir}")

X = data.drop(columns=variables_excluir + [variables_objetivo])
y = data[variables_objetivo]

print(f"Variables seleccionadas para el modelo: {X.shape[1]}")
print(X.columns.tolist())

variables_numericas = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
variables_categoricas = X.select_dtypes(include=['object']).columns.tolist()
print(f"Variables numericas ({len(variables_numericas)}): {variables_numericas}")
print(f"Variables categoricas ({len(variables_categoricas)}): {variables_categoricas}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(f"Tamaño conjunto entrenamiento: {X_train.shape[0]} ({100*X_train.shape[0]/X.shape[0]:.1f}%)")
print(f"Tamaño conjunto prueba: {X_test.shape[0]} ({100*X_test.shape[0]/X.shape[0]:.1f}%)")

print("Creación de pipeline")
transformador_numerico = Pipeline(steps=[
    ('imputador', SimpleImputer(strategy='median')),
    ('escalador', StandardScaler())
])

transformador_categorico = Pipeline(steps=[
    ('imputador', SimpleImputer(strategy='most_frequent')),
    ('codificador', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocesamiento = ColumnTransformer(
    transformers=[
        ('num', transformador_numerico, variables_numericas),
        ('cat', transformador_categorico, variables_categoricas)
    ])

print("Verificador de datos preparados")
X_train_preparado = preprocesamiento.fit_transform(X_train)
X_test_preparado = preprocesamiento.transform(X_test)

print(f"Dimensiones de X_train preparado: {X_train_preparado.shape}")
print(f"Dimensiones de X_test preparado: {X_test_preparado.shape}")
print(f"Variables finales: {X_train_preparado.shape[1]}")

fig, ejes = plt.subplots(1, 2, figsize=(14, 5))

ejes[0].hist(y_train, bins=30, edgecolor='black', alpha=0.7, label='Entrenamiento')
ejes[0].set_title('Distribución de CVD Risk Score - Entrenamiento')
ejes[0].set_xlabel('CVD Risk Score')
ejes[0].set_ylabel('Frecuencia')

ejes[1].hist(y_test, bins=30, edgecolor='black', alpha=0.7, color='orange', label='Prueba')
ejes[1].set_title('Distribución de CVD Risk Score - Prueba')
ejes[1].set_xlabel('CVD Risk Score')
ejes[1].set_ylabel('Frecuencia')

plt.tight_layout()
plt.show()

print("RESUMEN")
print(f"Registros iniciales: {data.shape[0] + duplicados}")
print(f"Registros finales: {X.shape[0]}")
print(f"Varibles finales: {X.shape[1]}")
print(f"Registros entrenamiento: {X_train.shape[0]}")
print(f"Registros prueba: {X_test.shape[0]}")