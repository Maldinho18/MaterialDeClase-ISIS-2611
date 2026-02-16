import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

data = pd.read_csv('./data/Datos Lab 1.csv')
data = data.copy()

data = data.drop_duplicates()
data = data.dropna(subset=['CVD Risk Score'])

variables_objetivo = 'CVD Risk Score'
variables_excluir = ['Patient ID', 'Date of Service', 'Blood Pressure (mmHg)', 
                     'Height (m)', 'Height (cm)', 'CVD Risk Level']

X = data.drop(columns=variables_excluir + [variables_objetivo])
y = data[variables_objetivo]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

with open('modelo1.pkl', 'rb') as f:
    m1_data = pickle.load(f)
    modelo1 = m1_data['modelo']
    prep1 = m1_data['preprocesamiento']
    var_num1 = m1_data['variables_num']
    var_cat1 = m1_data['variables_cat']
    
with open('modelo2.pkl', 'rb') as f:
    m2_data = pickle.load(f)
    modelo2 = m2_data['modelo']
    prep2 = m2_data['preprocesamiento']
    var_num2 = m2_data['variables_num']
    var_cat2 = m2_data['variables_cat']
    
X_train_prep1 = prep1.fit_transform(X_train)
X_test_prep1 = prep1.transform(X_test)

X_test_sel2 = X_test[var_num2 + var_cat2]
X_test_prep2 = prep2.transform(X_test_sel2)

y_pred_m1 = modelo1.predict(X_test_prep1)
y_pred_m2 = modelo2.predict(X_test_prep2)

def calcular_metricas(y_true, y_pred, nombre_modelo):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {'Modelo': nombre_modelo, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

metricas_m1 = calcular_metricas(y_test, y_pred_m1, 'Modelo 1 - Baseline')
metricas_m2 = calcular_metricas(y_test, y_pred_m2, 'Modelo 2 - Alternativas')

tabla_comparativa = pd.DataFrame([metricas_m1, metricas_m2])

print("Tabla comparativa de modelos")
print(tabla_comparativa.to_string(index=False))

tabla_comparativa.to_csv('comparacion_modelos.csv', index=False)
print("Tabla comparativa guardada como 'comparacion_modelos.csv'")

print(f"""
RMSE (Error Cuadrático Medio):
  - Mide el error promedio en unidades de CVD Risk Score
  - Penaliza más los errores grandes
  - Modelo 1 RMSE: {metricas_m1['RMSE']:.4f} (menor = mejor)
  - Modelo 2 RMSE: {metricas_m2['RMSE']:.4f}

MAE (Error Absoluto Medio):
  - Error promedio absoluto en unidades de CVD Risk Score
  - Interpretación directa: diferencia promedio en predicciones
  - Modelo 1 MAE: {metricas_m1['MAE']:.4f} (menor = mejor)
  - Modelo 2 MAE: {metricas_m2['MAE']:.4f}

R² (Coeficiente de Determinación):
  - Proporción de varianza en y explicada por el modelo
  - Rango: 0 a 1, donde 1 es perfecto
  - Modelo 1 R²: {metricas_m1['R2']:.4f} ({100*metricas_m1['R2']:.1f}% de varianza explicada)
  - Modelo 2 R²: {metricas_m2['R2']:.4f} ({100*metricas_m2['R2']:.1f}% de varianza explicada)
""")

mejor_idx = tabla_comparativa['R2'].idxmax()
mejor_fila = tabla_comparativa.loc[mejor_idx]
print(f"El mejor modelo es: {mejor_fila['Modelo']}")
print(f"  RMSE: {mejor_fila['RMSE']:.4f}")
print(f"  MAE: {mejor_fila['MAE']:.4f}")
print(f"  R²: {mejor_fila['R2']:.4f}")