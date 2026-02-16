print("="*70)
print("PREDICCIONES EN DATOS DE PRUEBA SIN ETIQUETAR")
print("="*70)
print()

# Usar el mejor modelo para predecir
if mejor_idx == 0:
    # Modelo 1: solo variables numéricas
    X_test_datos = data_test[variables_numericas].copy()
    X_test_datos_prep = prep_m1.transform(X_test_datos)
    predicciones_test = modelo1.predict(X_test_datos_prep)
else:
    # Modelo 2: todas las variables
    X_test_datos = data_test.copy()
    X_test_datos_prep = prep_m2.transform(X_test_datos)
    predicciones_test = modelo2.predict(X_test_datos_prep)

# Agregar predicciones al dataframe de test
data_test_con_predicciones = data_test.copy()
data_test_con_predicciones['CVD Risk Score'] = predicciones_test

print(f"✓ Predicciones realizadas para {len(predicciones_test)} registros sin etiquetar")
print(f"\nEstadísticas de predicciones:")
print(f"  Mínimo: {predicciones_test.min():.4f}")
print(f"  Máximo: {predicciones_test.max():.4f}")
print(f"  Media: {predicciones_test.mean():.4f}")
print(f"  Desviación Estándar: {predicciones_test.std():.4f}")
print()

# Mostrar primeras 10 predicciones
print("Primeras 10 predicciones:")
print(data_test_con_predicciones[['Patient ID', 'Age', 'BMI', 'CVD Risk Score']].head(10))
print()

# Guardar archivo con predicciones
data_test_con_predicciones.to_csv('Datos Test Lab 1.csv', index=False, sep=';')
print("✓ Archivo 'Datos Test Lab 1.csv' guardado con predicciones")

# Visualización de predicciones
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(predicciones_test, bins=30, edgecolor='black', alpha=0.7, color='green')
plt.xlabel('CVD Risk Score Predicho')
plt.ylabel('Frecuencia')
plt.title('Distribución de Predicciones en Datos de Prueba')

plt.subplot(1, 2, 2)
plt.scatter(data_test_con_predicciones['Age'], predicciones_test, alpha=0.5)
plt.xlabel('Edad')
plt.ylabel('CVD Risk Score Predicho')
plt.title('Riesgo Predicho vs Edad')

plt.tight_layout()
plt.show()