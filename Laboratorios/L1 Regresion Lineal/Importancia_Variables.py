# ===== TABLA DE IMPORTANCIA DE VARIABLES =====
print("="*60)
print("IMPORTANCIA DE VARIABLES - MEJOR MODELO")
print("="*60)
print()

# Cargar el mejor modelo
if mejor_idx == 0:
    mejor_mod = modelo1
    # Para modelo 1 (solo variables numéricas)
    nombres_features = variables_numericas
else:
    mejor_mod = modelo2
    # Para modelo 2 con OneHotEncoder
    # Necesitamos obtener los nombres de las features después del preprocesamiento
    # Características numéricas + nombres de categorías codificadas
    encoder = prep_m2.named_transformers_['cat'].named_steps['codificador']
    feature_names_cat = []
    for i, cat_var in enumerate(variables_categoricas):
        categories = encoder.categories_[i]
        feature_names_cat.extend([f"{cat_var}_{cat}" for cat in categories])
    nombres_features = variables_numericas + feature_names_cat

# Obtener coeficientes
coeficientes = mejor_mod.coef_
intercepto = mejor_mod.intercept_

# Crear tabla de importancia
importancia_vars = pd.DataFrame({
    'Variable': nombres_features,
    'Coeficiente': coeficientes,
    'Valor Absoluto': np.abs(coeficientes)
}).sort_values('Valor Absoluto', ascending=False)

print("Top 15 variables más importantes:")
print(importancia_vars.head(15).to_string(index=False))
print()
print(f"Intercepto (constante): {intercepto:.4f}")
print(f"Total de coeficientes: {len(coeficientes)}")

importancia_vars.to_csv('importancia_variables.csv', index=False)
print("\n✓ Importancia guardada como 'importancia_variables.csv'")

# Visualizar
plt.figure(figsize=(12, 6))
plt.barh(range(15), importancia_vars['Valor Absoluto'].head(15))
plt.yticks(range(15), importancia_vars['Variable'].head(15))
plt.xlabel('Valor Absoluto del Coeficiente')
plt.title('Top 15 Variables Más Importantes')
plt.tight_layout()
plt.show()