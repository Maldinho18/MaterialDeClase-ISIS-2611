# ===== TABLA COMPARATIVA =====
print("="*60)
print("TABLA COMPARATIVA DE MODELOS")
print("="*60)
print()

tabla_comparativa = pd.DataFrame({
    'Modelo': ['Modelo 1 - Baseline', 'Modelo 2 - Alternativas'],
    'Variables': [
        f'{len(variables_numericas)} numéricas',
        f'{len(variables_numericas)} numéricas + {len(variables_categoricas)} categóricas'
    ],
    'Estrategia Prep.': ['SimpleImputer + StandardScaler', 'SimpleImputer + MinMaxScaler + OneHotEncoder'],
    'RMSE': [rmse_m1, rmse_m2],
    'MAE': [mae_m1, mae_m2],
    'R²': [r2_m1, r2_m2]
})

print(tabla_comparativa.to_string(index=False))
print()

# Guardar tabla
tabla_comparativa.to_csv('comparacion_modelos.csv', index=False)
print("✓ Tabla guardada como 'comparacion_modelos.csv'")
print()

# Identificar mejor modelo
mejor_idx = tabla_comparativa['R²'].idxmax()
mejor_modelo_nombre = tabla_comparativa.loc[mejor_idx, 'Modelo']
mejor_rmse = tabla_comparativa.loc[mejor_idx, 'RMSE']
mejor_r2 = tabla_comparativa.loc[mejor_idx, 'R²']

print(f"🏆 MEJOR MODELO: {mejor_modelo_nombre}")
print(f"   RMSE: {mejor_rmse:.4f}")
print(f"   R²: {mejor_r2:.4f}")