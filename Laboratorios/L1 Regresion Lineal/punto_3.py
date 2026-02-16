# ===== MODELO 1: BASELINE =====
# Estrategia 1: Solo variables numéricas, sin escalado adicional
print("="*60)
print("CONSTRUCCIÓN MODELO 1 - BASELINE")
print("="*60)
print("Estrategia: Solo variables numéricas, Sin OneHotEncoder")
print()

# Seleccionar solo variables numéricas
X_train_m1 = X_train[variables_numericas].copy()
X_test_m1 = X_test[variables_numericas].copy()

# Pipeline Modelo 1: imputación + escalado
prep_m1 = Pipeline(steps=[
    ('imputador', SimpleImputer(strategy='median')),
    ('escalador', StandardScaler())
])

X_train_prep_m1 = prep_m1.fit_transform(X_train_m1)
X_test_prep_m1 = prep_m1.transform(X_test_m1)

# Entrenar modelo 1
modelo1 = LinearRegression()
modelo1.fit(X_train_prep_m1, y_train)

# Evaluar modelo 1
y_pred_m1 = modelo1.predict(X_test_prep_m1)
rmse_m1 = np.sqrt(mean_squared_error(y_test, y_pred_m1))
mae_m1 = mean_absolute_error(y_test, y_pred_m1)
r2_m1 = r2_score(y_test, y_pred_m1)

print(f"Modelo 1 - Rendimiento:")
print(f"  RMSE: {rmse_m1:.4f}")
print(f"  MAE: {mae_m1:.4f}")
print(f"  R²: {r2_m1:.4f}")
print()

# Guardar modelo 1
m1_data = {
    'modelo': modelo1,
    'preprocesamiento': prep_m1,
    'variables_num': variables_numericas,
    'variables_cat': [],
    'rmse': rmse_m1,
    'r2': r2_m1
}
with open('modelo1.pkl', 'wb') as f:
    pickle.dump(m1_data, f)
print("Modelo 1 guardado como 'modelo1.pkl'")
print()

# ===== MODELO 2: CON TODAS LAS VARIABLES =====
print("="*60)
print("CONSTRUCCIÓN MODELO 2 - ALTERNATIVAS")
print("="*60)
print("Estrategia: Variables numéricas + categóricas con OneHotEncoder")
print()

# Pipeline Modelo 2: ColumnTransformer con todas las variables
prep_m2 = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputador', SimpleImputer(strategy='median')),
            ('escalador', MinMaxScaler())  # Diferente escalador que Modelo 1
        ]), variables_numericas),
        ('cat', Pipeline(steps=[
            ('imputador', SimpleImputer(strategy='most_frequent')),
            ('codificador', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), variables_categoricas)
    ])

X_train_prep_m2 = prep_m2.fit_transform(X_train)
X_test_prep_m2 = prep_m2.transform(X_test)

# Entrenar modelo 2
modelo2 = LinearRegression()
modelo2.fit(X_train_prep_m2, y_train)

# Evaluar modelo 2
y_pred_m2 = modelo2.predict(X_test_prep_m2)
rmse_m2 = np.sqrt(mean_squared_error(y_test, y_pred_m2))
mae_m2 = mean_absolute_error(y_test, y_pred_m2)
r2_m2 = r2_score(y_test, y_pred_m2)

print(f"Modelo 2 - Rendimiento:")
print(f"  RMSE: {rmse_m2:.4f}")
print(f"  MAE: {mae_m2:.4f}")
print(f"  R²: {r2_m2:.4f}")
print()

# Guardar modelo 2
m2_data = {
    'modelo': modelo2,
    'preprocesamiento': prep_m2,
    'variables_num': variables_numericas,
    'variables_cat': variables_categoricas,
    'rmse': rmse_m2,
    'r2': r2_m2
}
with open('modelo2.pkl', 'wb') as f:
    pickle.dump(m2_data, f)
print("Modelo 2 guardado como 'modelo2.pkl'")