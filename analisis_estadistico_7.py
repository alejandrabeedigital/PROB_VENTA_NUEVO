import pandas as pd
import numpy as np
import statsmodels.api as sm

ARCHIVO = "todo_con_resultados_7.csv"
TARGET = "ganada"

# =========================
# 1) CARGA
# =========================
df = pd.read_csv(ARCHIVO, low_memory=False)
print(f"Filas leídas: {len(df):,}")

# Asegurar target 0/1
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df[df[TARGET].isin([0, 1])].copy()
df[TARGET] = df[TARGET].astype(int)

# =========================
# 2) VARIABLES (actualizadas)
# =========================
features = [
    "q_rk_score",
    "median_income_equiv",
    "origen_sc_o_no",
    "ct_merclie",
    "excliente",
    "outcome_forzado_autonomo",
    "habitat_3",
    "total_rating_imputado",
    "claim_business_imputado",
    "compe_cat_3"  # <-- NUEVA VARIABLE
]

# Validación mínima para no romperse
if "compe_cat_3" not in df.columns:
    raise ValueError(
        "No existe la columna 'compe_cat_3' en el CSV. "
        "Asegúrate de usar el CSV/modelo que ya la genera."
    )

# Asegurar numéricas correctamente
df["q_rk_score"] = pd.to_numeric(
    df["q_rk_score"].astype(str).str.replace(",", ".", regex=False),
    errors="coerce"
)

df["median_income_equiv"] = pd.to_numeric(
    df["median_income_equiv"],
    errors="coerce"
)

df["origen_sc_o_no"] = pd.to_numeric(
    df["origen_sc_o_no"],
    errors="coerce"
)

df["total_rating_imputado"] = pd.to_numeric(
    df["total_rating_imputado"].astype(str).str.replace(",", ".", regex=False),
    errors="coerce"
)

df["claim_business_imputado"] = pd.to_numeric(
    df["claim_business_imputado"].astype(str).str.replace(",", ".", regex=False),
    errors="coerce"
)

# (Categóricas) asegurar tipo "object" para dummies (evita raros con StringDtype)
df["habitat_3"] = df["habitat_3"].astype(object)
df["ct_merclie"] = df["ct_merclie"].astype(object)
df["excliente"] = df["excliente"].astype(object)
df["outcome_forzado_autonomo"] = df["outcome_forzado_autonomo"].astype(object)
df["compe_cat_3"] = df["compe_cat_3"].astype(object)

df_model = df[features + [TARGET]].dropna().copy()

print(f"Filas usadas para inferencia: {len(df_model):,}")
print(f"Tasa base: {df_model[TARGET].mean():.6f}")

# =========================
# 3) DUMMIES
# =========================
X = pd.get_dummies(
    df_model[features],
    drop_first=True
)

y = df_model[TARGET]

# Asegurar numérico
X = X.astype(float)

# Añadir constante
X = sm.add_constant(X)

print(X.dtypes)

# =========================
# 4) REGRESIÓN LOGÍSTICA
# =========================
model = sm.Logit(y, X)
result = model.fit(maxiter=2000, disp=True)

print("\n==============================")
print("      RESUMEN COMPLETO")
print("==============================\n")
print(result.summary())

# =========================
# 5) ODDS RATIOS
# =========================
odds_ratios = pd.DataFrame({
    "variable": result.params.index,
    "coef": result.params.values,
    "odds_ratio": np.exp(result.params.values),
    "p_value": result.pvalues.values
})

print("\n==============================")
print("      ODDS RATIOS")
print("==============================\n")
print(odds_ratios)

# =========================
# 6) EFECTOS MARGINALES
# =========================
marginal = result.get_margeff()
print("\n==============================")
print("    EFECTOS MARGINALES")
print("==============================\n")
print(marginal.summary())

# =========================
# 7) INTERPRETACIÓN AUTOMÁTICA
# =========================
print("\n==============================")
print("    INTERPRETACIÓN CLAVE")
print("==============================\n")

for var, coef, p in zip(result.params.index, result.params.values, result.pvalues.values):

    if var == "const":
        continue

    efecto = "AUMENTA" if coef > 0 else "DISMINUYE"
    signif = "SIGNIFICATIVO" if p < 0.05 else "NO SIGNIFICATIVO"

    print(f"{var}: {efecto} probabilidad | p-value={p:.4f} | {signif}")