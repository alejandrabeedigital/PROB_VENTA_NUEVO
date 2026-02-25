import pandas as pd
import numpy as np
import statsmodels.api as sm

ARCHIVO = "todo_con_resultados_2_nofilter.csv"
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
# 2) VARIABLES (las mismas del modelo + origen_sc_o_no)
# =========================
features = ["q_rk_score", "ct_merclie", "excliente", "outcome_forzado_autonomo", "origen_sc_o_no"]

# Por si q_rk_score viene con coma decimal o como texto
df["q_rk_score"] = pd.to_numeric(
    df["q_rk_score"].astype(str).str.replace(",", ".", regex=False),
    errors="coerce"
)

# Por si origen_sc_o_no viene como texto
df["origen_sc_o_no"] = pd.to_numeric(df["origen_sc_o_no"], errors="coerce")

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

# Asegurar que todo sea numérico
X = X.astype(float)

X = sm.add_constant(X)
print(X.dtypes)

# =========================
# 4) REGRESIÓN LOGÍSTICA
# =========================
model = sm.Logit(y, X)
result = model.fit()

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