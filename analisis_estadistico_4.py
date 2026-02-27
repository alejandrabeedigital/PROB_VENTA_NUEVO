import pandas as pd
import numpy as np
import statsmodels.api as sm

ARCHIVO = "todo_con_resultados_4.csv"
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
# 2) VARIABLES (actualizadas al nuevo modelo)
# =========================
# Numéricas + categóricas del script de sklearn
features_num = [
    "q_rk_score",
    "origen_sc_o_no",
    "median_income_equiv",
    "compe_empr_muni_ssubsec",
]
features_cat = [
    "ct_merclie",
    "excliente",
    "outcome_forzado_autonomo",
    "habitat",
]

features = features_num + features_cat

# Limpiezas / coerciones numéricas (igual que en el otro script)
df["q_rk_score"] = pd.to_numeric(
    df["q_rk_score"].astype(str).str.replace(",", ".", regex=False),
    errors="coerce"
)
df["origen_sc_o_no"] = pd.to_numeric(df["origen_sc_o_no"], errors="coerce")
df["median_income_equiv"] = pd.to_numeric(df["median_income_equiv"], errors="coerce")
df["compe_empr_muni_ssubsec"] = pd.to_numeric(df["compe_empr_muni_ssubsec"], errors="coerce")

# habitat como object (evitar pandas StringDtype)
df["habitat"] = df["habitat"].astype(object)

# excliente como categórica 0/1 (string) -> para que get_dummies la trate igual
df["excliente"] = df["excliente"].map(
    {True: "1", False: "0", "True": "1", "False": "0"}
).fillna(df["excliente"].astype(str)).astype(object)

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
model = sm.GLM(y, X, family=sm.families.Binomial())
result = model.fit(maxiter=1000)

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