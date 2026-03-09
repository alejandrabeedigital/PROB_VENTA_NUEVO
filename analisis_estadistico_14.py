import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

ARCHIVO = "todo_con_resultados_14.csv"
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
features_num = [
    "q_rk_score"
]

features_cat = [
    "ct_merclie",
    "excliente_cat",
    "sc_autonomo_pred_nombre",
    "con_web",
    "sin_gmb",
    "gmb_sin_owner",
    "movil",
    "sin_intentos_recientes",
    "ant_empresa"
]

features = features_num + features_cat

# =========================
# 2.1) TIPOS / LIMPIEZA MÍNIMA
# =========================
# numéricas
df["q_rk_score"] = pd.to_numeric(
    df["q_rk_score"].astype(str).str.replace(",", ".", regex=False),
    errors="coerce"
)

# categóricas -> object (evita líos con StringDtype y <NA>)
for c in features_cat:
    if c in df.columns:
        df[c] = df[c].astype(object)

# asegurar que existen todas
faltan = [c for c in features + [TARGET] if c not in df.columns]
if faltan:
    raise ValueError(f"Faltan columnas en el CSV: {faltan}")

# dataset final (sin NaN para Logit)
df_model = df[features + [TARGET]].dropna().copy()
print(f"Filas usadas para inferencia: {len(df_model):,}")
print(f"Tasa base: {df_model[TARGET].mean():.6f}")

# =========================
# 3) DUMMIES
# =========================
X = pd.get_dummies(df_model[features], drop_first=True)
y = df_model[TARGET]

# todo numérico
X = X.astype(float)

# Escalar SOLO numéricas dentro de X (mejora estabilidad numérica)
for col in ["q_rk_score"]:
    if col in X.columns:
        mu = X[col].mean()
        sd = X[col].std()
        if sd is not None and sd > 0:
            X[col] = (X[col] - mu) / sd

# constante
X = sm.add_constant(X, has_constant="add")
print(X.dtypes)

# =========================
# 4) REGRESIÓN LOGÍSTICA
# =========================
model = sm.Logit(y, X)

# Optimizador más estable para evitar overflow / no-convergencia
result = model.fit(method="lbfgs", maxiter=5000, disp=True)

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
}).sort_values("p_value")

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

# =========================
# 8) GRÁFICO EFECTOS MARGINALES
# =========================
mfx = marginal.margeff
variables = marginal.summary_frame().index

df_mfx = pd.DataFrame({
    "variable": variables,
    "dy_dx": mfx
})

# quitar constante si aparece
df_mfx = df_mfx[df_mfx["variable"] != "const"].copy()

# multiplicar por 10000 (lectura más cómoda con tasas bajas)
df_mfx["dy_dx_10000"] = df_mfx["dy_dx"] * 10000

# ordenar por impacto
df_mfx = df_mfx.sort_values("dy_dx_10000")

plt.figure(figsize=(10, 6))
plt.barh(df_mfx["variable"], df_mfx["dy_dx_10000"])
plt.xlabel("Cambio en probabilidad (por 10,000 clientes)")
plt.title("Efectos marginales del modelo logístico")
plt.tight_layout()
plt.show()