import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

ARCHIVO = "todo_con_resultados_12.csv"
TARGET = "ganada"

COL_MODELO_NUEVO = "prob_venta_modelo"
COL_MODELO_ANTIGUO = "prob_venta"

# =========================
# 1) CARGA
# =========================
df = pd.read_csv(ARCHIVO, low_memory=False)

print(f"Filas cargadas: {len(df):,}")

# limpiar target
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df[df[TARGET].isin([0,1])].copy()

# quitar filas sin predicción
df = df.dropna(subset=[COL_MODELO_NUEVO, COL_MODELO_ANTIGUO])

y = df[TARGET]

print(f"Filas usadas: {len(df):,}")
print(f"Tasa base: {y.mean():.6f}")

# =========================
# 2) MÉTRICAS GLOBALES
# =========================
print("\n==============================")
print("MÉTRICAS GLOBALES")
print("==============================")

auc_new = roc_auc_score(y, df[COL_MODELO_NUEVO])
auc_old = roc_auc_score(y, df[COL_MODELO_ANTIGUO])

auprc_new = average_precision_score(y, df[COL_MODELO_NUEVO])
auprc_old = average_precision_score(y, df[COL_MODELO_ANTIGUO])

print("\nAUC")
print(f"Modelo nuevo:   {auc_new:.4f}")
print(f"Modelo antiguo: {auc_old:.4f}")

print("\nAUPRC")
print(f"Modelo nuevo:   {auprc_new:.6f}")
print(f"Modelo antiguo: {auprc_old:.6f}")

# =========================
# 3) TOP-K COMPARACIÓN
# =========================
print("\n==============================")
print("TOP-K COMPARACIÓN")
print("==============================")

def top_metrics(score_col):

    tmp = df[[TARGET, score_col]].copy()
    tmp = tmp.sort_values(score_col, ascending=False)

    res = {}

    for pct in [0.01,0.02,0.05,0.10,0.20]:

        n = int(len(tmp)*pct)

        tasa = tmp.iloc[:n][TARGET].mean()

        res[pct] = tasa

    return res

top_new = top_metrics(COL_MODELO_NUEVO)
top_old = top_metrics(COL_MODELO_ANTIGUO)

print("\nTasa venta TOP X%")

for pct in top_new:

    print(f"\nTOP {int(pct*100)}%")
    print(f"Nuevo:   {top_new[pct]:.6f}")
    print(f"Antiguo: {top_old[pct]:.6f}")

# =========================
# 4) LIFT
# =========================
base_rate = y.mean()

print("\n==============================")
print("LIFT TOP")
print("==============================")

for pct in top_new:

    lift_new = top_new[pct]/base_rate
    lift_old = top_old[pct]/base_rate

    print(f"\nTOP {int(pct*100)}%")
    print(f"Nuevo lift:   {lift_new:.2f}x")
    print(f"Antiguo lift: {lift_old:.2f}x")

# =========================
# 5) DECILES
# =========================
def deciles(score):

    tmp = df[[TARGET]].copy()
    tmp["score"] = score

    tmp = tmp.sort_values("score", ascending=False).reset_index(drop=True)

    tmp["decil"] = pd.qcut(tmp.index,10,labels=False)+1

    tabla = tmp.groupby("decil")[TARGET].agg(
        clientes="count",
        ventas="sum",
        tasa="mean"
    )

    return tabla

print("\n==============================")
print("DECILES MODELO NUEVO")
print("==============================")
print(deciles(df[COL_MODELO_NUEVO]))

print("\n==============================")
print("DECILES MODELO ANTIGUO")
print("==============================")
print(deciles(df[COL_MODELO_ANTIGUO]))

print("\n\nComparación terminada.")