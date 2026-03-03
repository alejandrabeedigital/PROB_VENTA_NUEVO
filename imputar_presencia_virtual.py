import pandas as pd
import numpy as np

ARCHIVO_IN = "ventas_con_segmentacion_autonomo_qrk_origen_forzado_habitat_income.csv"
ARCHIVO_OUT = "ventas_con_segmentacion_autonomo_qrk_origen_forzado_habitat_income_gmb_imputado.csv"

# =========================
# 1) CARGA
# =========================
df = pd.read_csv(ARCHIVO_IN, low_memory=False)
print(f"\nFilas totales: {len(df):,}")

# Asegurar numérico
df["total_rating"] = pd.to_numeric(df.get("total_rating"), errors="coerce")
df["claim_business"] = pd.to_numeric(df.get("claim_business"), errors="coerce")

# =========================
# 2) ANALISIS INICIAL
# =========================
print("\n==============================")
print("ANÁLISIS INICIAL NaN")
print("==============================\n")

for col in ["total_rating", "claim_business"]:
    total_nan = df[col].isna().sum()
    pct_nan = total_nan / len(df) * 100
    print(f"{col}: {total_nan:,} NaN ({pct_nan:.2f}%)")

# =========================
# 3) IMPUTACIÓN
# =========================

# 3.1 Calcular mediana real del rating (solo valores existentes)
mediana_rating = df["total_rating"].median()

# 3.2 Rating imputado a la mediana
df["total_rating_imputado"] = df["total_rating"].fillna(mediana_rating)

# 3.3 Claim imputado a 0 (si es NaN asumimos no reclamado)
df["claim_business_imputado"] = df["claim_business"].fillna(0)
df["claim_business_imputado"] = df["claim_business_imputado"].astype(int)

# =========================
# 4) VALIDACIÓN
# =========================
print("\n==============================")
print("VALIDACIÓN POST-IMPUTACIÓN")
print("==============================\n")

for col in ["total_rating_imputado", "claim_business_imputado"]:
    print(f"{col} → NaN restantes: {df[col].isna().sum()}")

print("\nDistribución claim_business_imputado:")
print(df["claim_business_imputado"].value_counts())

print("\nDistribución total_rating_imputado:")
print(df["total_rating_imputado"].describe())

# =========================
# 5) GUARDAR
# =========================
df.to_csv(ARCHIVO_OUT, index=False, encoding="utf-8")

print(f"\n✅ Guardado correctamente en: {ARCHIVO_OUT}")
print(f"Filas guardadas: {len(df):,}")