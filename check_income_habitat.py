import pandas as pd
import numpy as np

ARCHIVO = r"C:\Users\AlejandradeSelysGóme\PycharmProjects\PROB_VENTA_NUEVO\ventas_con_segmentacion_autonomo_qrk_origen_forzado.csv"

# =========================
# 1) CARGA
# =========================
df = pd.read_csv(ARCHIVO, low_memory=False)
print(f"\nFilas totales: {len(df):,}")

# =========================
# 2) COMPROBAR EXISTENCIA
# =========================
for col in ["habitat", "median_income_equiv"]:
    if col in df.columns:
        print(f"\n✔ Columna encontrada: {col}")
    else:
        print(f"\n❌ ERROR: No existe la columna {col}")

# =========================
# 3) ANALISIS median_income_equiv
# =========================
print("\n==============================")
print("ANÁLISIS median_income_equiv")
print("==============================")

if "median_income_equiv" in df.columns:

    serie = df["median_income_equiv"]

    print(f"Tipo original: {serie.dtype}")
    print(f"NaN totales: {serie.isna().sum():,}")
    print(f"% NaN: {serie.isna().mean()*100:.2f}%")

    # Intentar convertir a numérico
    serie_num = pd.to_numeric(serie, errors="coerce")

    print(f"NaN tras conversión numérica: {serie_num.isna().sum():,}")

    print("\nResumen estadístico:")
    print(serie_num.describe())

    # Valores negativos (no deberían existir)
    negativos = (serie_num < 0).sum()
    print(f"\nValores negativos: {negativos}")

# =========================
# 4) ANALISIS habitat
# =========================
print("\n==============================")
print("ANÁLISIS habitat")
print("==============================")

if "habitat" in df.columns:

    serie = df["habitat"]

    print(f"Tipo original: {serie.dtype}")
    print(f"NaN totales: {serie.isna().sum():,}")
    print(f"% NaN: {serie.isna().mean()*100:.2f}%")

    print("\nValores únicos:")
    print(serie.value_counts(dropna=False).head(20))

    print(f"\nNúmero de categorías distintas: {serie.nunique()}")