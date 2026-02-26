import pandas as pd
import numpy as np

ARCHIVO_IN = "ventas_con_segmentacion_autonomo_qrk_origen_forzado_habitat_income.csv"
ARCHIVO_OUT = "ventas_con_segmentacion_autonomo_qrk_origen_forzado_imputado_habitat_ingresos_compe.csv"

COLS = ["compe_empr_muni_ssubsec", "compe_empr_prov_act"]

# =========================
# 1) CARGA
# =========================
df = pd.read_csv(ARCHIVO_IN, low_memory=False)
print(f"Filas totales: {len(df):,}")

# =========================
# 2) CONVERSIÓN A NUMÉRICO
# =========================
for col in COLS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        print(f"⚠️ Columna no encontrada: {col}")

# =========================
# 3) IMPUTACIÓN
# =========================
for col in COLS:
    if col not in df.columns:
        continue

    print("\n==============================")
    print(f"ANÁLISIS {col}")
    print("==============================")

    total_nan = df[col].isna().sum()
    pct_nan = total_nan / len(df) * 100

    print(f"NaN totales: {total_nan:,}")
    print(f"% NaN: {pct_nan:.3f}%")

    if total_nan == 0:
        print("✔ No hay NaN, no se imputa nada.")
        continue

    # Mediana global
    mediana_global = df[col].median()

    # Si existe provincia, intentamos imputar por grupo
    if "provincia" in df.columns:
        print("Imputando por mediana dentro de provincia...")

        df[col] = df.groupby("provincia")[col].transform(
            lambda x: x.fillna(x.median())
        )

        # Si alguna provincia estaba completamente vacía → mediana global
        df[col] = df[col].fillna(mediana_global)

    else:
        print("Imputando con mediana global...")
        df[col] = df[col].fillna(mediana_global)

    print(f"Mediana usada (global): {mediana_global:.4f}")
    print(f"NaN restantes tras imputación: {df[col].isna().sum()}")

# =========================
# 4) GUARDAR
# =========================
df.to_csv(ARCHIVO_OUT, index=False, encoding="utf-8")

print("\n=================================")
print("✅ Archivo guardado correctamente")
print(f"Ruta: {ARCHIVO_OUT}")
print("=================================")