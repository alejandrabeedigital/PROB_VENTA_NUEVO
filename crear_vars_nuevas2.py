import pandas as pd
import numpy as np
from datetime import datetime

ARCHIVO_IN = "ventas_con_segmentacion_autonomo_qrk_origen_forzado_habitat_income_presvirt_movil_vars_nuevas.csv"
ARCHIVO_OUT = "ventas_con_segmentacion_autonomo_qrk_origen_forzado_habitat_income_presvirt_movil_vars_nuevas2.csv"

# =========================
# 1) CARGA
# =========================
df = pd.read_csv(ARCHIVO_IN, low_memory=False)
print(f"Filas cargadas: {len(df):,}")

# =========================
# 2) sin_intentos_recientes
# =========================
# TRUE si no hay intentos registrados en los últimos 6 meses
df["sin_intentos_recientes"] = df["intentos_ult6m"].isna()

# =========================
# 3) ANTIGÜEDAD EMPRESA
# =========================

# Convertir fecha
df["fe_creacion_empresa"] = pd.to_datetime(
    df["fe_creacion_empresa"],
    errors="coerce"
)

# Año actual
anio_actual = datetime.today().year

# Calcular antigüedad
df["antig_empresa_years"] = anio_actual - df["fe_creacion_empresa"].dt.year

# Crear categorías tipo R cut()
df["ant_empresa"] = pd.cut(
    df["antig_empresa_years"],
    bins=[0, 2, 5, 10, np.inf],
    labels=[
        "0_2_años",
        "2_5_años",
        "5_10_años",
        "10+_años"
    ],
    include_lowest=True
)

# =========================
# 4) DESCRIPTIVOS
# =========================

def descriptivo(col):
    tabla = (
        df[col]
        .astype(object)
        .where(df[col].notna(), "NaN")
        .value_counts()
        .rename_axis("valor")
        .reset_index(name="total")
    )
    tabla["porcentaje"] = tabla["total"] / tabla["total"].sum() * 100
    return tabla

desc1 = descriptivo("sin_intentos_recientes")
desc1.to_csv("descriptivos_sin_intentos_recientes.csv", index=False)

desc2 = descriptivo("ant_empresa")
desc2.to_csv("descriptivos_ant_empresa.csv", index=False)

print("Descriptivos generados")

# =========================
# 5) GUARDAR
# =========================
df.to_csv(ARCHIVO_OUT, index=False, encoding="utf-8")

print(f"\nArchivo guardado: {ARCHIVO_OUT}")
print(f"Filas guardadas: {len(df):,}")