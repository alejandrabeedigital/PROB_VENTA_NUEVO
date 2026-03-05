import pandas as pd

# Archivos
ARCHIVO_IN = "ventas_con_segmentacion_autonomo_qrk_origen_forzado_habitat_income_presvirt_movil.csv"
ARCHIVO_OUT = "clientes_outcome_autonomo.csv"

# =========================
# 1) CARGA
# =========================
df = pd.read_csv(ARCHIVO_IN, low_memory=False)
print(f"Filas leídas: {len(df):,}")

# =========================
# 2) SELECCIÓN DE COLUMNAS
# =========================
columnas = ["co_cliente", "outcome_forzado_autonomo"]

df_out = df[columnas].copy()

print(f"Filas en salida: {len(df_out):,}")

# =========================
# 3) GUARDAR ARCHIVO
# =========================
df_out.to_csv(ARCHIVO_OUT, index=False, encoding="utf-8")

print(f"\nArchivo guardado: {ARCHIVO_OUT}")