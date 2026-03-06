import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

ARCHIVO_BASE = BASE_DIR / "seg_qlik_parcial_con_pred_todo.csv"
ARCHIVO_FORZADO = BASE_DIR / "ventas_con_segmentacion_autonomo_qrk_origen_forzado_habitat_income_presvirt_movil.csv"
ARCHIVO_OUT = BASE_DIR / "clientes_outcome_autonomo.csv"

# Leer con separador ;
df_base = pd.read_csv(ARCHIVO_BASE, sep=";", low_memory=False)
df_forzado = pd.read_csv(ARCHIVO_FORZADO, sep=",", low_memory=False)

print(f"Filas leídas base: {len(df_base):,}")
print(f"Filas leídas forzado: {len(df_forzado):,}")

columnas_base = [
    "co_cliente",
    "nombre_empresa",
    "no_comer",
    "email",
    "tx_actvad",
    "telefono",
    "website",
    "ct_sociedad",
    "ct_sociedad_pred",
    "p_autonomo",
    "outcome_pred",
]

columna_forzado = "outcome_forzado_autonomo"

faltan_base = [c for c in columnas_base if c not in df_base.columns]
if faltan_base:
    print("Columnas reales en base:")
    print(df_base.columns.tolist())
    raise KeyError(f"Faltan en {ARCHIVO_BASE.name}: {faltan_base}")

if "co_cliente" not in df_forzado.columns:
    print("Columnas reales en forzado:")
    print(df_forzado.columns.tolist())
    raise KeyError(f"Falta 'co_cliente' en {ARCHIVO_FORZADO.name}")

if columna_forzado not in df_forzado.columns:
    print("Columnas reales en forzado:")
    print(df_forzado.columns.tolist())
    raise KeyError(f"Falta '{columna_forzado}' en {ARCHIVO_FORZADO.name}")

df_base_sel = df_base[columnas_base].copy()
df_forzado_sel = df_forzado[["co_cliente", columna_forzado]].drop_duplicates(subset=["co_cliente"])

df_out = df_base_sel.merge(df_forzado_sel, on="co_cliente", how="inner")

print(f"Filas en salida: {len(df_out):,}")

df_out.to_csv(ARCHIVO_OUT, index=False, encoding="utf-8")
print(f"\nArchivo guardado: {ARCHIVO_OUT}")