import pandas as pd

archivo_ventas = "ventas_con_segmentacion_autonomo_qrk_origen_forzado_habitat_income_presvirt_movil_vars_nuevas2.csv"
archivo_segmentos = "seg_qlik_parcial_con_pred_todo.csv"
archivo_salida = "ventas_con_outcome_sin_con_pred.csv"

# Leer ventas
ventas = pd.read_csv(
    archivo_ventas,
    low_memory=False,
    dtype={"co_cliente": str}
)

# Leer segmentos
# Probamos primero con ; porque suele ser lo habitual en CSV exportados desde Excel/Qlik
segmentos = pd.read_csv(
    archivo_segmentos,
    sep=";",
    low_memory=False,
    dtype={"co_cliente": str}
)

# Limpiar espacios en nombres de columnas
ventas.columns = ventas.columns.str.strip()
segmentos.columns = segmentos.columns.str.strip()

# Comprobar que existen las columnas necesarias
if "co_cliente" not in ventas.columns:
    raise ValueError(f"No existe la columna 'co_cliente' en {archivo_ventas}")

if "co_cliente" not in segmentos.columns:
    raise ValueError(f"No existe la columna 'co_cliente' en {archivo_segmentos}")

if "outcome_sin_con_pred" not in segmentos.columns:
    raise ValueError(f"No existe la columna 'outcome_sin_con_pred' en {archivo_segmentos}")

# Quedarnos solo con las columnas necesarias y evitar duplicados de co_cliente
segmentos_reducido = (
    segmentos[["co_cliente", "outcome_sin_con_pred"]]
    .drop_duplicates(subset=["co_cliente"])
)

# Merge por co_cliente
resultado = ventas.merge(
    segmentos_reducido,
    on="co_cliente",
    how="left"
)

# Guardar salida
resultado.to_csv(archivo_salida, index=False)

print(f"Proceso terminado. Archivo generado: {archivo_salida}")
print(f"Filas ventas: {len(ventas)}")
print(f"Filas resultado: {len(resultado)}")
print(f"Clientes con outcome informado: {resultado['outcome_sin_con_pred'].notna().sum()}")