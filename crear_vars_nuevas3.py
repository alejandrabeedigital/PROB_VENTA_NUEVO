import pandas as pd

archivo_ventas = "ventas_con_segmentacion_autonomo_qrk_origen_forzado_habitat_income_presvirt_movil_vars_nuevas2.csv"
archivo_salida = "ventas_con_outcome_sin_con_pred.csv"

ventas = pd.read_csv(
    archivo_ventas,
    low_memory=False,
    dtype={"co_cliente": str}
)

ventas.columns = ventas.columns.str.strip()

if "co_cliente" not in ventas.columns:
    raise ValueError(f"No existe la columna 'co_cliente' en {archivo_ventas}")

if "outcome_sin_con_pred" not in ventas.columns:
    raise ValueError(f"No existe la columna 'outcome_sin_con_pred' en {archivo_ventas}")

ventas.to_csv(archivo_salida, index=False)

print(f"Proceso terminado. Archivo generado: {archivo_salida}")
print(f"Filas ventas: {len(ventas)}")
print(f"Clientes con outcome informado: {ventas['outcome_sin_con_pred'].notna().sum()}")