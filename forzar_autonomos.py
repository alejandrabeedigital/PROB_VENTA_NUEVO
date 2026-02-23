import pandas as pd

# --- RUTAS ---
archivo_entrada = r"C:\Users\AlejandradeSelysGóme\PycharmProjects\PROB_VENTA_NUEVO\ventas_con_segmentacion.csv"
archivo_salida = r"C:\Users\AlejandradeSelysGóme\PycharmProjects\PROB_VENTA_NUEVO\ventas_con_segmentacion_forzado_autonomo.csv"

# --- CARGA ---
df = pd.read_csv(archivo_entrada, encoding="utf-8", low_memory=False)

# --- VALIDACIONES ---
cols_necesarias = {"outcome_pred", "p_autonomo"}
faltan = cols_necesarias - set(df.columns)
if faltan:
    raise KeyError(f"Faltan columnas necesarias en el CSV: {faltan}. Columnas disponibles: {list(df.columns)[:50]}")

# Asegurar que p_autonomo es numérico
df["p_autonomo"] = pd.to_numeric(df["p_autonomo"], errors="coerce")

# Normalizar texto por si hay espacios/cambios de mayúsculas
df["outcome_pred"] = df["outcome_pred"].astype(str).str.strip().str.upper()

# --- LÓGICA ---
# Por defecto, copiamos outcome_pred
df["outcome_forzado_autonomo"] = df["outcome_pred"]

# Solo modificamos los "DESCONOCIDO"
mask_desconocido = df["outcome_pred"] == "DESCONOCIDO"

# Regla: si p_autonomo >= 0.5 -> AUTONOMO; si < 0.5 -> NO AUTONOMO
df.loc[mask_desconocido, "outcome_forzado_autonomo"] = df.loc[mask_desconocido, "p_autonomo"].apply(
    lambda p: "AUTONOMO" if pd.notna(p) and p >= 0.5 else "NO_AUTONOMO"
)

# --- GUARDAR ---
df.to_csv(archivo_salida, index=False, encoding="utf-8")

print("✅ Proceso completado.")
print("📄 Archivo generado:", archivo_salida)
print("📊 Filas/columnas:", df.shape)
print("ℹ️ DESCONOCIDO forzados:", int(mask_desconocido.sum()))