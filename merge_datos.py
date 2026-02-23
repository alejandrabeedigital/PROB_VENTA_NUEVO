import csv
import pandas as pd

# --- RUTAS (raw strings para evitar problemas con \ en Windows) ---
archivo_ventas = r"C:\Users\AlejandradeSelysGóme\PycharmProjects\PROB_VENTA_NUEVO\t_pr_venta_FINAL.csv"
archivo_segmentacion = r"C:\Users\AlejandradeSelysGóme\PycharmProjects\PROB_VENTA_NUEVO\seg_qlik_parcial_con_pred_todo.csv"
salida = r"C:\Users\AlejandradeSelysGóme\PycharmProjects\PROB_VENTA_NUEVO\ventas_con_segmentacion.csv"


def detectar_delimitador(path: str, encoding: str = "utf-8") -> str:
    """Intenta detectar el delimitador del CSV. Si falla, usa ';' como default típico en ES."""
    try:
        with open(path, "r", encoding=encoding, newline="") as f:
            sample = f.read(8192)
        dialect = csv.Sniffer().sniff(sample, delimiters=";,|\t,")
        return dialect.delimiter
    except Exception:
        return ";"


def leer_csv_robusto(path: str, encoding: str = "utf-8") -> pd.DataFrame:
    """
    Lee un CSV intentando:
    - detectar delimitador
    - usar low_memory=False para evitar DtypeWarning
    - si hay ParserError, reintenta con engine="python" y on_bad_lines="warn"
    """
    sep = detectar_delimitador(path, encoding=encoding)

    try:
        return pd.read_csv(path, sep=sep, encoding=encoding, low_memory=False)
    except UnicodeDecodeError:
        # fallback típico en Windows/Excel
        return pd.read_csv(path, sep=sep, encoding="latin-1", low_memory=False)
    except pd.errors.ParserError:
        # tolerante con líneas "rotas"
        try:
            return pd.read_csv(
                path,
                sep=sep,
                encoding=encoding,
                engine="python",
                on_bad_lines="warn",
            )
        except UnicodeDecodeError:
            return pd.read_csv(
                path,
                sep=sep,
                encoding="latin-1",
                engine="python",
                on_bad_lines="warn",
            )


# --- CARGA ---
df_ventas = leer_csv_robusto(archivo_ventas)
df_segmentacion = leer_csv_robusto(archivo_segmentacion)

# --- VALIDACIONES BÁSICAS ---
if "co_cliente" not in df_ventas.columns:
    raise KeyError(f"No existe la columna 'co_cliente' en {archivo_ventas}. Columnas: {list(df_ventas.columns)[:30]}")

if "co_cliente" not in df_segmentacion.columns:
    raise KeyError(f"No existe la columna 'co_cliente' en {archivo_segmentacion}. Columnas: {list(df_segmentacion.columns)[:30]}")

# --- NORMALIZAR CLAVE ---
df_ventas["co_cliente"] = df_ventas["co_cliente"].astype(str).str.strip()
df_segmentacion["co_cliente"] = df_segmentacion["co_cliente"].astype(str).str.strip()

# --- SI HAY DUPLICADOS EN SEGMENTACIÓN, NOS QUEDAMOS CON LA ÚLTIMA FILA POR CLIENTE ---
# (si prefieres otra lógica, dímelo)
df_segmentacion = df_segmentacion.drop_duplicates(subset=["co_cliente"], keep="last")

# --- MERGE ---
df_final = df_ventas.merge(df_segmentacion, on="co_cliente", how="left")

# --- GUARDAR ---
df_final.to_csv(salida, index=False, encoding="utf-8")

print("✅ Unión completada.")
print("📄 Archivo generado:", salida)
print("📊 Filas/columnas:", df_final.shape)