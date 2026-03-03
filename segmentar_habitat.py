import pandas as pd
import numpy as np

ARCHIVO_IN = "ventas_con_segmentacion_autonomo_qrk_origen_forzado_habitat_income_gmb_imputado.csv"
ARCHIVO_OUT = "ventas_con_segmentacion_autonomo_qrk_origen_forzadp_habitat2_income_gmb_imput.csv"
ARCHIVO_DESC = "descriptivos_habitat.csv"

COL_HABITAT = "habitat"

# =========================
# 1) CARGA
# =========================
df = pd.read_csv(ARCHIVO_IN, low_memory=False)
print(f"Filas totales: {len(df):,}")

if COL_HABITAT not in df.columns:
    raise ValueError(f"No existe la columna '{COL_HABITAT}' en el CSV de entrada.")

# Normalizar habitat (limpia espacios, NaN, etc.)
hab = df[COL_HABITAT].astype("string")
hab = hab.str.strip()

# Unificar casos "desconocidos"
# (incluye NaN reales + strings típicos)
mask_desconocido = (
    hab.isna() |
    (hab == "") |
    (hab.str.upper().isin(["DESCONOCIDO", "UNKNOWN", "NAN", "NONE"]))
)

# =========================
# 2) MAPEO A 3 CATEGORÍAS
# =========================
# Mapeo desde los labels originales a 3 buckets.
# Ajusta aquí si tienes etiquetas distintas.
map_3 = {
    "<1k": "MUNICIPIO_PEQUENO",
    "1k - 10k": "MUNICIPIO_PEQUENO",
    "10k - 20k": "MUNICIPIO_PEQUENO",

    "20k - 50k": "MUNICIPIO_MEDIANO",
    "50k - 100k": "MUNICIPIO_MEDIANO",

    "Capital o >100k": "MUNICIPIO_GRANDE",
}

hab_3 = hab.map(map_3)

# Los valores no mapeados (etiquetas raras) también los consideramos "desconocidos"
mask_no_mapeado = hab_3.isna() & (~mask_desconocido)

mask_desconocido_total = mask_desconocido | mask_no_mapeado

# =========================
# 3) IMPUTAR DESCONOCIDOS AL GRUPO MÁS COMÚN
# =========================
# Moda calculada SOLO con valores válidos (ya mapeados y no desconocidos)
validos = hab_3[~mask_desconocido_total].dropna()

if len(validos) == 0:
    raise ValueError("No hay valores válidos de habitat para calcular el grupo más común.")

grupo_mas_comun = validos.value_counts().idxmax()
print(f"Grupo más común (para imputar desconocidos): {grupo_mas_comun}")

hab_3 = hab_3.fillna(grupo_mas_comun)
hab_3.loc[mask_desconocido_total] = grupo_mas_comun

df["habitat_3"] = hab_3

# (Opcional) si quieres reemplazar la original por la nueva:
# df["habitat"] = df["habitat_3"]

# =========================
# 4) DESCRIPTIVOS
# =========================
conteos = df["habitat_3"].value_counts(dropna=False).rename("count")
porc = (conteos / len(df) * 100).rename("pct")

desc = pd.concat([conteos, porc], axis=1).reset_index().rename(columns={"index": "habitat_3"})
desc["pct"] = desc["pct"].round(2)

print("\n==============================")
print("DESCRIPTIVOS HABITAT_3")
print("==============================")
print(desc)

desc.to_csv(ARCHIVO_DESC, index=False, encoding="utf-8")
print(f"\n✅ Guardado descriptivos: {ARCHIVO_DESC}")

# =========================
# 5) GUARDAR DATASET NUEVO
# =========================
df.to_csv(ARCHIVO_OUT, index=False, encoding="utf-8")
print(f"✅ Guardado dataset: {ARCHIVO_OUT}")
print(f"Filas guardadas: {len(df):,}")