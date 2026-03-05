import pandas as pd
import numpy as np

ARCHIVO_IN = "ventas_con_segmentacion_autonomo_qrk_origen_forzado_habitat_income.csv"
ARCHIVO_OUT = "ventas_con_segmentacion_autonomo_qrk_origen_forzado_habitat_income_presvirt_movil.csv"

DESC_MOVIL = "descriptivos_movil.csv"
DESC_CLAIM = "descriptivos_claim_business.csv"
DESC_RANKING = "descriptivos_ranking_number.csv"


def guardar_descriptivo(df: pd.DataFrame, col: str, out_path: str) -> None:
    """
    Genera un CSV con:
    - count
    - porcentaje
    - (opcional) n_nan y pct_nan en una fila resumen
    """
    total = len(df)
    vc = df[col].value_counts(dropna=False)
    desc = vc.rename("count").to_frame()
    desc["pct"] = (desc["count"] / total) * 100
    desc = desc.reset_index().rename(columns={"index": col})

    # fila extra resumen NaN (solo si hay NaN reales)
    n_nan = df[col].isna().sum()
    if n_nan > 0:
        resumen = pd.DataFrame([{
            col: "__NaN__",
            "count": n_nan,
            "pct": (n_nan / total) * 100
        }])
        # ojo: value_counts(dropna=False) ya incluye NaN como NaN, esto es un “resumen” extra explícito
        desc = pd.concat([desc, resumen], ignore_index=True)

    desc.to_csv(out_path, index=False, encoding="utf-8")


# =========================
# 1) CARGA
# =========================
df = pd.read_csv(ARCHIVO_IN, low_memory=False)
print(f"\nFilas totales: {len(df):,}")

# Asegurar numérico
df["claim_business"] = pd.to_numeric(df.get("claim_business"), errors="coerce")
df["ranking_number"] = pd.to_numeric(df.get("ranking_number"), errors="coerce")

# =========================
# 2) ANALISIS INICIAL NaN
# =========================
print("\n==============================")
print("ANÁLISIS INICIAL NaN")
print("==============================\n")

for col in ["movil", "claim_business", "ranking_number"]:
    if col not in df.columns:
        print(f"{col}: ❌ NO EXISTE EN EL CSV")
        continue
    total_nan = df[col].isna().sum()
    pct_nan = total_nan / len(df) * 100
    print(f"{col}: {total_nan:,} NaN ({pct_nan:.2f}%)")

# =========================
# 3) TRANSFORMACIONES (CATEGÓRICAS)
# =========================

# 3.1 claim_business -> categórica: TRUE / FALSE / MEDIO
df["claim_business_cat"] = np.where(
    df["claim_business"].isna(),
    "MEDIO",
    np.where(df["claim_business"] == 1, "TRUE", "FALSE")
)

# 3.2 ranking_number -> categórica:
ranking = df["ranking_number"]

# cuantiles solo con valores existentes
ranking_notna = ranking.dropna()
if len(ranking_notna) > 0:
    q33 = ranking_notna.quantile(1/3)
    q66 = ranking_notna.quantile(2/3)
else:
    q33 = np.nan
    q66 = np.nan

def categoriza_ranking(x):
    if pd.isna(x):
        return "RANKING DESCONOCIDO"
    # si por lo que sea no se pudieron calcular cuantiles (todo NaN), lo dejamos desconocido
    if pd.isna(q33) or pd.isna(q66):
        return "RANKING DESCONOCIDO"
    # cuanto más bajo, mejor
    if x <= q33:
        return "RANKING BUENO"
    elif x <= q66:
        return "RANKING MEDIO"
    else:
        return "RANKING MALO"

df["ranking_cat"] = df["ranking_number"].apply(categoriza_ranking)

# 3.3 movil -> categórica TRUE/FALSE
if "movil" not in df.columns:
    raise ValueError("No existe la columna 'movil' en el CSV de entrada.")


# =========================
# 4) DESCRIPTIVOS (CSV)
# =========================
print("\n==============================")
print("GENERANDO DESCRIPTIVOS")
print("==============================\n")

# descriptivos de movil (sobre movil_cat + también sobre movil crudo si quieres ver NaN/vacíos)
guardar_descriptivo(df, "movil", DESC_MOVIL)
print(f"✅ Guardado: {DESC_MOVIL}")

# descriptivos de claim_business (sobre claim_business_cat)
guardar_descriptivo(df, "claim_business_cat", DESC_CLAIM)
print(f"✅ Guardado: {DESC_CLAIM}")

# descriptivos de ranking_number (sobre ranking_cat)
guardar_descriptivo(df, "ranking_cat", DESC_RANKING)
print(f"✅ Guardado: {DESC_RANKING}")

# =========================
# 5) VALIDACIÓN POR CONSOLA
# =========================
print("\n==============================")
print("VALIDACIÓN POST-TRANSFORMACIÓN")
print("==============================\n")

for col in ["claim_business_cat", "ranking_cat", "movil"]:
    print(f"{col} → NaN restantes: {df[col].isna().sum()}")

print("\nDistribución movil:")
print(df["movil"].value_counts(dropna=False))

print("\nDistribución claim_business_cat:")
print(df["claim_business_cat"].value_counts(dropna=False))

print("\nDistribución ranking_cat:")
print(df["ranking_cat"].value_counts(dropna=False))

if not (pd.isna(q33) or pd.isna(q66)):
    print("\nCortes ranking_number (cuantiles):")
    print(f"q33 = {q33:.4f} | q66 = {q66:.4f}")

# =========================
# 6) GUARDAR DATASET FINAL
# =========================
df.to_csv(ARCHIVO_OUT, index=False, encoding="utf-8")

print(f"\n✅ Guardado correctamente en: {ARCHIVO_OUT}")
print(f"Filas guardadas: {len(df):,}")