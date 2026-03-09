import pandas as pd
import numpy as np

archivo_entrada = "ventas_con_outcome_sin_con_pred_con_ct_sociedad.csv"
archivo_salida = "ventas_con_outcome_sin_con_pred_sc.csv"

df = pd.read_csv(
    archivo_entrada,
    low_memory=False,
    dtype={"origen": str, "ct_sociedad": str}
)

df.columns = df.columns.str.strip()

# Normalizar columnas
df["origen"] = df["origen"].fillna("").str.lower().str.strip()

df["ct_sociedad"] = (
    df["ct_sociedad"]
    .fillna("")
    .astype(str)
    .str.replace(".0", "", regex=False)
    .str.strip()
)

# Crear grupo
df["sc_autonomo_pred_nombre"] = np.where(
    (df["ct_sociedad"] == "22") | (df["origen"].str.startswith("sc")),
    "DESCONOCIDO",
    np.where(df["ct_sociedad"].isin(["0","00"]), "AUTONOMO", "NO_AUTONOMO")
)

# Guardar archivo
df.to_csv(archivo_salida, index=False)

# ======================
# RESUMEN
# ======================

resumen = (
    df.groupby("sc_autonomo_pred_nombre")
    .agg(
        clientes=("co_cliente", "count"),
        ganadas=("ganada", "sum"),
        pct_ganada=("ganada", "mean")
    )
)

resumen["pct_ganada"] = (resumen["pct_ganada"] * 100).round(2)

print("\nRESUMEN POR GRUPO:")
print(resumen)

print(f"\nArchivo generado: {archivo_salida}")