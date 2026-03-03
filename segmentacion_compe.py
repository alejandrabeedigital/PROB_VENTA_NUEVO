import pandas as pd
import numpy as np

ARCHIVO_IN = "ventas_con_segmentacion_autonomo_qrk_origen_forzadp_habitat2_income_gmb_imput.csv"
ARCHIVO_OUT = "ventas_con_segmentacion_autonomo_qrk_origen_forzadp_habitat2_income_gmb_imput_compe.csv"
DESCRIPTIVOS_OUT = "descriptivos_compe.csv"

COL_COMPE = "compe_pop_muni_ssubsec"   # <- la variable que quieres trocear
COL_TARGET = "ganada"                 # si existe, añadimos tasa/ventas por grupo

def main():
    df = pd.read_csv(ARCHIVO_IN, low_memory=False)
    print(f"Filas leídas: {len(df):,}")

    if COL_COMPE not in df.columns:
        raise ValueError(f"No existe la columna '{COL_COMPE}' en el CSV.")

    # 1) Asegurar numérico
    df[COL_COMPE] = pd.to_numeric(df[COL_COMPE], errors="coerce")
    nan_before = int(df[COL_COMPE].isna().sum())
    print(f"NaN en {COL_COMPE} (antes): {nan_before:,} ({nan_before/len(df)*100:.2f}%)")

    # 2) Imputación para poder categorizar SIN romper nada
    #    (como hiciste con q_rk_score): imputamos NaN a la MEDIANA global de la variable
    mediana = df[COL_COMPE].median(skipna=True)
    if pd.isna(mediana):
        raise ValueError(f"No se pudo calcular la mediana de '{COL_COMPE}' (todo NaN).")

    df[f"{COL_COMPE}_imputado"] = df[COL_COMPE].fillna(mediana)

    nan_after = int(df[f"{COL_COMPE}_imputado"].isna().sum())
    print(f"NaN en {COL_COMPE}_imputado (después): {nan_after:,}")

    # 3) Cortes en 3 grupos por terciles (33/33/33 aprox)
    #    - qcut puede fallar si hay muchos valores repetidos -> usamos duplicates='drop'
    try:
        cats, bins = pd.qcut(
            df[f"{COL_COMPE}_imputado"],
            q=[0, 1/3, 2/3, 1],
            labels=["BAJA", "MEDIA", "ALTA"],
            retbins=True,
            duplicates="drop"
        )
        df["compe_cat_3"] = cats.astype(str)
        print("Cortes (bins):", bins)
    except Exception as e:
        # fallback robusto si qcut no puede crear 3 bins por empates
        p33 = df[f"{COL_COMPE}_imputado"].quantile(1/3)
        p66 = df[f"{COL_COMPE}_imputado"].quantile(2/3)

        def asignar(v):
            if v <= p33:
                return "BAJA"
            elif v <= p66:
                return "MEDIA"
            return "ALTA"

        df["compe_cat_3"] = df[f"{COL_COMPE}_imputado"].apply(asignar)
        bins = np.array([df[f"{COL_COMPE}_imputado"].min(), p33, p66, df[f"{COL_COMPE}_imputado"].max()])
        print("qcut falló, usando cuantiles manuales.")
        print("Cortes (bins):", bins, "| error:", repr(e))

    # 4) Descriptivos por categoría
    desc = (
        df.groupby("compe_cat_3")[f"{COL_COMPE}_imputado"]
        .agg(count="count", mean="mean", median="median", min="min", max="max")
        .reset_index()
    )
    desc["pct"] = desc["count"] / len(df)

    # Si existe target, añadimos tasa y ventas por grupo (muy útil para decidir si tiene sentido)
    if COL_TARGET in df.columns:
        df[COL_TARGET] = pd.to_numeric(df[COL_TARGET], errors="coerce")
        df_ok = df[df[COL_TARGET].isin([0, 1])].copy()
        df_ok[COL_TARGET] = df_ok[COL_TARGET].astype(int)

        extra = (
            df_ok.groupby("compe_cat_3")[COL_TARGET]
            .agg(tasa_venta="mean", ventas="sum", n="count")
            .reset_index()
        )
        desc = desc.merge(extra, on="compe_cat_3", how="left")

    # Orden BAJA->MEDIA->ALTA
    orden = pd.Categorical(desc["compe_cat_3"], categories=["BAJA", "MEDIA", "ALTA"], ordered=True)
    desc = desc.assign(_ord=orden).sort_values("_ord").drop(columns="_ord")

    # 5) Guardar outputs
    df.to_csv(ARCHIVO_OUT, index=False, encoding="utf-8")
    desc.to_csv(DESCRIPTIVOS_OUT, index=False, encoding="utf-8")

    print(f"\n✅ Guardado dataset: {ARCHIVO_OUT}")
    print(f"✅ Guardado descriptivos: {DESCRIPTIVOS_OUT}")
    print("\nDescriptivos (preview):")
    print(desc)

if __name__ == "__main__":
    main()