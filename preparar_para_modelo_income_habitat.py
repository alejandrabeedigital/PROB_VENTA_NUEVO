import pandas as pd
import numpy as np

IN_PATH = "ventas_con_segmentacion_autonomo_qrk_origen_forzado.csv"
OUT_PATH = "ventas_con_segmentacion_autonomo_qrk_origen_forzado_habitat_income.csv"

def to_float_series(s: pd.Series) -> pd.Series:
    # Convierte strings con coma decimal a float y deja NaN donde no se pueda
    return pd.to_numeric(
        s.astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )

def main():
    df = pd.read_csv(IN_PATH, low_memory=False)
    print(f"Filas leídas: {len(df):,}")

    # =========================
    # 1) q_rk_score -> float + imputación a 0.5
    # =========================
    if "q_rk_score" not in df.columns:
        raise KeyError("No existe la columna q_rk_score en el CSV.")

    df["q_rk_score"] = to_float_series(df["q_rk_score"])
    n_nan_qrk = int(df["q_rk_score"].isna().sum())
    df["q_rk_score"] = df["q_rk_score"].fillna(0.5)
    print(f"q_rk_score: NaN antes={n_nan_qrk:,} -> imputados a 0.5")

    # =========================
    # 2) median_income_equiv -> float + imputación a mediana
    # =========================
    if "median_income_equiv" not in df.columns:
        raise KeyError("No existe la columna median_income_equiv en el CSV.")

    df["median_income_equiv"] = to_float_series(df["median_income_equiv"])
    n_nan_income = int(df["median_income_equiv"].isna().sum())

    mediana_income = float(df["median_income_equiv"].median(skipna=True))
    if np.isnan(mediana_income):
        # Caso extremo: todo NaN
        mediana_income = 0.0

    df["median_income_equiv"] = df["median_income_equiv"].fillna(mediana_income)

    # Si prefieres imputar con un valor fijo en vez de mediana, usa esto:
    # df["median_income_equiv"] = df["median_income_equiv"].fillna(20000)

    print(
        f"median_income_equiv: NaN antes={n_nan_income:,} -> imputados a mediana={mediana_income:.2f}"
    )

    # =========================
    # 3) habitat -> asegurar 'object' + imputación a "DESCONOCIDO"
    # =========================
    if "habitat" not in df.columns:
        raise KeyError("No existe la columna habitat en el CSV.")

    # IMPORTANTE: evitar pandas StringDtype (pd.NA). Queremos object + np.nan
    df["habitat"] = df["habitat"].astype(object)

    # Normaliza espacios, y convierte vacíos "" a NaN
    df["habitat"] = df["habitat"].astype(str).str.strip()
    df.loc[df["habitat"].isin(["", "nan", "None", "NaN"]), "habitat"] = np.nan

    n_nan_hab = int(pd.isna(df["habitat"]).sum())
    df["habitat"] = df["habitat"].fillna("DESCONOCIDO")
    print(f"habitat: NaN/vacíos antes={n_nan_hab:,} -> imputados a 'DESCONOCIDO'")

    # =========================
    # 4) Guardar
    # =========================
    df.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"\n✅ Guardado CSV limpio: {OUT_PATH}")
    print(f"Filas guardadas: {len(df):,}")

if __name__ == "__main__":
    main()