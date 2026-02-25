import pandas as pd

INPUT_PATH = "ventas_con_segmentacion_autonomo_forzado.csv"
OUTPUT_PATH = "ventas_con_segmentacion_autonomo_qrk_origen_forzado.csv"

def main():
    # Leer
    df = pd.read_csv(INPUT_PATH, low_memory=False)

    # 1) Rellenar q_rk_score vacío con 0.5
    if "q_rk_score" not in df.columns:
        raise KeyError("No existe la columna 'q_rk_score' en el CSV de entrada.")

    df["q_rk_score"] = pd.to_numeric(df["q_rk_score"], errors="coerce")
    df["q_rk_score"] = df["q_rk_score"].fillna(0.5)

    # 2) Crear origen_sc_o_no: 1 si origen empieza por 'sc' (ignorando mayúsculas/espacios), si no 0
    if "origen" not in df.columns:
        raise KeyError("No existe la columna 'origen' en el CSV de entrada.")

    origen_norm = (
        df["origen"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    df["origen_sc_o_no"] = origen_norm.str.startswith("sc").astype(int)

    # Guardar
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"✅ Guardado: {OUTPUT_PATH}")
    print(f"Filas: {len(df):,}")
    print(f"q_rk_score rellenados con 0.5: {(df['q_rk_score'] == 0.5).sum():,}")
    print(f"origen_sc_o_no = 1: {df['origen_sc_o_no'].sum():,}")

if __name__ == "__main__":
    main()