import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
RANKING_FILE = r"C:\Users\AlejandradeSelysGóme\PycharmProjects\PROB_VENTA_NUEVO\clientes_priorizados_rank.csv"
OUT_DIR = r"C:\Users\AlejandradeSelysGóme\PycharmProjects\PROB_VENTA_NUEVO"

# Columnas esperadas en el ranking
COL_SCORE = "score_priorizacion"
COL_Y = "ganada"
COL_DECIL = "decil"

# Variables para "perfil"
COL_AUT = "outcome_forzado_autonomo"
COL_HAB = "habitat"
COL_QRK = "q_rk_score"
COL_RENTA = "median_income_equiv"

# =========================
# HELPERS
# =========================
def savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def ensure_numeric(series):
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")

def make_bins(s, q=5):
    """Crea bins por cuantiles (etiquetas Q1..Qq) evitando errores si hay pocos valores."""
    s = ensure_numeric(s)
    s = s.replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() < 50:
        return pd.Series(["NA"] * len(s), index=s.index)
    # qcut puede fallar si hay muchos empates; duplicates="drop" lo arregla
    try:
        b = pd.qcut(s, q=q, duplicates="drop")
        return b.astype(str)
    except Exception:
        return pd.Series(["NA"] * len(s), index=s.index)

# =========================
# LOAD
# =========================
df = pd.read_csv(RANKING_FILE, low_memory=False)

missing = [c for c in [COL_SCORE, COL_Y] if c not in df.columns]
if missing:
    raise KeyError(f"Faltan columnas en el ranking: {missing}. "
                   f"Necesito al menos {COL_SCORE} y {COL_Y} en {RANKING_FILE}")

df[COL_Y] = pd.to_numeric(df[COL_Y], errors="coerce")
df = df[df[COL_Y].isin([0, 1])].copy()
df[COL_Y] = df[COL_Y].astype(int)

df[COL_SCORE] = pd.to_numeric(df[COL_SCORE], errors="coerce")
df = df.dropna(subset=[COL_SCORE]).copy()

# Si no existe 'decil', lo creamos (1=TOP 10%, 10=bottom 10%)
if COL_DECIL not in df.columns:
    df = df.sort_values(COL_SCORE, ascending=False).copy()
    df[COL_DECIL] = pd.qcut(np.arange(len(df)), q=10, labels=False) + 1

# Asegurar tipos útiles
if COL_AUT in df.columns:
    df[COL_AUT] = df[COL_AUT].astype(str)
if COL_HAB in df.columns:
    df[COL_HAB] = df[COL_HAB].astype(str)
if COL_QRK in df.columns:
    df[COL_QRK] = ensure_numeric(df[COL_QRK])
if COL_RENTA in df.columns:
    df[COL_RENTA] = ensure_numeric(df[COL_RENTA])

# =========================
# 0) PRINT KPIs básicos
# =========================
tasa_global = df[COL_Y].mean()
top10 = df[df[COL_DECIL] == 1]
top5 = df.sort_values(COL_SCORE, ascending=False).head(int(0.05 * len(df)))
print(f"Tasa global: {tasa_global:.6f} (~{tasa_global*1000:.2f} ventas/1000)")
print(f"Top 10%:     {top10[COL_Y].mean():.6f} (~{top10[COL_Y].mean()*1000:.2f} ventas/1000)")
print(f"Top 5%:      {top5[COL_Y].mean():.6f} (~{top5[COL_Y].mean()*1000:.2f} ventas/1000)")

# =========================
# 1) Tasa de venta por decil
# =========================
decil_table = (
    df.groupby(COL_DECIL)
      .agg(tasa_venta=(COL_Y, "mean"), n=(COL_Y, "count"), ventas=(COL_Y, "sum"))
      .sort_index()
)
plt.figure()
plt.bar(decil_table.index.astype(str), decil_table["tasa_venta"])
plt.title("Tasa de venta por decil (1 = Top 10% del ranking)")
plt.xlabel("Decil")
plt.ylabel("Tasa de venta (ganada=1)")
savefig(f"{OUT_DIR}\\grafico_1_tasa_por_decil.png")

# =========================
# 2) Lift por decil
# =========================
decil_table["lift_vs_media"] = decil_table["tasa_venta"] / tasa_global
plt.figure()
plt.bar(decil_table.index.astype(str), decil_table["lift_vs_media"])
plt.title("Lift por decil (tasa_decil / tasa_global)")
plt.xlabel("Decil")
plt.ylabel("Lift vs media")
savefig(f"{OUT_DIR}\\grafico_2_lift_por_decil.png")

# =========================
# 3) Curva acumulada (Cumulative Gains)
#    % ventas capturadas vs % llamadas si llamas en orden del score
# =========================
df_sorted = df.sort_values(COL_SCORE, ascending=False).copy()
df_sorted["cum_calls_pct"] = np.arange(1, len(df_sorted)+1) / len(df_sorted)
total_sales = df_sorted[COL_Y].sum()
df_sorted["cum_sales_pct"] = df_sorted[COL_Y].cumsum() / total_sales if total_sales > 0 else 0.0

plt.figure()
plt.plot(df_sorted["cum_calls_pct"], df_sorted["cum_sales_pct"], label="Modelo")
plt.plot([0, 1], [0, 1], linestyle="--", label="Aleatorio")
plt.title("Curva acumulada: % ventas capturadas vs % llamadas")
plt.xlabel("% de llamadas (ordenadas por score)")
plt.ylabel("% de ventas capturadas")
plt.legend()
savefig(f"{OUT_DIR}\\grafico_3_curva_acumulada.png")

# Métrica rápida: ventas capturadas en top 5% / 10% / 20%
def captured_at(p):
    k = int(p * len(df_sorted))
    if k <= 0:
        return 0.0
    return df_sorted.head(k)[COL_Y].sum() / total_sales if total_sales > 0 else 0.0

cap5 = captured_at(0.05)
cap10 = captured_at(0.10)
cap20 = captured_at(0.20)
print("\n% ventas capturadas llamando al top:")
print(f"Top 5%  -> {cap5:.2%} de las ventas")
print(f"Top 10% -> {cap10:.2%} de las ventas")
print(f"Top 20% -> {cap20:.2%} de las ventas")

# =========================
# 4) “Perfil” TOP10 vs RESTO (visual y entendible)
#    4.1 AUTONOMO (barras)
# =========================
resto = df[df[COL_DECIL] != 1]

# Perfil AUTONOMO
if COL_AUT in df.columns:
    dist_top = top10[COL_AUT].value_counts(normalize=True)
    dist_res = resto[COL_AUT].value_counts(normalize=True)
    prof = pd.DataFrame({"Top10": dist_top, "Resto": dist_res}).fillna(0)

    prof.plot(kind="bar")
    plt.title("Perfil: outcome_forzado_autonomo (Top 10% vs resto)")
    plt.xlabel("Categoría")
    plt.ylabel("Proporción")
    savefig(f"{OUT_DIR}\\grafico_4_perfil_autonomo.png")

# Perfil HABITAT (top categorías)
if COL_HAB in df.columns:
    top_cats = (
        pd.concat([top10[COL_HAB], resto[COL_HAB]])
          .value_counts()
          .head(10)
          .index
    )
    dist_top = top10[top10[COL_HAB].isin(top_cats)][COL_HAB].value_counts(normalize=True)
    dist_res = resto[resto[COL_HAB].isin(top_cats)][COL_HAB].value_counts(normalize=True)
    prof = pd.DataFrame({"Top10": dist_top, "Resto": dist_res}).fillna(0)

    prof.plot(kind="bar")
    plt.title("Perfil: habitat (Top 10% vs resto) — top 10 categorías")
    plt.xlabel("Habitat")
    plt.ylabel("Proporción")
    savefig(f"{OUT_DIR}\\grafico_5_perfil_habitat.png")

# Perfil q_rk_score por bins
if COL_QRK in df.columns:
    df["q_rk_bin"] = make_bins(df[COL_QRK], q=5)
    top10["q_rk_bin"] = df.loc[top10.index, "q_rk_bin"]
    resto["q_rk_bin"] = df.loc[resto.index, "q_rk_bin"]

    dist_top = top10["q_rk_bin"].value_counts(normalize=True).sort_index()
    dist_res = resto["q_rk_bin"].value_counts(normalize=True).sort_index()
    prof = pd.DataFrame({"Top10": dist_top, "Resto": dist_res}).fillna(0)

    prof.plot(kind="bar")
    plt.title("Perfil: q_rk_score (bins por cuantiles) — Top 10% vs resto")
    plt.xlabel("Bin de q_rk_score")
    plt.ylabel("Proporción")
    savefig(f"{OUT_DIR}\\grafico_6_perfil_qrk.png")

# Perfil renta por bins
if COL_RENTA in df.columns:
    df["renta_bin"] = make_bins(df[COL_RENTA], q=5)
    top10["renta_bin"] = df.loc[top10.index, "renta_bin"]
    resto["renta_bin"] = df.loc[resto.index, "renta_bin"]

    dist_top = top10["renta_bin"].value_counts(normalize=True).sort_index()
    dist_res = resto["renta_bin"].value_counts(normalize=True).sort_index()
    prof = pd.DataFrame({"Top10": dist_top, "Resto": dist_res}).fillna(0)

    prof.plot(kind="bar")
    plt.title("Perfil: median_income_equiv (bins por cuantiles) — Top 10% vs resto")
    plt.xlabel("Bin de renta")
    plt.ylabel("Proporción")
    savefig(f"{OUT_DIR}\\grafico_7_perfil_renta.png")

print("\n✅ Gráficos guardados en:")
print(OUT_DIR)
print(" - grafico_1_tasa_por_decil.png")
print(" - grafico_2_lift_por_decil.png")
print(" - grafico_3_curva_acumulada.png")
print(" - grafico_4..7_perfil_*.png (según columnas disponibles)")