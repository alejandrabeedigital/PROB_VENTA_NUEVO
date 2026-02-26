import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

ARCHIVO_IN = "ventas_con_segmentacion_autonomo_qrk_origen_forzado_imputado_habitat_ingresos_compe.csv"
ARCHIVO_OUT = "todo_con_resultados_4.csv"
TARGET = "ganada"

# =========================
# 1) CARGA
# =========================
df_raw = pd.read_csv(ARCHIVO_IN, low_memory=False)
print(f"Filas al cargar: {len(df_raw):,}")

df_raw[TARGET] = pd.to_numeric(df_raw[TARGET], errors="coerce")
df_raw = df_raw[df_raw[TARGET].isin([0, 1])].copy()
df_raw[TARGET] = df_raw[TARGET].astype(int)

print(f"Filas tras filtrar target 0/1: {len(df_raw):,}")

# =========================
# 2) FILTRO DESCUEGUES CAMPAÑA
# =========================
df_raw["camp_total_descuelgues"] = pd.to_numeric(df_raw["camp_total_descuelgues"], errors="coerce")
df = df_raw[df_raw["camp_total_descuelgues"].fillna(0) > 0].copy()
print(f"Filas tras filtrar camp_total_descuelgues > 0: {len(df):,}")

# =========================
# 3) FEATURES
# =========================
features_num = [
    "q_rk_score",
    "origen_sc_o_no",
    "median_income_equiv",
    "compe_empr_muni_ssubsec",
    "compe_empr_prov_act",
]
features_cat = ["ct_merclie", "excliente", "outcome_forzado_autonomo", "habitat"]

df["q_rk_score"] = pd.to_numeric(
    df["q_rk_score"].astype(str).str.replace(",", ".", regex=False),
    errors="coerce"
)
df["origen_sc_o_no"] = pd.to_numeric(df["origen_sc_o_no"], errors="coerce")
df["median_income_equiv"] = pd.to_numeric(df["median_income_equiv"], errors="coerce")
df["compe_empr_muni_ssubsec"] = pd.to_numeric(df["compe_empr_muni_ssubsec"], errors="coerce")
df["compe_empr_prov_act"] = pd.to_numeric(df["compe_empr_prov_act"], errors="coerce")

df["habitat"] = df["habitat"].astype("string")

df["excliente"] = df["excliente"].map(
    {True: "1", False: "0", "True": "1", "False": "0"}
).fillna(df["excliente"].astype(str))

# =========================
# 4) DATASET PARA ENTRENAR
# =========================
df_model = df[features_num + features_cat + [TARGET]].copy()
df_model = df_model.dropna(subset=[TARGET])

pos = int(df_model[TARGET].sum())
neg = int((df_model[TARGET] == 0).sum())
tasa_global = pos / (pos + neg) if (pos + neg) > 0 else 0.0

print(f"Filas para modelar: {len(df_model):,}")
print(f"Positivos={pos}, Negativos={neg}, tasa={tasa_global:.6f}")

# =========================
# 5) PIPELINE
# =========================
preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            # CLAVE: escalar numéricas para que LBFGS converja
            ("scaler", StandardScaler(with_mean=False)),
        ]), features_num),
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), features_cat),
    ]
)

clf = LogisticRegression(
    max_iter=5000,          # subimos iteraciones
    class_weight="balanced",
    solver="lbfgs"
)

pipeline = Pipeline(steps=[
    ("preprocess", preprocess),
    ("clf", clf)
])

X = df_model[features_num + features_cat]
y = df_model[TARGET]

use_stratify = y.nunique() == 2 and y.value_counts().min() >= 2

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y if use_stratify else None
)

pipeline.fit(X_train, y_train)

# =========================
# 6) MÉTRICAS
# =========================
proba_test = pipeline.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba_test)
auprc = average_precision_score(y_test, proba_test)

print(f"\nAUC: {auc:.4f}")
print(f"AUPRC: {auprc:.6f}")

# =========================
# 7) RANKING Y LIFT
# =========================
df_eval = X_test.copy()
df_eval["ganada"] = y_test.values
df_eval["score"] = proba_test

df_eval = df_eval.sort_values("score", ascending=False).reset_index(drop=True)

df_eval["percentil"] = np.linspace(0, 1, len(df_eval))
df_eval["decil"] = pd.qcut(df_eval.index, 10, labels=False) + 1

tabla = df_eval.groupby("decil")["ganada"].agg(["count", "mean", "sum"])
tabla = tabla.rename(columns={"mean": "tasa_venta", "sum": "ventas"})
tabla["lift_vs_media"] = (tabla["tasa_venta"] / tasa_global) if tasa_global > 0 else np.nan

print("\n--- MÉTRICAS TOP ---")
for pct in [0.05, 0.10, 0.20, 0.30, 0.40]:
    corte = int(len(df_eval) * pct)
    if corte <= 0:
        continue
    tasa_top = df_eval.iloc[:corte]["ganada"].mean()
    lift = (tasa_top / tasa_global) if tasa_global > 0 else np.nan
    print(f"TOP {int(pct*100)}%: tasa={tasa_top:.6f} | lift={lift:.2f}x")

print("\n--- LIFT POR DECILES (1 = TOP) ---")
print(tabla)

# =========================
# 8) PROBABILIDAD PARA TODOS LOS FILTRADOS
# =========================
X_all = df[features_num + features_cat]
df["prob_venta_modelo"] = pipeline.predict_proba(X_all)[:, 1]

# =========================
# 10) GRÁFICOS
# =========================
plt.figure()
plt.bar(tabla.index.astype(str), tabla["tasa_venta"])
plt.title("Tasa de venta por decil (1 = TOP)")
plt.xlabel("Decil")
plt.ylabel("Tasa de venta")
plt.tight_layout()
plt.show()

ventas_totales = df_eval["ganada"].sum()
df_eval["ventas_acum"] = df_eval["ganada"].cumsum()
df_eval["pct_clientes"] = np.arange(1, len(df_eval) + 1) / len(df_eval)
df_eval["pct_ventas"] = df_eval["ventas_acum"] / ventas_totales if ventas_totales > 0 else 0

plt.figure()
plt.plot(df_eval["pct_clientes"], df_eval["pct_ventas"])
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("Curva acumulada de captación de ventas")
plt.xlabel("% clientes llamados")
plt.ylabel("% ventas captadas")
plt.tight_layout()
plt.show()

# =========================
# 9) GUARDAR CSV FINAL
# =========================
cols_out = [c for c in df.columns if c != "prob_venta_modelo"] + ["prob_venta_modelo"]
df[cols_out].to_csv(ARCHIVO_OUT, index=False, encoding="utf-8")

print(f"\n✅ Guardado: {ARCHIVO_OUT}")
print(f"Filas guardadas: {len(df):,}")