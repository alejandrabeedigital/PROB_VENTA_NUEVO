import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb

# -------- PATHS --------
archivo = r"C:\Users\AlejandradeSelysGóme\PycharmProjects\PROB_VENTA_NUEVO\ventas_con_segmentacion_forzado_autonomo.csv"
salida_rank = r"C:\Users\AlejandradeSelysGóme\PycharmProjects\PROB_VENTA_NUEVO\clientes_priorizados_rank.csv"

# -------- CARGA --------
df = pd.read_csv(archivo, low_memory=False)

# -------- PREPROCESADO ROBUSTO --------

# cat_contact a numérico SOLO PARA FILTRAR
# - si viene "1"/"2" OK
# - si viene "A1"/"A2"/"A3" extrae 1/2/3
# - si viene "B"/"C" -> NaN (se elimina por filtro)
df["cat_contact"] = pd.to_numeric(
    df["cat_contact"].astype(str).str.extract(r"(\d+)")[0],
    errors="coerce"
)

# Filtrar contactados
df = df[df["cat_contact"] >= 1].copy()

# subsector_entero (6 dígitos: 3 de sector + 3 de subsector)
df["subsector_entero"] = (
    df["co_sector"].astype(str).str.extract(r"(\d+)")[0].fillna("").str.zfill(3) +
    df["co_subsector"].astype(str).str.extract(r"(\d+)")[0].fillna("").str.zfill(3)
)

# edad_empresa (años)
df["fe_creacion_empresa"] = pd.to_datetime(df["fe_creacion_empresa"], errors="coerce")
df["edad_empresa"] = (pd.Timestamp.today() - df["fe_creacion_empresa"]).dt.days / 365.25

# numéricos (por si vienen como texto o con coma decimal)
numeric_cols = [
    "q_rk_score",
    "median_income_equiv",
    "densidad",
    "muni_pob",
    "compe_empr_muni_ssubsec",
    "edad_empresa",
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", ".", regex=False),
            errors="coerce"
        )

# objetivo ganada a 0/1
df["ganada"] = pd.to_numeric(df["ganada"], errors="coerce")
df = df[df["ganada"].isin([0, 1])].copy()
df["ganada"] = df["ganada"].astype(int)

# -------- FEATURES (SIN cat_contact) --------
features = [
    "subsector_entero",
    "ct_merclie",
    "con_local",
    "retail",
    "outcome_forzado_autonomo",
    "q_rk_score",
    "habitat",
    "median_income_equiv",
    "densidad",
    "muni_pob",
    "compe_empr_muni_ssubsec",
    "edad_empresa",
    "excliente",
]
target = "ganada"

faltan = [c for c in features + [target, "co_cliente"] if c not in df.columns]
if faltan:
    raise KeyError(f"Faltan columnas en el dataset: {faltan}")

df_model = df[["co_cliente"] + features + [target]].copy()

# categóricas (LightGBM las maneja mejor como category)
categorical_cols = [
    "subsector_entero",
    "ct_merclie",
    "con_local",
    "retail",
    "outcome_forzado_autonomo",
    "habitat",
    "excliente",
]
for col in categorical_cols:
    df_model[col] = df_model[col].astype("category")

# quitamos filas sin target; en numéricas LightGBM tolera NaN
df_model = df_model.dropna(subset=[target])

X = df_model[features]
y = df_model[target]

# -------- TRAIN/TEST --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------- PESO POSITIVOS (MEJOR PARA DESBALANCE EXTREMO) --------
pos = int((y_train == 1).sum())
neg = int((y_train == 0).sum())
if pos == 0:
    raise ValueError("No hay positivos (ganada=1) en el train. Revisa el filtro o la columna ganada.")
scale_pos_weight = neg / pos

print(f"[Info] Train positives={pos}, negatives={neg}, scale_pos_weight={scale_pos_weight:.2f}")

# -------- MODELO (AJUSTADO PARA RANKING) --------
# Notas:
# - Más árboles + lr bajo suele mejorar ranking estable
# - num_leaves mayor captura interacciones sector/entorno
# - min_child_samples protege contra overfitting con pocos positivos
# - subsample/colsample mejoran generalización
model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=2000,
    learning_rate=0.02,
    num_leaves=127,
    max_depth=-1,
    min_child_samples=80,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.9,
    reg_alpha=0.0,
    reg_lambda=2.0,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train,
    y_train,
    categorical_feature=categorical_cols,
)

# -------- EVALUACIÓN --------
preds_test = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, preds_test)
auprc = average_precision_score(y_test, preds_test)

print("AUC:", round(auc, 4))
print("AUPRC (Average Precision):", round(auprc, 6))

# -------- RANKING GLOBAL --------
df_model["score_priorizacion"] = model.predict_proba(X)[:, 1]
df_model = df_model.sort_values("score_priorizacion", ascending=False)

# -------- MÉTRICAS DE NEGOCIO: TOP X% --------
base_rate = df_model[target].mean()

def tasa_top(pct: float) -> float:
    n = int(len(df_model) * pct)
    if n <= 0:
        return np.nan
    return df_model.head(n)[target].mean()

print("\n--- MÉTRICAS TOP ---")
print("Tasa venta global:", round(base_rate, 6), f"(~{base_rate*1000:.2f} ventas / 1000 llamadas)")
for pct in [0.05, 0.10, 0.20, 0.30, 0.40]:
    tr = tasa_top(pct)
    lift = tr / base_rate if base_rate > 0 else np.nan
    print(f"TOP {int(pct*100)}%: tasa={tr:.6f} | lift={lift:.2f}x | ~{tr*1000:.2f} ventas/1000 llamadas")

# -------- LIFT POR DECILES (1 = TOP) --------
df_lift = df_model[["score_priorizacion", target]].copy()
df_lift["decil"] = pd.qcut(df_lift["score_priorizacion"], 10, labels=False, duplicates="drop")
df_lift["decil"] = df_lift["decil"].max() - df_lift["decil"] + 1  # 1=mejor

lift_table = (
    df_lift.groupby("decil")[target]
    .agg(["count", "mean", "sum"])
    .rename(columns={"mean": "tasa_venta", "sum": "ventas"})
    .sort_index()
)
lift_table["lift_vs_media"] = lift_table["tasa_venta"] / base_rate

print("\n--- LIFT POR DECILES (1 = TOP) ---")
print(lift_table)

# -------- GUARDAR RANKING --------
df_model.to_csv(salida_rank, index=False, encoding="utf-8")
print("\n✅ Ranking generado:", salida_rank)