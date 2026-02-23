import pandas as pd
import numpy as np

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

import matplotlib.pyplot as plt


# =========================
# PATHS
# =========================
ARCHIVO = r"C:\Users\AlejandradeSelysGóme\PycharmProjects\PROB_VENTA_NUEVO\ventas_con_segmentacion_forzado_autonomo.csv"

OUT_GAIN = r"C:\Users\AlejandradeSelysGóme\PycharmProjects\PROB_VENTA_NUEVO\importancia_gain_lgbm.csv"
OUT_PERM = r"C:\Users\AlejandradeSelysGóme\PycharmProjects\PROB_VENTA_NUEVO\importancia_permutacion_auc.csv"

OUT_AUT_WIF_CSV = r"C:\Users\AlejandradeSelysGóme\PycharmProjects\PROB_VENTA_NUEVO\impacto_autonomo_whatif.csv"
OUT_AUT_WIF_PNG = r"C:\Users\AlejandradeSelysGóme\PycharmProjects\PROB_VENTA_NUEVO\impacto_autonomo_whatif.png"

OUT_PDP_QRK = r"C:\Users\AlejandradeSelysGóme\PycharmProjects\PROB_VENTA_NUEVO\pdp_q_rk_score.png"
OUT_PDP_EDAD = r"C:\Users\AlejandradeSelysGóme\PycharmProjects\PROB_VENTA_NUEVO\pdp_edad_empresa.png"


# =========================
# CARGA + PREPROCESADO
# =========================
df = pd.read_csv(ARCHIVO, low_memory=False)

# cat_contact SOLO PARA FILTRAR
df["cat_contact"] = pd.to_numeric(
    df["cat_contact"].astype(str).str.extract(r"(\d+)")[0],
    errors="coerce"
)
df = df[df["cat_contact"] >= 1].copy()

# subsector_entero
df["subsector_entero"] = (
    df["co_sector"].astype(str).str.extract(r"(\d+)")[0].fillna("").str.zfill(3) +
    df["co_subsector"].astype(str).str.extract(r"(\d+)")[0].fillna("").str.zfill(3)
)

# edad_empresa
df["fe_creacion_empresa"] = pd.to_datetime(df["fe_creacion_empresa"], errors="coerce")
df["edad_empresa"] = (pd.Timestamp.today() - df["fe_creacion_empresa"]).dt.days / 365.25

# numéricos robustos
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

# target
df["ganada"] = pd.to_numeric(df["ganada"], errors="coerce")
df = df[df["ganada"].isin([0, 1])].copy()
df["ganada"] = df["ganada"].astype(int)

# =========================
# FEATURES (sin cat_contact como feature)
# =========================
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

faltan = [c for c in features + [target] if c not in df.columns]
if faltan:
    raise KeyError(f"Faltan columnas: {faltan}")

df_model = df[features + [target]].copy()

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

X = df_model[features]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# scale_pos_weight
pos = int((y_train == 1).sum())
neg = int((y_train == 0).sum())
if pos == 0:
    raise ValueError("No hay positivos en train.")
scale_pos_weight = neg / pos
print(f"[Info] Train positives={pos}, negatives={neg}, scale_pos_weight={scale_pos_weight:.2f}")


# =========================
# MODELO (el “bueno” de ranking)
# =========================
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

model.fit(X_train, y_train, categorical_feature=categorical_cols)

preds = model.predict_proba(X_test)[:, 1]
print("AUC:", round(roc_auc_score(y_test, preds), 4))
print("AUPRC:", round(average_precision_score(y_test, preds), 6))


# =========================
# 1) IMPORTANCIA GAIN (LightGBM)
# =========================
booster = model.booster_
gain = booster.feature_importance(importance_type="gain")
feat_names = booster.feature_name()

imp_gain = pd.DataFrame({"variable": feat_names, "importance_gain": gain}) \
    .sort_values("importance_gain", ascending=False)

imp_gain.to_csv(OUT_GAIN, index=False, encoding="utf-8")
print("✅ Guardado:", OUT_GAIN)


# =========================
# 2) PERMUTATION IMPORTANCE (AUC, en test)
# =========================
perm = permutation_importance(
    model, X_test, y_test,
    n_repeats=3,
    random_state=42,
    scoring="roc_auc"
)

imp_perm = pd.DataFrame({
    "variable": X_test.columns,
    "perm_importance_mean": perm.importances_mean,
    "perm_importance_std": perm.importances_std
}).sort_values("perm_importance_mean", ascending=False)

imp_perm.to_csv(OUT_PERM, index=False, encoding="utf-8")
print("✅ Guardado:", OUT_PERM)


# =========================
# 3) WHAT-IF AUTÓNOMO (efecto de cambiar solo esa variable)
# =========================
# Tomamos una muestra para hacerlo rápido
n_sample = min(50000, len(X_test))
Xw = X_test.sample(n=n_sample, random_state=42).copy()

# Predicción base
p_base = model.predict_proba(Xw)[:, 1]

# Predicciones con outcome_forzado_autonomo forzado
X_aut = Xw.copy()
X_aut["outcome_forzado_autonomo"] = "AUTONOMO"
X_aut["outcome_forzado_autonomo"] = X_aut["outcome_forzado_autonomo"].astype("category")

X_no = Xw.copy()
X_no["outcome_forzado_autonomo"] = "NO AUTONOMO"
X_no["outcome_forzado_autonomo"] = X_no["outcome_forzado_autonomo"].astype("category")

p_aut = model.predict_proba(X_aut)[:, 1]
p_no = model.predict_proba(X_no)[:, 1]

impact_df = pd.DataFrame({
    "p_base": p_base,
    "p_si_AUTONOMO": p_aut,
    "p_si_NO_AUTONOMO": p_no,
    "delta_AUTONOMO_vs_base": p_aut - p_base,
    "delta_NO_AUTONOMO_vs_base": p_no - p_base,
    "delta_AUTONOMO_vs_NO": p_aut - p_no,
})

resumen = impact_df[[
    "delta_AUTONOMO_vs_base",
    "delta_NO_AUTONOMO_vs_base",
    "delta_AUTONOMO_vs_NO"
]].agg(["mean", "median"])

print("\n--- IMPACTO WHAT-IF outcome_forzado_autonomo ---")
print(resumen)

impact_df.to_csv(OUT_AUT_WIF_CSV, index=False, encoding="utf-8")
print("✅ Guardado:", OUT_AUT_WIF_CSV)

# Plot del delta AUTONOMO vs NO
plt.figure()
impact_df["delta_AUTONOMO_vs_NO"].clip(-0.05, 0.05).plot(kind="hist", bins=60)
plt.title("Impacto what-if: AUTONOMO vs NO AUTONOMO (delta prob, recortado ±0.05)")
plt.xlabel("p(AUTONOMO) - p(NO AUTONOMO)")
plt.tight_layout()
plt.savefig(OUT_AUT_WIF_PNG, dpi=200)
plt.close()
print("✅ Guardado:", OUT_AUT_WIF_PNG)


# =========================
# 4) PDP (cómo cambia la probabilidad promedio con una variable)
# =========================
# Nota: scikit-learn PDP funciona mejor con features numéricas
# y tarda más en modelos grandes; hacemos PDP solo de 2 variables clave.

for var, out_png in [
    ("q_rk_score", OUT_PDP_QRK),
    ("edad_empresa", OUT_PDP_EDAD),
]:
    if var in X_test.columns:
        fig, ax = plt.subplots()
        PartialDependenceDisplay.from_estimator(
            model,
            X_test,
            [var],
            kind="average",
            grid_resolution=30,
            ax=ax
        )
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        print("✅ Guardado:", out_png)

print("\nListo. Revisa CSVs y PNGs generados en la carpeta del proyecto.")