import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

ARCHIVO = "ventas_con_segmentacion_autonomo_qrk_origen_forzado_habitat_income.csv"
TARGET = "ganada"

# =========================
# VARIABLES PRESENCIA DIGITAL
# =========================
vars_virtual = [
    "numero_de_paginas",
    "no_responsive",
    "errores_graves_web",
    "num_kw_en_top_10",
    "no_en_1_pagina",
    "ranking_number",
    "total_rating",
    "claim_business",
    "seguidores_facebook",
]

# =========================
# 1) CARGA
# =========================
df = pd.read_csv(ARCHIVO, low_memory=False)
print(f"\nFilas totales: {len(df):,}")

df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df[df[TARGET].isin([0,1])].copy()
df[TARGET] = df[TARGET].astype(int)

# Filtro campaña como tu modelo
df["camp_total_descuelgues"] = pd.to_numeric(df["camp_total_descuelgues"], errors="coerce")
df = df[df["camp_total_descuelgues"].fillna(0) > 0].copy()

print(f"Filas tras filtro campaña: {len(df):,}")

# =========================
# 2) EXISTENCIA + NaN
# =========================
print("\n==============================")
print("      ANÁLISIS DE NaN")
print("==============================\n")

nan_report = []

for v in vars_virtual:
    if v not in df.columns:
        print(f"{v} → NO EXISTE")
        continue

    total_nan = df[v].isna().sum()
    pct_nan = total_nan / len(df) * 100

    print(f"{v}")
    print(f"  NaN: {total_nan:,} ({pct_nan:.2f}%)")

    nan_report.append((v, total_nan, pct_nan))

# =========================
# 3) CONVERSIÓN NUMÉRICA
# =========================
for v in vars_virtual:
    if v in df.columns:
        df[v] = pd.to_numeric(df[v], errors="coerce")

# =========================
# 4) RESUMEN ESTADÍSTICO
# =========================
print("\n==============================")
print("      RESUMEN NUMÉRICO")
print("==============================\n")

for v in vars_virtual:
    if v in df.columns:
        print(f"\n--- {v} ---")
        print(df[v].describe())

# =========================
# 5) CORRELACIONES
# =========================
print("\n==============================")
print("      CORRELACIONES (Pearson)")
print("==============================\n")

df_corr = df[vars_virtual].dropna()

if len(df_corr) > 1000:
    corr_matrix = df_corr.corr()
    print(corr_matrix.round(3))

# =========================
# 6) VIF (COLINEALIDAD)
# =========================
print("\n==============================")
print("      VIF")
print("==============================\n")

df_vif = df[vars_virtual].dropna()

if len(df_vif) > 1000:
    X_vif = sm.add_constant(df_vif)
    vif_data = pd.DataFrame()
    vif_data["variable"] = X_vif.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X_vif.values, i)
        for i in range(X_vif.shape[1])
    ]
    print(vif_data.sort_values("VIF", ascending=False))

# =========================
# 7) PODER PREDICTIVO INDIVIDUAL (AUC)
# =========================
print("\n==============================")
print("      AUC INDIVIDUAL")
print("==============================\n")

for v in vars_virtual:
    if v not in df.columns:
        continue

    df_temp = df[[v, TARGET]].dropna()
    if len(df_temp) < 1000:
        continue

    X = df_temp[[v]]
    y = df_temp[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, proba)

    print(f"{v}: AUC = {auc:.4f}")

# =========================
# 8) MEJORA SOBRE MODELO BASE
# =========================
print("\n==============================")
print("   MEJORA SOBRE BASE (AUC)")
print("==============================\n")

# Modelo base reducido (igual que el tuyo actual)
base_vars = [
    "q_rk_score",
    "origen_sc_o_no",
    "median_income_equiv",
    "compe_empr_muni_ssubsec"
]

df_base = df[base_vars + [TARGET]].dropna()

X_base = df_base[base_vars]
y_base = df_base[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X_base, y_base, test_size=0.3, random_state=42, stratify=y_base
)

pipe_base = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
])

pipe_base.fit(X_train, y_train)
base_auc = roc_auc_score(y_test, pipe_base.predict_proba(X_test)[:,1])

print(f"AUC BASE: {base_auc:.4f}\n")

for v in vars_virtual:
    if v not in df.columns:
        continue

    df_ext = df[base_vars + [v] + [TARGET]].dropna()
    if len(df_ext) < 1000:
        continue

    X = df_ext[base_vars + [v]]
    y = df_ext[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    pipe.fit(X_train, y_train)
    auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:,1])

    print(f"{v}: AUC = {auc:.4f} | Δ vs base = {auc - base_auc:.5f}")