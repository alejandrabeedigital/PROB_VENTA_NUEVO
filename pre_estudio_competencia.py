import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

ARCHIVO_IN = "ventas_con_segmentacion_autonomo_qrk_origen_forzado.csv"
TARGET = "ganada"
RANDOM_STATE = 42

# Para que vaya rápido: número máximo de filas para el pre-estudio
MAX_FILAS = 120_000   # prueba 80k-150k; si va lento baja a 80k
TEST_SIZE = 0.30

# =========================
# 1) CARGA + FILTROS (igual que tu modelo)
# =========================
df_raw = pd.read_csv(ARCHIVO_IN, low_memory=False)
print(f"Filas al cargar: {len(df_raw):,}")

df_raw[TARGET] = pd.to_numeric(df_raw[TARGET], errors="coerce")
df_raw = df_raw[df_raw[TARGET].isin([0, 1])].copy()
df_raw[TARGET] = df_raw[TARGET].astype(int)
print(f"Filas tras filtrar target 0/1: {len(df_raw):,}")

df_raw["camp_total_descuelgues"] = pd.to_numeric(df_raw["camp_total_descuelgues"], errors="coerce")
df = df_raw[df_raw["camp_total_descuelgues"].fillna(0) > 0].copy()
print(f"Filas tras filtrar camp_total_descuelgues > 0: {len(df):,}")

# Si quieres además filtrar contactados, descomenta:
# df = df[df["cat_contact"].isin(["A1","A2","A3"])].copy()
# print(f"Filas tras filtrar cat_contact A1/A2/A3: {len(df):,}")

# =========================
# 2) LIMPIEZA MINIMA (para no romper imputers)
# =========================
# q_rk_score
df["q_rk_score"] = pd.to_numeric(
    df["q_rk_score"].astype(str).str.replace(",", ".", regex=False),
    errors="coerce"
)

# origen_sc_o_no
df["origen_sc_o_no"] = pd.to_numeric(df.get("origen_sc_o_no"), errors="coerce")

# median_income_equiv
df["median_income_equiv"] = pd.to_numeric(df.get("median_income_equiv"), errors="coerce")

# habitat como string + NaN -> np.nan (lo imputa el pipeline)
df["habitat"] = df.get("habitat").astype("string")

# excliente a string "1"/"0"
df["excliente"] = df["excliente"].map({True: "1", False: "0", "True": "1", "False": "0"}).fillna(df["excliente"].astype(str))

# outcome_forzado_autonomo y ct_merclie (por si vienen raros)
df["outcome_forzado_autonomo"] = df["outcome_forzado_autonomo"].astype("string")
df["ct_merclie"] = df["ct_merclie"].astype("string")

# Competencia -> numéricas
compe_cols = [
    "compe_pop_muni_act","compe_empr_muni_act",
    "compe_pop_muni_ssubsec","compe_empr_muni_ssubsec",
    "compe_pop_prov_act","compe_empr_prov_act",
    "compe_pop_prov_ssubsec","compe_empr_prov_ssubsec"
]
for c in compe_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# =========================
# 3) MUESTRA PARA QUE SEA RÁPIDO
# =========================
if len(df) > MAX_FILAS:
    # muestreo estratificado simple: mezclamos y cortamos
    df = df.sample(n=MAX_FILAS, random_state=RANDOM_STATE)
    print(f"⚡ Usando muestra para pre-estudio: {len(df):,} filas")
else:
    print(f"⚡ Dataset completo (no supera MAX_FILAS): {len(df):,} filas")

# Tasa base
tasa_base = df[TARGET].mean()
print(f"Tasa base (muestra): {tasa_base:.6f}")

# =========================
# 4) DEFINIMOS COMBINACIONES A PROBAR (pocas)
# =========================
base_num = ["q_rk_score", "origen_sc_o_no", "median_income_equiv"]
base_cat = ["ct_merclie", "excliente", "outcome_forzado_autonomo", "habitat"]

combos = [
    ("BASE + compe_empr_muni_ssubsec", ["compe_empr_muni_ssubsec"]),
    ("BASE + compe_empr_prov_act", ["compe_empr_prov_act"]),
    ("BASE + compe_empr_muni_ssubsec + compe_empr_prov_act", ["compe_empr_muni_ssubsec", "compe_empr_prov_act"]),  # ✅ recomendada
    ("BASE + compe_empr_muni_act + compe_empr_prov_act", ["compe_empr_muni_act", "compe_empr_prov_act"]),
    ("BASE + compe_pop_muni_ssubsec + compe_pop_prov_act", ["compe_pop_muni_ssubsec", "compe_pop_prov_act"]),
]

# =========================
# 5) FUNCIÓN ENTRENAR + MEDIR
# =========================
def fit_and_score(df_in: pd.DataFrame, extra_num: list[str], title: str) -> dict:
    features_num = base_num + extra_num
    features_cat = base_cat

    # Quedarnos con columnas necesarias
    need = features_num + features_cat + [TARGET]
    missing = [c for c in need if c not in df_in.columns]
    if missing:
        return {"modelo": title, "error": f"Faltan columnas: {missing}"}

    df_model = df_in[need].copy()

    # OJO: en pre-estudio mejor no dropear por NaN en numéricas (imputer lo arregla)
    # Solo aseguramos que TARGET existe y es 0/1 (ya filtrado)
    df_model = df_model.dropna(subset=[TARGET])

    X = df_model[features_num + features_cat]
    y = df_model[TARGET]

    # Split
    use_stratify = y.nunique() == 2 and y.value_counts().min() >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y if use_stratify else None
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]), features_num),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), features_cat),
        ]
    )

    # SAGA suele ir más rápido con muchas dummies
    clf = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        solver="saga",
        n_jobs=-1
    )

    pipe = Pipeline(steps=[("preprocess", preprocess), ("clf", clf)])
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    auprc = average_precision_score(y_test, proba)

    # Lift TOP 10% (rápido)
    df_eval = pd.DataFrame({"y": y_test.values, "score": proba})
    df_eval = df_eval.sort_values("score", ascending=False).reset_index(drop=True)
    corte = max(1, int(len(df_eval) * 0.10))
    tasa_top10 = df_eval.iloc[:corte]["y"].mean()
    lift_top10 = (tasa_top10 / tasa_base) if tasa_base > 0 else np.nan

    return {
        "modelo": title,
        "auc": auc,
        "auprc": auprc,
        "tasa_top10": tasa_top10,
        "lift_top10": lift_top10,
        "n_test": len(df_eval),
    }

# =========================
# 6) EJECUTAR COMBOS
# =========================
results = []
for title, extra in combos:
    print(f"\n=== Probando: {title} ===")
    out = fit_and_score(df, extra, title)
    results.append(out)
    if "error" in out:
        print("❌", out["error"])
    else:
        print(f"AUC={out['auc']:.4f} | AUPRC={out['auprc']:.6f} | Lift TOP10={out['lift_top10']:.2f}x")

# =========================
# 7) RESUMEN ORDENADO
# =========================
res_df = pd.DataFrame(results)
print("\n==============================")
print("RESUMEN (ordenado por AUPRC)")
print("==============================")
if "error" in res_df.columns:
    # separa errores
    ok = res_df[~res_df["auc"].isna()].copy()
    err = res_df[res_df["auc"].isna()].copy()
else:
    ok = res_df.copy()
    err = pd.DataFrame()

if len(ok) > 0:
    ok = ok.sort_values(["auprc", "auc", "lift_top10"], ascending=False)
    print(ok[["modelo","auc","auprc","lift_top10","tasa_top10","n_test"]].to_string(index=False))

if len(err) > 0:
    print("\nModelos con error:")
    print(err[["modelo","error"]].to_string(index=False))