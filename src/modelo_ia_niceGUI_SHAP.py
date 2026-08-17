"""
SURROGATE MODEL — CALIDAD DEL AIRE | NiceGUI
Versión: 11.4

Cambios principales (11.4):
- FIX: el panel IQCA conserva sus componentes y actualiza explícitamente sus
  propiedades en cada click. Esto evita que `clear()` y la recreación de nodos
  dejen al navegador mostrando la primera versión del panel.

Cambios principales (11.3):
- DIAGNÓSTICO: se revisó a fondo `handle_predict()` porque se reportó que el
  panel de resultados no cambiaba entre clicks. Los inputs (fecha_input,
  hora_input, temp_input, hum_input, vvel_input, vdir_input, precip_input)
  ya se leían con `.value` en vivo dentro del closure de `build_app()`, así
  que NO había un problema de binding/closure real. Se añadió:
    1) Un log de depuración en `log_box` que imprime los valores exactos
       leídos en cada click de "Predecir", para verificar en la propia UI
       que sí cambian entre ejecuciones.
    2) Un mensaje más explícito cuando `SESION["modelo"] is None` (la causa
       más probable de "el resultado no cambia": si no se ha entrenado un
       modelo, `predecir_detalle()` siempre devuelve el mismo texto de aviso,
       sin importar qué inputs se manden).

Cambios principales (11.2):
- FIX: la sección "Predicción IQCA" ahora admite predicciones ILIMITADAS.
  Se introdujo un `result_container` dedicado que se limpia y reconstruye
  en cada click (los inputs y el botón permanecen intactos fuera de él),
  y el handler quedó envuelto en try/except/finally con notificaciones
  de error, para que ningún fallo deje el botón en estado "muerto".

Cambios principales (11.1):
- Dashboard rediseñado: gauge circular IQCA + estado del modelo (izquierda)
  y mapa grande de Quito (derecha).
- Se mantiene la paleta teal original del código (#0f766e, #14b8a6, #12303a).
"""

from __future__ import annotations

import base64
import inspect
import io
import logging
import os
import pickle
import re
import subprocess
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

MPLCONFIG_DIR = Path(__file__).resolve().parent / "matplotlib_cache"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nicegui import events, run, ui
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

try:
    import shap
    SHAP_DISPONIBLE = True
except ImportError:
    shap = None
    SHAP_DISPONIBLE = False

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "resultados_surrogate"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

SESION: dict[str, Any] = {
    "modelo": None,
    "scaler": None,
    "feat_cols": [],
    "target": "",
    "parroquia": "",
    "feat_stats": {},
    "lag_lookup": {},
    "ultimo_timestamp": None,
}

ANOS_PANDEMIA = [2020, 2021]
CONTAMINANTES_Y = ["PM25", "PM10", "O3", "CO", "NO2", "SO2"]
METEO_COLS = ["Temperatura", "Humedad", "Viento_Velocidad", "Viento_Direccion", "Precipitacion"]
MAX_FEATURES_SEL = 40

DEFAULT_ALGORITHM = "XGBoost  (auto GPU/CPU)"
DEFAULT_TRAIN_RATIO = 0.80
DEFAULT_K_SPLITS = 5
DEFAULT_PLOT_DAYS = 7
DEFAULT_EXCLUIR_PANDEMIA = True
DEFAULT_USAR_FEATURE_ENGINEERING = True
DEFAULT_XGB_PARAMS = {
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 2.0,
    "reg_alpha": 0.5,
    "gamma": 0.0,
    "min_child_weight": 1,
}

COLOR_REAL = "#0F766E"
COLOR_PRED = "#F59E0B"
COLOR_POS = "#16A34A"
COLOR_NEG = "#DC2626"
COLOR_SEC = "#2563EB"
BG_PLOT = "#FFFFFF"
TEXT_PLOT = "#12303A"
GRID_PLOT = "#D8E4E7"
SURF_PLOT = "#F5FAFA"


# ---------------- BACKEND (idéntico a v11.0) ---------------- #

def detectar_gpu() -> tuple[bool, str]:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=8,
        )
        if r.returncode == 0 and r.stdout.strip():
            return True, f"GPU detectada (nvidia-smi): {r.stdout.strip().split(chr(10))[0]}"
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return True, f"GPU detectada (torch.cuda): {torch.cuda.get_device_name(0)}"
    except ImportError:
        pass
    try:
        import cupy  # noqa: F401
        return True, "GPU detectada (cupy disponible)"
    except ImportError:
        pass
    return False, "No se detectó GPU — se usará CPU"


GPU_DISPONIBLE, GPU_MSG = detectar_gpu()
log.info(GPU_MSG)


def _xgb_tree_method() -> dict[str, str]:
    return {"tree_method": "hist", "device": "cuda"} if GPU_DISPONIBLE else {"tree_method": "hist", "device": "cpu"}


def _safe_model_basename(nombre: str) -> str:
    nombre = (nombre or "surrogate_calidad_aire").strip()
    nombre = re.sub(r"\.pkl$", "", nombre, flags=re.IGNORECASE)
    nombre = re.sub(r"[^\w\-.]+", "_", nombre).strip("._-")
    return nombre or "surrogate_calidad_aire"


def _cargar_csv(ruta: str | Path) -> pd.DataFrame:
    try:
        return pd.read_csv(ruta, comment="#", low_memory=False, on_bad_lines="warn")
    except Exception as e:
        raise RuntimeError(f"Error al leer '{ruta}': {e}") from e


def _detectar_timestamp(df: pd.DataFrame) -> str | None:
    keywords = ("time", "fecha", "date", "hora", "datetime", "timestamp")
    candidatos = [c for c in df.columns if any(k in c.lower() for k in keywords)]
    if candidatos:
        return candidatos[0]
    for c in df.columns:
        try:
            pd.to_datetime(df[c].dropna().astype(str).iloc[:10], infer_datetime_format=True)
            return c
        except Exception:
            continue
    return None


def detectar_targets_csv(ruta: str | Path | None) -> list[str]:
    if not ruta:
        return CONTAMINANTES_Y.copy()
    try:
        df_head = pd.read_csv(ruta, comment="#", nrows=3, low_memory=False)
        disponibles = [c for c in CONTAMINANTES_Y if c in set(df_head.columns)]
        return disponibles or CONTAMINANTES_Y.copy()
    except Exception:
        return CONTAMINANTES_Y.copy()


def resumen_csv(ruta: str | Path | None) -> dict[str, Any]:
    if not ruta:
        return {"ok": False, "message": "Sin archivo cargado", "targets": CONTAMINANTES_Y.copy()}
    try:
        df = pd.read_csv(ruta, comment="#", low_memory=False)
        ts = _detectar_timestamp(df)
        contams = [c for c in CONTAMINANTES_Y if c in df.columns]
        return {
            "ok": True,
            "name": Path(ruta).name,
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "timestamp": ts or "No detectado",
            "contaminantes": contams,
            "targets": contams or CONTAMINANTES_Y.copy(),
        }
    except Exception as e:
        return {"ok": False, "message": str(e), "targets": CONTAMINANTES_Y.copy()}


def _preprocesar(df: pd.DataFrame, target_col: str, timestamp_col: str, excluir_pandemia: bool = True) -> pd.DataFrame:
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], infer_datetime_format=True, errors="coerce")
    df = df.dropna(subset=[timestamp_col]).set_index(timestamp_col).sort_index()
    if excluir_pandemia:
        df = df[~df.index.year.isin(ANOS_PANDEMIA)]
    obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    if obj_cols:
        df[obj_cols] = df[obj_cols].apply(pd.to_numeric, errors="coerce")
    return df.dropna(subset=[target_col])


def _dividir_cronologico(X: pd.DataFrame, y: pd.Series, ratio: float = DEFAULT_TRAIN_RATIO):
    corte = int(len(X) * ratio)
    return X.iloc[:corte], X.iloc[corte:], y.iloc[:corte], y.iloc[corte:]


def _resolver_features_x(df: pd.DataFrame, target_col: str) -> list[str]:
    cols = set(df.select_dtypes(include=[np.number]).columns)
    excluir = set(CONTAMINANTES_Y)
    return [c for c in cols if c not in excluir and c != target_col]


def _agregar_features_avanzadas(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = df.copy()
    contaminantes = [c for c in CONTAMINANTES_Y if c in df.columns]
    for col in contaminantes:
        for window in [3, 6, 12, 24]:
            df[f"{col}_roll_mean_{window}h"] = df[col].rolling(window, min_periods=1).mean()
            if window >= 6:
                df[f"{col}_roll_std_{window}h"] = df[col].rolling(window, min_periods=2).std()
        for lag in [1, 3, 6, 12, 24]:
            df[f"{col}_diff_{lag}h"] = df[col].diff(lag)
    if "Temperatura" in df.columns and "Humedad" in df.columns:
        df["temp_hum"] = df["Temperatura"] * df["Humedad"]
    if "Temperatura" in df.columns and "Viento_Velocidad" in df.columns:
        df["temp_wind"] = df["Temperatura"] * df["Viento_Velocidad"]
    if "Viento_Direccion" in df.columns and "Viento_Velocidad" in df.columns:
        dir_rad = np.radians(df["Viento_Direccion"])
        df["wind_u"] = df["Viento_Velocidad"] * np.cos(dir_rad)
        df["wind_v"] = df["Viento_Velocidad"] * np.sin(dir_rad)
    if hasattr(df.index, "weekday"):
        df["dia_semana"] = df.index.weekday
        df["dia_semana_sin"] = np.sin(2 * np.pi * df["dia_semana"] / 7)
        df["dia_semana_cos"] = np.cos(2 * np.pi * df["dia_semana"] / 7)
        df["es_finde"] = (df["dia_semana"] >= 5).astype(int)
    return df.dropna()


def _seleccionar_top_features(X_tr: pd.DataFrame, y_tr: pd.Series, n_features: int = MAX_FEATURES_SEL, log_fn=None) -> list[str]:
    log_fn = log_fn or log.info
    all_cols = X_tr.columns.tolist()
    if len(all_cols) <= n_features:
        log_fn(f"Todas las {len(all_cols)} features disponibles.")
        return all_cols
    try:
        mask_ok = X_tr.notna().all(axis=1) & y_tr.notna()
        X_ok, y_ok = X_tr[mask_ok].values, y_tr[mask_ok].values
        if len(X_ok) < 50:
            log_fn("Datos insuficientes para selección; usando todas.")
            return all_cols
        rf_sel = RandomForestRegressor(n_estimators=80, max_depth=6, min_samples_leaf=20, n_jobs=-1, random_state=42)
        rf_sel.fit(X_ok, y_ok)
        selector = SelectFromModel(rf_sel, threshold=-np.inf, max_features=n_features, prefit=True)
        selected = [c for c, k in zip(all_cols, selector.get_support()) if k]
        log_fn(f"SelectFromModel: {len(all_cols)} → {len(selected)} features.")
        return selected or all_cols
    except Exception as exc:
        log_fn(f"Selección falló ({exc}); usando todas las features.")
        return all_cols


def _crear_lag_lookup(X_train: pd.DataFrame, feat_cols: list[str]) -> dict[str, dict[tuple[int, int], float]]:
    lookup: dict[str, dict[tuple[int, int], float]] = {}
    if not hasattr(X_train.index, "hour"):
        return lookup
    df_tmp = X_train[feat_cols].copy()
    df_tmp["_hora"] = X_train.index.hour
    df_tmp["_mes"] = X_train.index.month
    for col in feat_cols:
        if col not in df_tmp.columns:
            continue
        grupo = df_tmp.groupby(["_hora", "_mes"])[col].mean().dropna()
        lookup[col] = {(int(h), int(m)): float(v) for (h, m), v in grupo.items()}
    return lookup


IQCA_BREAKPOINTS: dict[str, list[tuple[float, float, int, int]]] = {
    "PM25": [(0.0, 12.0, 0, 50), (12.1, 37.4, 51, 100), (37.5, 55.4, 101, 150), (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 500.4, 301, 500)],
    "PM10": [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150), (255, 354, 151, 200), (355, 424, 201, 300), (425, 604, 301, 500)],
    "O3": [(0, 54, 0, 50), (55, 124, 51, 100), (125, 164, 101, 150), (165, 204, 151, 200), (205, 404, 201, 300), (405, 604, 301, 500)],
    "CO": [(0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150), (12.5, 15.4, 151, 200), (15.5, 30.4, 201, 300), (30.5, 50.4, 301, 500)],
    "NO2": [(0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150), (361, 649, 151, 200), (650, 1249, 201, 300), (1250, 2049, 301, 500)],
    "SO2": [(0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150), (186, 304, 151, 200), (305, 604, 201, 300), (605, 1004, 301, 500)],
}
CATEGORIAS_IQCA = [
    (0, 50, "Deseable", "#10B981"),
    (51, 100, "Aceptable", "#EAB308"),
    (101, 150, "Precaución", "#F97316"),
    (151, 200, "Alerta", "#EF4444"),
    (201, 300, "Alarma", "#A855F7"),
    (301, 500, "Emergencia", "#111827"),
]


def calcular_iqca(contaminante: str, concentracion: float) -> float | None:
    bp_list = IQCA_BREAKPOINTS.get(contaminante)
    if bp_list is None:
        return None
    for c_lo, c_hi, i_lo, i_hi in bp_list:
        if c_lo <= concentracion <= c_hi:
            return i_lo + (concentracion - c_lo) * (i_hi - i_lo) / (c_hi - c_lo)
    if concentracion > bp_list[-1][1]:
        return 500.0
    return 0.0


def categoria_iqca(iqca: float) -> tuple[str, str]:
    for lo, hi, label, color in CATEGORIAS_IQCA:
        if lo <= iqca <= hi:
            return label, color
    return "Emergencia", "#111827"


def _fig_to_data_uri(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _aplicar_estilo_ax(ax, titulo: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(titulo, fontsize=12, fontweight="bold", color=TEXT_PLOT, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_PLOT, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_PLOT, fontsize=10)
    ax.tick_params(colors=TEXT_PLOT, labelsize=9)
    ax.grid(True, linestyle="--", alpha=0.55, color=GRID_PLOT)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_PLOT)


def _fig_prediccion(y_test, y_pred, target_col: str, days: int, parroquia: str = ""):
    df_p = pd.DataFrame({"Real": y_test.values, "Predicho": y_pred}, index=y_test.index)
    df_p = df_p[df_p.index >= df_p.index.max() - pd.Timedelta(days=days)]
    titulo = f"Real vs Predicho — {target_col} · últimos {days} días"
    if parroquia:
        titulo = f"{parroquia} · {titulo}"
    fig, ax = plt.subplots(figsize=(12, 4.4), facecolor=BG_PLOT)
    ax.set_facecolor(BG_PLOT)
    ax.plot(df_p.index, df_p["Real"], label="Real", color=COLOR_REAL, lw=2.0, alpha=0.95)
    ax.plot(df_p.index, df_p["Predicho"], label="Predicho", color=COLOR_PRED, lw=1.8, linestyle="--", alpha=0.9)
    ax.fill_between(df_p.index, df_p["Real"], df_p["Predicho"], alpha=0.12, color=COLOR_PRED)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    plt.xticks(rotation=28, ha="right", color=TEXT_PLOT, fontsize=9)
    plt.yticks(color=TEXT_PLOT, fontsize=9)
    _aplicar_estilo_ax(ax, titulo, "Fecha", target_col)
    ax.legend(framealpha=0.92, facecolor=SURF_PLOT, edgecolor=GRID_PLOT, fontsize=10)
    plt.tight_layout()
    return fig


def _fig_feature_importance(fi_df: pd.DataFrame, target_col: str, parroquia: str = ""):
    n = min(len(fi_df), 25)
    plot_df = fi_df.head(n).copy()
    fig, ax = plt.subplots(figsize=(9, max(4, n * 0.46)), facecolor=BG_PLOT)
    ax.set_facecolor(BG_PLOT)
    colores = [COLOR_POS if v >= 0 else COLOR_NEG for v in plot_df["Importance"]]
    ax.barh(plot_df["Feature"][::-1], plot_df["Importance"][::-1], xerr=plot_df["Std"][::-1], color=colores[::-1], alpha=0.88, ecolor="#94A3B8", capsize=3, height=0.65)
    ax.axvline(0, color="#94A3B8", linewidth=0.9, linestyle="--")
    titulo = f"Feature Importance · {target_col}"
    if parroquia:
        titulo = f"{parroquia} · {titulo}"
    _aplicar_estilo_ax(ax, titulo, "Importancia media (Δ R²)", "")
    plt.tight_layout()
    return fig


def _fig_heatmap(df_full: pd.DataFrame, target_col: str, parroquia: str = ""):
    num_df = df_full.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return None
    corr = num_df.corr(method="pearson")
    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.58), max(7, n * 0.54)), facecolor=BG_PLOT)
    ax.set_facecolor(BG_PLOT)
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    cmap = sns.diverging_palette(220, 25, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, annot=True, fmt=".2f", annot_kws={"size": 7.2, "color": TEXT_PLOT}, linewidths=0.4, linecolor=GRID_PLOT, square=True, ax=ax, cbar_kws={"shrink": 0.72})
    if target_col in corr.columns:
        idx = list(corr.columns).index(target_col)
        ax.add_patch(plt.Rectangle((idx, 0), 1, n, fill=False, edgecolor=COLOR_PRED, lw=2.3, clip_on=False))
        ax.add_patch(plt.Rectangle((0, idx), n, 1, fill=False, edgecolor=COLOR_PRED, lw=2.3, clip_on=False))
    titulo = f"Correlación de Pearson — {target_col}"
    if parroquia:
        titulo = f"{parroquia} · {titulo}"
    ax.set_title(titulo, fontsize=12, fontweight="bold", color=TEXT_PLOT, pad=14)
    ax.tick_params(colors=TEXT_PLOT, labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    ax.collections[0].colorbar.ax.tick_params(colors=TEXT_PLOT, labelsize=8)
    plt.tight_layout()
    return fig


def _fig_curva_aprendizaje(curvas, target_col: str, algoritmo: str, parroquia: str = "", k_splits: int = 5):
    fig, ax = plt.subplots(figsize=(12, 4.6), facecolor=BG_PLOT)
    ax.set_facecolor(BG_PLOT)
    if not curvas or not curvas.get("train"):
        ax.text(0.5, 0.5, "No hay datos de curva.", ha="center", va="center", color=TEXT_PLOT, fontsize=11, transform=ax.transAxes)
        _aplicar_estilo_ax(ax, f"Curva de Aprendizaje — {target_col}", "", "")
        plt.tight_layout()
        return fig
    train_curves = curvas["train"]
    val_curves = curvas.get("val", [])
    metric_name = curvas.get("metric", "Score")
    is_r2 = metric_name == "R²"
    lengths = [len(c) for c in train_curves + (val_curves if val_curves else []) if c]
    if not lengths or min(lengths) < 2:
        ax.text(0.5, 0.5, "Historial demasiado corto.", ha="center", va="center", color=TEXT_PLOT, transform=ax.transAxes)
        _aplicar_estilo_ax(ax, f"Curva de Aprendizaje — {target_col}", "", "")
        plt.tight_layout()
        return fig
    min_len = min(lengths)
    x = np.arange(min_len)
    tr_arr = np.array([c[:min_len] for c in train_curves])
    tr_mean = tr_arr.mean(axis=0)
    tr_std = tr_arr.std(axis=0)
    for c in tr_arr:
        ax.plot(x, c, color=COLOR_REAL, alpha=0.10, lw=0.75)
    ax.plot(x, tr_mean, color=COLOR_REAL, lw=2.2, label=f"Train — {metric_name}")
    ax.fill_between(x, tr_mean - tr_std, tr_mean + tr_std, alpha=0.12, color=COLOR_REAL)
    if val_curves:
        va_arr = np.array([c[:min_len] for c in val_curves])
        va_mean = va_arr.mean(axis=0)
        va_std = va_arr.std(axis=0)
        for c in va_arr:
            ax.plot(x, c, color=COLOR_PRED, alpha=0.10, lw=0.75)
        ax.plot(x, va_mean, color=COLOR_PRED, lw=2.2, linestyle="--", label=f"Validación — {metric_name}")
        ax.fill_between(x, va_mean - va_std, va_mean + va_std, alpha=0.12, color=COLOR_PRED)
        best_iter = int(np.argmax(va_mean) if is_r2 else np.argmin(va_mean))
        best_val = va_mean[best_iter]
        ax.axvline(best_iter, color=COLOR_SEC, lw=1.5, linestyle=":", alpha=0.8, label=f"Mejor iter: {best_iter} ({best_val:.4f})")
        ax.scatter([best_iter], [best_val], color=COLOR_SEC, s=58, zorder=8)
    titulo = f"Curva de Aprendizaje — {target_col} · XGBoost · TSS K={k_splits}"
    if parroquia:
        titulo = f"{parroquia} · {titulo}"
    ylabel = "R² (mayor es mejor)" if is_r2 else f"{metric_name} (menor es mejor)"
    _aplicar_estilo_ax(ax, titulo, "Iteración / Boosting Round", ylabel)
    ax.legend(framealpha=0.92, facecolor=SURF_PLOT, edgecolor=GRID_PLOT, fontsize=9, loc="best")
    plt.tight_layout()
    return fig


def _fig_mae_comparativo(kf_resumen: pd.DataFrame, target_col: str, parroquia: str = ""):
    if kf_resumen.empty:
        return None
    contams = kf_resumen["Contaminante"].tolist()
    mae_m = kf_resumen["MAE_mean"].values
    mae_s = kf_resumen["MAE_std"].values
    rmse_m = kf_resumen["RMSE_mean"].values
    rmse_s = kf_resumen["RMSE_std"].values
    x = np.arange(len(contams))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10.5, 4.8), facecolor=BG_PLOT)
    ax.set_facecolor(BG_PLOT)
    b1 = ax.bar(x - w / 2, mae_m, w, yerr=mae_s, label="MAE", color=COLOR_REAL, alpha=0.86, ecolor="#94A3B8", capsize=4)
    b2 = ax.bar(x + w / 2, rmse_m, w, yerr=rmse_s, label="RMSE", color=COLOR_PRED, alpha=0.86, ecolor="#94A3B8", capsize=4)
    if target_col in contams:
        idx = contams.index(target_col)
        for bar in (b1[idx], b2[idx]):
            bar.set_edgecolor(COLOR_SEC)
            bar.set_linewidth(2.4)
    ax.set_xticks(x)
    ax.set_xticklabels(contams, color=TEXT_PLOT, fontsize=10)
    titulo = "Comparativo MAE / RMSE por Contaminante"
    if parroquia:
        titulo = f"{parroquia} · {titulo}"
    _aplicar_estilo_ax(ax, titulo, "Contaminante", "Error promedio")
    ax.legend(framealpha=0.92, facecolor=SURF_PLOT, edgecolor=GRID_PLOT, fontsize=10)
    plt.tight_layout()
    return fig


def _fig_r2_comparativo(kf_resumen: pd.DataFrame, target_col: str, parroquia: str = ""):
    if kf_resumen.empty:
        return None
    df = kf_resumen.sort_values("R2_mean", ascending=True).reset_index(drop=True)
    contams = df["Contaminante"].tolist()
    r2_m = df["R2_mean"].values
    r2_s = df["R2_std"].values
    colores = ["#10B981" if v >= 0.70 else ("#F59E0B" if v >= 0.50 else "#EF4444") for v in r2_m]
    fig, ax = plt.subplots(figsize=(9.8, max(3.4, len(contams) * 0.66)), facecolor=BG_PLOT)
    ax.set_facecolor(BG_PLOT)
    bars = ax.barh(contams, r2_m, xerr=r2_s, color=colores, alpha=0.90, ecolor="#94A3B8", capsize=4, height=0.58)
    if target_col in contams:
        idx = contams.index(target_col)
        bars[idx].set_edgecolor(COLOR_SEC)
        bars[idx].set_linewidth(2.4)
    for xv, lbl, col, ls in [(0.5, "Mínimo 0.5", "#F59E0B", ":"), (0.7, "Bueno 0.7", "#10B981", "--"), (0.9, "Excelente 0.9", COLOR_SEC, "-.")]:
        ax.axvline(xv, color=col, lw=1.0, linestyle=ls, alpha=0.62, label=lbl)
    for i, (v, s) in enumerate(zip(r2_m, r2_s)):
        ax.text(min(v + 0.01, 1.0), i, f"{v:.3f}±{s:.3f}", va="center", color=TEXT_PLOT, fontsize=8.5)
    ax.set_xlim(0, 1.05)
    titulo = "R² por Contaminante · TimeSeriesSplit"
    if parroquia:
        titulo = f"{parroquia} · {titulo}"
    _aplicar_estilo_ax(ax, titulo, "R²", "")
    ax.legend(framealpha=0.92, facecolor=SURF_PLOT, edgecolor=GRID_PLOT, fontsize=8.5, loc="lower right")
    plt.tight_layout()
    return fig


def _fig_pie_importancia(fi_df: pd.DataFrame, target_col: str, parroquia: str = ""):
    if fi_df.empty:
        return None
    fi_pos = fi_df[fi_df["Importance"] > 0].reset_index(drop=True)
    if fi_pos.empty:
        return None
    top_n = fi_pos.head(10)
    otros = fi_pos.iloc[10:]
    labels = top_n["Feature"].tolist()
    values = top_n["Importance"].tolist()
    if len(otros) > 0 and otros["Importance"].sum() > 0:
        labels.append(f"Otros ({len(otros)})")
        values.append(float(otros["Importance"].sum()))
    palette = ["#0F766E", "#14B8A6", "#F59E0B", "#2563EB", "#7C3AED", "#EC4899", "#22C55E", "#F97316", "#06B6D4", "#64748B", "#84CC16"]
    fig, ax = plt.subplots(figsize=(8.6, 6.7), facecolor=BG_PLOT)
    ax.set_facecolor(BG_PLOT)
    wedges, _, autotexts = ax.pie(values, labels=None, colors=palette[: len(labels)], autopct="%1.1f%%", startangle=140, pctdistance=0.80, wedgeprops={"edgecolor": BG_PLOT, "linewidth": 2.3})
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color("white")
        at.set_fontweight("bold")
    ax.add_patch(plt.Circle((0, 0), 0.55, fc=BG_PLOT))
    ax.text(0, 0.08, target_col, ha="center", va="center", color=TEXT_PLOT, fontsize=14, fontweight="bold")
    ax.text(0, -0.12, "top features", ha="center", va="center", color="#64748B", fontsize=9)
    ax.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=3, framealpha=0.92, facecolor=SURF_PLOT, edgecolor=GRID_PLOT, fontsize=8.2)
    titulo = f"Distribución de Importancia — {target_col}"
    if parroquia:
        titulo = f"{parroquia} · {titulo}"
    ax.set_title(titulo, fontsize=11.5, fontweight="bold", color=TEXT_PLOT, pad=14)
    plt.tight_layout(rect=[0, 0.10, 1, 1])
    return fig


def _generar_explicabilidad_shap(
    modelo: Any,
    X_te_sc: np.ndarray,
    feat_cols: list[str],
    tag: str,
    output_dir: Path,
    n_samples: int = 5000,
    log_fn: Any = None,
) -> dict[str, Any]:
    """Calcula valores SHAP y genera gráficos de interpretabilidad (summary, beeswarm, waterfall).

    Guarda los artefactos en `output_dir` ({tag}_shap_*.png / .csv / .parquet)
    y retorna Data URIs para renderizado directo en la UI NiceGUI.
    """
    t0 = time.time()
    log_fn = log_fn or log.info

    if not SHAP_DISPONIBLE or shap is None:
        raise RuntimeError("La librería 'shap' no está instalada. Instala con `pip install shap`.")

    np.random.seed(42)
    X_df = pd.DataFrame(X_te_sc, columns=feat_cols)
    if len(X_df) > n_samples:
        X_sample = X_df.sample(n=n_samples, random_state=42)
    else:
        X_sample = X_df

    log_fn(f"Calculando SHAP para {len(X_sample):,} muestras... (esto puede tomar unos segundos)")

    # ---- Selección inteligente del Explainer ----
    from sklearn.ensemble import HistGradientBoostingRegressor
    es_hist_gbm = isinstance(modelo, HistGradientBoostingRegressor)

    explainer = None
    if es_hist_gbm:
        # HistGradientBoosting no es compatible con TreeExplainer nativo.
        # Se intenta shap.Explainer (path automático) y luego PermutationExplainer como fallback.
        try:
            explainer = shap.Explainer(modelo, X_sample)
        except Exception as e_hist:
            log.warning("shap.Explainer falló para HistGBM (%s); usando PermutationExplainer.", e_hist)
            try:
                # PermutationExplainer: toma background reducido para no ser demasiado lento
                background = X_sample.sample(min(50, len(X_sample)), random_state=42)
                explainer = shap.PermutationExplainer(modelo.predict, background, max_evals=2 * len(feat_cols) + 1)
            except Exception as e_perm:
                log.warning("PermutationExplainer también falló (%s).", e_perm)
                raise RuntimeError(f"No se pudo construir Explainer SHAP para HistGBM ({e_hist}).") from e_perm
    else:
        # XGBoost / LightGBM: TreeExplainer SIN background explícito = método path-dependent
        # (mucho más rápido que el método interventional que usa el background como masker)
        try:
            explainer = shap.TreeExplainer(modelo)
        except Exception as e_tree:
            raise RuntimeError(f"No se pudo construir TreeExplainer ({e_tree}).") from e_tree

    # ---- Cálculo de valores SHAP ----
    try:
        exp = explainer(X_sample)
    except Exception:
        # Fallback: método legacy shap_values()
        sv = explainer.shap_values(X_sample)
        base_v = getattr(explainer, "expected_value", 0.0)
        if isinstance(base_v, (list, np.ndarray)):
            base_v = base_v[0]
        exp = shap.Explanation(
            values=sv,
            base_values=float(base_v),
            data=X_sample.values,
            feature_names=feat_cols,
        )

    figs_uri: dict[str, str | None] = {}

    # 1. Summary Plot (Bar Chart global feature importance)
    try:
        fig = plt.figure(figsize=(10, 6), facecolor=BG_PLOT)
        ax = plt.gca()
        ax.set_facecolor(BG_PLOT)
        vals_matrix = exp.values if hasattr(exp, "values") else exp
        shap.summary_plot(vals_matrix, X_sample, plot_type="bar", show=False)
        plt.title(f"SHAP Summary Plot (Importancia Global) · {tag}", fontsize=11, fontweight="bold", color=TEXT_PLOT)
        plt.tight_layout()
        summary_path = output_dir / f"{tag}_shap_summary.png"
        fig.savefig(summary_path, dpi=140, bbox_inches="tight")
        figs_uri["shap_summary"] = _fig_to_data_uri(fig)
    except Exception as e_sum:
        log.warning("Error al generar SHAP summary plot: %s", e_sum)
        figs_uri["shap_summary"] = None

    # 2. Beeswarm Plot (Impacto y distribución por variable)
    try:
        fig = plt.figure(figsize=(10, 6), facecolor=BG_PLOT)
        ax = plt.gca()
        ax.set_facecolor(BG_PLOT)
        if hasattr(exp, "values"):
            shap.plots.beeswarm(exp, show=False)
        else:
            shap.summary_plot(exp, X_sample, show=False)
        plt.title(f"SHAP Beeswarm Plot (Distribución de Impacto) · {tag}", fontsize=11, fontweight="bold", color=TEXT_PLOT)
        plt.tight_layout()
        beeswarm_path = output_dir / f"{tag}_shap_beeswarm.png"
        fig.savefig(beeswarm_path, dpi=140, bbox_inches="tight")
        figs_uri["shap_beeswarm"] = _fig_to_data_uri(fig)
    except Exception as e_bee:
        log.warning("Error al generar SHAP beeswarm plot: %s", e_bee)
        figs_uri["shap_beeswarm"] = None

    # 3. Waterfall Plot (Primera predicción de prueba)
    try:
        fig = plt.figure(figsize=(10, 6), facecolor=BG_PLOT)
        ax = plt.gca()
        ax.set_facecolor(BG_PLOT)
        if hasattr(exp, "__getitem__"):
            shap.plots.waterfall(exp[0], show=False)
        else:
            base_v = getattr(explainer, "expected_value", 0.0)
            if isinstance(base_v, (list, np.ndarray)):
                base_v = base_v[0]
            shap.plots.waterfall(
                shap.Explanation(
                    values=exp[0],
                    base_values=float(base_v),
                    data=X_sample.iloc[0].values,
                    feature_names=feat_cols,
                ),
                show=False,
            )
        plt.title(f"SHAP Waterfall (Primera Predicción de Prueba) · {tag}", fontsize=11, fontweight="bold", color=TEXT_PLOT)
        plt.tight_layout()
        waterfall_path = output_dir / f"{tag}_shap_waterfall.png"
        fig.savefig(waterfall_path, dpi=140, bbox_inches="tight")
        figs_uri["shap_waterfall"] = _fig_to_data_uri(fig)
    except Exception as e_wat:
        log.warning("Error al generar SHAP waterfall plot: %s", e_wat)
        figs_uri["shap_waterfall"] = None

    # 4. Guardar valores SHAP en CSV y Parquet
    try:
        vals_matrix = exp.values if hasattr(exp, "values") else exp
        shap_df = pd.DataFrame(vals_matrix, columns=feat_cols, index=X_sample.index)
        csv_path = output_dir / f"{tag}_shap_values.csv"
        shap_df.to_csv(csv_path)

        try:
            parquet_path = output_dir / f"{tag}_shap_values.parquet"
            shap_df.to_parquet(parquet_path)
        except Exception:
            pass
    except Exception as e_save:
        log.warning("Error al guardar valores SHAP en disco: %s", e_save)

    elapsed = time.time() - t0
    log_fn(f"SHAP completado en {elapsed:.2f} segundos.")

    return {"figs": figs_uri, "elapsed": elapsed}


def fig_mapa_monitoreo(df_prep: pd.DataFrame, target_col: str = "PM25"):
    QUITO_LAT, QUITO_LON = -0.180653, -78.467834

    lat_cols = [c for c in df_prep.columns if c.lower() in ("lat", "latitud", "latitude")]
    lon_cols = [c for c in df_prep.columns if c.lower() in ("lon", "lng", "longitud", "longitude")]

    puntos = []

    if lat_cols and lon_cols and target_col in df_prep.columns:
        try:
            grp = df_prep.groupby([lat_cols[0], lon_cols[0]])[target_col].mean().reset_index()
            for _, r in grp.iterrows():
                puntos.append(dict(
                    lat=float(r[lat_cols[0]]),
                    lon=float(r[lon_cols[0]]),
                    val=float(r[target_col]),
                    name="Estación",
                ))
        except Exception:
            puntos = []

    if not puntos:
        estaciones = [
            ("Centro", -0.220, -78.510),
            ("Cotocollao", -0.106, -78.497),
            ("Belisario", -0.180, -78.490),
            ("El Camal", -0.250, -78.520),
            ("Los Chillos", -0.310, -78.450),
            ("Tumbaco", -0.210, -78.400),
        ]

        media = float(df_prep[target_col].mean()) if target_col in df_prep.columns else 25.0
        rng = np.random.default_rng(42)

        for name, lat, lon in estaciones:
            val = float(media * rng.uniform(0.5, 1.6))
            puntos.append(dict(lat=lat, lon=lon, val=val, name=name))

    def _color_iqca(val):
        iq = calcular_iqca(target_col, val)
        if iq is None:
            return "#6B7280"
        _, col = categoria_iqca(iq)
        return col

    lats = [p["lat"] for p in puntos]
    lons = [p["lon"] for p in puntos]
    vals = [p["val"] for p in puntos]
    names = [p["name"] for p in puntos]
    cols = [_color_iqca(v) for v in vals]
    iqcas = [calcular_iqca(target_col, v) or 0 for v in vals]

    fig = go.Figure(go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode="markers",
        marker=dict(size=18, color=cols, opacity=0.85),
        text=[f"<b>{n}</b><br>{target_col}: {v:.1f}<br>IQCA: {iq:.0f}" for n, v, iq in zip(names, vals, iqcas)],
        hoverinfo="text",
    ))

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_center=dict(lat=QUITO_LAT, lon=QUITO_LON),
        mapbox_zoom=10,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#FFFFFF",
        height=None,
        autosize=True,
    )

    return fig


def _extraer_curvas_fold(modelo, algoritmo: str) -> tuple[list, list, str]:
    train_hist, val_hist, metric_name = [], [], "Score"
    try:
        if "XGBoost" in algoritmo:
            evals = modelo.evals_result()
            ks = list(evals.keys())
            met = list(evals[ks[0]].keys())[0]
            metric_name = met.upper()
            train_hist = list(evals[ks[0]][met])
            val_hist = list(evals[ks[1]][met]) if len(ks) > 1 else []
        else:
            if hasattr(modelo, "train_score_") and modelo.train_score_ is not None:
                train_hist = list(modelo.train_score_)
                metric_name = "R²"
            if hasattr(modelo, "validation_score_") and modelo.validation_score_ is not None:
                val_hist = list(modelo.validation_score_)
    except Exception as exc:
        log.warning("_extraer_curvas_fold: %s", exc)
    return train_hist, val_hist, metric_name


def _construir_modelo(algoritmo: str = DEFAULT_ALGORITHM, params: dict[str, Any] | None = None):
    params = params or DEFAULT_XGB_PARAMS
    if "XGBoost" in algoritmo:
        import xgboost as xgb
        return xgb.XGBRegressor(
            n_estimators=2000,
            max_depth=int(params.get("max_depth", 5)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            subsample=float(params.get("subsample", 0.8)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            reg_lambda=float(params.get("reg_lambda", 2.0)),
            reg_alpha=float(params.get("reg_alpha", 0.5)),
            gamma=float(params.get("gamma", 0.0)),
            min_child_weight=float(params.get("min_child_weight", 1)),
            early_stopping_rounds=50,
            eval_metric="rmse",
            random_state=42,
            verbosity=0,
            **_xgb_tree_method(),
        )
    return HistGradientBoostingRegressor(
        max_iter=1000,
        early_stopping=True,
        n_iter_no_change=30,
        validation_fraction=0.1,
        max_depth=int(params.get("max_depth", 5)),
        min_samples_leaf=int(params.get("min_child_weight", 30)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        l2_regularization=float(params.get("reg_lambda", 1.0)),
        random_state=42,
    )


def _entrenar_kfold(df: pd.DataFrame, algoritmo: str, n_splits: int = DEFAULT_K_SPLITS, log_fn=None, xgb_params: dict[str, Any] | None = None):
    log_fn = log_fn or log.info
    tscv = TimeSeriesSplit(n_splits=n_splits)
    filas, all_curves = [], {}
    cols_df = set(df.columns)
    for contaminante in CONTAMINANTES_Y:
        if contaminante not in cols_df:
            log_fn(f"{contaminante} no disponible; omitido.")
            continue
        feat_cols = _resolver_features_x(df, contaminante)
        y_vals = df[contaminante].dropna()
        X_vals = df.loc[y_vals.index, feat_cols]
        mask_ok = X_vals.notna().all(axis=1) & y_vals.notna()
        X_vals, y_vals = X_vals[mask_ok], y_vals[mask_ok]
        if len(X_vals) < n_splits * 20:
            log_fn(f"{contaminante}: {len(X_vals)} filas; omitido.")
            continue
        X_arr, y_arr = X_vals.values, y_vals.values
        fold_metrics, fold_curves_tr, fold_curves_va = [], [], []
        metric_name_cv = "Score"
        for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(X_arr), 1):
            X_tr, X_va = X_arr[tr_idx], X_arr[va_idx]
            y_tr, y_va = y_arr[tr_idx], y_arr[va_idx]
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_va_sc = scaler.transform(X_va)
            modelo = _construir_modelo(algoritmo, xgb_params)
            if "XGBoost" in algoritmo:
                modelo.fit(X_tr_sc, y_tr, eval_set=[(X_tr_sc, y_tr), (X_va_sc, y_va)], verbose=False)
            else:
                modelo.fit(X_tr_sc, y_tr)
            tr_h, va_h, met = _extraer_curvas_fold(modelo, algoritmo)
            if tr_h:
                fold_curves_tr.append(tr_h)
            if va_h:
                fold_curves_va.append(va_h)
            metric_name_cv = met
            y_pred = modelo.predict(X_va_sc)
            fold_metrics.append({"MAE": mean_absolute_error(y_va, y_pred), "RMSE": float(np.sqrt(mean_squared_error(y_va, y_pred))), "R2": r2_score(y_va, y_pred)})
            log_fn(f"{contaminante} | Fold {fold_idx}/{n_splits} → MAE={fold_metrics[-1]['MAE']:.3f} RMSE={fold_metrics[-1]['RMSE']:.3f} R²={fold_metrics[-1]['R2']:.3f}")
        all_curves[contaminante] = {"train": fold_curves_tr, "val": fold_curves_va, "metric": metric_name_cv}
        mf = pd.DataFrame(fold_metrics)
        filas.append({"Contaminante": contaminante, "Features_X": len(feat_cols), "MAE_mean": mf["MAE"].mean(), "MAE_std": mf["MAE"].std(), "RMSE_mean": mf["RMSE"].mean(), "RMSE_std": mf["RMSE"].std(), "R2_mean": mf["R2"].mean(), "R2_std": mf["R2"].std()})
    return (pd.DataFrame(filas) if filas else pd.DataFrame()), all_curves


def entrenar_nicegui(csv_path: str, target_col: str, nombre_modelo: str, usar_shap: bool = False, n_samples_shap: int = 5000) -> dict[str, Any]:
    logs: list[str] = []

    def info(m):
        log.info(m)
        logs.append(f"OK: {m}")

    def warn(m):
        log.warning(m)
        logs.append(f"AVISO: {m}")

    def fail_payload(msg: str) -> dict[str, Any]:
        return {"ok": False, "metricas_md": msg, "logs": "\n".join(logs[-40:]), "figures": {}, "kpis": {"r2": "—", "mae": "—", "estado": "Error"}}

    try:
        if not csv_path:
            return fail_payload("### Error\nSube un archivo CSV antes de entrenar.")

        target_col = (target_col or "PM25").strip()
        nombre_modelo = _safe_model_basename(nombre_modelo)
        parroquia = Path(csv_path).stem.replace("_", " ").title()
        algoritmo = DEFAULT_ALGORITHM
        xgb_params = DEFAULT_XGB_PARAMS.copy()

        info(f"Archivo: {Path(csv_path).name}")
        df = _cargar_csv(csv_path)
        info(f"CSV cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")

        ts_col = _detectar_timestamp(df)
        if ts_col is None:
            return fail_payload("### Error\nNo se encontró columna Timestamp/Date/Fecha.")
        info(f"Timestamp detectado: {ts_col}")

        if target_col not in df.columns:
            return fail_payload(f"### Error\nTarget `{target_col}` no encontrado. Columnas disponibles: `{', '.join(df.columns)}`")

        df_prep = _preprocesar(df, target_col, ts_col, DEFAULT_EXCLUIR_PANDEMIA)
        if DEFAULT_EXCLUIR_PANDEMIA:
            info(f"Pandemia excluida: {ANOS_PANDEMIA}")

        y = df_prep[target_col]
        X = df_prep.drop(columns=[target_col])

        info("Aplicando Feature Engineering avanzado")
        df_full = _agregar_features_avanzadas(pd.concat([X, y], axis=1), target_col)
        y = df_full[target_col]
        X = df_full.drop(columns=[target_col])
        X_tr_sel, _, y_tr_sel, _ = _dividir_cronologico(X, y, DEFAULT_TRAIN_RATIO)
        info(f"Seleccionando top-{MAX_FEATURES_SEL} features entre {X_tr_sel.shape[1]}")
        selected = _seleccionar_top_features(X_tr_sel, y_tr_sel, MAX_FEATURES_SEL, log_fn=info)
        X = X[selected]
        feat_cols = list(X.columns)
        info(f"Features seleccionadas: {len(feat_cols)}")

        n_nulos = int(X.isna().sum().sum())
        if n_nulos:
            warn(f"{n_nulos:,} NaN en features; eliminando filas")
        mask = X.notna().all(axis=1)
        X = X.loc[mask]
        y = y.loc[mask]
        df_clean = pd.concat([X, y], axis=1)
        info(f"Dataset limpio: {df_clean.shape[0]:,} filas")

        info(f"TimeSeriesSplit K={DEFAULT_K_SPLITS} · Algoritmo: XGBoost")
        kfold_logs: list[str] = []

        def kf_log(m):
            log.info(m)
            kfold_logs.append(m)

        kf_resumen, kf_curvas = _entrenar_kfold(df_clean, algoritmo, n_splits=DEFAULT_K_SPLITS, log_fn=kf_log, xgb_params=xgb_params)
        logs.extend(kfold_logs)
        if kf_resumen.empty:
            warn("TSS K-Fold no produjo resultados")

        X_tr, X_te, y_tr, y_te = _dividir_cronologico(X, y, DEFAULT_TRAIN_RATIO)
        sub_corte = int(len(X_tr) * 0.80)
        X_sub_tr = X_tr.iloc[:sub_corte]
        X_val_sub = X_tr.iloc[sub_corte:]
        y_sub_tr = y_tr.iloc[:sub_corte]
        y_val_sub = y_tr.iloc[sub_corte:]

        scaler_final = StandardScaler()
        X_sub_tr_sc = scaler_final.fit_transform(X_sub_tr)
        X_val_sub_sc = scaler_final.transform(X_val_sub)
        X_te_sc = scaler_final.transform(X_te)
        X_tr_sc = np.vstack([X_sub_tr_sc, X_val_sub_sc])
        y_tr_np = y_tr.values

        modelo_final = _construir_modelo(algoritmo, xgb_params)
        modelo_final.fit(X_sub_tr_sc, y_sub_tr.values, eval_set=[(X_val_sub_sc, y_val_sub.values)], verbose=False)

        y_pred_tr = modelo_final.predict(X_tr_sc)
        y_pred_te = modelo_final.predict(X_te_sc)

        def _m(yt, yp):
            return {"MAE": mean_absolute_error(yt, yp), "RMSE": float(np.sqrt(mean_squared_error(yt, yp))), "R2": r2_score(yt, yp)}

        m_tr = _m(y_tr_np, y_pred_tr)
        m_te = _m(y_te.values, y_pred_te)
        gap = m_tr["R2"] - m_te["R2"]
        if gap > 0.15:
            warn(f"Posible overfitting: ΔR² = {gap:.3f}")

        info("Generando tabla de lags climatológicos")
        lag_lookup = _crear_lag_lookup(X_tr, feat_cols)
        ultimo_timestamp = X_tr.index.max() if hasattr(X_tr.index, "max") else None
        info(f"lag_lookup: {len(lag_lookup)} columnas · último dato: {ultimo_timestamp}")

        if not kf_resumen.empty:
            filas_kf = []
            for _, row in kf_resumen.iterrows():
                nivel = "Bueno" if row["R2_mean"] >= 0.7 else ("Aceptable" if row["R2_mean"] >= 0.5 else "Mejorable")
                filas_kf.append(
                    f"| **{row['Contaminante']}** | `{int(row['Features_X'])}` | `{row['MAE_mean']:.3f} ± {row['MAE_std']:.3f}` | `{row['RMSE_mean']:.3f} ± {row['RMSE_std']:.3f}` | `{row['R2_mean']:.3f} ± {row['R2_std']:.3f}` | {nivel} |"
                )
            tabla_kf = "\n".join(filas_kf)
        else:
            tabla_kf = "| — | — | — | — | — | — |"

        metricas_md = f"""
## Surrogate Model — {parroquia}

### TimeSeriesSplit K={DEFAULT_K_SPLITS}

| Contaminante | Features X | MAE | RMSE | R² | Estado |
|:------------:|:----------:|:---:|:----:|:--:|:------:|
{tabla_kf}

### Modelo final — `{target_col}`

| Métrica | Train | Test ciego |
|---------|:-----:|:----------:|
| **MAE** | `{m_tr['MAE']:.4f}` | `{m_te['MAE']:.4f}` |
| **RMSE** | `{m_tr['RMSE']:.4f}` | `{m_te['RMSE']:.4f}` |
| **R²** | `{m_tr['R2']:.4f}` | `{m_te['R2']:.4f}` |
| **Precisión (R² %)** | `{m_tr['R2']*100:.2f}%` | `{m_te['R2']*100:.2f}%` |

{f"**Aviso:** posible overfitting, ΔR² = `{gap:.3f}`" if gap > 0.15 else "Sin señales fuertes de overfitting."}
"""

        perm = permutation_importance(modelo_final, X_te_sc, y_te.values, n_repeats=8, random_state=42, scoring="r2")
        fi_df = pd.DataFrame({"Feature": feat_cols, "Importance": perm.importances_mean, "Std": perm.importances_std}).sort_values("Importance", ascending=False).reset_index(drop=True)
        info("Permutation Importance lista")

        tag = f"{nombre_modelo}_{parroquia.replace(' ', '_')}"
        pkl_path = OUTPUT_DIR / f"{tag}.pkl"
        with open(pkl_path, "wb") as fh:
            pickle.dump({"modelo": modelo_final, "scaler": scaler_final, "features": feat_cols, "target": target_col, "parroquia": parroquia, "kfold_resumen": kf_resumen, "kf_curvas": kf_curvas, "lag_lookup": lag_lookup, "ultimo_timestamp": ultimo_timestamp}, fh)
        fi_df.to_csv(OUTPUT_DIR / f"{tag}_feature_importance.csv", index=False)
        if not kf_resumen.empty:
            kf_resumen.to_csv(OUTPUT_DIR / f"{tag}_kfold_resumen.csv", index=False)
        info(f"Modelo guardado: {pkl_path.name}")

        SESION.update({
            "modelo": modelo_final,
            "scaler": scaler_final,
            "feat_cols": feat_cols,
            "target": target_col,
            "parroquia": parroquia,
            "feat_stats": {col: {"min": float(X[col].min()), "max": float(X[col].max()), "mean": float(X[col].mean())} for col in feat_cols},
            "lag_lookup": lag_lookup,
            "ultimo_timestamp": ultimo_timestamp,
        })

        # Bloque de interpretabilidad SHAP (opcional)
        shap_figs: dict[str, str | None] = {"shap_summary": None, "shap_beeswarm": None, "shap_waterfall": None}
        if usar_shap:
            if not SHAP_DISPONIBLE:
                warn("SHAP no disponible. Instala con pip install shap para activar la interpretabilidad.")
            else:
                try:
                    shap_res = _generar_explicabilidad_shap(
                        modelo_final, X_te_sc, feat_cols, tag, OUTPUT_DIR, n_samples=n_samples_shap, log_fn=info
                    )
                    shap_figs = shap_res["figs"]
                    info("Valores y gráficos de interpretabilidad SHAP generados exitosamente.")
                except Exception as exc_shap:
                    warn(f"Cálculo SHAP no disponible o falló ({exc_shap}); continuando sin SHAP.")

        y_te_series = pd.Series(y_te.values, index=X_te.index, name=target_col)
        fig_pie = _fig_pie_importancia(fi_df, target_col, parroquia)
        figs = {
            "mae": _fig_to_data_uri(_fig_mae_comparativo(kf_resumen, target_col, parroquia)) if not kf_resumen.empty else None,
            "pred": _fig_to_data_uri(_fig_prediccion(y_te_series, y_pred_te, target_col, DEFAULT_PLOT_DAYS, parroquia)),
            "lc": _fig_to_data_uri(_fig_curva_aprendizaje(kf_curvas.get(target_col, {}), target_col, algoritmo, parroquia, DEFAULT_K_SPLITS)),
            "r2": _fig_to_data_uri(_fig_r2_comparativo(kf_resumen, target_col, parroquia)) if not kf_resumen.empty else None,
            "fi": _fig_to_data_uri(_fig_feature_importance(fi_df, target_col, parroquia)),
            "pie": _fig_to_data_uri(fig_pie) if fig_pie is not None else None,
        }
        figs.update(shap_figs)
        hm = _fig_heatmap(pd.concat([X, y], axis=1), target_col, parroquia)
        figs["hm"] = _fig_to_data_uri(hm) if hm is not None else None
        info("Todas las figuras generadas")

        return {
            "ok": True,
            "metricas_md": metricas_md,
            "logs": "\n".join(logs[-80:]),
            "figures": figs,
            "kpis": {"r2": f"{m_te['R2']:.3f}", "mae": f"{m_te['MAE']:.3f}", "estado": "Completado", "model_file": pkl_path.name},
        }
    except ImportError as e:
        logs.append(f"ERROR: {e}")
        return fail_payload(f"### Librería faltante\n```\n{e}\n```")
    except Exception:
        logs.append("ERROR: excepción inesperada")
        return fail_payload(f"### Error\n```\n{traceback.format_exc()}\n```")


def predecir_detalle(fecha_hora, temperatura, humedad, viento_vel, viento_dir, precipitacion) -> dict[str, Any]:
    if SESION["modelo"] is None:
        return {
            "ok": False,
            "markdown": (
                "### Sin modelo entrenado\n\n"
                "No hay ningún modelo cargado en memoria todavía. "
                "Ve a la pestaña **Entrenamiento**, sube un CSV y presiona "
                "**Entrenar Modelo** antes de predecir.\n\n"
                "*(Si ya entrenaste uno y sigues viendo este mensaje, revisa "
                "el `log_box` de la pestaña de Entrenamiento: probablemente "
                "el entrenamiento terminó con error.)*"
            ),
            "categoria": "Sin modelo",
            "iqca": None,
            "color": "#94A3B8",
            "pred": None,
        }
    try:
        if fecha_hora is None or str(fecha_hora).strip() == "":
            return {"ok": False, "markdown": "Ingresa una fecha y hora válida.", "categoria": "Fecha requerida", "iqca": None, "color": "#F59E0B", "pred": None}
        try:
            ts = pd.Timestamp(str(fecha_hora))
        except Exception:
            return {"ok": False, "markdown": f"Formato inválido: `{fecha_hora}`. Usa `YYYY-MM-DD HH:MM`.", "categoria": "Fecha inválida", "iqca": None, "color": "#EF4444", "pred": None}
        hora, mes = ts.hour, ts.month
        ultimo = SESION.get("ultimo_timestamp")
        advertencia = ""
        if ultimo is not None:
            delta_h = (ts - ultimo).total_seconds() / 3600
            if delta_h > 48:
                advertencia = f"\n\nAviso: extrapolación de {delta_h:.0f}h después del último dato ({ultimo.strftime('%Y-%m-%d %H:%M')})."
        ciclicas = {"hora_sin": float(np.sin(2 * np.pi * hora / 24)), "hora_cos": float(np.cos(2 * np.pi * hora / 24)), "mes_sin": float(np.sin(2 * np.pi * mes / 12)), "mes_cos": float(np.cos(2 * np.pi * mes / 12))}
        feat_cols = SESION["feat_cols"]
        feat_stats = SESION["feat_stats"]
        lag_lookup = SESION.get("lag_lookup", {})
        meteo_vals = {"Temperatura": float(temperatura), "Humedad": float(humedad), "Viento_Velocidad": float(viento_vel), "Viento_Direccion": float(viento_dir), "Precipitacion": float(precipitacion)}
        dir_rad = np.radians(float(viento_dir))
        derivadas = {
            "temp_hum": float(temperatura) * float(humedad),
            "temp_wind": float(temperatura) * float(viento_vel),
            "wind_u": float(viento_vel) * np.cos(dir_rad),
            "wind_v": float(viento_vel) * np.sin(dir_rad),
            "dia_semana": float(ts.weekday()),
            "dia_semana_sin": float(np.sin(2 * np.pi * ts.weekday() / 7)),
            "dia_semana_cos": float(np.cos(2 * np.pi * ts.weekday() / 7)),
            "es_finde": float(1 if ts.weekday() >= 5 else 0),
        }
        row: dict[str, float] = {}
        for col in feat_cols:
            if col in meteo_vals:
                row[col] = meteo_vals[col]
            elif col in ciclicas:
                row[col] = ciclicas[col]
            elif col in derivadas:
                row[col] = derivadas[col]
            elif col in lag_lookup and (hora, mes) in lag_lookup[col]:
                row[col] = lag_lookup[col][(hora, mes)]
            else:
                row[col] = feat_stats.get(col, {}).get("mean", 0.0)
        X_input = pd.DataFrame([row])[feat_cols]
        scaler = SESION["scaler"]
        X_sc = scaler.transform(X_input) if scaler else X_input.values
        pred = float(SESION["modelo"].predict(X_sc)[0])
        target = SESION["target"]
        parroquia = SESION["parroquia"]
        iqca_val = calcular_iqca(target, pred)
        if iqca_val is not None:
            cat_label, color = categoria_iqca(iqca_val)
            iqca_line = f"IQCA: `{iqca_val:.1f} / 500` · Categoría: **{cat_label}**"
        else:
            cat_label, color = "N/D", "#64748B"
            iqca_line = f"IQCA no disponible para {target}"
        n_lookup = sum(1 for col in feat_cols if col in lag_lookup and (hora, mes) in lag_lookup.get(col, {}))
        markdown = (
            f"### Predicción IQCA — `{target}` · {parroquia}\n\n"
            f"| Campo | Valor |\n|-------|-------|\n"
            f"| Fecha / Hora | `{ts.strftime('%Y-%m-%d %H:%M')}` |\n"
            f"| {target} estimado | `{pred:.3f}` |\n"
            f"| Resultado | {iqca_line} |\n"
            f"| Lags imputados | `{n_lookup}` desde tabla climatológica |\n"
            f"{advertencia}"
        )
        return {"ok": True, "markdown": markdown, "categoria": cat_label, "iqca": iqca_val, "color": color, "pred": pred}
    except Exception:
        return {"ok": False, "markdown": f"Error:\n```\n{traceback.format_exc()}\n```", "categoria": "Error", "iqca": None, "color": "#EF4444", "pred": None}


# ---------------- CSS ---------------- #
APP_CSS = """
:root {
    --teal: #0f766e;
    --teal-2: #14b8a6;
    --ink: #12303a;
    --muted: #6b7f86;
    --line: #d9e7e8;
    --paper: #ffffff;
    --soft: #f3faf9;
    --amber: #f59e0b;
    --red: #ef4444;
}
body.body--light { background: #eef7f6; color: var(--ink); }
.q-drawer { background: linear-gradient(180deg, #0f766e 0%, #0b5f59 100%) !important; color: white; }
.q-drawer *,
.q-drawer .q-btn__content,
.q-drawer .q-btn__content span,
.q-drawer .q-icon {
    color: #ffffff !important;
    opacity: 1 !important;
}
.brand-mark { width: 42px; height: 42px; border-radius: 14px; background: rgba(255,255,255,.18); display: grid; place-items: center; }
.nav-button { width: 100%; justify-content: flex-start; color: #ffffff !important; border-radius: 14px; padding: 10px 12px; font-weight: 700 !important; }
.nav-button:hover { background: rgba(255,255,255,.18) !important; }
.content-wrap { max-width: 1440px; margin: 0 auto; padding: 22px; }
.hero-card { background: radial-gradient(circle at 10% 20%, rgba(20,184,166,.18), transparent 30%), linear-gradient(135deg, #ffffff 0%, #f4fbfa 100%); border: 1px solid var(--line); border-radius: 26px; box-shadow: 0 22px 60px rgba(15,118,110,.09); }
.card { background: rgba(255,255,255,.92); border: 1px solid var(--line); border-radius: 22px; box-shadow: 0 18px 44px rgba(18,48,58,.07); }
.card-flat { background: #f8fcfb; border: 1px solid var(--line); border-radius: 18px; }
.kpi { background: linear-gradient(180deg, #ffffff 0%, #f7fbfb 100%); border: 1px solid var(--line); border-radius: 20px; box-shadow: 0 12px 30px rgba(18,48,58,.06); }
.kpi-label { color: var(--muted); font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: .08em; }
.kpi-value { color: var(--ink); font-size: 30px; line-height: 1; font-weight: 900; }
.badge { border-radius: 999px; padding: 5px 10px; background: #e6f7f4; color: var(--teal); font-size: 12px; font-weight: 800; }
.badge-muted { border-radius: 999px; padding: 5px 10px; background: #eef3f4; color: #60767d; font-size: 12px; font-weight: 800; }
.upload-box .q-uploader { width: 100%; border: 2px dashed #8edbd1; border-radius: 20px; background: linear-gradient(135deg,#f7fffe,#ffffff); box-shadow: none; }
.upload-box .q-uploader__header { background: linear-gradient(90deg, #0f766e, #14b8a6); }
.upload-box .q-uploader--uploading { animation: uploadPulse 1.15s ease-in-out infinite; }
@keyframes uploadPulse { 0%,100% { box-shadow: 0 0 0 rgba(20,184,166,0); } 50% { box-shadow: 0 0 0 8px rgba(20,184,166,.16); } }
.result-band { height: 14px; border-radius: 999px; background: linear-gradient(90deg,#10B981 0 10%,#EAB308 10% 20%,#F97316 20% 30%,#EF4444 30% 40%,#A855F7 40% 60%,#111827 60% 100%); overflow: hidden; }
.log-box { background: #0d2f2b; color: #dffcf8; border-radius: 16px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; max-height: 210px; overflow: auto; white-space: pre-wrap; }
.plot-img { width: 100%; border-radius: 18px; border: 1px solid var(--line); background: white; }
.q-tab { border-radius: 14px; }

/* Dashboard rediseñado */
.dash-gauge-card {
    background: linear-gradient(180deg, #ffffff 0%, #f7fbfb 100%);
    border: 1px solid var(--line);
    border-radius: 22px;
    box-shadow: 0 18px 44px rgba(18,48,58,.07);
    padding: 22px;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
}
.dash-status-card {
    background: linear-gradient(180deg, #0f766e 0%, #0b5f59 100%);
    color: #ffffff;
    border-radius: 22px;
    padding: 22px;
    box-shadow: 0 18px 44px rgba(15,118,110,.18);
    display: flex;
    flex-direction: column;
    gap: 14px;
}
.dash-map-card {
    background: #ffffff;
    border: 1px solid var(--line);
    border-radius: 22px;
    box-shadow: 0 18px 44px rgba(18,48,58,.07);
    overflow: hidden;
    display: flex;
    flex-direction: column;
    height: 540px;
}
.dash-map-header {
    padding: 18px 22px;
    border-bottom: 1px solid var(--line);
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
}
.dash-legend-dot {
    width: 10px;
    height: 10px;
    border-radius: 999px;
    display: inline-block;
}
.section-label {
    color: var(--teal);
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .12em;
}

/* Upload estilo moderno */
.upload-shell { background: linear-gradient(135deg, #FFFFFF 0%, #F0FDFA 100%); border: 1px solid #D9F99D; border-radius: 20px; padding: 18px; box-shadow: 0 10px 30px rgba(15, 118, 110, 0.08); }
.upload-dropzone { border: 2px dashed #BAE6FD; border-radius: 18px; padding: 22px; background: rgba(255,255,255,0.72); }
.upload-icon { color: #10B981; font-size: 36px; }
.upload-title { color: #1E3A5F; font-size: 18px; font-weight: 800; }
.upload-subtitle { color: #31547A; font-size: 14px; font-weight: 600; }
.upload-help { color: #64748B; font-size: 12px; font-weight: 700; text-align: center; }
.upload-pretty .q-uploader { width: 100%; background: transparent !important; box-shadow: none !important; border: none !important; }
.upload-pretty .q-uploader__header { width: 220px; margin: 12px auto 8px auto; border-radius: 10px !important; background: linear-gradient(135deg, #34D399 0%, #06B6D4 100%) !important; box-shadow: 0 8px 18px rgba(6, 182, 212, 0.25); }
.upload-pretty .q-uploader__title,
.upload-pretty .q-uploader__subtitle,
.upload-pretty .q-uploader__header .q-icon { color: #FFFFFF !important; font-weight: 800 !important; }
.upload-pretty .q-uploader__list { display: none !important; }

/* Card configuración modelo */
.model-shell { background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%); border: 1px solid #E5E7EB; border-radius: 20px; padding: 20px; box-shadow: 0 10px 30px rgba(15, 118, 110, 0.08); }
.model-title { color: #1E3A5F; font-size: 18px; font-weight: 900; }
.train-gradient-btn { background: linear-gradient(135deg, #34D399 0%, #06B6D4 100%) !important; color: #FFFFFF !important; border-radius: 12px !important; font-weight: 900 !important; height: 48px; box-shadow: 0 8px 18px rgba(6, 182, 212, 0.25); }
.train-gradient-btn * { color: #FFFFFF !important; }
"""


def _set_image(img, src: str | None) -> None:
    if src:
        img.set_source(src)
        img.visible = True
    else:
        img.visible = False


def _chip(text: str, color: str = "teal"):
    return ui.label(text).classes("badge" if color == "teal" else "badge-muted")


def _upload_event_filename(e: events.UploadEventArguments) -> str:
    for attr in ("name", "filename"):
        value = getattr(e, attr, None)
        if value:
            return Path(str(value)).name

    args = getattr(e, "args", None)
    if isinstance(args, dict):
        for key in ("name", "filename"):
            value = args.get(key)
            if value:
                return Path(str(value)).name

    file_obj = getattr(e, "file", None)
    if file_obj is not None:
        for attr in ("filename", "name"):
            value = getattr(file_obj, attr, None)
            if value:
                return Path(str(value)).name

    content = getattr(e, "content", None)
    value = getattr(content, "name", None)
    if value:
        return Path(str(value)).name

    return "dataset.csv"


async def _upload_event_bytes(e: events.UploadEventArguments) -> bytes:
    content = getattr(e, "content", None)
    if content is not None:
        data = content.read() if hasattr(content, "read") else content
    else:
        file_obj = getattr(e, "file", None)
        if file_obj is None:
            raise RuntimeError("No se pudo leer el contenido del archivo subido.")
        if hasattr(file_obj, "read"):
            data = file_obj.read()
        elif hasattr(file_obj, "file") and hasattr(file_obj.file, "read"):
            data = file_obj.file.read()
        else:
            raise RuntimeError("No se pudo leer el contenido del archivo subido.")

    if inspect.isawaitable(data):
        data = await data
    if isinstance(data, str):
        data = data.encode("utf-8")
    return data


# ---------------- FRONTEND ---------------- #
def build_app() -> None:
    ui.add_head_html(f"<style>{APP_CSS}</style>")
    ui.colors(primary="#0f766e", secondary="#14b8a6", accent="#f59e0b", positive="#10b981")

    state: dict[str, Any] = {"csv_path": None}

    # Sidebar
    with ui.left_drawer(value=True).props("show-if-above bordered").classes("p-4"):
        with ui.row().classes("items-center gap-3 mb-6"):
            with ui.element("div").classes("brand-mark"):
                ui.icon("air", size="26px")
            with ui.column().classes("gap-0"):
                ui.label("Quito Air ML").classes("text-lg font-black")
                ui.label("NiceGUI dashboard").classes("text-xs opacity-80")

        ui.label("Navegación").classes("text-xs uppercase tracking-widest opacity-70 mb-2")
        nav_items: list[tuple[str, str, str]] = [
            ("dashboard", "dashboard", "Dashboard"),
            ("train", "model_training", "Entrenamiento"),
            ("metrics", "analytics", "Métricas"),
            ("predict", "tips_and_updates", "Predicción"),
        ]
        nav_buttons = []
        for tab_name, icon, label in nav_items:
            nav_buttons.append((tab_name, ui.button(label, icon=icon).props("flat no-caps align=left").classes("nav-button")))

        ui.separator().classes("my-5 opacity-30")
        ui.label("Parámetros internos bloqueados").classes("text-xs uppercase tracking-widest opacity-70")
        ui.label("XGBoost · Feature Engineering activo · TSS K=5").classes("text-sm opacity-90 leading-snug")
        ui.label(GPU_MSG).classes("text-xs opacity-75 mt-4 leading-snug")

    with ui.header().classes("bg-white text-slate-800 border-b border-slate-200"):
        ui.label("Surrogate Model — Calidad del Aire").classes("font-bold")
        ui.space()
        ui.label("Interfaz clara · NiceGUI").classes("text-sm text-slate-500")

    with ui.column().classes("content-wrap w-full gap-5"):
        with ui.element("section").classes("hero-card w-full p-6"):
            with ui.row().classes("items-center justify-between gap-4 w-full"):
                with ui.column().classes("gap-1"):
                    ui.label("Modelo sustituto para calidad del aire").classes("text-3xl font-black text-slate-900")
                    ui.label("Entrena XGBoost con ingeniería de variables activa y evalúa IQCA REMMAQ desde un dashboard limpio.").classes("text-slate-600")
                with ui.row().classes("gap-2"):
                    _chip("XGBoost fijo")
                    _chip("FE activo")
                    _chip("Tema claro")

        with ui.tabs().classes("w-full bg-transparent") as tabs:
            ui.tab("dashboard", label="Dashboard", icon="dashboard")
            ui.tab("train", label="Entrenamiento", icon="model_training")
            ui.tab("metrics", label="Métricas", icon="analytics")
            ui.tab("predict", label="Predicción IQCA", icon="tips_and_updates")

        def go(tab_name: str):
            tabs.set_value(tab_name)

        for tab_name, btn in nav_buttons:
            btn.on("click", lambda _=None, t=tab_name: go(t))

        with ui.tab_panels(tabs, value="dashboard").classes("w-full bg-transparent"):
            # ---- DASHBOARD (rediseñado) ---- #
            with ui.tab_panel("dashboard").classes("p-0"):
                with ui.row().classes("w-full gap-5 items-start flex-nowrap"):
                    # Columna izquierda (320px)
                    with ui.column().classes("w-full lg:w-[320px] gap-5 flex-shrink-0"):
                        # Gauge IQCA promedio
                        with ui.element("div").classes("dash-gauge-card w-full"):
                            ui.label("Promedio IQCA (Actual)").classes("section-label mb-4")
                            with ui.element("div").classes("relative flex items-center justify-center").style("width: 176px; height: 176px;"):
                                avg_iqca_progress = ui.circular_progress(
                                    value=0, min=0, max=500, show_value=False,
                                ).props("size=170px thickness=0.08 color=teal track-color=grey-3")
                                with ui.column().classes("absolute items-center justify-center gap-0 inset-0"):
                                    avg_iqca_val = ui.label("—").classes("text-5xl font-black leading-none").style("color: var(--ink);")
                                    avg_iqca_cat = ui.label("SIN DATOS").classes("text-[10px] font-bold tracking-wider mt-1 uppercase").style("color: var(--teal);")
                            ui.label("Calidad del aire promedio de las estaciones de monitoreo activas.").classes("text-xs mt-5 px-2").style("color: var(--muted);")

                        # Estado del modelo AI
                        with ui.element("div").classes("dash-status-card w-full"):
                            ui.label("Estado del Modelo AI").classes("text-[11px] uppercase tracking-widest font-bold").style("color:#a7f3d0;")
                            with ui.row().classes("items-start gap-3"):
                                ui.icon("verified_user", size="24px", color="white")
                                with ui.column().classes("gap-0"):
                                    ui.label("ARCHIVO ACTIVO").classes("text-[9px] tracking-wider").style("color: rgba(255,255,255,.55);")
                                    model_active_file = ui.label("Sin modelo").classes("text-xs font-bold text-white")
                            with ui.row().classes("items-start gap-3"):
                                ui.icon("update", size="24px", color="white")
                                with ui.column().classes("gap-0"):
                                    ui.label("ÚLTIMO ENTRENAMIENTO").classes("text-[9px] tracking-wider").style("color: rgba(255,255,255,.55);")
                                    model_active_date = ui.label("No entrenado").classes("text-xs font-bold text-white")

                    # Columna derecha: mapa grande
                    with ui.element("div").classes("dash-map-card flex-grow"):
                        with ui.element("div").classes("dash-map-header"):
                            with ui.column().classes("gap-0"):
                                ui.label("Mapa de Monitoreo · Quito").classes("text-lg font-black").style("color: var(--ink);")
                                ui.label("Sensores estratégicos en tiempo real").classes("text-xs").style("color: var(--muted);")
                            with ui.row().classes("gap-3 items-center text-xs").style("color: var(--muted);"):
                                for color, label_text in [("#10B981", "Bueno"), ("#EAB308", "Acep"), ("#EF4444", "Alerta")]:
                                    with ui.row().classes("items-center gap-1"):
                                        ui.element("span").classes("dash-legend-dot").style(f"background: {color};")
                                        ui.label(label_text)
                        map_plot = ui.plotly(fig_mapa_monitoreo(pd.DataFrame(), "PM25")).classes("w-full flex-grow").style("min-height: 0;")

            # ---- TRAIN ---- #
            with ui.tab_panel("train").classes("p-0"):
                with ui.grid(columns=2).classes("w-full gap-5"):
                    with ui.element("div").classes("upload-shell"):
                        with ui.element("div").classes("upload-dropzone"):
                            with ui.row().classes("items-center gap-4"):
                                ui.icon("cloud_upload").classes("upload-icon")
                                with ui.column().classes("gap-1"):
                                    ui.label("Sube tu dataset").classes("upload-title")
                                    ui.label("Arrastra y suelta tu archivo CSV aquí").classes("upload-subtitle")
                                    ui.label("o selecciona un archivo").classes("upload-subtitle")

                            with ui.element("div").classes("upload-pretty mt-3"):
                                upload = ui.upload(
                                    auto_upload=True,
                                    multiple=False,
                                    on_upload=lambda e: None,
                                ).props("accept=.csv label='Seleccionar archivo'")

                            ui.label("Formato soportado: CSV · Tamaño máx: 50MB ⓘ").classes("upload-help mt-2")

                        with ui.row().classes("items-center gap-2 mt-3"):
                            upload_spinner = ui.spinner("dots", size="md", color="primary")
                            upload_spinner.visible = False
                            upload_status = ui.label("Esperando archivo").classes("text-slate-600 font-bold")

                        with ui.element("div").classes("card-flat p-4 mt-4 w-full"):
                            csv_name = ui.label("Sin archivo").classes("font-bold text-slate-800")
                            csv_shape = ui.label("Filas: — · Columnas: —").classes("text-slate-600")
                            csv_timestamp = ui.label("Timestamp: —").classes("text-slate-600")
                            with ui.row().classes("gap-2 flex-wrap mt-2") as badges_box:
                                _chip("Sin contaminantes", "muted")

                    with ui.element("div").classes("model-shell"):
                        with ui.row().classes("items-center gap-2 mb-3"):
                            ui.label("Configuración del Modelo").classes("model-title")
                            ui.icon("info", color="blue-grey")

                        with ui.grid(columns=2).classes("w-full gap-4"):
                            target_select = ui.select(
                                CONTAMINANTES_Y,
                                value="PM25",
                                label="Target (contaminante)",
                            ).props("outlined dense").classes("w-full")

                            model_name_input = ui.input(
                                "Nombre del modelo (.pkl)",
                                value="surrogate_calidad_aire.pkl",
                            ).props("outlined dense").classes("w-full")

                        with ui.row().classes("w-full items-center gap-2 mt-3"):
                            shap_checkbox = ui.checkbox(
                                "Calcular SHAP (interpretabilidad)",
                                value=False,
                            ).classes("text-slate-700 font-bold")
                            if SHAP_DISPONIBLE:
                                shap_checkbox.tooltip("Calcula gráficos de interpretabilidad SHAP para el modelo entrenado. Puede añadir unos segundos al entrenamiento.")
                            else:
                                shap_checkbox.disable()
                                shap_checkbox.tooltip("SHAP no disponible. Instala con 'pip install shap' para activar la interpretabilidad.")
                                ui.label("⚠️ SHAP no instalado (pip install shap)").classes("text-xs text-amber-600 font-semibold ml-2")

                        train_btn = ui.button(
                            "Entrenar Modelo",
                            icon="model_training",
                        ).props("unelevated no-caps").classes("train-gradient-btn w-full mt-4")

                        with ui.row().classes("items-center gap-3 mt-4"):
                            train_spinner = ui.spinner("hourglass", size="md", color="primary")
                            train_spinner.visible = False
                            train_status = ui.label("Listo para entrenar").classes("text-slate-600 font-bold")

                with ui.element("div").classes("card p-5 mt-5"):
                    ui.label("Registro de ejecución").classes("text-lg font-black text-slate-900 mb-2")
                    log_box = ui.label("Esperando ejecución…").classes("log-box w-full p-4")

            # ---- METRICS ---- #
            with ui.tab_panel("metrics").classes("p-0"):
                with ui.grid(columns=3).classes("w-full gap-4 mb-5"):
                    with ui.element("div").classes("kpi p-5"):
                        ui.label("R² Score").classes("kpi-label")
                        kpi_r2 = ui.label("—").classes("kpi-value")
                    with ui.element("div").classes("kpi p-5"):
                        ui.label("MAE").classes("kpi-label")
                        kpi_mae = ui.label("—").classes("kpi-value")
                    with ui.element("div").classes("kpi p-5"):
                        ui.label("Estado del sistema").classes("kpi-label")
                        kpi_estado = ui.label("Esperando CSV").classes("kpi-value text-lg")

                with ui.element("div").classes("card p-5"):
                    ui.label("Resumen de métricas").classes("text-xl font-black text-slate-900")
                    metrics_md = ui.markdown("Las métricas aparecerán tras entrenar.").classes("w-full")

                # Sección de figuras estándar del modelo
                with ui.grid(columns=2).classes("w-full gap-5 mt-5"):
                    std_figure_specs = [
                        ("MAE / RMSE", "mae"),
                        ("Real vs Predicho", "pred"),
                        ("Curva de Aprendizaje", "lc"),
                        ("R² comparativo", "r2"),
                        ("Feature Importance", "fi"),
                        ("Donut de importancia", "pie"),
                        ("Correlaciones", "hm"),
                    ]
                    figure_images: dict[str, Any] = {}
                    for title, key in std_figure_specs:
                        with ui.element("div").classes("card p-4"):
                            ui.label(title).classes("font-black text-slate-900 mb-2")
                            img = ui.image().classes("plot-img")
                            img.visible = False
                            figure_images[key] = img

                # ---- Sección de Explicabilidad SHAP ----
                with ui.element("div").classes("card p-5 mt-5 w-full"):
                    with ui.row().classes("items-center gap-3 mb-4"):
                        ui.icon("psychology", color="teal", size="28px")
                        ui.label("Explicabilidad del Modelo (SHAP)").classes("text-xl font-black text-slate-900")
                        if SHAP_DISPONIBLE:
                            ui.label("SHAP disponible ✓").classes("badge")
                        else:
                            ui.label("SHAP no instalado").classes("badge-muted")

                    if not SHAP_DISPONIBLE:
                        ui.label(
                            "⚠️ La librería SHAP no está instalada. Instala con 'pip install shap' y reinicia la aplicación para activar la interpretabilidad."
                        ).classes("text-amber-700 bg-amber-50 border border-amber-200 rounded-xl p-4 w-full text-sm font-semibold")
                    else:
                        ui.label(
                            "Activa el checkbox 'Calcular SHAP' en la pestaña de Entrenamiento para generar los gráficos de interpretabilidad."
                        ).classes("text-slate-500 text-sm mb-4")

                    shap_figure_specs = [
                        ("Importancia Global SHAP (Summary Plot)", "shap_summary"),
                        ("Distribución de Impacto (Beeswarm Plot)", "shap_beeswarm"),
                        ("Descomposición de Predicción Individual (Waterfall Plot)", "shap_waterfall"),
                    ]
                    with ui.grid(columns=3).classes("w-full gap-5 mt-2"):
                        for title, key in shap_figure_specs:
                            with ui.element("div").classes("card-flat p-4"):
                                ui.label(title).classes("font-black text-slate-900 mb-2 text-sm")
                                img = ui.image().classes("plot-img")
                                img.visible = False
                                figure_images[key] = img


            # ---- PREDICT ---- #
            with ui.tab_panel("predict").classes("p-0"):
                with ui.grid(columns=2).classes("w-full gap-5"):
                    with ui.element("div").classes("card p-5"):
                        ui.label("Panel de predicción").classes("text-xl font-black text-slate-900")
                        ui.label("Ingresa variables meteorológicas; los lags se imputan automáticamente.").classes("text-slate-500 mb-3")
                        with ui.row().classes("w-full gap-3"):
                            fecha_input = ui.input("Fecha").props("type=date").classes("w-full")
                            hora_input = ui.input("Hora").props("type=time").classes("w-full")
                        temp_input = ui.slider(min=-10, max=40, step=0.1, value=18).props("label label-always").classes("w-full mt-4")
                        ui.label("Temperatura (°C)").classes("text-xs text-slate-500")
                        hum_input = ui.slider(min=0, max=100, step=1, value=70).props("label label-always").classes("w-full mt-4")
                        ui.label("Humedad (%)").classes("text-xs text-slate-500")
                        vvel_input = ui.slider(min=0, max=30, step=0.1, value=2).props("label label-always").classes("w-full mt-4")
                        ui.label("Viento velocidad (m/s)").classes("text-xs text-slate-500")
                        vdir_input = ui.slider(min=0, max=360, step=1, value=180).props("label label-always").classes("w-full mt-4")
                        ui.label("Viento dirección (°)").classes("text-xs text-slate-500")
                        precip_input = ui.slider(min=0, max=200, step=0.1, value=0).props("label label-always").classes("w-full mt-4")
                        ui.label("Precipitación (mm)").classes("text-xs text-slate-500")
                        pred_btn = ui.button("Estimar calidad del aire + IQCA", icon="bolt").props("unelevated no-caps").classes("mt-5 px-5 py-3 text-white")

                    # Los componentes del resultado se crean una sola vez. En cada
                    # predicción sólo se actualizan sus propiedades; de esta forma no
                    # quedan referencias a elementos eliminados del navegador.
                    with ui.element("div").classes("card p-5"):
                        ui.label("Resultado").classes("text-xl font-black text-slate-900")
                        result_container = ui.column().classes("w-full gap-2")
                        with result_container:
                            result_category = ui.label("Sin predicción").classes(
                                "text-4xl font-black text-slate-700 mt-4"
                            )
                            result_iqca = ui.label("IQCA: — / 500").classes(
                                "text-2xl font-bold text-slate-700"
                            )
                            with ui.element("div").classes("result-band w-full my-4"):
                                iqca_pointer = ui.element("div").style(
                                    "width: 4px; height: 22px; background: #12303A; "
                                    "margin-left: 0%; transform: translateY(-4px); border-radius: 999px;"
                                )
                            result_md = ui.markdown(
                                "Entrena primero un modelo para activar esta sección."
                            ).classes("w-full")

    # ---------------- HANDLERS ---------------- #
    async def handle_upload(e: events.UploadEventArguments):
        upload_spinner.visible = True
        upload_status.text = "Subiendo y procesando CSV…"
        try:
            filename = _upload_event_filename(e)
            if not filename.lower().endswith(".csv"):
                ui.notify("El archivo debe ser CSV", type="negative")
                return
            safe_name = re.sub(r"[^\w\-.]+", "_", filename)
            dest = UPLOAD_DIR / safe_name
            with open(dest, "wb") as f:
                f.write(await _upload_event_bytes(e))
            state["csv_path"] = str(dest)
            summary = resumen_csv(dest)
            try:
                df_raw = _cargar_csv(dest)
                ts_col = _detectar_timestamp(df_raw)

                if ts_col:
                    df_map = df_raw.copy()
                    df_map[ts_col] = pd.to_datetime(df_map[ts_col], infer_datetime_format=True, errors="coerce")
                    df_map = df_map.dropna(subset=[ts_col]).set_index(ts_col).sort_index()
                else:
                    df_map = df_raw

                target_for_map = target_select.value or "PM25"
                map_plot.update_figure(fig_mapa_monitoreo(df_map, target_for_map))

                # Actualizar gauge IQCA promedio
                if target_for_map in df_raw.columns:
                    mean_val = float(df_raw[target_for_map].dropna().mean())
                    iq = calcular_iqca(target_for_map, mean_val)
                    if iq is not None:
                        avg_iqca_val.text = f"{iq:.0f}"
                        cat_label, color_cat = categoria_iqca(iq)
                        avg_iqca_cat.text = cat_label.upper()
                        avg_iqca_cat.style(f"color: {color_cat};")
                        avg_iqca_progress.set_value(iq)
            except Exception:
                map_plot.update_figure(fig_mapa_monitoreo(pd.DataFrame(), "PM25"))
            targets = summary.get("targets", CONTAMINANTES_Y.copy())
            target_select.options = targets
            target_select.value = targets[0]
            target_select.update()
            if summary.get("ok"):
                csv_name.text = summary["name"]
                csv_shape.text = f"Filas: {summary['rows']:,} · Columnas: {summary['cols']}"
                csv_timestamp.text = f"Timestamp: {summary['timestamp']}"
                badges_box.clear()
                with badges_box:
                    for c in summary.get("contaminantes", []):
                        _chip(c)
                    if not summary.get("contaminantes"):
                        _chip("Sin contaminantes", "muted")
                upload_status.text = "CSV cargado ✅"
                kpi_estado.text = "CSV listo"
                ui.notify("CSV cargado", type="positive")
            else:
                upload_status.text = f"Error: {summary.get('message')}"
                ui.notify("No se pudo leer el CSV", type="negative")
        finally:
            upload_spinner.visible = False

    upload.on_upload(handle_upload)

    async def handle_train():
        if not state.get("csv_path"):
            ui.notify("Sube un CSV antes de entrenar", type="warning")
            return
        train_btn.disable()
        train_spinner.visible = True
        train_status.text = "Procesando modelo… ⏳"
        kpi_estado.text = "Entrenando"
        log_box.text = "Ejecutando pipeline…"
        try:
            usar_shap_val = shap_checkbox.value if SHAP_DISPONIBLE else False
            result = await run.io_bound(entrenar_nicegui, state["csv_path"], target_select.value, model_name_input.value, usar_shap_val)

            # Actualizar la UI — guardado en try/except para soportar reconexiones largas:
            # si el navegador se desconectó durante el cálculo (ej. SHAP de larga duración),
            # NiceGUI pierde el slot/client. Registramos la advertencia pero no rompemos la app.
            try:
                metrics_md.content = result["metricas_md"]
                log_box.text = result.get("logs") or "Sin logs"
                kpi_r2.text = result["kpis"].get("r2", "—")
                kpi_mae.text = result["kpis"].get("mae", "—")
                kpi_estado.text = result["kpis"].get("estado", "—")
                train_status.text = "Modelo listo 🚀" if result.get("ok") else "Entrenamiento con error"
                for key, img in figure_images.items():
                    _set_image(img, result.get("figures", {}).get(key))

                # Actualizar card "Estado del Modelo AI"
                if result.get("ok"):
                    model_active_file.text = result["kpis"].get("model_file") or (model_name_input.value or "modelo.pkl")
                    model_active_date.text = datetime.now().strftime("%Y-%m-%d %H:%M")

                ui.notify("Entrenamiento completado" if result.get("ok") else "Revisa los errores", type="positive" if result.get("ok") else "negative")
                tabs.set_value("metrics")
            except RuntimeError as e_ui:
                # El cliente se desconectó durante el entrenamiento largo (normal con SHAP).
                # El modelo y resultados están correctamente guardados en disco.
                log.warning("Cliente desconectado durante entrenamiento; UI no actualizada: %s", e_ui)
        finally:
            try:
                train_spinner.visible = False
                train_btn.enable()
            except RuntimeError:
                pass  # Cliente ya desconectado; se ignora de forma segura

    train_btn.on("click", handle_train)

    def handle_predict():
        """Handler de predicción IQCA — pensado para ejecutarse un número
        ILIMITADO de veces.

        Claves del diseño (verificadas en v11.3):
        1. `pred_btn` se deshabilita al entrar y se vuelve a habilitar en el
           `finally`, evitando dobles clicks mientras se calcula y
           garantizando que SIEMPRE quede operativo, incluso si algo falla.
        2. Los componentes del resultado son persistentes y se actualizan
           explícitamente; no se eliminan ni se recrean nodos del navegador.
        3. Todo el cálculo va envuelto en try/except: si `predecir_detalle`
           o cualquier otro paso lanza una excepción inesperada, se notifica
           al usuario en vez de dejar el click "muerto" silenciosamente.
        4. Los valores de fecha_input/hora_input/temp_input/hum_input/
           vvel_input/vdir_input/precip_input se leen con `.value` DENTRO
           de este handler (no se capturan antes), así que siempre reflejan
           lo último que el usuario ajustó. Se registra un log de estos
           valores en `log_box` para poder verificarlo visualmente si algún
           día vuelve a sospecharse un problema de "valores congelados".
        """
        pred_btn.disable()
        try:
            fecha_hora = f"{fecha_input.value} {hora_input.value}" if fecha_input.value and hora_input.value else ""

            # --- Log de depuración: confirma en la propia UI que los inputs
            # leídos en este click son los actuales, no valores viejos. ---
            debug_line = (
                f"[Predicción] fecha_hora='{fecha_hora}' | "
                f"temp={temp_input.value} | hum={hum_input.value} | "
                f"vviento={vvel_input.value} | dviento={vdir_input.value} | "
                f"precip={precip_input.value}"
            )
            log.info(debug_line)
            log_box.text = debug_line

            result = predecir_detalle(
                fecha_hora,
                temp_input.value,
                hum_input.value,
                vvel_input.value,
                vdir_input.value,
                precip_input.value,
            )

            categoria = result.get("categoria") or "N/D"
            color = result.get("color") or "#94A3B8"
            iqca_value = result.get("iqca")
            iqca = max(0, min(500, float(iqca_value))) if iqca_value is not None else None
            pointer_position = iqca / 5 if iqca is not None else 0

            # Actualizar los mismos elementos y forzar su sincronización con
            # el cliente. Los valores se calculan de nuevo en cada ejecución.
            result_category.set_text(categoria)
            result_category.style(replace=f"color: {color};")
            result_iqca.set_text(
                f"IQCA: {iqca:.1f} / 500" if iqca is not None else "IQCA: — / 500"
            )
            iqca_pointer.style(
                replace=(
                    "width: 4px; height: 22px; background: #12303A; "
                    f"margin-left: {pointer_position:.1f}%; transform: translateY(-4px); "
                    "border-radius: 999px;"
                )
            )
            result_md.set_content(result["markdown"])

            result_category.update()
            result_iqca.update()
            iqca_pointer.update()
            result_md.update()

            if not result.get("ok"):
                ui.notify("Revisa los datos ingresados", type="warning")
            else:
                ui.notify("Predicción calculada", type="positive")
        except Exception:
            log.warning("Error en handle_predict: %s", traceback.format_exc())
            ui.notify("Ocurrió un error al calcular la predicción. Intenta nuevamente.", type="negative")
        finally:
            pred_btn.enable()

    pred_btn.on("click", handle_predict)


build_app()

if __name__ in {"__main__", "__mp_main__"}:
    port = int(os.environ.get("PORT", "3000"))
    ui.run(host="0.0.0.0", port=port, title="Quito Air ML — NiceGUI", reload=False)
