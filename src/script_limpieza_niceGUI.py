"""
================================================================================
  REMMAQ — Dataset ML Builder  |  Interfaz NiceGUI (v2 · Alta Velocidad)
  Autor : Senior Data Engineer
  Uso   : python remmaq_nicegui.py
================================================================================
  Mejoras aplicadas:
    · Carga asíncrona de archivos con barra de progreso real.
    · Procesamiento en segundo plano sin congelar la UI (run.cpu_bound).
    · Campo personalizado para el nombre del dataset de salida.
    · Botón de descarga rediseñado con estilo premium y estado reactivo.
    · Validación de completitud de parroquias en tiempo real.
================================================================================
"""

from __future__ import annotations

import asyncio
import io
import os
import re
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from nicegui import events, run, ui

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

CONTAMINANTES: list[str] = ["PM25", "PM10", "O3", "CO", "NO2", "SO2"]
METEOROLOGICAS: list[str] = [
    "Temperatura", "Humedad", "Viento_Velocidad", "Viento_Direccion", "Precipitacion",
]
VARIABLES_ESPERADAS: list[str] = CONTAMINANTES + METEOROLOGICAS
TOTAL_VARIABLES: int = len(VARIABLES_ESPERADAS)  # 11
MAX_FALTANTES: int = 1

TARGET: str = "PM25"
PANDEMIA_INICIO = "2020-01-01"
PANDEMIA_FIN = "2021-12-31"

# Umbral de gap fijo (24 horas) — NO se expone en la interfaz.
GAP_UMBRAL: int = 24

RANGOS_VALIDOS: dict = {
    "PM25": (0, 400), "PM10": (0, 999), "O3": (0, 500),
    "CO": (0, 50), "NO2": (0, 500), "SO2": (0, 500),
    "Temperatura": (-10, 50), "Humedad": (0, 100),
    "Viento_Velocidad": (0, 50), "Viento_Direccion": (0, 360),
    "Precipitacion": (0, 200),
}

NOMBRES_ARCHIVOS: dict = {
    "PM25": ["PM2.5.xlsx", "PM25.xlsx", "PM25.csv"],
    "PM10": ["PM10.xlsx", "PM10.csv"],
    "O3": ["O3.xlsx", "O3.csv"],
    "CO": ["CO.xlsx", "CO.csv"],
    "NO2": ["NO2.xlsx", "NO2.csv"],
    "SO2": ["SO2.xlsx", "SO2.csv"],
    "Temperatura": ["TMP.xlsx", "Temperatura.xlsx", "Temperatura.csv"],
    "Humedad": ["HUM.xlsx", "Humedad.xlsx", "Humedad.csv"],
    "Viento_Velocidad": ["VEL.xlsx", "Viento_Velocidad.xlsx"],
    "Viento_Direccion": ["DIR.xlsx", "Viento_Direccion.xlsx"],
    "Precipitacion": ["LLU.xlsx", "Precipitacion.xlsx"],
}

# ══════════════════════════════════════════════════════════════════════════════
# LÓGICA DE PROCESAMIENTO (sin cambios)
# ══════════════════════════════════════════════════════════════════════════════

def _leer_archivo(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        for sep in (",", ";", "\t"):
            try:
                df = pd.read_csv(path, sep=sep, low_memory=False)
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue
        raise ValueError(f"No se pudo leer el CSV: {path}")
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Extensión no soportada: {ext}")


def _preparar_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={df.columns[0]: "Fecha"})
    mask_u = df["Fecha"].astype(str).str.contains(
        r"unidad|unit|ug|mg|%|m/s|°|grados", case=False, na=False, regex=True
    )
    df = df[~mask_u].reset_index(drop=True)
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df = df.dropna(subset=["Fecha"]).set_index("Fecha").sort_index()
    df.columns = [str(c).strip() for c in df.columns]
    return df


EXTENSIONES_VALIDAS = (".xlsx", ".xls", ".csv")


def _normalizar_stem(nombre: str) -> str:
    stem = Path(nombre).stem.upper()
    return re.sub(r"[^A-Z0-9]", "", stem)


_NOMBRES_NORMALIZADOS: dict[str, set[str]] = {
    variable: {_normalizar_stem(n) for n in candidatos}
    for variable, candidatos in NOMBRES_ARCHIVOS.items()
}


def buscar_archivos(data_dir: Path) -> dict[str, Path]:
    if not data_dir.exists():
        return {}
    disponibles = [
        p for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONES_VALIDAS
    ]
    encontrados: dict[str, Path] = {}
    for variable, stems_validos in _NOMBRES_NORMALIZADOS.items():
        for p in disponibles:
            if _normalizar_stem(p.name) in stems_validos:
                encontrados[variable] = p
                break
    return encontrados


def obtener_parroquias(archivos: dict[str, Path]) -> list[str]:
    parroquias: set[str] = set()
    for _, ruta in archivos.items():
        try:
            df = _leer_archivo(ruta)
            df = _preparar_df(df)
            for c in df.columns:
                if c.upper() not in ("FECHA", "DATE"):
                    parroquias.add(c.upper())
        except Exception:
            continue
    return sorted(parroquias)


def detectar_parroquias_por_variable(archivos: dict[str, Path]) -> dict[str, list[str]]:
    mapa: dict[str, list[str]] = {}
    for variable, ruta in archivos.items():
        try:
            df = _leer_archivo(ruta)
            df = _preparar_df(df)
        except Exception:
            continue
        for col in df.columns:
            nombre = str(col).strip().upper()
            if nombre in ("FECHA", "DATE") or not nombre:
                continue
            serie = pd.to_numeric(df[col], errors="coerce")
            if serie.notna().sum() == 0:
                continue
            mapa.setdefault(nombre, [])
            if variable not in mapa[nombre]:
                mapa[nombre].append(variable)
    return mapa


def validar_parroquias(parroquias_detectadas: dict) -> tuple[list, list]:
    parroquias_validas: list[dict] = []
    parroquias_excluidas: list[dict] = []
    for parroquia, presentes in parroquias_detectadas.items():
        presentes_unicas = [v for v in VARIABLES_ESPERADAS if v in presentes]
        faltantes = [v for v in VARIABLES_ESPERADAS if v not in presentes_unicas]
        registro = {
            "parroquia": parroquia,
            "presentes": presentes_unicas,
            "faltantes": faltantes,
        }
        if len(faltantes) <= MAX_FALTANTES:
            parroquias_validas.append(registro)
        else:
            parroquias_excluidas.append(registro)
    parroquias_validas.sort(key=lambda r: r["parroquia"])
    parroquias_excluidas.sort(key=lambda r: r["parroquia"])
    return parroquias_validas, parroquias_excluidas


def _extraer_columna(path: Path, variable: str, parroquia: str, log: list) -> pd.Series:
    if not path.exists():
        log.append(f"  ✗  [{variable:<22}] No encontrado: {path.name}")
        return pd.Series(dtype=float, name=variable)
    df = _leer_archivo(path)
    df = _preparar_df(df)
    pu = parroquia.strip().upper()
    col_match = None
    for col in df.columns:
        if col.upper() == pu:
            col_match = col
            break
    if col_match is None:
        candidatos = [c for c in df.columns if pu in c.upper()]
        if candidatos:
            col_match = candidatos[0]
            log.append(f"  ⚠  [{variable:<22}] parcial → '{col_match}'")
    if col_match is None:
        log.append(f"  ✗  [{variable:<22}] '{parroquia}' no encontrada")
        return pd.Series(dtype=float, name=variable)
    serie = pd.to_numeric(df[col_match], errors="coerce")
    serie.name = variable
    n_val = serie.notna().sum()
    rango = f"{serie.index.min().date()} → {serie.index.max().date()}"
    log.append(f"  ✓  [{variable:<22}] {n_val:>8,} valores  |  {rango}")
    return serie


def construir_dataset(
    archivos: dict[str, Path],
    parroquia: str,
    excluir_pandemia: bool,
) -> tuple[pd.DataFrame, str]:
    """Pipeline principal. Usa siempre `GAP_UMBRAL = 24` horas."""
    log: list[str] = []
    SEP = "═" * 62
    log.append(SEP)
    log.append(f"  REMMAQ · Dataset ML — Parroquia: {parroquia.upper()}")
    log.append(
        f"  Target: {TARGET}  |  Gap: {GAP_UMBRAL}h (fijo)  |  Pandemia excluida: {excluir_pandemia}"
    )
    log.append(SEP)
    log.append("")

    # PASO 0
    log.append("▶ PASO 0 · Validación de completitud por parroquia")
    mapa = detectar_parroquias_por_variable(archivos)
    validas, excluidas = validar_parroquias(mapa)
    if validas:
        resumen_val = ", ".join(
            f"{r['parroquia']} ({len(r['presentes'])}/{TOTAL_VARIABLES})" for r in validas
        )
        log.append(f"  ✅ Parroquias válidas (0-1 variables faltantes): {resumen_val}")
    else:
        log.append("  ⚠ No hay parroquias válidas.")
    if excluidas:
        for r in excluidas:
            faltan = ", ".join(r["faltantes"]) if r["faltantes"] else "—"
            log.append(f"  ❌ Parroquia excluida: {r['parroquia']} (faltan: {faltan})")
    else:
        log.append("  ✓ Ninguna parroquia excluida por completitud.")
    log.append("")

    # PASO 1
    log.append("▶ PASO 1 · Cargando archivos")
    series = []
    for var, ruta in archivos.items():
        s = _extraer_columna(ruta, var, parroquia, log)
        if not s.empty:
            series.append(s)
    if not series:
        raise RuntimeError(f"No se encontró '{parroquia}' en ningún archivo.")
    df = pd.concat(series, axis=1, join="outer").sort_index()
    df.index.name = "Timestamp"
    log.append(f"\n  Dataset bruto: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    log.append(f"  Período: {df.index.min()} → {df.index.max()}")

    n_total = len(df)
    etapas: list[tuple[str, int]] = []

    def reg(etapa: str, n_a: int, n_d: int) -> None:
        elim = n_a - n_d
        pct = elim / n_total * 100 if n_total else 0
        etapas.append((etapa, elim))
        ico = "✂" if elim else "✓"
        log.append(f"  {ico}  {etapa:<46}  −{elim:>7,} ({pct:.1f}%)")

    # PASO 2
    log.append("\n▶ PASO 2 · Limpiando errores de sensor")
    df_c = df.copy()
    n_errores = 0
    for col in df_c.columns:
        n0 = df_c[col].notna().sum()
        df_c.loc[df_c[col].isin([-999, -9999, 9999]), col] = np.nan
        rng = RANGOS_VALIDOS.get(col)
        if rng:
            vmin, vmax = rng
            df_c.loc[(df_c[col] < vmin) | (df_c[col] > vmax), col] = np.nan
        n_errores += n0 - df_c[col].notna().sum()
    df = df_c
    log.append(f"  ✓  {n_errores:,} valores anómalos → NaN")

    # PASO 3
    log.append("\n▶ PASO 3 · Filtros de coherencia ML")
    if excluir_pandemia:
        n = len(df)
        mask = (df.index >= PANDEMIA_INICIO) & (df.index <= PANDEMIA_FIN)
        df = df[~mask].copy()
        reg(f"Pandemia ({PANDEMIA_INICIO}→{PANDEMIA_FIN})", n, len(df))
    if TARGET not in df.columns:
        raise RuntimeError(f"Columna '{TARGET}' no encontrada. ¿Subiste PM2.5.xlsx o PM25.xlsx?")
    n = len(df)
    df = df.dropna(subset=[TARGET]).copy()
    reg("R1 — Sin target (PM25=NaN)", n, len(df))
    cols = [c for c in CONTAMINANTES if c in df.columns]
    if len(cols) >= 2:
        n = len(df)
        df = df[df[cols].notna().any(axis=1)].copy()
        reg("R2 — Sin ningún contaminante (falla total)", n, len(df))
    if cols:
        n = len(df)
        silencio = df[cols].isna().all(axis=1)
        cambio = silencio != silencio.shift()
        id_blq = cambio.cumsum()
        df_sil = df[silencio].copy()
        blqs_malos: set = set()
        if not df_sil.empty:
            df_sil["_blq_"] = id_blq[silencio]
            durs = df_sil.groupby("_blq_").apply(
                lambda g: (g.index.max() - g.index.min()).total_seconds() / 3600
            )
            blqs_malos = set(durs[durs > GAP_UMBRAL].index.tolist())
            if blqs_malos:
                log.append(f"\n  Gaps detectados (>{GAP_UMBRAL}h):")
                for bid in sorted(blqs_malos):
                    g2 = df_sil[df_sil["_blq_"] == bid]
                    log.append(f"    • {g2.index.min()} → {g2.index.max()} ({durs[bid]:.1f}h)")
            mascara = silencio & id_blq.isin(blqs_malos)
            df = df[~mascara].copy()
        reg(f"R3 — Gaps >{GAP_UMBRAL}h", n, len(df))

    n_final = len(df)
    ret = n_final / n_total * 100 if n_total else 0
    log.append(f"\n{'─' * 62}")
    log.append(f"  {'Filas originales':<46}  {n_total:>10,}")
    for e, elim in etapas:
        log.append(f"  {e:<46}  −{elim:>9,}")
    log.append(f"{'─' * 62}")
    log.append(f"  {'Filas finales':<46}  {n_final:>10,}  ({ret:.1f}%)")
    log.append(SEP)
    if df.empty:
        raise RuntimeError("El dataset quedó vacío tras el filtrado.")
    assert df[TARGET].isna().sum() == 0, "FALLO: PM25 tiene NaN residuales."

    # PASO 4
    log.append("\n▶ PASO 4 · Generando features ML")
    df["PM25_lag_1h"] = df[TARGET].shift(1)
    df["PM25_lag_3h"] = df[TARGET].shift(3)
    df["PM25_lag_24h"] = df[TARGET].shift(24)
    hour = df.index.hour
    month = df.index.month
    df["hora_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hora_cos"] = np.cos(2 * np.pi * hour / 24)
    df["mes_sin"] = np.sin(2 * np.pi * month / 12)
    df["mes_cos"] = np.cos(2 * np.pi * month / 12)
    log.append("  ✓  +7 columnas (lags PM25 + cíclicas hora/mes)")
    log.append(f"  ✓  Total columnas: {df.shape[1]}")
    log.append(f"\n  Dataset listo — {len(df):,} filas · {df.shape[1]} columnas")
    log.append(SEP + "\n")
    return df, "\n".join(log)


def df_a_csv(df: pd.DataFrame, parroquia: str) -> str:
    meta = [
        "# ═══════════════════════════════════════════════════════════════",
        "# DATASET ML — Red REMMAQ / Quito",
        f"# Parroquia    : {parroquia.upper()}",
        "# Target       : PM25 (µg/m³)",
        f"# Filas        : {len(df):,}",
        f"# Columnas     : {df.shape[1]}  →  {list(df.columns)}",
        f"# Inicio       : {df.index.min()}",
        f"# Fin          : {df.index.max()}",
        f"# Generado     : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "#",
        "# Uso: pd.read_csv('...csv', comment='#', index_col='Timestamp', parse_dates=True)",
        "# ═══════════════════════════════════════════════════════════════",
    ]
    buf = io.StringIO()
    for line in meta:
        buf.write(line + "\n")
    df.to_csv(buf, date_format="%Y-%m-%d %H:%M")
    return buf.getvalue()


def _tabla_archivos(archivos: dict) -> pd.DataFrame:
    rows = []
    for var, _ in NOMBRES_ARCHIVOS.items():
        encontrado = var in archivos
        rows.append({
            "Variable": var,
            "Estado": "Cargado" if encontrado else "No encontrado",
            "Archivo": archivos[var].name if encontrado else "—",
        })
    return pd.DataFrame(rows)


def _perfil_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.select_dtypes(include=[np.number]).columns:
        rows.append({
            "Variable": col,
            "N válidos": int(df[col].notna().sum()),
            "% NaN": round(df[col].isna().mean() * 100, 1),
            "Media": round(df[col].mean(), 3),
            "Mín": round(df[col].min(), 3),
            "Máx": round(df[col].max(), 3),
        })
    return pd.DataFrame(rows)


def _df_to_table_payload(
    df: pd.DataFrame, max_rows: int | None = None
) -> tuple[list[dict], list[dict]]:
    work = df.copy()
    if max_rows is not None:
        work = work.head(max_rows)
    if not isinstance(work.index, pd.RangeIndex):
        work = work.reset_index()
    for col in work.columns:
        if pd.api.types.is_datetime64_any_dtype(work[col]):
            work[col] = work[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    work = work.astype(object).where(pd.notnull(work), None)
    rows = []
    for record in work.to_dict("records"):
        clean = {}
        for k, v in record.items():
            if isinstance(v, pd.Timestamp):
                clean[k] = v.strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(v, (np.integer, np.floating)):
                clean[k] = v.item()
            else:
                clean[k] = v
        rows.append(clean)
    columns = [
        {"name": str(c), "label": str(c), "field": str(c), "align": "left", "sortable": True}
        for c in work.columns
    ]
    return columns, rows


def diagnostico_uploads(tmp_dir: Path | None) -> str:
    lineas: list[str] = []
    SEP = "═" * 62
    lineas.append(SEP)
    lineas.append("  DIAGNÓSTICO · Comprobación de datos subidos")
    lineas.append(SEP)
    if tmp_dir is None or not tmp_dir.exists():
        lineas.append("  ✗ No hay archivos subidos todavía.")
        return "\n".join(lineas)
    archivos_en_disco = sorted(
        [p for p in tmp_dir.iterdir() if p.is_file()],
        key=lambda p: p.name.lower(),
    )
    if not archivos_en_disco:
        lineas.append("  ✗ El directorio de subida está vacío.")
        return "\n".join(lineas)
    lineas.append(f"\n▶ Archivos recibidos ({len(archivos_en_disco)}):")
    for p in archivos_en_disco:
        try:
            size_kb = p.stat().st_size / 1024
        except Exception:
            size_kb = 0
        lineas.append(f"    • {p.name}  ({size_kb:,.1f} KB)")
    lineas.append(
        "\n▶ Reconocimiento contra nombres esperados "
        "(tolerante a mayúsculas/minúsculas y separadores . _ - ):"
    )
    reconocidos: dict[str, Path] = buscar_archivos(tmp_dir)
    for var in NOMBRES_ARCHIVOS.keys():
        if var in reconocidos:
            lineas.append(f"  ✓  {var:<20} → {reconocidos[var].name}")
        else:
            candidatos = ", ".join(NOMBRES_ARCHIVOS[var])
            lineas.append(f"  ✗  {var:<20} — no reconocido (esperado, p.ej.: {candidatos})")
    rutas_reconocidas = set(reconocidos.values())
    no_reconocidos = [p.name for p in archivos_en_disco if p not in rutas_reconocidas]
    if no_reconocidos:
        lineas.append("\n▶ Archivos que NO coinciden con ningún nombre esperado:")
        for nombre in no_reconocidos:
            lineas.append(f"    • {nombre}")
        lineas.append(
            "  ⚠ Renombra estos archivos usando alguno de los nombres esperados "
            "(p.ej. PM2.5.xlsx, TMP.xlsx, HUM.xlsx…)."
        )
    if reconocidos:
        lineas.append("\n▶ Columnas encontradas por archivo reconocido:")
        for var, ruta in reconocidos.items():
            try:
                df = _leer_archivo(ruta)
                df = _preparar_df(df)
                cols = [c for c in df.columns if c.upper() not in ("FECHA", "DATE")]
                lineas.append(f"  • {var} ({ruta.name}) → {len(cols)} columna(s):")
                if cols:
                    lineas.append(f"      {', '.join(cols)}")
                else:
                    lineas.append("      (sin columnas de parroquia detectables)")
            except Exception as exc:
                lineas.append(f"  ✗  {var} ({ruta.name}) — error al leer: {exc}")
        lineas.append("\n▶ Completitud por parroquia (11 variables requeridas):")
        mapa = detectar_parroquias_por_variable(reconocidos)
        if not mapa:
            lineas.append("  ✗ No se detectó ninguna parroquia.")
        else:
            validas, excluidas = validar_parroquias(mapa)
            for r in validas:
                lineas.append(
                    f"  ✓  {r['parroquia']:<20} "
                    f"{len(r['presentes'])}/{TOTAL_VARIABLES}  "
                    f"(faltan: {', '.join(r['faltantes']) if r['faltantes'] else '—'})"
                )
            for r in excluidas:
                lineas.append(
                    f"  ✗  {r['parroquia']:<20} "
                    f"{len(r['presentes'])}/{TOTAL_VARIABLES}  "
                    f"(faltan: {', '.join(r['faltantes'])})"
                )
    else:
        lineas.append("\n  ⚠ Ningún archivo fue reconocido — imposible analizar parroquias.")
    lineas.append("\n" + SEP)
    return "\n".join(lineas)


def _escanear_y_detectar(tmp: Path):
    archivos = buscar_archivos(tmp)
    mapa = detectar_parroquias_por_variable(archivos) if archivos else {}
    return archivos, mapa


def _diagnostico_y_deteccion(tmp: Path):
    reporte = diagnostico_uploads(tmp)
    archivos, mapa = _escanear_y_detectar(tmp)
    return reporte, archivos, mapa


# ══════════════════════════════════════════════════════════════════════════════
# CSS — paleta teal / esmeralda con menú lateral (sin cambios relevantes)
# ══════════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Nunito+Sans:wght@400;600;700;800&display=swap');

:root {
    --sidebar-1: #0E6F66;
    --sidebar-2: #0B5E57;
    --accent-1:  #1F8F7A;
    --accent-2:  #2AAE9B;
    --bg:        #F5F7F7;
    --panel:     #FFFFFF;
    --ink:       #2E2E2E;
    --ink-soft:  #3A3A3A;
    --muted:     #556;
    --line:      #E4EBEA;
    --shadow:    0 14px 40px rgba(14, 111, 102, 0.08);
    --danger:    #b73d39;
    --amber:     #c88a00;
}

* { box-sizing: border-box; }

html, body, #app {
    background: var(--bg) !important;
    color: var(--ink);
    font-family: 'Nunito Sans', sans-serif;
    margin: 0;
    padding: 0;
    width: 100%;
    min-height: 100vh;
    overflow-x: hidden;
}

.q-page {
    min-height: 100vh;
    background: var(--bg) !important;
    padding: 0 !important;
}

.q-page-container {
    padding: 0 !important;
}

.nicegui-content {
    padding: 0 !important;
    gap: 0 !important;
    margin: 0 !important;
    max-width: none !important;
    width: 100% !important;
    min-height: 100vh;
    display: block !important;
}

/* ── Layout global ────────────────────────────────────────────────────── */
.app-frame {
    width: 100%;
    min-height: 100vh;
    display: grid;
    grid-template-columns: 280px minmax(0, 1fr);
    gap: 0;
    background: var(--bg);
}

/* ── Sidebar ──────────────────────────────────────────────────────────── */
.side-panel {
    background: linear-gradient(180deg, var(--sidebar-1) 0%, var(--sidebar-2) 100%);
    color: #ffffff;
    min-height: 100vh;
    padding: 30px 22px 26px;
    box-shadow: 8px 0 30px rgba(11, 94, 87, 0.14);
    display: flex;
    flex-direction: column;
    position: sticky;
    top: 0;
}

.brand-mark {
    width: 46px;
    height: 46px;
    border-radius: 14px;
    display: grid;
    place-items: center;
    background: rgba(255, 255, 255, 0.16);
    border: 1px solid rgba(255, 255, 255, 0.22);
}

.nav-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(255, 255, 255, 0.76);
    padding: 0 6px;
    margin-bottom: 4px;
}

.side-link {
    width: 100%;
    padding: 11px 14px;
    border-radius: 12px;
    color: #ffffff;
    transition: all 0.2s ease;
    background: transparent;
    margin-bottom: 4px;
}
.side-link:hover {
    background: rgba(255, 255, 255, 0.10);
    transform: translateX(2px);
}
.side-link-active {
    background: linear-gradient(135deg, var(--accent-1), var(--accent-2)) !important;
    box-shadow: 0 8px 22px rgba(42, 174, 155, 0.35);
}
.side-link-active .q-icon,
.side-link-active .text-white {
    color: #ffffff !important;
}

.locked-card {
    margin-top: 24px;
    padding: 16px;
    border-radius: 16px;
    background: rgba(4, 60, 55, 0.32);
    border: 1px solid rgba(255, 255, 255, 0.14);
}

/* ── Contenido principal ──────────────────────────────────────────────── */
.content-panel {
    min-width: 0;
    width: 100%;
    max-width: 1680px;
    margin: 0 auto;
    padding: 30px 42px 48px;
    background: var(--bg);
}

.top-strip {
    height: 62px;
    border-radius: 20px;
    background: var(--panel);
    border: 1px solid var(--line);
    box-shadow: var(--shadow);
    padding: 0 24px;
}

/* ── Hero ─────────────────────────────────────────────────────────────── */
.hero-card {
    position: relative;
    overflow: hidden;
    border-radius: 24px;
    padding: 32px 36px;
    background:
        linear-gradient(135deg, var(--panel) 0%, #F3FBF9 100%);
    border: 1px solid var(--line);
    box-shadow: var(--shadow);
}
.hero-card::after {
    content: "";
    position: absolute;
    width: 240px;
    height: 240px;
    right: -80px;
    top: -85px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(42, 174, 155, 0.22), transparent 68%);
}

.eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    color: var(--accent-1);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}
.hero-title {
    font-size: clamp(30px, 3.6vw, 48px);
    line-height: 1;
    font-weight: 800;
    letter-spacing: -0.035em;
    color: var(--ink);
}
.hero-copy {
    max-width: 760px;
    color: var(--muted);
    font-size: 15px;
    line-height: 1.65;
}

.pill {
    border-radius: 999px;
    padding: 7px 12px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    background: #E3F5F0;
    color: var(--sidebar-2);
    border: 1px solid rgba(31, 143, 122, 0.2);
}

/* ── Tarjetas ─────────────────────────────────────────────────────────── */
.metric-card, .work-card {
    border-radius: 20px;
    background: var(--panel);
    border: 1px solid var(--line);
    box-shadow: 0 10px 32px rgba(28, 75, 85, 0.06);
}
.metric-card {
    padding: 20px 22px;
    min-height: 118px;
}
.metric-value {
    font-size: 34px;
    line-height: 1;
    font-weight: 800;
    color: var(--ink);
    letter-spacing: -0.035em;
}
.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--sidebar-2);
}
.work-card {
    padding: 26px 28px;
}
.section-title {
    font-size: 18px;
    font-weight: 800;
    color: var(--ink);
    letter-spacing: -0.02em;
}

/* ── Inputs ───────────────────────────────────────────────────────────── */
.q-field--outlined .q-field__control {
    border-radius: 14px;
    background: #fbfefe;
}

/* ── Botones ──────────────────────────────────────────────────────────── */
.primary-btn {
    min-height: 46px;
    border-radius: 14px;
    background: linear-gradient(135deg, var(--accent-1), var(--sidebar-2)) !important;
    color: #ffffff !important;
    font-weight: 800;
    box-shadow: 0 12px 26px rgba(31, 143, 122, 0.28);
}
.secondary-btn {
    min-height: 42px;
    border-radius: 12px;
    color: var(--sidebar-2) !important;
    border: 1px solid rgba(11, 94, 87, 0.28) !important;
    background: var(--panel) !important;
    font-weight: 700;
}

/* ── Botón de descarga premium ────────────────────────────────────────── */
.download-premium {
    min-height: 52px;
    border-radius: 16px;
    background: linear-gradient(135deg, #ffffff, #f0fdf9) !important;
    color: #0B5E57 !important;
    border: 2px solid #0B5E57 !important;
    font-weight: 800;
    font-size: 1rem;
    letter-spacing: 0.02em;
    box-shadow: 0 8px 24px rgba(11, 94, 87, 0.18);
    transition: all 0.25s ease;
}
.download-premium:hover {
    background: linear-gradient(135deg, #0B5E57, #0E6F66) !important;
    color: #ffffff !important;
    border-color: #0E6F66 !important;
    transform: translateY(-2px);
    box-shadow: 0 14px 32px rgba(11, 94, 87, 0.32);
}

/* ── Consola de log ───────────────────────────────────────────────────── */
.console-box textarea {
    min-height: 480px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    line-height: 1.65 !important;
    color: #b8ffee !important;
    background: #0d2b2a !important;
    border-radius: 16px !important;
    padding: 18px !important;
}

/* ── Tablas ───────────────────────────────────────────────────────────── */
.result-table .q-table__container {
    border-radius: 16px;
    border: 1px solid var(--line);
    box-shadow: none;
    background: var(--panel);
}
.result-table thead tr {
    background: #EAF6F3;
}
.result-table th {
    color: var(--sidebar-2);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Descarga ─────────────────────────────────────────────────────────── */
.download-card {
    border-radius: 22px;
    padding: 26px 28px;
    background: linear-gradient(135deg, var(--sidebar-1) 0%, var(--sidebar-2) 100%);
    color: #ffffff;
    box-shadow: 0 20px 40px rgba(11, 94, 87, 0.24);
}

/* ── Avisos de validación ─────────────────────────────────────────────── */
.validation-warning {
    border-radius: 16px;
    padding: 16px 20px;
    background: #FFF7DE;
    border: 1px solid #F0D480;
    border-left: 5px solid var(--amber);
    color: #7a5200;
}
.validation-warning .warning-title {
    font-size: 15px;
    font-weight: 800;
    color: #7a5200;
    letter-spacing: -0.01em;
}
.validation-warning .warning-body {
    font-size: 13.5px;
    line-height: 1.6;
    color: #5c3d00;
}
.validation-warning .warning-list {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12.5px;
    color: #4c3300;
    background: rgba(255, 255, 255, 0.55);
    padding: 8px 12px;
    border-radius: 10px;
    border: 1px dashed #dcae4a;
    word-break: break-word;
}
.validation-empty {
    border-radius: 16px;
    padding: 14px 18px;
    background: #FDECEC;
    border: 1px solid #f0b3b1;
    border-left: 5px solid var(--danger);
    color: #7a1f1c;
    font-weight: 700;
}

/* ── Aviso de reconexión ──────────────────────────────────────────────── */
.processing-banner {
    border-radius: 14px;
    padding: 12px 18px;
    background: #EAF6F3;
    border: 1px solid rgba(31, 143, 122, 0.28);
    color: var(--sidebar-2);
    font-weight: 700;
    font-size: 13.5px;
}

/* ── Responsive ───────────────────────────────────────────────────────── */
@media (max-width: 1000px) {
    .app-frame {
        grid-template-columns: 1fr;
    }
    .side-panel {
        min-height: auto;
        position: static;
    }
    .content-panel {
        padding: 20px 16px 32px;
    }
}
"""


# ══════════════════════════════════════════════════════════════════════════════
# INTERFAZ NICEGUI (v2 · rápida)
# ══════════════════════════════════════════════════════════════════════════════

@ui.page("/")
def main_page() -> None:
    """Página principal con carga asíncrona y pipeline optimizado."""

    # ── Estado reactivo por cliente ─────────────────────────────────────────
    _estado = {
        "archivos": {},
        "parroquias": [],
        "parroquias_excluidas": [],
        "temp_dir": None,
        "uploads": [],
    }
    _ui_state = {"csv_path": None}
    _debounce = {"timer": None, "pending": 0}

    ui.add_head_html(f"<style>{CUSTOM_CSS}</style>")
    ui.query(".nicegui-content").classes("p-0 gap-0").style("max-width:none; width:100%;")

    refs: dict = {}

    # ── Helpers UI ─────────────────────────────────────────────────────────
    def actualizar_tabla_archivos(archivos: dict) -> None:
        cols, rows = _df_to_table_payload(_tabla_archivos(archivos))
        refs["tabla_archivos"].columns = cols
        refs["tabla_archivos"].rows = rows
        refs["tabla_archivos"].update()

    def actualizar_select_parroquias(parroquias: list[str]) -> None:
        sel = refs["parroquia_select"]
        sel.set_options(parroquias, value=parroquias[0] if parroquias else None)

    def actualizar_metricas() -> None:
        refs["metric_archivos"].set_text(str(len(_estado.get("archivos", {}))))
        refs["metric_parroquias"].set_text(str(len(_estado.get("parroquias", []))))
        refs["metric_excluidas"].set_text(str(len(_estado.get("parroquias_excluidas", []))))

    def mensaje_estado(txt: str, tone: str = "info") -> None:
        clase_map = {
            "info": "text-[#0B5E57]",
            "ok": "text-[#0B5E57]",
            "warning": "text-[#9a6a00]",
            "error": "text-[#b73d39]",
        }
        refs["status_label"].classes(remove="text-[#0B5E57] text-[#9a6a00] text-[#b73d39]")
        refs["status_label"].classes(add=clase_map.get(tone, clase_map["info"]))
        refs["status_label"].set_text(txt)

    def actualizar_validacion_ui(validas: list[dict], excluidas: list[dict]) -> None:
        hay_validas = len(validas) > 0
        hay_excluidas = len(excluidas) > 0

        refs["warning_container"].visible = hay_excluidas
        if hay_excluidas:
            texto = ", ".join(
                f"{r['parroquia']} (faltan: {', '.join(r['faltantes']) if r['faltantes'] else '—'})"
                for r in excluidas
            )
            refs["warning_list_label"].set_text(texto)

        refs["empty_container"].visible = not hay_validas

    def _aplicar_desde_mapa(archivos: dict[str, Path], mapa: dict, origen: str) -> None:
        _estado["archivos"] = archivos
        validas, excluidas = validar_parroquias(mapa)
        nombres_validos = [r["parroquia"] for r in validas]
        _estado["parroquias"] = nombres_validos
        _estado["parroquias_excluidas"] = excluidas

        actualizar_select_parroquias(nombres_validos)
        actualizar_tabla_archivos(archivos)
        actualizar_metricas()
        actualizar_validacion_ui(validas, excluidas)

        if archivos:
            partes = [
                f"{len(archivos)} archivo(s) reconocido(s) desde {origen}: "
                f"{', '.join(archivos.keys())}"
            ]
            if nombres_validos:
                partes.append(f"{len(nombres_validos)} parroquia(s) válida(s).")
            if excluidas:
                partes.append(
                    f"{len(excluidas)} parroquia(s) excluida(s) por completitud insuficiente."
                )
            mensaje_estado(" · ".join(partes), "ok" if nombres_validos else "warning")
            ui.notify("Archivos REMMAQ cargados.", type="positive")
        else:
            mensaje_estado("No se encontraron archivos REMMAQ reconocidos.", "warning")
            ui.notify("No se encontraron archivos reconocidos.", type="warning")

    # ── Callbacks optimizados ──────────────────────────────────────────────
    async def _procesar_lote_upload() -> None:
        """
        Escanea y detecta parroquias en segundo plano usando run.cpu_bound.
        La barra de progreso se actualiza durante la espera para dar feedback real.
        """
        tmp = Path(_estado["temp_dir"]) if _estado["temp_dir"] else None
        if tmp is None or not tmp.exists():
            return
        mensaje_estado("Analizando archivos subidos…", "info")
        # Mostrar barra de progreso indeterminada mientras se procesa
        refs["progress_upload"].visible = True
        try:
            archivos, mapa = await run.cpu_bound(_escanear_y_detectar, tmp)
        except Exception as exc:
            ui.notify(f"Error al analizar archivos: {exc}", type="negative")
            mensaje_estado(f"Error al analizar: {exc}", "error")
            return
        finally:
            refs["progress_upload"].visible = False
        _aplicar_desde_mapa(archivos, mapa, origen="subida manual")

    def _programar_deteccion() -> None:
        if _debounce["timer"] is not None:
            try:
                _debounce["timer"].cancel()
            except Exception:
                pass
        # Debounce de 0.4 segundos para agrupar subidas rápidas
        _debounce["timer"] = ui.timer(0.4, _procesar_lote_upload, once=True)

    async def handle_upload(e: events.UploadEventArguments) -> None:
        if _estado["temp_dir"] is None:
            _estado["temp_dir"] = tempfile.mkdtemp(prefix="remmaq_uploads_")
        tmp = Path(_estado["temp_dir"])
        nombre = e.file.name
        dest = tmp / Path(nombre).name
        # Guardado asíncrono
        await e.file.save(dest)
        _estado["uploads"].append(dest.name)
        _debounce["pending"] += 1
        mensaje_estado(
            f"Recibiendo archivos… ({_debounce['pending']} subido/s)", "info"
        )
        _programar_deteccion()

    async def comprobar_datos_ui() -> None:
        tmp = Path(_estado["temp_dir"]) if _estado["temp_dir"] else None
        directorio = (refs["txt_dir"].value or "").strip() if "txt_dir" in refs else ""
        if directorio:
            candidato = Path(directorio)
            if candidato.exists():
                tmp = candidato
        if tmp is None or not tmp.exists():
            ui.notify("No hay archivos que comprobar todavía.", type="warning")
            return
        mensaje_estado("Analizando archivos…", "info")
        refs["diagnostico_container"].visible = True
        refs["diagnostico_out"].set_value("Analizando archivos, espera un momento…")
        try:
            reporte, archivos, mapa = await run.cpu_bound(_diagnostico_y_deteccion, tmp)
        except Exception as exc:
            refs["diagnostico_out"].set_value(f"Error durante el análisis:\n{exc}")
            mensaje_estado(f"Error durante el análisis: {exc}", "error")
            return
        refs["diagnostico_out"].set_value(reporte)
        _aplicar_desde_mapa(archivos, mapa, origen="comprobación")

    async def escanear_directorio_ui() -> None:
        directorio = (refs["txt_dir"].value or "").strip()
        if not directorio:
            mensaje_estado("Ingresa una ruta de directorio válida.", "warning")
            ui.notify("Falta la ruta del directorio.", type="warning")
            return
        data_dir = Path(directorio)
        if not data_dir.exists():
            mensaje_estado(f"El directorio no existe: {directorio}", "error")
            ui.notify("El directorio no existe.", type="negative")
            _aplicar_desde_mapa({}, {}, origen="directorio")
            return
        mensaje_estado("Escaneando directorio…", "info")
        btn_scan = refs["btn_scan"]
        btn_scan.props("loading")
        try:
            archivos, mapa = await run.cpu_bound(_escanear_y_detectar, data_dir)
        except Exception as exc:
            ui.notify(f"Error al escanear: {exc}", type="negative")
            mensaje_estado(f"Error al escanear: {exc}", "error")
            return
        finally:
            btn_scan.props(remove="loading")
        _aplicar_desde_mapa(archivos, mapa, origen="directorio")

    async def ejecutar_pipeline_ui() -> None:
        archivos = _estado.get("archivos", {})
        sel = refs["parroquia_select"]
        parroquia = (sel.value or "").strip() if sel.value else ""
        if not archivos:
            ui.notify("Primero carga o escanea archivos REMMAQ.", type="warning")
            refs["log_out"].set_value("No hay archivos cargados.")
            navegar_a("resultados")
            return
        if not parroquia:
            ui.notify("Selecciona una parroquia válida.", type="warning")
            refs["log_out"].set_value("Selecciona una parroquia válida primero.")
            return

        refs["btn_run"].disable()
        refs["progress_bar"].visible = True
        refs["progress_bar"].set_value(0.08)
        refs["processing_banner"].visible = True
        navegar_a("resultados")
        await asyncio.sleep(0)

        try:
            df, log_text = await run.cpu_bound(
                construir_dataset,
                archivos,
                parroquia.upper(),
                bool(refs["excluir_pandemia"].value),
            )
            refs["progress_bar"].set_value(0.72)
            await asyncio.sleep(0)

            # ── Usar nombre personalizado si se ingresó ─────────────────────
            nombre_custom = (refs["nombre_dataset"].value or "").strip()
            if nombre_custom:
                nombre_base = re.sub(r"[^\w\-]+", "_", nombre_custom).strip("._-")
            else:
                nombre_base = f"dataset_ml_{parroquia.lower().replace(' ', '_')}"
            nombre_archivo = f"{nombre_base}.csv"

            tmp_out = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".csv",
                prefix=f"{nombre_base}_",
            )
            tmp_out.write(df_a_csv(df, parroquia).encode("utf-8"))
            tmp_out.close()
            # Copiar a una ruta con el nombre definitivo para que la descarga sea legible
            ruta_final = Path(tempfile.gettempdir()) / nombre_archivo
            shutil.copy2(tmp_out.name, ruta_final)
            _ui_state["csv_path"] = str(ruta_final)

            refs["log_out"].set_value(log_text)

            pcols, prows = _df_to_table_payload(_perfil_dataframe(df))
            refs["perfil_table"].columns = pcols
            refs["perfil_table"].rows = prows
            refs["perfil_table"].update()

            vcols, vrows = _df_to_table_payload(df, max_rows=100)
            refs["preview_table"].columns = vcols
            refs["preview_table"].rows = vrows
            refs["preview_table"].update()

            refs["download_btn"].enable()
            refs["download_info"].set_text(
                f"CSV listo: {nombre_archivo} · {len(df):,} filas × {df.shape[1]} columnas"
            )
            refs["progress_bar"].set_value(1.0)
            ui.notify("Dataset listo para descargar.", type="positive")
        except Exception as exc:
            refs["log_out"].set_value(str(exc))
            refs["download_btn"].disable()
            _ui_state["csv_path"] = None
            ui.notify("No se pudo construir el dataset.", type="negative")
        finally:
            refs["btn_run"].enable()
            refs["processing_banner"].visible = False

    def descargar_csv_ui() -> None:
        path = _ui_state.get("csv_path")
        if not path or not Path(path).exists():
            ui.notify("Todavía no hay dataset generado.", type="warning")
            return
        ui.download(path, filename=Path(path).name)

    def navegar_a(seccion: str) -> None:
        for key, panel in refs["paneles"].items():
            panel.visible = (key == seccion)
        for key, link in refs["nav_links"].items():
            if key == seccion:
                link.classes(add="side-link-active", remove="")
            else:
                link.classes(add="", remove="side-link-active")

    # ── LAYOUT ──────────────────────────────────────────────────────────────
    with ui.element("div").classes("app-frame"):
        # ═══ Sidebar ═════════════════════════════════════════════════════════
        with ui.column().classes("side-panel"):
            with ui.row().classes("items-center gap-3 mb-6"):
                with ui.element("div").classes("brand-mark"):
                    ui.icon("air", size="26px").classes("text-white")
                with ui.column().classes("gap-0"):
                    ui.label("REMMAQ").classes("text-xl font-extrabold leading-tight text-white")
                    ui.label("Dataset ML Builder").classes("text-xs text-white/70")

            ui.label("Navegación").classes("nav-label mt-2")

            nav_items = [
                ("inicio", "home", "Inicio"),
                ("carga", "cloud_upload", "Carga de datos"),
                ("config", "tune", "Configuración"),
                ("resultados", "analytics", "Resultados"),
                ("descarga", "download", "Descarga"),
            ]
            refs["nav_links"] = {}
            for key, icon, label in nav_items:
                link = ui.row().classes("side-link items-center gap-3 cursor-pointer")
                with link:
                    ui.icon(icon, size="20px").classes("text-white")
                    ui.label(label).classes("font-bold text-white")
                link.on("click", lambda k=key: navegar_a(k))
                refs["nav_links"][key] = link

            with ui.column().classes("locked-card gap-2 mt-auto"):
                ui.label("Parámetros internos").classes("nav-label text-white/80")
                with ui.row().classes("items-center gap-2"):
                    ui.icon("lock", size="15px").classes("text-white/85")
                    ui.label("Gap sensores: 24 h (fijo)").classes("text-white font-bold text-sm")
                ui.label(f"Target: {TARGET}").classes("text-white/85 text-xs")
                ui.label("Feature engineering activo").classes("text-white/85 text-xs")

        # ═══ Contenido principal ═════════════════════════════════════════════
        with ui.column().classes("content-panel gap-6"):
            with ui.row().classes("top-strip w-full items-center justify-between"):
                ui.label("Red Metropolitana de Monitoreo Atmosférico de Quito").classes(
                    "font-extrabold text-[#2E2E2E]"
                )
                with ui.row().classes("items-center gap-2"):
                    ui.icon("bolt", size="18px").classes("text-[#1F8F7A]")
                    ui.label("NiceGUI · scikit-learn").classes("text-sm text-[#556]")

            refs["paneles"] = {}

            # ─── Panel: Inicio ───────────────────────────────────────────────
            panel_inicio = ui.column().classes("gap-6 w-full")
            with panel_inicio:
                with ui.column().classes("hero-card gap-4"):
                    ui.label("Dataset ML Builder").classes("eyebrow")
                    ui.label("Limpieza y creación de dataset para IA").classes("hero-title")
                    ui.label(
                        "Carga archivos REMMAQ, valida completitud por parroquia, "
                        "ejecuta el pipeline y descarga un CSV listo para modelos de machine learning."
                    ).classes("hero-copy")
                    with ui.row().classes("gap-2 mt-2"):
                        ui.label("PM25 target").classes("pill")
                        ui.label("Gap fijo 24 h").classes("pill")
                        ui.label("COVID 2020–2021 opcional").classes("pill")
                        ui.label("Completitud 10/11").classes("pill")

                with ui.grid(columns=3).classes("w-full gap-5"):
                    for icon, label, key in [
                        ("insert_drive_file", "Archivos", "metric_archivos"),
                        ("place", "Parroquias válidas", "metric_parroquias"),
                        ("block", "Parroquias excluidas", "metric_excluidas"),
                    ]:
                        with ui.column().classes("metric-card gap-2"):
                            with ui.row().classes("items-center gap-2"):
                                ui.icon(icon, size="20px").classes("text-[#1F8F7A]")
                                ui.label(label).classes("metric-label")
                            refs[key] = ui.label("0").classes("metric-value")

                with ui.column().classes("work-card gap-3"):
                    ui.label("Cómo usar la herramienta").classes("section-title")
                    for step in [
                        "1. Ve a “Carga de datos” y escanea el directorio o sube los archivos.",
                        "2. Abre “Configuración” para elegir la parroquia y las opciones.",
                        "3. Ejecuta el pipeline — el sistema aplica el gap fijo de 24 h.",
                        "4. Revisa “Resultados” y descarga el CSV en “Descarga”.",
                    ]:
                        with ui.row().classes("items-center gap-2"):
                            ui.icon("check_circle", size="18px").classes("text-[#2AAE9B]")
                            ui.label(step).classes("text-[#3A3A3A] text-sm")

            refs["paneles"]["inicio"] = panel_inicio

            # ─── Panel: Carga de datos ───────────────────────────────────────
            panel_carga = ui.column().classes("gap-6 w-full")
            with panel_carga:
                with ui.column().classes("work-card gap-4"):
                    ui.label("Carga de datos").classes("section-title")
                    ui.label(
                        "Elige el origen de los archivos REMMAQ. El sistema detecta "
                        "automáticamente las variables y las parroquias disponibles."
                    ).classes("text-sm text-[#556]")

                    with ui.tabs().classes("text-[#2E2E2E]") as tabs_carga:
                        tab_dir = ui.tab("Directorio local", icon="folder_open")
                        tab_upload = ui.tab("Subir archivos", icon="upload_file")
                    with ui.tab_panels(tabs_carga, value=tab_dir).classes(
                        "w-full bg-transparent"
                    ):
                        with ui.tab_panel(tab_dir).classes("p-0 pt-4"):
                            with ui.row().classes("w-full items-center gap-3"):
                                refs["txt_dir"] = ui.input(
                                    "Ruta del directorio",
                                    placeholder="/ruta/a/tus/archivos/remmaq",
                                ).props("outlined clearable").classes("flex-1")
                                refs["btn_scan"] = ui.button(
                                    "Escanear", icon="search"
                                ).classes("secondary-btn")
                                refs["btn_scan"].on_click(escanear_directorio_ui)
                        with ui.tab_panel(tab_upload).classes("p-0 pt-4"):
                            ui.label(
                                "Nombres esperados: PM2.5.xlsx, PM10.xlsx, SO2.xlsx, "
                                "CO.xlsx, O3.xlsx, NO2.xlsx, TMP.xlsx, HUM.xlsx, "
                                "VEL.xlsx, DIR.xlsx, LLU.xlsx."
                            ).classes("text-sm text-[#556] leading-relaxed")
                            ui.upload(
                                label="Selecciona archivos REMMAQ",
                                multiple=True,
                                auto_upload=True,
                                max_file_size=200 * 1024 * 1024,
                                on_upload=handle_upload,
                                on_rejected=lambda: ui.notify(
                                    "Archivo rechazado (formato o tamaño no soportado).",
                                    type="warning",
                                ),
                            ).props('accept=".xlsx,.xls,.csv"').classes("w-full")
                            # Barra de progreso durante la carga
                            refs["progress_upload"] = ui.linear_progress(value=0).classes("w-full mt-2")
                            refs["progress_upload"].visible = False

                    refs["status_label"] = ui.label(
                        "Carga un directorio o sube archivos para comenzar."
                    ).classes("text-sm font-bold text-[#0B5E57]")

                    with ui.row().classes("w-full items-center gap-3 mt-2"):
                        btn_check = ui.button(
                            "Comprobar datos", icon="fact_check"
                        ).classes("primary-btn")
                        btn_check.on_click(comprobar_datos_ui)
                        ui.label(
                            "Analiza los archivos subidos: cuáles se reconocen, qué "
                            "columnas tienen y qué variables faltan por parroquia."
                        ).classes("text-sm text-[#556]")

                refs["diagnostico_container"] = ui.column().classes("work-card gap-3 w-full")
                with refs["diagnostico_container"]:
                    ui.label("Diagnóstico de archivos").classes("section-title")
                    refs["diagnostico_out"] = ui.textarea(
                        placeholder="Pulsa “Comprobar datos” para ver el análisis detallado."
                    ).props("readonly outlined").classes("console-box w-full")
                refs["diagnostico_container"].visible = False

                with ui.column().classes("work-card gap-3"):
                    ui.label("Archivos detectados").classes("section-title")
                    cols0, rows0 = _df_to_table_payload(_tabla_archivos({}))
                    refs["tabla_archivos"] = ui.table(
                        columns=cols0,
                        rows=rows0,
                        row_key="Variable",
                        pagination=12,
                    ).classes("result-table w-full")

            panel_carga.visible = False
            refs["paneles"]["carga"] = panel_carga

            # ─── Panel: Configuración ────────────────────────────────────────
            panel_config = ui.column().classes("gap-6 w-full")
            with panel_config:
                with ui.column().classes("work-card gap-4"):
                    ui.label("Configuración del pipeline").classes("section-title")

                    with ui.column().classes("w-full gap-3"):
                        refs["warning_container"] = ui.column().classes(
                            "validation-warning w-full gap-2"
                        )
                        with refs["warning_container"]:
                            with ui.row().classes("items-center gap-2"):
                                ui.icon("warning_amber", size="22px").classes("text-[#c88a00]")
                                ui.label(
                                    "Parroquias excluidas por completitud insuficiente"
                                ).classes("warning-title")
                            ui.label(
                                "Las siguientes parroquias tienen 2 o más variables "
                                "faltantes y NO pueden ser utilizadas para el modelo:"
                            ).classes("warning-body")
                            refs["warning_list_label"] = ui.label("").classes("warning-list")
                            ui.label(
                                "Para usar estas parroquias, asegúrate de que los archivos "
                                "contengan al menos 10 de las 11 variables requeridas "
                                "(6 contaminantes + 5 meteorológicas)."
                            ).classes("warning-body")
                        refs["warning_container"].visible = False

                        refs["empty_container"] = ui.column().classes(
                            "validation-empty w-full gap-1"
                        )
                        with refs["empty_container"]:
                            with ui.row().classes("items-center gap-2"):
                                ui.icon("info", size="20px").classes("text-[#b73d39]")
                                ui.label(
                                    "No hay parroquias con suficientes datos. "
                                    "Sube archivos adicionales."
                                ).classes("font-extrabold")
                        refs["empty_container"].visible = False

                    refs["parroquia_select"] = ui.select(
                        options=[],
                        label="Parroquia / Estación",
                        with_input=True,
                    ).props("outlined clearable use-input").classes("w-full")

                    refs["excluir_pandemia"] = ui.checkbox(
                        "Excluir período pandemia COVID-19 (2020–2021)",
                        value=True,
                    ).classes("text-[#2E2E2E]")

                    with ui.row().classes("items-center gap-2 mt-2"):
                        ui.icon("lock", size="16px").classes("text-[#1F8F7A]")
                        ui.label(
                            "El umbral de gap de sensores es fijo (24 horas) y no "
                            "puede modificarse desde la interfaz."
                        ).classes("text-sm text-[#556]")

                    # ── Campo para nombre del dataset (NUEVO) ──────────────
                    refs["nombre_dataset"] = ui.input(
                        label="Nombre base del CSV de salida",
                        placeholder="dataset_ml_carapungo",
                        value="",
                    ).props("outlined clearable").classes("w-full")

                    refs["progress_bar"] = ui.linear_progress(value=0).classes("w-full")
                    refs["progress_bar"].visible = False

                    refs["btn_run"] = ui.button(
                        "Construir Dataset ML", icon="play_arrow"
                    ).classes("primary-btn w-full")
                    refs["btn_run"].on_click(ejecutar_pipeline_ui)

            panel_config.visible = False
            refs["paneles"]["config"] = panel_config

            # ─── Panel: Resultados ───────────────────────────────────────────
            panel_res = ui.column().classes("gap-6 w-full")
            with panel_res:
                refs["processing_banner"] = ui.row().classes(
                    "processing-banner items-center gap-2 w-full"
                )
                with refs["processing_banner"]:
                    ui.spinner(size="18px")
                    ui.label(
                        "Procesando en segundo plano — la interfaz sigue disponible, "
                        "puedes navegar entre secciones mientras termina."
                    )
                refs["processing_banner"].visible = False

                with ui.column().classes("work-card gap-4"):
                    ui.label("Resultados del pipeline").classes("section-title")
                    with ui.tabs().classes("text-[#2E2E2E]") as tabs_res:
                        tab_log = ui.tab("Log", icon="terminal")
                        tab_perfil = ui.tab("Perfil", icon="bar_chart")
                        tab_preview = ui.tab("Vista previa", icon="visibility")
                    with ui.tab_panels(tabs_res, value=tab_log).classes(
                        "w-full bg-transparent"
                    ):
                        with ui.tab_panel(tab_log).classes("p-0 pt-4"):
                            refs["log_out"] = ui.textarea(
                                placeholder="El log aparecerá aquí al ejecutar el pipeline."
                            ).props("readonly outlined").classes("console-box w-full")
                        with ui.tab_panel(tab_perfil).classes("p-0 pt-4"):
                            refs["perfil_table"] = ui.table(
                                columns=[], rows=[], pagination=10
                            ).classes("result-table w-full")
                        with ui.tab_panel(tab_preview).classes("p-0 pt-4"):
                            refs["preview_table"] = ui.table(
                                columns=[], rows=[], pagination=10
                            ).classes("result-table w-full")

            panel_res.visible = False
            refs["paneles"]["resultados"] = panel_res

            # ─── Panel: Descarga ─────────────────────────────────────────────
            panel_desc = ui.column().classes("gap-6 w-full")
            with panel_desc:
                with ui.column().classes("download-card gap-3"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("download", size="24px").classes("text-white")
                        ui.label("Dataset generado").classes("text-xl font-extrabold")
                    ui.label(
                        "El CSV incluye metadatos comentados con # y se puede leer con "
                        "pandas usando index_col='Timestamp'."
                    ).classes("opacity-90")
                    refs["download_info"] = ui.label("Aún no hay dataset generado.").classes(
                        "opacity-90 text-sm"
                    )
                    # Botón de descarga premium rediseñado
                    refs["download_btn"] = ui.button(
                        "Descargar CSV", icon="download"
                    ).classes("download-premium w-full")
                    refs["download_btn"].disable()
                    refs["download_btn"].on_click(descargar_csv_ui)

                with ui.column().classes("work-card gap-3"):
                    ui.label("Uso del CSV").classes("section-title")
                    ui.code(
                        "import pandas as pd\n"
                        "df = pd.read_csv('dataset_ml_*.csv',\n"
                        "                 comment='#',\n"
                        "                 index_col='Timestamp',\n"
                        "                 parse_dates=True)\n"
                        "X = df.drop(columns=['PM25'])\n"
                        "y = df['PM25']",
                        language="python",
                    ).classes("w-full")

            panel_desc.visible = False
            refs["paneles"]["descarga"] = panel_desc

            # Footer
            ui.label(
                "REMMAQ · Red Metropolitana de Monitoreo Atmosférico de Quito · "
                "Dataset para scikit-learn"
            ).classes("w-full text-center text-xs text-[#7a8a92] pt-4 pb-6")

    # Activar sección inicial
    navegar_a("inicio")


# ══════════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA
# ══════════════════════════════════════════════════════════════════════════════

if __name__ in ("__main__", "__mp_main__"):
    ui.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 3002)),
        title="REMMAQ · Dataset ML Builder",
        reload=False,
        show=True,
        reconnect_timeout=180,
    )