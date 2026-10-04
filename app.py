"""
Formula Kite Analytics Dashboard
Telemetría Sailmon Max / Vakaros Atlas · Python 3 · Streamlit
"""

import io
import os
import struct

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go

CARTO_API_KEY = os.environ.get("CARTO_API_KEY", "cb1_47g1_1_e3de9d8bb58aa32a792ce32c")


def carto_mapbox(**kwargs):
    """Config `map` (MapLibre) de Plotly con el basemap oscuro de CARTO (requiere API key)."""
    return dict(
        style="white-bg",
        layers=[dict(
            below="traces",
            sourcetype="raster",
            source=[f"https://basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}.png?key={CARTO_API_KEY}"],
        )],
        **kwargs,
    )


_COMPONENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "components")
_time_range_selector_component = components.declare_component(
    "time_range_selector",
    path=os.path.join(_COMPONENTS_DIR, "time_range_selector"),
)


def time_range_selector(series, duration_ms, start_clock_s, start_ms, end_ms, height=240, key=None):
    """Selector de rango temporal (HTML/JS puro): arrastra el bloque completo para
    mover inicio y fin a la vez, o arrastra un borde para ajustar solo ese extremo."""
    return _time_range_selector_component(
        series=series,
        duration_ms=duration_ms,
        start_clock_s=start_clock_s,
        start_ms=start_ms,
        end_ms=end_ms,
        height=height,
        ref_lines=[FOIL_FLIGHT_KTS],
        key=key,
        default=None,
    )


_track_replay_component = components.declare_component(
    "track_replay",
    path=os.path.join(_COMPONENTS_DIR, "track_replay"),
)


REPLAY_METRICS = [  # (etiqueta, columna, unidad) — datos elegibles en la gráfica del replay
    ("Velocidad (SOG)", "SOG_kts", "kts"),
    ("VMG", "VMG_kts", "kts"),
    ("TWA", "TWA", "°"),
    ("Heel", "Heel", "°"),
    ("Rumbo (HDT)", "HDT", "°"),
    ("COG", "COG", "°"),
    ("TWD", "TWD", "°"),
    ("Trim proa/popa", "Trim Fore / Aft", "°"),
    ("Pérdida vs ghost", "Ghost_loss_m", "m"),  # solo en los replays de maniobra
]


def build_replay_racers(dfs, max_points=8000):
    """Serializa los tracks para el componente de replay: tiempos relativos (ms) al
    primer instante de `dfs`, lat/lon, SOG y rumbo (grados) calculado entre puntos."""
    valid = [d for d in dfs if {"time", "latitude", "longitude"} <= set(d.columns)]
    if not valid:
        return [], 0.0, 0.0
    t_min = min(d["time"].min() for d in valid)
    t_max = max(d["time"].max() for d in valid)
    racers = []
    for i, d in enumerate(valid):
        dd = d.dropna(subset=["time", "latitude", "longitude"]).sort_values("time")
        if len(dd) > max_points:
            dd = dd.iloc[:: int(np.ceil(len(dd) / max_points))]
        if dd.empty:
            continue
        lat = dd["latitude"].to_numpy(float)
        lon = dd["longitude"].to_numpy(float)
        dlat = np.gradient(lat)
        dlon = np.gradient(lon) * np.cos(np.radians(lat))
        hdg = (np.degrees(np.arctan2(dlon, dlat)) + 360) % 360
        metrics = {}
        for label, col, unit in REPLAY_METRICS:
            if col in dd.columns:
                vals = pd.to_numeric(dd[col], errors="coerce").round(1)
                metrics[label] = {"unit": unit, "v": [None if pd.isna(v) else float(v) for v in vals]}
        racers.append({
            "name": str(d["Regatista"].iloc[0]) if "Regatista" in d.columns else f"Regatista {i + 1}",
            "color": RACER_PALETTE[i % len(RACER_PALETTE)],
            "t": ((dd["time"] - t_min).dt.total_seconds() * 1000).round().tolist(),
            "lat": lat.round(6).tolist(),
            "lon": lon.round(6).tolist(),
            "metrics": metrics,
            "fase": dd["Fase"].fillna("—").astype(str).tolist() if "Fase" in dd.columns else None,
            "hdg": np.nan_to_num(hdg).round(0).tolist(),
        })
    duration_ms = (t_max - t_min).total_seconds() * 1000
    start_clock_s = t_min.hour * 3600 + t_min.minute * 60 + t_min.second
    return racers, duration_ms, start_clock_s


def track_replay(racers, duration_ms, start_clock_s, ahead_s=5, key=None):
    """Mapa representativo (solo línea por delante) + gráfica inferior que fija la
    posición del track, con Play y velocidades 1x/2x/4x."""
    return _track_replay_component(
        racers=racers,
        duration_ms=duration_ms,
        start_clock_s=start_clock_s,
        ahead_s=ahead_s,
        key=key,
        default=None,
    )


def build_ghost_racers(df, man_row):
    """Replay de una maniobra (fila de detect_maneuvers): tramo del barco y ghost que va con él
    hasta el inicio de la maniobra y luego sigue recto (ver maneuver_ghost), ambos con la TWD
    de entrada fija para que la línea de líder mida lo mismo que la tabla."""
    df = df.reset_index(drop=True)
    _opt = lambda v: None if pd.isna(v) else int(v)
    g, _ = maneuver_ghost(df, int(man_row["_i"]), leg_vmg_ref(df),
                          _opt(man_row["_recovered"]), _opt(man_row["_start"]))
    if g is None:
        return None
    boat = df.iloc[g["a"]:g["b"] + 1].copy()
    boat["TWD"] = g["twd"]
    boat["Ghost_loss_m"] = g["loss"]
    same = np.arange(len(boat)) <= g["start"]  # hasta el inicio de la maniobra, igual que el barco
    _col = lambda c, after: np.where(same, boat[c].to_numpy(float), after) if c in boat else after
    ghost = pd.DataFrame({
        "time": boat["time"].to_numpy(),
        "latitude": g["lat"],
        "longitude": g["lon"],
        "SOG_kts": _col("SOG_kts", g["sog_kts"]),
        "VMG_kts": _col("VMG_kts", g["vmg_kts"]),
        "TWA": _col("TWA", g["twa"]),
        "TWD": g["twd"],
        "Regatista": "Ghost",
    })
    racers, duration_ms, start_clock_s = build_replay_racers([boat, ghost])
    racers[1]["color"] = "#cbd5e1"
    racers[1]["ghost"] = True
    return racers, duration_ms, start_clock_s

# ─── Configuración de página ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Formula Kite Analytics",
    page_icon="🪁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS mínimo para toque visual ────────────────────────────────────────────
st.markdown(
    """
    <style>
    .section-title {
        font-size: 1.2rem; font-weight: 700;
        border-left: 3px solid #e94560;
        padding-left: 10px;
        margin: 1.4rem 0 0.7rem 0;
        color: #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Constantes ──────────────────────────────────────────────────────────────
MS_TO_KNOTS = 1.94384
LOCAL_UTC_OFFSET = pd.Timedelta(hours=2)  # Sailmon exporta en UTC; se muestra en hora local UTC+2
RECOVERY_FRAC   = 0.90  # recovery: hasta recuperar el 90 % de la SOG de entrada
DROP_START_FRAC = 0.95  # la caída empieza cuando la SOG baja del 95 % de la de entrada
ENTRY_WINDOW_S  = (15, 5)  # SOG de entrada: media entre 15 y 5 s antes del cambio de amura
EXIT_WINDOW_S   = (20, 40)  # SOG de salida: mediana entre 20 y 40 s después (rumbo ya estable)
GHOST_PRE_S     = 15  # el ghost sale de la posición del barco 15 s antes del cambio de amura
GHOST_POST_S    = 30  # pérdida medida 30 s después del cambio de amura (ventana fija, comparable)
GHOST_EXIT_S    = (20, 30)  # VMG de salida del ghost: entre 20 y 30 s después (10 s, como la entrada)
GHOST_REC_MARGIN_S = 5      # si recupera tarde, la salida empieza 5 s después del recovery
GHOST_MAX_COURSE_DEV = 15   # ° máx. de variación del rumbo en las ventanas de entrada/salida
GHOST_MIN_VMG_FRAC   = 0.5  # VMG de entrada/salida mínimo, como fracción del normal de la borda
FOIL_FLIGHT_KTS = 6   # por debajo de este umbral el foil no vuela. Estimado de los CSV: el vuelo
                      # sostenido empieza en ~9 kts, 5-8 kts son solo transitorios y hay maniobras
                      # sin caída con SOG mín de 6.7 kts
FALL_KTS        = 3   # maniobra fallida (caída al agua) solo si la SOG baja de aquí; una maniobra
                      # lenta puede pasar por 6-8 kts sin caerse
REFLY_S         = 5   # tras una caída, s seguidos en vuelo para considerar que vuelve a volar
PHASE_SMOOTH_S  = 5   # ventana (s) de la mediana móvil aplicada a SOG y |TWA|
PHASE_MIN_S     = 5   # un tramo más corto entre dos tramos de la misma fase se absorbe (parpadeo A-B-A)
TWA_UPWIND_MAX   = 70   # |TWA| < 70° → Ceñida
TWA_DOWNWIND_MIN = 110  # |TWA| > 110° → Popa
MANEUVER_WINDOW_S   = 8     # s antes y después que se comparan para confirmar el cambio de amura
MANEUVER_SIDE_MIN   = 0.25  # |media de sin(TWA)| mínima a cada lado (≈ TWA a >15° de proa/popa)
MANEUVER_TURN_MIN   = 40    # ° de giro del TWA medio para confirmar maniobras muy cerradas o muy
MANEUVER_DEEP_SIDE  = 0.5   # abiertas (|sin| < 0.25 en un lado) si el otro lado tiene |sin| ≥ 0.5
MANEUVER_RATE_MIN   = 4     # °/s: la maniobra dura mientras el TWA gire más rápido que esto
MANEUVER_MAX_EXT_S  = 10    # máx. segundos que se extiende la maniobra a cada lado del cruce
MANEUVER_PAD_S      = 1     # segundos añadidos al principio y al final
TWD_MAN_MIN_KTS      = 12    # TWD por maniobras (Atlas): en vuelo antes y después
TWD_MAN_BEFORE_S     = 10    # s de rumbo estable antes de la maniobra (acaba 4 s antes del centro)
TWD_MAN_AFTER_S      = 10    # s de rumbo estable después (empieza 3 s después del centro)
TWD_MAN_DELAY_S      = 12    # s tras el centro en que se aplica la nueva TWD (como Sailmon)
TWD_MAN_TURN_MIN     = 50    # ° de giro mínimo/máximo para que cuente como virada/trasluchada
TWD_MAN_TURN_MAX     = 130
TWD_MAN_MAX_SD       = 20    # ° de dispersión máxima del rumbo en cada ventana
TWD_MAN_MAX_DEV      = 35    # ° máximos entre la bisectriz y la TWD vigente (si no, es arribada/orzada)
TWD_REF_MIN_FLYING_S = 300   # s en vuelo mínimos para orientar las maniobras
TWD_BORROW_TOL_S     = 60    # s máximos hasta el dato de Sailmon más cercano al copiar la TWD
_TWD_HARMONICS = np.arange(1, 9)
_TWD_AXES = np.radians(np.arange(0, 180, 1.0))
DEVICE_LABELS = {"sailmon": "Sailmon Max", "atlas": "Vakaros Atlas"}
PHASE_COLORS = {
    "Popa":       "#EF553B",
    "Ceñida":     "#00CC96",
    "Través":     "#19D3F3",
    "Transición": "#FFA15A",
    "Caída":      "#636EFA",
}

RACER_PALETTE = [
    "#00B4D8", "#FF6B6B", "#A8DADC", "#FFD166",
    "#06D6A0", "#EF476F", "#118AB2", "#073B4C",
]


# ─── Lógica de negocio ───────────────────────────────────────────────────────
def _maneuver_mask(twa: pd.Series) -> np.ndarray:
    """
    Marca las viradas/trasluchadas (cambios de amura) en una serie de TWA a 1 Hz.
    1. Cambio de amura confirmado: la media de sin(TWA) en los MANEUVER_WINDOW_S
       anteriores y posteriores tiene signo opuesto (las eses a popa no lo cumplen).
       Una trasluchada desde popa profunda (TWA ≈ ±175°) o una virada desde un TWA muy cerrado
       solo tiene |sin| alto en un lado: vale también si el TWA medio gira ≥ MANEUVER_TURN_MIN.
    2. Desde el cruce del TWA, la maniobra se extiende a cada lado mientras el TWA
       siga girando a más de MANEUVER_RATE_MIN °/s (máx. MANEUVER_MAX_EXT_S).
    """
    n = len(twa)
    mask = np.zeros(n, dtype=bool)
    if twa.notna().sum() < 2:
        return mask

    rad = np.radians(twa)
    before = np.sin(rad).rolling(MANEUVER_WINDOW_S, min_periods=MANEUVER_WINDOW_S // 2).mean()
    after = before.shift(-MANEUVER_WINDOW_S)
    cos_b = np.cos(rad).rolling(MANEUVER_WINDOW_S, min_periods=MANEUVER_WINDOW_S // 2).mean()
    cos_a = cos_b.shift(-MANEUVER_WINDOW_S)
    turn = _wrap180(np.degrees(np.arctan2(after, cos_a) - np.arctan2(before, cos_b))).abs()
    flip = before * after < 0
    confirmed = (
        (flip & (before.abs() >= MANEUVER_SIDE_MIN) & (after.abs() >= MANEUVER_SIDE_MIN))
        | (flip & (turn >= MANEUVER_TURN_MIN)
           & (np.maximum(before.abs(), after.abs()) >= MANEUVER_DEEP_SIDE))
    ).to_numpy()

    t = twa.ffill().bfill().to_numpy(dtype=float)
    unwrapped = np.degrees(np.unwrap(np.radians(t)))
    rate = (
        pd.Series(np.abs(np.diff(unwrapped, prepend=unwrapped[0])))
        .rolling(3, center=True, min_periods=1).mean().to_numpy()
    )
    sign = np.sign(t)

    i = 0
    while i < n:
        if not confirmed[i]:
            i += 1
            continue
        j = i
        while j < n and confirmed[j]:
            j += 1
        lo, hi = max(1, i - MANEUVER_WINDOW_S), min(n, j + MANEUVER_WINDOW_S)
        flips = lo + np.nonzero((sign[lo:hi] != sign[lo - 1:hi - 1]) & (sign[lo:hi] != 0))[0]
        # Dos maniobras seguidas (< 2·MANEUVER_WINDOW_S) dan un solo tramo confirmado:
        # se marcan todos los cruces confirmados, no solo el primero
        centers = [c for c in flips if confirmed[max(0, c - 3):c + 3].any()]
        if not centers:
            centers = [flips[0] if len(flips) else (i + j) // 2]
        for c in centers:
            s = c
            while s > 0 and c - s < MANEUVER_MAX_EXT_S and rate[s - 1] > MANEUVER_RATE_MIN:
                s -= 1
            e = c
            while e < n - 1 and e - c < MANEUVER_MAX_EXT_S and rate[e + 1] > MANEUVER_RATE_MIN:
                e += 1
            mask[max(0, s - MANEUVER_PAD_S): e + MANEUVER_PAD_S + 1] = True
        i = j
    return mask


def classify_phases(sog_kts: pd.Series, twa: pd.Series) -> pd.Series:
    """
    Clasifica la fase de navegación de cada muestra (1 Hz).
    - SOG y |TWA| se suavizan con una mediana móvil para eliminar ruido.
    - En vuelo: SOG suavizada ≥ FOIL_FLIGHT_KTS.
    - Viradas y trasluchadas (ver _maneuver_mask) → Transición.
    - Fuera de maniobra, la fase la decide solo |TWA|; sin dato de viento → Transición.
    - Un tramo de menos de PHASE_MIN_S entre dos tramos de la misma fase se absorbe.
    """
    sog = sog_kts.rolling(PHASE_SMOOTH_S, center=True, min_periods=1).median().to_numpy()
    twa_abs = twa.abs().rolling(PHASE_SMOOTH_S, center=True, min_periods=1).median().to_numpy()

    flying = sog >= FOIL_FLIGHT_KTS
    turning = _maneuver_mask(twa)

    phase = np.select(
        [~flying, turning, np.isnan(twa_abs),
         twa_abs < TWA_UPWIND_MAX, twa_abs > TWA_DOWNWIND_MIN],
        ["Caída", "Transición", "Transición", "Ceñida", "Popa"],
        default="Través",
    )
    phase = pd.Series(phase, index=sog_kts.index, dtype=object)

    run_id = (phase != phase.shift()).cumsum()
    runs = phase.groupby(run_id).agg(["first", "size"])
    labels, sizes = runs["first"].tolist(), runs["size"].tolist()
    for k in range(1, len(labels) - 1):
        if sizes[k] < PHASE_MIN_S and labels[k - 1] == labels[k + 1]:
            labels[k] = labels[k - 1]
    return run_id.map(dict(zip(runs.index, labels)))


def detect_device(filename: str, data: bytes):
    """Devuelve "sailmon", "atlas" o None según la extensión y la cabecera del archivo."""
    if filename.lower().endswith(".vkx"):
        return "atlas"
    header = data[:4096].decode("utf-8", errors="ignore").lstrip("﻿").splitlines()
    if not header:
        return None
    cols = [c.strip().strip('"').lower() for c in header[0].split(",")]
    if "timestamp" in cols and "sog_kts" in cols:
        return "atlas"
    if "time" in cols and any(c.startswith("sog") for c in cols):
        return "sailmon"
    return None


def _read_sailmon(data: bytes) -> pd.DataFrame:
    """Lee un CSV de Sailmon y normaliza sus nombres de columna al esquema interno."""
    df = pd.read_csv(io.BytesIO(data))
    df.columns = df.columns.str.strip().str.replace('"', '')

    # ── Normalizar columnas Sailmon (nombre completo → nombre corto) ──────
    # Mapeo: si el nombre de columna CONTIENE la clave, se renombra al valor.
    SAILMON_MAP = {
        "SOG":       "SOG",   # "SOG - Speed over Ground"
        "VMG":       "VMG",   # "VMG - Velocity Made Good"
        "TWA":       "TWA",   # "TWA - True Wind Angle"
        "TWD":       "TWD",   # "TWD - True Wind Direction"
        "HDT":       "HDT",   # "HDT - Heading True"
        "COG":       "COG",   # "COG - Course over Ground"
    }
    rename = {}
    for c in df.columns:
        cu = c.upper()
        # GPS
        if c.lower() in ("lat", "latitude"):
            rename[c] = "latitude"
        elif c.lower() in ("lon", "lng", "longitude"):
            rename[c] = "longitude"
        else:
            for key, short in SAILMON_MAP.items():
                if cu.startswith(key) and short not in df.columns and c != short:
                    rename[c] = short
                    break
    if rename:
        df.rename(columns=rename, inplace=True)
    if "time" not in df.columns or "SOG" not in df.columns:
        raise ValueError("no parece un export de Sailmon Max (faltan las columnas time / SOG)")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for col in ("COG", "HDT", "TWD"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# Columnas del Atlas CSV → esquema interno (el mismo que Sailmon)
ATLAS_CSV_MAP = {
    "latitude": "latitude", "longitude": "longitude", "cog": "COG",
    "hdg_true": "HDT", "heel": "Heel", "trim": "Trim Fore / Aft",
}
ATLAS_ANGLE_COLS = ("COG", "HDT", "Heel", "Trim Fore / Aft")

# Tamaño del payload de cada tipo de fila VKX (https://github.com/vakaros/vkx, v1.4)
VKX_ROW_SIZES = {
    0xFF: 7, 0xFE: 2, 0x01: 32, 0x02: 44, 0x03: 20, 0x04: 13, 0x05: 17, 0x06: 18,
    0x07: 12, 0x08: 13, 0x0A: 16, 0x0B: 16, 0x0C: 12, 0x0E: 16, 0x0F: 16, 0x10: 12,
    0x20: 13, 0x21: 52,
}
VKX_PVO = struct.Struct("<Qiifffffff")  # 0x02 Posición, velocidad y orientación


def _read_atlas_csv(data: bytes) -> pd.DataFrame:
    """Lee un CSV del Vakaros Atlas (2 Hz) al esquema interno, SOG en m/s y hora UTC."""
    raw = pd.read_csv(io.BytesIO(data))
    raw.columns = raw.columns.str.strip().str.replace('"', '')
    if "timestamp" not in raw.columns or "sog_kts" not in raw.columns:
        raise ValueError("no parece un CSV del Vakaros Atlas (faltan las columnas timestamp / sog_kts)")
    df = raw[[c for c in ATLAS_CSV_MAP if c in raw.columns]].rename(columns=ATLAS_CSV_MAP)
    df["time"] = pd.to_datetime(raw["timestamp"], errors="coerce", utc=True).dt.tz_localize(None)
    df["SOG"] = pd.to_numeric(raw["sog_kts"], errors="coerce") / MS_TO_KNOTS
    return df


def _read_vkx(data: bytes) -> pd.DataFrame:
    """Lee un .vkx del Vakaros Atlas (2 Hz) al esquema interno. Rumbo, escora y trimado se
    obtienen del cuaternión (marco NED verdadero) como ángulos de Euler ZYX, igual que el CSV."""
    rows, i = [], 0
    while i < len(data):
        key = data[i]
        size = VKX_ROW_SIZES.get(key)
        if size is None:
            raise ValueError(f"tipo de fila VKX desconocido 0x{key:02X} en el byte {i}")
        if key == 0x02 and i + 1 + size <= len(data):
            rows.append(VKX_PVO.unpack_from(data, i + 1))
        i += 1 + size
    if not rows:
        raise ValueError("el .vkx no contiene datos de posición")
    a = np.array(rows, dtype=float)
    w, x, y, z = a[:, 6], a[:, 7], a[:, 8], a[:, 9]
    return pd.DataFrame({
        "time":      pd.to_datetime(a[:, 0].astype("int64"), unit="ms"),
        "latitude":  a[:, 1] * 1e-7,
        "longitude": a[:, 2] * 1e-7,
        "SOG":       a[:, 3],
        "COG":       np.degrees(a[:, 4]) % 360,
        "HDT":       np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))) % 360,
        "Heel":      np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))),
        "Trim Fore / Aft": np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))),
    })


def _to_1hz(df: pd.DataFrame) -> pd.DataFrame:
    """Remuestrea a 1 Hz (todas las ventanas de la app cuentan muestras como segundos).
    Media aritmética para posición/velocidad y media circular para los ángulos."""
    d = df.dropna(subset=["time"]).set_index("time").sort_index()
    out = d[[c for c in d.columns if c not in ATLAS_ANGLE_COLS]].resample("1s").mean()
    for col in ATLAS_ANGLE_COLS:
        if col in d.columns:
            rad = np.radians(d[col])
            ang = np.degrees(np.arctan2(np.sin(rad).resample("1s").mean(),
                                        np.cos(rad).resample("1s").mean()))
            out[col] = ang % 360 if col in ("COG", "HDT") else ang
    return out.dropna(subset=["latitude", "longitude", "SOG"]).reset_index()


@st.cache_data(show_spinner=False)
def _read_atlas_raw(data: bytes, filename: str) -> pd.DataFrame:
    """Muestras originales (2 Hz) del Atlas, desde CSV o .vkx."""
    return _read_vkx(data) if filename.lower().endswith(".vkx") else _read_atlas_csv(data)


@st.cache_data(show_spinner=False)
def load_track(data: bytes, filename: str, device: str) -> pd.DataFrame:
    """Lee un archivo de cualquier dispositivo al esquema interno (hora UTC, SOG en m/s, 1 Hz)."""
    if device == "atlas":
        df = _to_1hz(_read_atlas_raw(data, filename))
    elif filename.lower().endswith(".vkx"):
        raise ValueError("un .vkx solo puede ser del Vakaros Atlas")
    else:
        df = _read_sailmon(data)
    df["Dispositivo"] = DEVICE_LABELS[device]
    return df


def _wrap180(deg):
    return (deg + 180) % 360 - 180


def wind_from_twd(twd, cog, sog):
    """Mismas fórmulas que Sailmon: TWA = TWD − COG (±180°) y VMG = SOG·cos(TWA)."""
    twa = _wrap180(twd - cog)
    return twa, sog * np.cos(np.radians(twa))


def apply_wind(df: pd.DataFrame, twd) -> pd.DataFrame:
    """Añade TWD, TWA y VMG (m/s) a un track sin viento a partir de una TWD (serie o constante)."""
    df = df.copy()
    df["TWD"] = twd
    df["TWA"], df["VMG"] = wind_from_twd(df["TWD"], df["COG"], df["SOG"])
    return df


def _symmetry_axis(cog_rad: np.ndarray) -> float:
    """Eje (0–180°, en rad) respecto al que la distribución de rumbos es más simétrica: los
    bordos de ceñida y popa en ambas amuras se reflejan en la línea del viento."""
    C = np.cos(np.outer(_TWD_HARMONICS, cog_rad)).mean(axis=1)
    S = np.sin(np.outer(_TWD_HARMONICS, cog_rad)).mean(axis=1)
    a = 2 * np.outer(_TWD_AXES, _TWD_HARMONICS)
    score = ((C * C - S * S) * np.cos(a) + 2 * C * S * np.sin(a)).sum(axis=1)
    return float(_TWD_AXES[np.argmax(score)])


def _twd_reference(df: pd.DataFrame):
    """TWD aproximada de toda la sesión, solo para orientar cada maniobra (virada o trasluchada).
    Eje: el de máxima simetría de los COG en vuelo. Sentido: barlovento es el lado con menor
    SOG media (en FK se va más rápido a popa). None si no hay datos suficientes."""
    sog_kts = df["SOG"] * MS_TO_KNOTS
    fl = df[(sog_kts >= TWD_MAN_MIN_KTS) & df["COG"].notna()]
    if len(fl) < TWD_REF_MIN_FLYING_S:
        return None
    cog = np.radians(fl["COG"].to_numpy(dtype=float))
    sog = fl["SOG"].to_numpy(dtype=float)
    axis = _symmetry_axis(cog)
    candidates = []
    for d in (axis, axis + np.pi):
        up = np.cos(d - cog) > 0  # |TWA| < 90°
        if up.any() and (~up).any():
            candidates.append((sog[up].mean() - sog[~up].mean(), d))
    return np.degrees(min(candidates)[1]) % 360 if candidates else None


def _window_stats(series: pd.Series, n: int, end: int) -> pd.Series:
    """Media de la ventana de n muestras que termina `end` muestras después de cada posición."""
    return series.rolling(n).mean().shift(-end)


@st.cache_data(show_spinner=False)
def estimate_twd(df: pd.DataFrame):
    """
    TWD por maniobras, como Sailmon: en cada virada/trasluchada la TWD es la bisectriz del COG
    medio antes y después (+180° si es trasluchada) y se aplica TWD_MAN_DELAY_S después del
    cambio de amura. Antes de la primera maniobra no hay viento.
    Solo cuentan maniobras limpias: en vuelo a ambos lados, rumbo estable antes y después,
    giro entre TWD_MAN_TURN_MIN y TWD_MAN_TURN_MAX y bisectriz a menos de TWD_MAN_MAX_DEV de la
    TWD vigente (descarta arribadas/orzadas en boya, cuya bisectriz es perpendicular al viento).
    Devuelve (serie de TWD alineada con df, nº de maniobras usadas, TWD de referencia).
    """
    ref = _twd_reference(df)
    if ref is None:
        return None, 0, None
    d = df.reset_index(drop=True)
    rad = np.radians(d["COG"].astype(float))
    sin, cos, kts = np.sin(rad), np.cos(rad), d["SOG"] * MS_TO_KNOTS
    nb, na = TWD_MAN_BEFORE_S, TWD_MAN_AFTER_S
    # Ventana "antes": [i-3-nb, i-4]; ventana "después": [i+3, i+2+na]
    sb, cb = _window_stats(sin, nb, -4), _window_stats(cos, nb, -4)
    sa, ca = _window_stats(sin, na, 2 + na), _window_stats(cos, na, 2 + na)
    before = np.degrees(np.arctan2(sb, cb))
    after = np.degrees(np.arctan2(sa, ca))
    sd_b = np.degrees(np.sqrt(-2 * np.log(np.hypot(sb, cb).clip(1e-9, 1))))
    sd_a = np.degrees(np.sqrt(-2 * np.log(np.hypot(sa, ca).clip(1e-9, 1))))
    turn = _wrap180(after - before).abs()
    span = (d["time"].shift(-(2 + na)) - d["time"].shift(3 + nb)).dt.total_seconds()
    ok = (
        turn.between(TWD_MAN_TURN_MIN, TWD_MAN_TURN_MAX)
        & (sd_b <= TWD_MAN_MAX_SD) & (sd_a <= TWD_MAN_MAX_SD)
        & (_window_stats(kts, nb, -4) >= TWD_MAN_MIN_KTS)
        & (_window_stats(kts, na, 2 + na) >= TWD_MAN_MIN_KTS)
        & (span <= 5 + nb + na + 2)  # sin huecos grandes en el registro
    ).to_numpy()

    twd = np.full(len(d), np.nan)
    idx = np.flatnonzero(ok)
    n_man, last = 0, ref
    for group in np.split(idx, np.flatnonzero(np.diff(idx) > 10) + 1) if len(idx) else []:
        i = group[np.argmax(turn.to_numpy()[group])]  # centro de la maniobra: giro máximo
        bis = np.degrees(np.arctan2(sb[i] + sa[i], cb[i] + ca[i])) % 360
        if abs(_wrap180(bis - last)) > 90:  # trasluchada: la bisectriz apunta a sotavento
            bis = (bis + 180) % 360
        if abs(_wrap180(bis - last)) > TWD_MAN_MAX_DEV:  # arribada/orzada: no cruza el viento
            continue
        twd[min(i + TWD_MAN_DELAY_S, len(d) - 1)] = last = bis
        n_man += 1
    if not n_man:
        return None, 0, ref
    return pd.Series(twd, index=df.index).ffill(), n_man, ref


def borrow_twd(df: pd.DataFrame, sailmon_dfs) -> pd.Series:
    """TWD de los Sailmon de la sesión interpolada a los tiempos de df (NaN si no hay dato
    de Sailmon a menos de TWD_BORROW_TOL_S)."""
    src = pd.concat(
        [d[["time", "TWD"]] for d in sailmon_dfs if "TWD" in d.columns], ignore_index=True
    ).dropna()
    if src.empty:
        return pd.Series(np.nan, index=df.index)
    rad = np.radians(src["TWD"].to_numpy(dtype=float))
    src = (
        pd.DataFrame({"sin": np.sin(rad), "cos": np.cos(rad)}, index=src["time"].dt.floor("1s"))
        .groupby(level=0).median()
    )
    epoch = pd.Timestamp(0)
    t_src = (src.index - epoch).total_seconds().to_numpy()
    t_dst = (df["time"] - epoch).dt.total_seconds().to_numpy()
    twd = np.degrees(np.arctan2(np.interp(t_dst, t_src, src["sin"]),
                                np.interp(t_dst, t_src, src["cos"]))) % 360
    nearest = np.searchsorted(t_src, t_dst).clip(1, len(t_src) - 1)
    gap = np.minimum(np.abs(t_dst - t_src[nearest - 1]), np.abs(t_dst - t_src[nearest]))
    return pd.Series(np.where(gap <= TWD_BORROW_TOL_S, twd, np.nan), index=df.index)


def finalize_track(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Preprocesado común a todos los dispositivos: nudos, fases, hora local y regatista."""
    df = df.copy()

    # Convertir m/s → nudos
    for col in ("SOG", "VMG"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[f"{col}_kts"] = df[col] * MS_TO_KNOTS

    # Convertir ángulos y Heel a numérico (pueden tener celdas vacías)
    for col in ("TWA", "Heel"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clasificar fases (TWA opcional: sin viento, en vuelo → Transición)
    if "SOG_kts" in df.columns:
        twa_col = df["TWA"] if "TWA" in df.columns else pd.Series(np.nan, index=df.index)
        df["Fase"] = classify_phases(df["SOG_kts"], twa_col)

    # Tiempo en UTC → hora local UTC+2
    if "time" in df.columns:
        df["time"] = df["time"] + LOCAL_UTC_OFFSET

    df["Regatista"] = name
    return df


def compute_kpis(df: pd.DataFrame) -> dict:
    k = {}
    # Velocidades y VMG solo en navegación limpia: sin maniobras (Transición) ni
    # caídas. Sin dato de viento todo el vuelo es Transición, así que entonces
    # solo se descartan las caídas.
    nav = df
    if "Fase" in df.columns:
        has_wind = "TWA" in df.columns and df["TWA"].notna().any()
        nav = df[df["Fase"].isin(["Popa", "Ceñida", "Través"])] if has_wind else df[df["Fase"] != "Caída"]
    if "SOG_kts" in nav.columns:
        k["sog_max"]  = nav["SOG_kts"].max()
        k["sog_mean"] = nav["SOG_kts"].mean()
    if "VMG_kts" in nav.columns:
        k["vmg_max"] = nav["VMG_kts"].max()
        # VMG es + en ceñida y − en popa: se promedia en valor absoluto
        k["vmg_mean"] = nav["VMG_kts"].abs().mean()
    if "Fase" in df.columns:
        n = len(df)
        k["pct_popa"]    = 100 * (df["Fase"] == "Popa").sum() / n
        k["pct_cenida"]  = 100 * (df["Fase"] == "Ceñida").sum() / n
        k["pct_traves"]  = 100 * (df["Fase"] == "Través").sum() / n
        k["pct_trans"]   = 100 * (df["Fase"] == "Transición").sum() / n
    return k


def leg_vmg_ref(df: pd.DataFrame) -> dict:
    """VMG normal (m/s, mediana de |VMG|) en Ceñida (+1) y Popa (−1), para descartar maniobras
    con entrada o salida a un VMG muy por debajo del habitual en la borda."""
    if not {"VMG", "Fase"} <= set(df.columns):
        return {}
    vmg = pd.to_numeric(df["VMG"], errors="coerce").abs()
    ref = {1: vmg[df["Fase"] == "Ceñida"].median(), -1: vmg[df["Fase"] == "Popa"].median()}
    return {k: float(v) for k, v in ref.items() if pd.notna(v)}


def maneuver_ghost(df: pd.DataFrame, i: int, vmg_ref=None, recovered=None, start=None):
    """
    Rival fantasma que no maniobra: hasta el inicio de la maniobra `start` (última muestra antes
    de que caiga la SOG; como tarde ENTRY_WINDOW_S[1] s antes del cambio de amura `i`) va exactamente con el
    barco, y desde ahí sigue recto con el rumbo GPS de entrada, a la velocidad que le da un
    VMG igual a la media entre el VMG de entrada (ENTRY_WINDOW_S) y el de salida (GHOST_EXIT_S)
    del barco. Así la pérdida es solo el coste de la maniobra, no el de cambiar de ángulo al
    viento antes o después. Se mide en el eje del viento (como la línea de líder del mapa de
    track), con la TWD de entrada fija: metros que el ghost avanza hacia barlovento (ceñida) o
    sotavento (popa) más que el barco, GHOST_POST_S s después del cambio de amura. La ventana
    devuelta empieza GHOST_PRE_S s antes del cambio de amura.
    Devuelve (ghost, None) o (None, motivo) si no es comparable: sin viento o GPS, ventana fuera
    del track, otra maniobra antes de acabar la ventana, rodeo de baliza, rumbo irregular
    (> GHOST_MAX_COURSE_DEV) o VMG por debajo de GHOST_MIN_VMG_FRAC del normal de la borda
    (`vmg_ref`, ver leg_vmg_ref) antes o después.
    `df` con índice 0..n-1 a 1 Hz.
    """
    if not {"time", "latitude", "longitude", "TWD", "TWA"} <= set(df.columns):
        return None, "sin viento o GPS"
    e0, e1 = ENTRY_WINDOW_S
    x0, x1 = GHOST_EXIT_S
    if recovered is not None and recovered + GHOST_REC_MARGIN_S > i + x0:
        # salida medida ya recuperado: si no, el ghost frenaría con la recuperación lenta
        x0 = recovered + GHOST_REC_MARGIN_S - i
        x1 = x0 + GHOST_EXIT_S[1] - GHOST_EXIT_S[0]
    a, b = i - GHOST_PRE_S, i + GHOST_POST_S
    end = max(b, i + x1)
    if a < 0 or a > i - e0 or end >= len(df):
        return None, "demasiado cerca del inicio o fin del tramo"

    lat = df["latitude"].to_numpy(float)[a:end + 1]
    lon = df["longitude"].to_numpy(float)[a:end + 1]
    twa = df["TWA"].to_numpy(float)
    twd = np.radians(df["TWD"].to_numpy(float)[i - e0:i - e1])
    cos_in = np.nanmean(np.cos(np.radians(twa[i - e0:i - e1])))
    cos_out = np.nanmean(np.cos(np.radians(twa[i + x0:i + x1])))
    if (not np.isfinite(twd).any() or not np.isfinite([cos_in, cos_out]).all()
            or not np.isfinite(lat).all() or not np.isfinite(lon).all()):
        return None, "sin viento o GPS"
    after = twa[i + MANEUVER_WINDOW_S:i + x1]
    if (np.sign(after[1:]) * np.sign(after[:-1]) < 0).any():  # otro cambio de amura
        return None, f"otra maniobra antes de {x1} s"
    if cos_in * cos_out < 0:
        return None, "rodeo de baliza"
    twd0 = float(np.arctan2(np.nanmean(np.sin(twd)), np.nanmean(np.cos(twd))))
    sign = 1.0 if cos_in > 0 else -1.0  # ceñida: gana el de barlovento; popa: el de sotavento
    ux, uy = sign * np.sin(twd0), sign * np.cos(twd0)

    # Plano local en metros con origen en la salida del ghost (misma aproximación que drawLadder)
    kx, ky = np.cos(np.radians(lat[0])) * 111320, 110540
    e, n = (lon - lon[0]) * kx, (lat - lat[0]) * ky
    up = e * ux + n * uy
    secs = (df["time"].iloc[a:end + 1] - df["time"].iloc[a]).dt.total_seconds().to_numpy()

    def _course(j0, j1):
        """Rumbo medio (cuerda) y desviación máx. del rumbo segundo a segundo en [j0, j1]."""
        k0, k1 = j0 - a, j1 - a
        hdg = np.arctan2(e[k1] - e[k0], n[k1] - n[k0])
        steps = np.arctan2(np.diff(e[k0:k1 + 1]), np.diff(n[k0:k1 + 1]))
        dev = np.abs(_wrap180(np.degrees(steps - hdg))).max()
        vmg = (up[k1] - up[k0]) / (secs[k1] - secs[k0])
        return hdg, dev, vmg

    hdg, dev_in, vmg_in = _course(i - e0, i - e1)
    _, dev_out, vmg_out = _course(i + x0, i + x1)
    if max(dev_in, dev_out) > GHOST_MAX_COURSE_DEV:
        return None, "rumbo irregular antes o después"
    ref = (vmg_ref or {}).get(int(sign))
    if ref and min(vmg_in, vmg_out) < GHOST_MIN_VMG_FRAC * ref:
        return None, "VMG bajo antes o después"
    along = np.sin(hdg) * ux + np.cos(hdg) * uy  # fracción del rumbo del ghost que es VMG
    if along < 0.1:
        return None, "rumbo irregular antes o después"

    # Hasta el inicio de la maniobra el ghost es el barco; desde ahí, recto con el rumbo de
    # entrada y SOG tal que su VMG sea la media entrada/salida
    ks = (max(a, min(start, i - e1)) if start is not None else i - e1) - a
    vmg_target = (vmg_in + vmg_out) / 2
    speed = vmg_target / along
    t = np.maximum(secs[:b - a + 1] - secs[ks], 0)
    ge = np.where(t > 0, e[ks] + speed * t * np.sin(hdg), e[:b - a + 1])
    gn = np.where(t > 0, n[ks] + speed * t * np.cos(hdg), n[:b - a + 1])
    loss = np.where(t > 0, vmg_target * t - (up[:b - a + 1] - up[ks]), 0.0)
    return {
        "a": a, "b": b, "start": ks,
        "twd": float(np.degrees(twd0) % 360),
        "twa": float(np.nanmean(twa[i - e0:i - e1])),
        "sog_kts": float(speed * MS_TO_KNOTS),
        "vmg_kts": float(sign * vmg_target * MS_TO_KNOTS),
        "lat": lat[0] + gn / ky,
        "lon": lon[0] + ge / kx,
        "loss": loss,
    }, None


def detect_maneuvers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta viradas y trasluchadas por cambio de signo en TWA.
    Para cada maniobra calcula:
      - SOG antes: media entre ENTRY_WINDOW_S s antes del cambio de amura (antes de frenar)
      - SOG mínima (ventana ±15s)
      - Caída de SOG
      - Recovery time: segundos desde que la SOG empieza a caer (< DROP_START_FRAC de la de
        entrada) hasta recuperar RECOVERY_FRAC de la velocidad de referencia: la menor entre la
        de entrada y la de salida (una trasluchada en la boya de sotavento que sale a ceñida no
        puede recuperar la velocidad de popa)
      - Pérdida (m): metros que saca en el eje del viento un ghost que sigue recto sin maniobrar
        (ver maneuver_ghost), GHOST_POST_S s después del cambio de amura. NaN en las fallidas o
        si no es comparable (`_motivo`). `_i` es el índice del cambio de amura, para
        reconstruir el ghost.
    """
    if "TWA" not in df.columns or "SOG_kts" not in df.columns:
        return pd.DataFrame()

    df = df.reset_index(drop=True)
    twa = df["TWA"].values.astype(float)
    sog = df["SOG_kts"].values.astype(float)

    rows = []
    i = 1
    GAP = 8  # mínimo de segundos entre maniobras detectadas
    vmg_ref = leg_vmg_ref(df)

    while i < len(twa) - 1:
        prev, curr = twa[i - 1], twa[i]
        if not (np.isnan(prev) or np.isnan(curr)) and prev * curr < 0:
            # SOG de entrada, antes de empezar a frenar
            e0, e1 = ENTRY_WINDOW_S
            entry = sog[max(0, i - e0):max(0, i - e1)]
            if not np.isfinite(entry).any():
                i += 1
                continue
            sog_before = float(np.nanmean(entry))

            # Descartar si el foil no estaba volando antes de la maniobra
            if sog_before < FOIL_FLIGHT_KTS:
                i += 1
                continue

            # Ventana ±15s para SOG mínima
            w0, w1 = max(0, i - 15), min(len(df), i + 15)
            min_idx = w0 + int(np.nanargmin(sog[w0:w1]))
            min_sog = float(sog[min_idx])

            # Tipo: virada (upwind) o trasluchada (downwind)
            avg_abs = float(np.nanmean(np.abs(twa[max(0, i - 5):i])))
            mtype = "Virada" if avg_abs <= 90 else "Trasluchada"

            # Recovery time: desde el inicio de la caída hasta recuperar la velocidad de entrada
            drop_start = min_idx
            while drop_start > max(0, i - 25) and sog[drop_start - 1] < DROP_START_FRAC * sog_before:
                drop_start -= 1
            x0, x1 = EXIT_WINDOW_S
            exit_win = sog[i + x0:i + x1]
            target = sog_before
            exit_sog = float(np.nanmedian(exit_win)) if np.isfinite(exit_win).any() else np.nan
            if exit_sog >= FOIL_FLIGHT_KTS:  # si sale sin volar (caída), se mide contra la entrada
                target = min(sog_before, exit_sog)
            rec = ">90"
            for j in range(min_idx, min(len(sog), min_idx + 90)):
                if sog[j] >= RECOVERY_FRAC * target:
                    rec = j - drop_start
                    break

            failed = min_sog < FALL_KTS
            recovered = drop_start + rec if isinstance(rec, int) else None
            ghost, motivo = ((None, "caída") if failed
                             else maneuver_ghost(df, i, vmg_ref, recovered, drop_start - 1))
            row = {
                "Tipo":            mtype,
                "Estado":          "🔴 Fallida" if failed else "✅ OK",
                "SOG antes (kts)": round(sog_before, 1),
                "SOG mín (kts)":   round(min_sog, 1),
                "Caída (kts)":     round(sog_before - min_sog, 1),
                "Recovery (s)":    rec,
                "Pérdida (m)":     round(float(ghost["loss"][-1]), 1) if ghost else np.nan,
                "_i":              i,
                "_motivo":         motivo,
                "_recovered":      recovered,
                "_start":          drop_start - 1,  # última muestra antes de frenar
            }
            if "time" in df.columns:
                row["Tiempo"] = df["time"].iloc[i]
            if "latitude" in df.columns:
                row["lat"] = float(df["latitude"].iloc[i])
            if "longitude" in df.columns:
                row["lon"] = float(df["longitude"].iloc[i])

            rows.append(row)
            if min_sog < FALL_KTS:
                # Tras una caída el TWA oscila con el regatista en el agua: no buscar otra
                # maniobra hasta que vuelva a volar de forma sostenida (no un pico suelto)
                flying = np.convolve(sog[min_idx:] >= FOIL_FLIGHT_KTS,
                                     np.ones(REFLY_S, dtype=int), "valid") == REFLY_S
                back = np.flatnonzero(flying)
                i = max(i + GAP, min_idx + int(back[0])) if back.size else len(twa)
            else:
                i += GAP
            continue
        i += 1

    return pd.DataFrame(rows)


# ─── Constructores de gráficos ───────────────────────────────────────────────
def _dark_layout(extra=None):
    base = dict(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1a1a2e",
        font=dict(color="#e2e8f0"),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(color="white")),
    )
    if extra:
        base.update(extra)
    return base


def build_map(dfs, color_by="Velocidad", view=None):
    """`view`, si se pasa, fija el centro/zoom inicial (p. ej. calculado una sola vez
    sobre la traza completa) en vez de recalcularlo del subconjunto `dfs` actual —
    así el mapa no se recentra cada vez que `dfs` cambia (p. ej. al mover el slider
    de tramo)."""
    fig = go.Figure()

    for idx, df in enumerate(dfs):
        name  = df["Regatista"].iloc[0]
        color = RACER_PALETTE[idx % len(RACER_PALETTE)]

        def _build_customdata(src):
            """Construye customdata con cols: [0]SOG [1]VMG [2]TWA [3]Fase [4]Hora [5]Heel."""
            n = pd.RangeIndex(len(src))
            _idx = src.index

            def _fmt(series, spec):
                return series.apply(lambda v: f"{v:{spec}}" if pd.notna(v) else "—")

            sog  = _fmt(src["SOG_kts"], ".1f") if "SOG_kts" in src.columns else pd.Series(["—"] * len(src), index=_idx)
            vmg  = _fmt(src["VMG_kts"], ".1f") if "VMG_kts" in src.columns else pd.Series(["—"] * len(src), index=_idx)
            twa  = _fmt(src["TWA"],     ".1f") if "TWA"     in src.columns else pd.Series(["—"] * len(src), index=_idx)
            fase = src["Fase"].fillna("—")      if "Fase"   in src.columns else pd.Series(["—"] * len(src), index=_idx)
            hora = (src["time"].dt.strftime("%H:%M:%S").fillna("—")
                    if "time" in src.columns else pd.Series(["—"] * len(src), index=_idx))
            heel = _fmt(src["Heel"],   ".1f") if "Heel"    in src.columns else pd.Series(["—"] * len(src), index=_idx)
            return np.column_stack([sog.values, vmg.values, twa.values,
                                    fase.values, hora.values, heel.values])

        if color_by == "Velocidad" and "SOG_kts" in df.columns:
            fig.add_trace(go.Scattermap(
                lat=df["latitude"], lon=df["longitude"], mode="markers",
                marker=dict(
                    size=5,
                    color=df["SOG_kts"],
                    colorscale="Viridis",
                    showscale=(idx == 0),
                    colorbar=dict(title="SOG (kts)") if idx == 0 else None,
                    cmin=0,
                    cmax=float(df["SOG_kts"].quantile(0.99)),
                ),
                name=name,
                customdata=_build_customdata(df),
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    "SOG: %{customdata[0]} kts<br>"
                    "VMG: %{customdata[1]} kts<br>"
                    "TWA: %{customdata[2]}°<br>"
                    "Fase: %{customdata[3]}<br>"
                    "Heel: %{customdata[5]}°<br>"
                    "⏱ %{customdata[4]}<extra></extra>"
                ),
            ))
        elif color_by == "Heel" and "Heel" in df.columns:
            _heel_max = float(df["Heel"].abs().quantile(0.99)) or 1.0
            fig.add_trace(go.Scattermap(
                lat=df["latitude"], lon=df["longitude"], mode="markers",
                marker=dict(
                    size=5,
                    color=df["Heel"],
                    colorscale="RdBu_r",
                    showscale=(idx == 0),
                    colorbar=dict(title="Heel (°)") if idx == 0 else None,
                    cmin=-_heel_max,
                    cmax=_heel_max,
                    cmid=0,
                ),
                name=name,
                customdata=_build_customdata(df),
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    "Heel: %{customdata[5]}°<br>"
                    "SOG: %{customdata[0]} kts<br>"
                    "VMG: %{customdata[1]} kts<br>"
                    "TWA: %{customdata[2]}°<br>"
                    "Fase: %{customdata[3]}<br>"
                    "⏱ %{customdata[4]}<extra></extra>"
                ),
            ))
        elif color_by == "Fase" and "Fase" in df.columns:
            for fase, col in PHASE_COLORS.items():
                sub = df[df["Fase"] == fase]
                if sub.empty:
                    continue
                fig.add_trace(go.Scattermap(
                    lat=sub["latitude"], lon=sub["longitude"], mode="markers",
                    marker=dict(size=5, color=col),
                    name=f"{name} – {fase}",
                    customdata=_build_customdata(sub),
                    hovertemplate=(
                        f"<b>{name}</b> · {fase}<br>"
                        "SOG: %{customdata[0]} kts<br>"
                        "VMG: %{customdata[1]} kts<br>"
                        "TWA: %{customdata[2]}°<br>"
                        "Heel: %{customdata[5]}°<br>"
                        "⏱ %{customdata[4]}<extra></extra>"
                    ),
                ))
        else:
            fig.add_trace(go.Scattermap(
                lat=df["latitude"], lon=df["longitude"], mode="markers",
                marker=dict(size=5, color=color), name=name,
                customdata=_build_customdata(df),
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    "SOG: %{customdata[0]} kts<br>"
                    "VMG: %{customdata[1]} kts<br>"
                    "TWA: %{customdata[2]}°<br>"
                    "Heel: %{customdata[5]}°<br>"
                    "⏱ %{customdata[4]}<extra></extra>"
                ),
            ))

    if view is not None:
        center = dict(lat=view["lat"], lon=view["lon"])
        zoom = view["zoom"]
    else:
        all_lat = pd.concat([d["latitude"] for d in dfs])
        all_lon = pd.concat([d["longitude"] for d in dfs])
        center = dict(lat=float(all_lat.mean()), lon=float(all_lon.mean()))
        zoom = 12
    fig.update_layout(
        map=carto_mapbox(
            center=center,
            zoom=zoom,
            uirevision="race_map",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
        paper_bgcolor="#0e1117",
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)", font=dict(color="white"),
            x=0.01, y=0.99, xanchor="left", yanchor="top",
        ),
    )
    return fig


def build_polar(dfs):
    fig = go.Figure()
    BIN = 5
    for idx, df in enumerate(dfs):
        if "TWA" not in df.columns or "SOG_kts" not in df.columns:
            continue
        tmp = df.copy()
        if "Fase" in tmp.columns:
            tmp = tmp[tmp["Fase"].isin(["Ceñida", "Popa", "Través"])]
        tmp["TWA_abs"] = tmp["TWA"].abs()
        tmp["TWA_bin"] = (tmp["TWA_abs"] // BIN) * BIN + BIN / 2
        polar = tmp.groupby("TWA_bin")["SOG_kts"].mean().reset_index()

        fig.add_trace(go.Scatterpolar(
            r=polar["SOG_kts"],
            theta=polar["TWA_bin"],
            mode="lines+markers",
            name=df["Regatista"].iloc[0],
            line=dict(color=RACER_PALETTE[idx % len(RACER_PALETTE)], width=2),
            marker=dict(size=4),
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(title="SOG media (kts)", gridcolor="#2d3748", color="#a0aec0"),
            angularaxis=dict(direction="clockwise", rotation=90, gridcolor="#2d3748", color="#a0aec0"),
            bgcolor="#1a1a2e",
        ),
        **_dark_layout({
            "height": 460,
            "title": dict(text="Polar Comparativa · SOG media por ángulo de viento", font=dict(color="#e2e8f0")),
        }),
    )
    return fig


def build_speed_timeline(dfs):
    fig = go.Figure()
    for idx, df in enumerate(dfs):
        if "SOG_kts" not in df.columns:
            continue
        x = df["time"] if "time" in df.columns else df.index
        fig.add_trace(go.Scatter(
            x=x, y=df["SOG_kts"], mode="lines",
            name=df["Regatista"].iloc[0],
            line=dict(color=RACER_PALETTE[idx % len(RACER_PALETTE)], width=1.5),
        ))
    for kts, label, color in [
        (FOIL_FLIGHT_KTS, f"Umbral de vuelo ({FOIL_FLIGHT_KTS} kts)", "#FFA15A"),
    ]:
        fig.add_hline(
            y=kts, line_dash="dot", line_color=color, opacity=0.5,
            annotation_text=label, annotation_font_color=color,
        )
    fig.update_layout(
        xaxis=dict(title="Tiempo", gridcolor="#2d3748", color="#a0aec0"),
        yaxis=dict(title="SOG (kts)", gridcolor="#2d3748", color="#a0aec0"),
        **_dark_layout({
            "height": 350,
            "title": dict(text="Velocidad en el Tiempo", font=dict(color="#e2e8f0")),
        }),
    )
    return fig


def build_maneuver_recovery_chart(man_df):
    if man_df.empty:
        return None
    fig = go.Figure()
    for mtype, color in [("Virada", "#00CC96"), ("Trasluchada", "#EF553B")]:
        sub = man_df[man_df["Tipo"] == mtype]
        if sub.empty:
            continue
        num_r = pd.to_numeric(sub["Recovery (s)"], errors="coerce")
        fig.add_trace(go.Bar(
            x=[mtype], y=[num_r.mean()],
            name=mtype, marker_color=color,
            text=[f"{num_r.mean():.0f}s"], textposition="outside",
            error_y=dict(type="data", array=[num_r.std()], visible=True, color="#a0aec0"),
        ))
    fig.update_layout(
        yaxis=dict(title="Recovery time (s)", gridcolor="#2d3748", color="#a0aec0"),
        xaxis=dict(color="#a0aec0"),
        showlegend=False,
        **_dark_layout({
            "height": 280,
            "title": dict(text="Recovery Time medio por tipo de maniobra", font=dict(color="#e2e8f0")),
        }),
    )
    return fig


def build_vmg_polar(dfs):
    """Polar de VMG medio por ángulo de viento (TWA absoluto, bins de 5°)."""
    fig = go.Figure()
    BIN = 5
    for idx, df in enumerate(dfs):
        if "TWA" not in df.columns or "VMG_kts" not in df.columns:
            continue
        tmp = df.dropna(subset=["TWA", "VMG_kts"]).copy()
        if "Fase" in tmp.columns:
            tmp = tmp[tmp["Fase"].isin(["Ceñida", "Popa", "Través"])]
        tmp["TWA_abs"]  = tmp["TWA"].abs()
        tmp["VMG_abs"]  = tmp["VMG_kts"].abs()
        tmp["TWA_bin"]  = (tmp["TWA_abs"] // BIN) * BIN + BIN / 2
        polar = tmp.groupby("TWA_bin")["VMG_abs"].mean().reset_index()
        fig.add_trace(go.Scatterpolar(
            r=polar["VMG_abs"],
            theta=polar["TWA_bin"],
            mode="lines+markers",
            name=df["Regatista"].iloc[0],
            line=dict(color=RACER_PALETTE[idx % len(RACER_PALETTE)], width=2),
            marker=dict(size=4),
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(title="VMG media (kts)", gridcolor="#2d3748", color="#a0aec0"),
            angularaxis=dict(direction="clockwise", rotation=90,
                             gridcolor="#2d3748", color="#a0aec0"),
            bgcolor="#1a1a2e",
        ),
        **_dark_layout({
            "height": 460,
            "title": dict(
                text="Polar de VMG · Velocidad hacia el destino por ángulo",
                font=dict(color="#e2e8f0"),
            ),
        }),
    )
    return fig


def build_phase_boxplot(dfs):
    """Box plot de SOG agrupado por Fase y regatista."""
    fig = go.Figure()
    phase_order = ["Popa", "Ceñida", "Través"]
    for idx, df in enumerate(dfs):
        if "SOG_kts" not in df.columns or "Fase" not in df.columns:
            continue
        color = RACER_PALETTE[idx % len(RACER_PALETTE)]
        filtered = df[df["Fase"].isin(phase_order)]
        fig.add_trace(go.Box(
            x=filtered["Fase"],
            y=filtered["SOG_kts"],
            name=df["Regatista"].iloc[0],
            marker_color=color,
            line_color=color,
            boxmean="sd",          # muestra media ± desviación típica
            legendgroup=df["Regatista"].iloc[0],
        ))
    fig.update_layout(
        xaxis=dict(
            title="Fase",
            categoryorder="array",
            categoryarray=phase_order,
            gridcolor="#2d3748", color="#a0aec0",
        ),
        yaxis=dict(title="SOG (kts)", gridcolor="#2d3748", color="#a0aec0"),
        boxmode="group",
        **_dark_layout({
            "height": 460,
            "title": dict(
                text="Consistencia por Fase · Distribución de Velocidad",
                font=dict(color="#e2e8f0"),
            ),
        }),
    )
    return fig


def build_heel_analysis(dfs):
    """
    SOG media binned por ángulo de Heel, separado por fase (Ceñida / Popa).
    Color por regatista, estilo de línea por fase.
    """
    fig = go.Figure()
    BIN = 2  # bins de 2 grados

    # Fase → estilo de línea (color lo pone el regatista)
    FASE_DASH = {
        "Ceñida": "solid",
        "Popa":   "dash",
        "Través": "dot",
    }

    any_data = False
    for idx, df in enumerate(dfs):
        if "Heel" not in df.columns or "SOG_kts" not in df.columns or "Fase" not in df.columns:
            continue

        name  = df["Regatista"].iloc[0]
        color = RACER_PALETTE[idx % len(RACER_PALETTE)]
        symbol = ["circle", "square", "diamond", "cross"][idx % 4]

        tmp = df[["Heel", "SOG_kts", "Fase"]].dropna()
        tmp = tmp[tmp["SOG_kts"] >= FOIL_FLIGHT_KTS].copy()
        if tmp.empty:
            continue

        tmp["Heel"] = tmp["Heel"].abs()  # babor y estribor equivalentes
        tmp["Heel_bin"] = (tmp["Heel"] // BIN) * BIN + BIN / 2

        for fase, dash in FASE_DASH.items():
            sub = tmp[tmp["Fase"] == fase]
            if sub.empty:
                continue
            stats = (
                sub.groupby("Heel_bin")["SOG_kts"]
                .agg(media="mean", n="count")
                .reset_index()
            )
            stats = stats[stats["n"] >= 10]
            if stats.empty:
                continue
            any_data = True
            label = f"{name} · {fase}"
            fig.add_trace(go.Scatter(
                x=stats["Heel_bin"],
                y=stats["media"],
                mode="lines+markers",
                name=label,
                line=dict(color=color, dash=dash, width=2),
                marker=dict(size=5, symbol=symbol, color=color),
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "Heel: %{x:.0f}°<br>"
                    "SOG media: %{y:.1f} kts<extra></extra>"
                ),
            ))

    if not any_data:
        return None

    fig.update_layout(
        xaxis=dict(
            title="Heel (°)  ·  valor absoluto (babor y estribor combinados)",
            gridcolor="#2d3748", color="#a0aec0",
        ),
        yaxis=dict(title="SOG media (kts)", gridcolor="#2d3748", color="#a0aec0"),
        **_dark_layout({
            "height": 420,
            "title": dict(
                text=f"Escora vs Velocidad por Fase · SOG media (foil en vuelo ≥ {FOIL_FLIGHT_KTS} kts)",
                font=dict(color="#e2e8f0"),
            ),
        }),
    )
    return fig


def build_maneuver_sog_chart(man_df):
    """Gráfico de caída de SOG por maniobra."""
    if man_df.empty:
        return None
    fig = go.Figure()
    for mtype, color in [("Virada", "#00CC96"), ("Trasluchada", "#EF553B")]:
        sub = man_df[man_df["Tipo"] == mtype].reset_index(drop=True)
        if sub.empty:
            continue
        x_labels = [f"{mtype} {i+1}" for i in range(len(sub))]
        fig.add_trace(go.Bar(
            x=x_labels, y=sub["Caída (kts)"],
            name=mtype, marker_color=color, opacity=0.8,
        ))
    fig.update_layout(
        xaxis=dict(title="Maniobra", gridcolor="#2d3748", color="#a0aec0", tickangle=-45),
        yaxis=dict(title="Caída SOG (kts)", gridcolor="#2d3748", color="#a0aec0"),
        barmode="group",
        **_dark_layout({
            "height": 280,
            "title": dict(text="Caída de Velocidad por Maniobra", font=dict(color="#e2e8f0")),
        }),
    )
    return fig


# ─── Animated replay map ─────────────────────────────────────────────────────
def build_animated_map(dfs, step_s=15, duration_ms=700,
                        trail_behind_s=60, trail_ahead_s=30):
    """
    Mapa de replay animado.
    - Trail de ±N segundos alrededor de la posición actual (sin track completo).
    - Dot grande para posición actual, dots pequeños para el trail.
    - SOG en hover. Sin texto ni icono sobre el marcador.
    - Botones Play y Pausa separados (más fiables que toggle).
    """
    valid = [d for d in dfs
             if "time" in d.columns and "latitude" in d.columns
             and d["time"].notna().any()]
    if not valid:
        return None

    all_t  = pd.concat([d["time"].dropna() for d in valid])
    t_min  = all_t.min()
    t_max  = all_t.max()
    timestamps = pd.date_range(t_min, t_max, freq=f"{step_s}s")

    racer_list = [(idx, df) for idx, df in enumerate(dfs)
                  if "latitude" in df.columns and "time" in df.columns]
    n_racers = len(racer_list)
    all_trace_indices = list(range(2 * n_racers))

    # Precompute sorted dfs y posiciones de barco
    positions  = {}
    sorted_dfs = {}
    for idx, df in racer_list:
        df_s = (df.dropna(subset=["latitude", "longitude", "time"])
                  .sort_values("time").reset_index(drop=True))
        sorted_dfs[idx] = df_s
        positions[idx] = (
            df_s.set_index("time")[["latitude", "longitude"]]
                .reindex(timestamps, method="ffill")
        )

    def _trail_slices(idx, t):
        """Devuelve (lats, lons, sogs) para la ventana de trail."""
        df_s = sorted_dfs[idx]
        t_ns = t.value
        tarr = df_s["time"].values.astype(np.int64)
        lo   = int(np.searchsorted(tarr, t_ns - int(trail_behind_s * 1e9)))
        hi   = int(np.searchsorted(tarr, t_ns + int(trail_ahead_s  * 1e9), side="right"))
        seg  = df_s.iloc[lo:hi]
        if seg.empty:
            return [None], [None], [None]
        lats = seg["latitude"].tolist()
        lons = seg["longitude"].tolist()
        sogs = (seg["SOG_kts"].round(1).tolist()
                if "SOG_kts" in seg.columns else [None] * len(lats))
        return lats, lons, sogs

    def _boat_pos(idx, t):
        row = positions[idx].loc[t]
        lat_, lon_ = row["latitude"], row["longitude"]
        ok = not (pd.isna(lat_) or pd.isna(lon_))
        return ([float(lat_)], [float(lon_)]) if ok else ([None], [None])

    # ── Traces iniciales (t0) — incluyen info de leyenda ─────────────────────
    t0 = timestamps[0]
    initial_data = []
    for idx, df in racer_list:
        color = RACER_PALETTE[idx % len(RACER_PALETTE)]
        name  = df["Regatista"].iloc[0]
        lats, lons, sogs = _trail_slices(idx, t0)
        blat, blon       = _boat_pos(idx, t0)
        initial_data.append(go.Scattermap(
            lat=lats, lon=lons,
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=3, color=color),
            customdata=sogs,
            hovertemplate=f"<b>{name}</b><br>SOG: %{{customdata}} kts<extra></extra>",
            opacity=0.9,
            name=name, legendgroup=name, showlegend=True,
        ))
        initial_data.append(go.Scattermap(
            lat=blat, lon=blon,
            mode="markers",
            marker=dict(size=14, color=color, opacity=1),
            hovertemplate=f"<b>{name}</b><extra></extra>",
            legendgroup=name, showlegend=False,
        ))

    # ── Frames ───────────────────────────────────────────────────────────────
    frames       = []
    slider_steps = []
    for t in timestamps:
        frame_data = []
        for idx, df in racer_list:
            color = RACER_PALETTE[idx % len(RACER_PALETTE)]
            name  = df["Regatista"].iloc[0]
            lats, lons, sogs = _trail_slices(idx, t)
            blat, blon       = _boat_pos(idx, t)
            # Trail — sin props de leyenda para que Plotly preserve las iniciales
            frame_data.append(go.Scattermap(
                lat=lats, lon=lons,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=3, color=color),
                customdata=sogs,
                hovertemplate=f"<b>{name}</b><br>SOG: %{{customdata}} kts<extra></extra>",
                opacity=0.9,
            ))
            # Barco — dot grande
            frame_data.append(go.Scattermap(
                lat=blat, lon=blon,
                mode="markers",
                marker=dict(size=14, color=color, opacity=1),
                hovertemplate=f"<b>{name}</b><extra></extra>",
            ))

        t_label = t.strftime("%H:%M:%S")
        frames.append(go.Frame(
            data=frame_data, name=t_label, traces=all_trace_indices,
        ))
        slider_steps.append(dict(
            args=[[t_label], {"frame": {"duration": duration_ms, "redraw": True},
                              "mode": "immediate"}],
            label=t_label, method="animate",
        ))

    fig = go.Figure(data=initial_data, frames=frames)

    all_lat = pd.concat([d["latitude"].dropna() for d in dfs if "latitude" in d.columns])
    all_lon = pd.concat([d["longitude"].dropna() for d in dfs if "longitude" in d.columns])

    fig.update_layout(
        map=carto_mapbox(
            center=dict(lat=float(all_lat.mean()), lon=float(all_lon.mean())),
            zoom=12,
        ),
        margin=dict(l=0, r=0, t=50, b=60),
        height=600,
        paper_bgcolor="#0e1117",
        legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(color="white")),
        uirevision="animated_map",
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.07, x=0.0, xanchor="left",
            pad=dict(r=10, t=5),
            buttons=[
                dict(
                    label="▶ Play",
                    method="animate",
                    args=[None, {
                        "frame": {"duration": duration_ms, "redraw": True},
                        "fromcurrent": True, "mode": "immediate",
                        "transition": {"duration": 0},
                    }],
                ),
                dict(
                    label="⏸ Pausa",
                    method="animate",
                    args=[[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                    }],
                ),
            ],
        )],
        sliders=[dict(
            active=0,
            steps=slider_steps,
            x=0.0, y=0, len=1.0,
            pad=dict(b=10, t=5),
            currentvalue=dict(
                prefix="⏱ ",
                visible=True,
                xanchor="center",
                font=dict(color="#e2e8f0", size=14),
            ),
            bgcolor="#1a1a2e",
            bordercolor="#4a5568",
            activebgcolor="#e94560",
            font=dict(color="#a0aec0", size=8),
            ticklen=3,
        )],
    )
    return fig


# ─── Legs & Peak speed ───────────────────────────────────────────────────────
def _haversine_total(lats, lons) -> float:
    """Distancia total recorrida en millas náuticas."""
    valid = [d for d in dfs
             if "time" in d.columns and "latitude" in d.columns
             and d["time"].notna().any()]
    if not valid:
        return None

    all_t  = pd.concat([d["time"].dropna() for d in valid])
    t_min  = all_t.min()
    t_max  = all_t.max()
    timestamps = pd.date_range(t_min, t_max, freq=f"{step_s}s")

    racer_list = [(idx, df) for idx, df in enumerate(dfs)
                  if "latitude" in df.columns and "time" in df.columns]
    n_racers = len(racer_list)
    all_trace_indices = list(range(2 * n_racers))  # trail + boat per racer

    # Precompute: sorted df por tiempo + posiciones de barco en cada timestamp
    positions  = {}
    sorted_dfs = {}
    for idx, df in racer_list:
        df_s = (df.dropna(subset=["latitude", "longitude", "time"])
                  .sort_values("time")
                  .reset_index(drop=True))
        sorted_dfs[idx] = df_s
        positions[idx] = (
            df_s.set_index("time")[["latitude", "longitude"]]
                .reindex(timestamps, method="ffill")
        )

    def _build_traces(t, include_legend=False):
        traces = []
        t_ns  = t.value
        lo_ns = t_ns - int(trail_behind_s * 1_000_000_000)
        hi_ns = t_ns + int(trail_ahead_s  * 1_000_000_000)

        for i, (idx, df) in enumerate(racer_list):
            color = RACER_PALETTE[idx % len(RACER_PALETTE)]
            name  = df["Regatista"].iloc[0]
            df_s  = sorted_dfs[idx]
            tarr  = df_s["time"].values.astype(np.int64)
            lo    = int(np.searchsorted(tarr, lo_ns, side="left"))
            hi    = int(np.searchsorted(tarr, hi_ns, side="right"))
            trail = df_s.iloc[lo:hi]

            # Trail (ventana temporal)
            tr = go.Scattermap(
                lat=trail["latitude"].tolist(),
                lon=trail["longitude"].tolist(),
                mode="lines",
                line=dict(color=color, width=3),
                opacity=0.85,
            )
            if include_legend:
                tr.name       = name
                tr.legendgroup = name
                tr.showlegend  = True
            traces.append(tr)

            # Barco
            row = positions[idx].loc[t]
            lat_, lon_ = row["latitude"], row["longitude"]
            ok = not (pd.isna(lat_) or pd.isna(lon_))
            bt = go.Scattermap(
                lat=[float(lat_)] if ok else [None],
                lon=[float(lon_)] if ok else [None],
                mode="markers+text",
                marker=dict(size=16, color=color, opacity=1),
                text=[f"⛵ {name}"] if ok else [""],
                textposition="top right",
                textfont=dict(color="white", size=11),
                showlegend=False,
            )
            if include_legend:
                bt.legendgroup = name
            traces.append(bt)

        return traces

    initial_data = _build_traces(timestamps[0], include_legend=True)

    frames = []
    slider_steps = []
    for t in timestamps:
        t_label = t.strftime("%H:%M:%S")
        frames.append(go.Frame(
            data=_build_traces(t),
            name=t_label,
            traces=all_trace_indices,
        ))
        slider_steps.append(dict(
            args=[[t_label], {"frame": {"duration": duration_ms, "redraw": True},
                              "mode": "immediate"}],
            label=t_label,
            method="animate",
        ))

    fig = go.Figure(data=initial_data, frames=frames)

    all_lat = pd.concat([d["latitude"].dropna() for d in dfs if "latitude" in d.columns])
    all_lon = pd.concat([d["longitude"].dropna() for d in dfs if "longitude" in d.columns])

    fig.update_layout(
        map=carto_mapbox(
            center=dict(lat=float(all_lat.mean()), lon=float(all_lon.mean())),
            zoom=12,
        ),
        margin=dict(l=0, r=0, t=50, b=60),
        height=600,
        paper_bgcolor="#0e1117",
        legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(color="white")),
        uirevision="animated_map",
        updatemenus=[dict(
            type="buttons",
            showactive=True,        # resalta cuando está reproduciendo
            y=1.07, x=0.0, xanchor="left",
            pad=dict(r=10, t=5),
            buttons=[
                dict(
                    label="▶ Play",
                    method="animate",
                    # 1º clic → reproduce
                    args=[None, {
                        "frame": {"duration": duration_ms, "redraw": True},
                        "fromcurrent": True, "mode": "immediate",
                        "transition": {"duration": 0},
                    }],
                    # 2º clic → pausa (toggle)
                    args2=[[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                    }],
                ),
            ],
        )],
        sliders=[dict(
            active=0,
            steps=slider_steps,
            x=0.0, y=0, len=1.0,
            pad=dict(b=10, t=5),
            currentvalue=dict(
                prefix="⏱ ",
                visible=True,
                xanchor="center",
                font=dict(color="#e2e8f0", size=14),
            ),
            bgcolor="#1a1a2e",
            bordercolor="#4a5568",
            activebgcolor="#e94560",
            font=dict(color="#a0aec0", size=8),
            ticklen=3,
        )],
    )
    return fig


# ─── Legs ─────────────────────────────────────────────────────────────────────
def _haversine_total(lats, lons) -> float:
    """Distancia total recorrida en millas náuticas."""
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    if len(lats) < 2:
        return 0.0
    R_nm = 3440.065
    rlat = np.radians(lats)
    rlon = np.radians(lons)
    dlat = np.diff(rlat)
    dlon = np.diff(rlon)
    a = np.sin(dlat / 2) ** 2 + np.cos(rlat[:-1]) * np.cos(rlat[1:]) * np.sin(dlon / 2) ** 2
    return float(np.sum(R_nm * 2 * np.arcsin(np.sqrt(a).clip(0, 1))))


def detect_legs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta bordes continuos de Ceñida, Popa o Través (mínimo 5 s) y calcula:
    duración, SOG media, VMG media y distancia recorrida.
    """
    if "Fase" not in df.columns or "SOG_kts" not in df.columns:
        return pd.DataFrame()

    tmp = df.copy().reset_index(drop=True)
    tmp["_ph"] = tmp["Fase"].where(tmp["Fase"].isin(["Ceñida", "Popa", "Través"]))
    tmp["_lg"] = (tmp["_ph"] != tmp["_ph"].shift()).cumsum()

    rows = []
    leg_num = 0
    for _, grp in tmp.dropna(subset=["_ph"]).groupby("_lg", sort=True):
        if len(grp) < 5:
            continue
        leg_num += 1
        fase     = grp["_ph"].iloc[0]
        dur_s    = len(grp)
        sog_mean = grp["SOG_kts"].mean()
        vmg_mean = grp["VMG_kts"].mean() if "VMG_kts" in grp.columns else float("nan")

        has_gps  = "latitude" in grp.columns and "longitude" in grp.columns
        if has_gps:
            dist_nm = _haversine_total(
                grp["latitude"].ffill().values,
                grp["longitude"].ffill().values,
            )
        else:
            dist_nm = sog_mean * dur_s / 3600.0

        if "time" in grp.columns and not grp["time"].isna().all():
            t0 = grp["time"].iloc[0]
            inicio = t0.strftime("%H:%M:%S") if pd.notna(t0) else "—"
        else:
            inicio = "—"

        rows.append({
            "#":                leg_num,
            "Fase":             fase,
            "Inicio":           inicio,
            "Duración":         f"{dur_s // 60}:{dur_s % 60:02d}",
            "SOG media (kts)":  round(sog_mean, 1),
            "VMG media (kts)":  round(vmg_mean, 1) if not pd.isna(vmg_mean) else None,
            "Distancia (m)":    round(dist_nm * 1852, 0),
        })

    return pd.DataFrame(rows)


def compute_peak_speeds(df: pd.DataFrame, windows=(10, 30, 60)) -> dict:
    """Mejor SOG media en ventanas deslizantes de N segundos (1 Hz)."""
    if "SOG_kts" not in df.columns:
        return {}
    result = {}
    for w in windows:
        rolled = df["SOG_kts"].rolling(w, min_periods=w).mean()
        val = rolled.max()
        result[w] = round(float(val), 2) if not pd.isna(val) else None
    return result


# ─── Estado de la sesión ─────────────────────────────────────────────────────
def _fkey(f) -> str:
    """Identificador estable de un archivo subido (las keys de sus widgets no dependen
    de su posición, así que quitar o añadir archivos no mezcla nombres ni ajustes)."""
    return f"{f.name}_{f.size}"


def _reset_app():
    """Vuelve a empezar de cero: vacía el uploader (cambiando su key) y todo el estado."""
    nonce = st.session_state.get("_uploader_nonce", 0)
    st.session_state.clear()
    st.session_state["_uploader_nonce"] = nonce + 1


def _open_trim():
    """Reabre el selector de tramo partiendo del tramo actual."""
    st.session_state.pop("trim_confirmed_for", None)
    st.session_state["_offer_last_range"] = False
    st.session_state["_trim_nonce"] = st.session_state.get("_trim_nonce", 0) + 1


def _use_full_session():
    st.session_state["trim_range"] = None


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🪁 Formula Kite")
    st.caption("Dashboard de Telemetría · v1.0")
    st.divider()

    uploaded_files = st.file_uploader(
        "Archivos (Sailmon Max CSV · Vakaros Atlas CSV/VKX)",
        type=["csv", "vkx"],
        accept_multiple_files=True,
        help="Un archivo por regatista. El dispositivo se detecta automáticamente.",
        key=f"uploader_{st.session_state.get('_uploader_nonce', 0)}",
    )

    racer_names, racer_devices = [], []
    wind_box = None
    if uploaded_files:
        st.button(
            "🗑️ Reiniciar (quitar archivos)", on_click=_reset_app, use_container_width=True,
        )
        st.divider()
        st.subheader("Regatistas")
        for i, f in enumerate(uploaded_files):
            default = os.path.splitext(f.name)[0].replace("_", " ").title()
            name = st.text_input(f"Regatista {i + 1}", value=default, key=f"n_{_fkey(f)}")
            detected = detect_device(f.name, f.getvalue())
            choice = st.selectbox(
                "Dispositivo", ["Auto", *DEVICE_LABELS.values()], key=f"dev_{_fkey(f)}",
            )
            if choice == "Auto":
                st.caption(f"Detectado: **{DEVICE_LABELS.get(detected, 'desconocido')}**")
                device = detected
            else:
                device = next(k for k, v in DEVICE_LABELS.items() if v == choice)
            racer_names.append(name)
            racer_devices.append(device)
        wind_box = st.container()

        if st.session_state.get("trim_confirmed_for"):
            st.button("🔁 Cambiar tramo seleccionado", on_click=_open_trim)

    st.divider()
    st.markdown(
        "**Fases de Navegación:**\n\n"
        f"🔴 **Popa** · en vuelo · TWA > {TWA_DOWNWIND_MIN}°  \n"
        f"🟢 **Ceñida** · en vuelo · TWA < {TWA_UPWIND_MAX}°  \n"
        f"🔵 **Través** · en vuelo · TWA {TWA_UPWIND_MAX}–{TWA_DOWNWIND_MIN}°  \n"
        "🟠 **Transición** · virada/trasluchada (cambio de amura) o sin dato de viento  \n"
        "🟣 **Caída** · fuera de vuelo\n\n"
        f"_En vuelo: SOG ≥ {FOIL_FLIGHT_KTS} kts._"
    )


# ─── Landing page (sin datos) ─────────────────────────────────────────────────
if not uploaded_files:
    st.title("🪁 Formula Kite Analytics")
    st.markdown(
        """
        Analiza y compara la telemetría de regatistas de **Formula Kite** con datos de
        **Sailmon Max** y **Vakaros Atlas**.

        **← Carga uno o más archivos** desde el panel lateral para comenzar. El dispositivo
        se detecta automáticamente (también se puede elegir a mano).

        ---

        ### Formatos admitidos

        | Dispositivo | Archivo | Frecuencia | Viento |
        |-------------|---------|------------|--------|
        | Sailmon Max | export `.csv` (`time`, `SOG`, `COG`, `TWA`, `TWD`, `VMG`…) | 1 Hz | incluido |
        | Vakaros Atlas | export `.csv` (`timestamp`, `sog_kts`, `cog`, `hdg_true`, `heel`, `trim`) o binario `.vkx` | 2 Hz → se remuestrea a 1 Hz | calculado |

        > El Atlas no registra viento. Su **TWD** se copia de un Sailmon de la misma sesión,
        > se calcula **por maniobras como Sailmon** (bisectriz del rumbo antes y después de cada
        > virada/trasluchada) o se introduce a mano. **TWA** y **VMG** usan las mismas fórmulas
        > que Sailmon: TWA = TWD − COG y VMG = SOG · cos(TWA).
        """
    )
    st.stop()


# ─── Carga de datos ───────────────────────────────────────────────────────────
tracks = []  # (índice, archivo, nombre, dispositivo, df en esquema interno)
for i, (f, nm, dev) in enumerate(zip(uploaded_files, racer_names, racer_devices)):
    with st.spinner(f"Procesando {f.name}…"):
        try:
            if dev is None:
                raise ValueError("formato no reconocido; elige el dispositivo en el panel lateral")
            tracks.append((i, f, nm, dev, load_track(f.getvalue(), f.name, dev)))
        except Exception as exc:
            st.error(f"❌ Error al cargar **{f.name}**: {exc}")

# Viento común: una sola TWD para todos los regatistas, para que TWA, VMG y fases sean
# coherentes entre ellos. Se toma de los Sailmon, por maniobras o manual, y se aplica a
# todos los tracks (también a los Sailmon, cuya TWA/VMG se recalculan con esa TWD).
def _circ_mean(deg) -> float:
    rad = np.radians(np.asarray(deg, dtype=float))
    rad = rad[~np.isnan(rad)]
    return float(np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360)


if tracks:
    _sailmon_raw = {
        _fkey(f): (nm, df) for _, f, nm, dev, df in tracks
        if dev == "sailmon" and "TWD" in df.columns and df["TWD"].notna().any()
    }
    _wind_labels = {}
    if len(_sailmon_raw) > 1:
        _wind_labels["sailmon_all"] = "Media de los Sailmon"
    for fk, (nm, _) in _sailmon_raw.items():
        _wind_labels[f"sailmon:{fk}"] = f"Sailmon · {nm}"
    _wind_labels["estimate"] = "Por maniobras (todos los regatistas)"
    _wind_labels["manual"] = "Manual (constante)"

    # Estimación combinada: TWD por maniobras de cada track, mediana circular por segundo
    _estimates = []
    for _, _, _, _, df in tracks:
        _est, _n, _ = estimate_twd(df)
        if _est is not None:
            _estimates.append((pd.DataFrame({"time": df["time"], "TWD": _est}), _n))
    _n_man = sum(n for _, n in _estimates)

    with wind_box:
        st.divider()
        st.subheader("Viento (común a todos)")
        src = st.selectbox(
            "Origen de la TWD", list(_wind_labels), format_func=_wind_labels.get, key="wind_src",
        )
        if src.startswith("sailmon"):
            _src_dfs = [d for _, d in _sailmon_raw.values()] if src == "sailmon_all" \
                else [_sailmon_raw[src.split(":", 1)[1]][1]]
            twd_for = lambda df: borrow_twd(df, _src_dfs)
        elif src == "estimate":
            twd_for = (lambda df: borrow_twd(df, [d for d, _ in _estimates])) if _estimates else (lambda df: None)
        else:
            _all_est = pd.concat([d["TWD"] for d, _ in _estimates]) if _estimates else []
            _default = _circ_mean(_all_est) if len(_all_est) else 0.0
            _twd_const = st.number_input(
                "TWD (°)", min_value=0.0, max_value=359.0, step=1.0,
                value=float(round(_default)) % 360, key="wind_twd",
            )
            twd_for = lambda df: _twd_const

        _coverage = []
        for k, (i, f, nm, dev, df) in enumerate(tracks):
            twd = twd_for(df) if "COG" in df.columns else None
            if twd is None or (isinstance(twd, pd.Series) and twd.isna().all()):
                # Sin TWD común para este track: se descarta también su viento propio
                df = df.drop(columns=["TWD", "TWA", "VMG"], errors="ignore")
                tracks[k] = (i, f, nm, dev, df)
                _coverage.append((nm, 0.0))
            else:
                tracks[k] = (i, f, nm, dev, apply_wind(df, twd))
                _coverage.append((nm, twd.notna().mean() if isinstance(twd, pd.Series) else 1.0))

        if src == "estimate" and _estimates:
            st.caption(
                f"TWD de **{_n_man}** maniobras de todos los regatistas · media "
                f"**{_circ_mean(pd.concat([d['TWD'] for d, _ in _estimates])):.0f}°**. "
                "Sin viento hasta la primera maniobra."
            )
        if src != "manual":
            st.caption("Cobertura: " + " · ".join(f"{nm} {c:.0%}" for nm, c in _coverage))
        if any(c == 0 for _, c in _coverage):
            st.info("Sin viento en algún track: sus fases solo distinguen En vuelo / Caída, sin maniobras ni polares.")

dfs_full = []
files_full = []  # archivo original de cada df en dfs_full (para exportar el recorte)
for _, f, nm, dev, df in tracks:
    dfs_full.append(finalize_track(df, nm))
    files_full.append(f)

if not dfs_full:
    st.warning("No se pudieron cargar archivos válidos.")
    st.stop()


# ─── Helper UI ───────────────────────────────────────────────────────────────
def section(title):
    st.markdown(f'<p class="section-title">{title}</p>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PASO 2 · SELECCIONA EL TRAMO A ANALIZAR (modal, antes de mostrar cualquier análisis)
# ═══════════════════════════════════════════════════════════════════════════════
_has_gps_full = all("latitude" in d.columns and "longitude" in d.columns for d in dfs_full)
_files_key = tuple((f.name, f.size) for f in uploaded_files)

# Si han cambiado los archivos, descarta el rango recordado de la sesión anterior
# (podría quedar fuera de los límites de tiempo del nuevo archivo). El selector
# usa una key derivada de _files_key, así que se remonta solo con el archivo nuevo.
if st.session_state.get("_trim_files_key") != _files_key:
    st.session_state.pop("trim_range", None)
    st.session_state["_trim_files_key"] = _files_key
    st.session_state["_offer_last_range"] = True


@st.dialog("✂️ Selecciona el tramo a analizar", width="large")
def _trim_dialog():
    st.caption(
        "Recorta la sesión al tramo que quieres analizar (p. ej. una prueba completa) "
        "y descarta el resto. **Arrastra el bloque verde** para mover todo el tramo "
        "(misma duración), o **arrastra uno de sus bordes** para ajustar solo ese "
        "extremo. El mapa de abajo se actualiza con cada ajuste."
    )

    _time_dfs = [d for d in dfs_full if "time" in d.columns and d["time"].notna().any()]
    if not _time_dfs:
        st.info("Los archivos no tienen columna de tiempo; se usará la sesión completa.")
        if st.button("Continuar", type="primary"):
            st.session_state["trim_range"] = None
            st.session_state["trim_confirmed_for"] = _files_key
            st.rerun()
        return

    _all_t = pd.concat([d["time"].dropna() for d in _time_dfs])
    _t_min, _t_max = _all_t.min(), _all_t.max()
    _duration_ms = (_t_max - _t_min).total_seconds() * 1000
    _start_clock_s = _t_min.hour * 3600 + _t_min.minute * 60 + _t_min.second

    # Si ya se había confirmado un tramo antes (p. ej. al añadir otro regatista a
    # una sesión ya recortada), ofrece reutilizarlo directamente en vez de tener
    # que volver a arrastrar el slider.
    _last_range = st.session_state.get("_last_confirmed_range")
    if (_last_range and not st.session_state.get("trim_range")
            and st.session_state.get("_offer_last_range")):
        _lo, _hi = _last_range
        if _lo < _t_max and _hi > _t_min:  # solapa con la sesión de los archivos actuales
            _clamped_lo = max(_lo, _t_min)
            _clamped_hi = min(_hi, _t_max)
            st.info(
                "📎 Ya habías recortado un tramo antes: "
                f"**{_clamped_lo.strftime('%H:%M:%S')} → {_clamped_hi.strftime('%H:%M:%S')}**"
            )
            if st.button("✅ Usar ese mismo tramo para todos los regatistas", type="primary"):
                st.session_state["trim_range"] = (_clamped_lo, _clamped_hi)
                st.session_state["_last_confirmed_range"] = (_clamped_lo, _clamped_hi)
                st.session_state["trim_confirmed_for"] = _files_key
                st.rerun()
            st.divider()

    _prev_range = st.session_state.get("trim_range")
    if _prev_range:
        _default_start_ms = (_prev_range[0] - _t_min).total_seconds() * 1000
        _default_end_ms = (_prev_range[1] - _t_min).total_seconds() * 1000
    else:
        _default_start_ms, _default_end_ms = 0, _duration_ms

    _series = []
    for i, d in enumerate(dfs_full):
        if "time" not in d.columns or "SOG_kts" not in d.columns:
            continue
        dd = d[["time", "SOG_kts"]].dropna()
        if len(dd) > 700:
            dd = dd.iloc[:: max(1, len(dd) // 700)]
        _series.append({
            "name": str(d["Regatista"].iloc[0]) if "Regatista" in d.columns else f"Regatista {i + 1}",
            "color": RACER_PALETTE[i % len(RACER_PALETTE)],
            "points": [
                [(t - _t_min).total_seconds() * 1000, float(v)]
                for t, v in zip(dd["time"], dd["SOG_kts"])
            ],
        })

    _result = time_range_selector(
        series=_series,
        duration_ms=_duration_ms,
        start_clock_s=_start_clock_s,
        start_ms=_default_start_ms,
        end_ms=_default_end_ms,
        # El nonce remonta el componente en cada apertura, para que arranque en el
        # tramo actual (el JS solo lee start_ms/end_ms en el primer render).
        key=f"trim_range_selector_{hash(_files_key)}_{st.session_state.get('_trim_nonce', 0)}",
    )

    if _result:
        _t0 = _t_min + pd.Timedelta(milliseconds=_result["start_ms"])
        _t1 = _t_min + pd.Timedelta(milliseconds=_result["end_ms"])
    else:
        _t0 = _t_min + pd.Timedelta(milliseconds=_default_start_ms)
        _t1 = _t_min + pd.Timedelta(milliseconds=_default_end_ms)

    _preview = [d[d["time"].between(_t0, _t1)].copy() for d in dfs_full if "time" in d.columns]
    _preview = [d for d in _preview if not d.empty]

    if not _preview:
        st.warning("Sin datos en el tramo seleccionado.")
    elif _has_gps_full:
        # Centro/zoom fijos, calculados sobre la traza COMPLETA (no el tramo
        # visible), para que el mapa no se recentre al mover el slider.
        _full_lat = pd.concat([d["latitude"] for d in dfs_full])
        _full_lon = pd.concat([d["longitude"] for d in dfs_full])
        _map_view = {
            "lat": float(_full_lat.mean()),
            "lon": float(_full_lon.mean()),
            "zoom": 12,
        }
        st.plotly_chart(
            build_map(_preview, "Velocidad", view=_map_view),
            use_container_width=True,
            config={"scrollZoom": True},
            key="trim_preview_map",
        )
    else:
        st.info("Los archivos no tienen GPS; no se puede previsualizar en el mapa.")
        st.plotly_chart(
            build_speed_timeline(_preview),
            use_container_width=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Confirmar tramo y analizar", type="primary", disabled=not _preview):
            st.session_state["trim_range"] = (_t0, _t1)
            st.session_state["_last_confirmed_range"] = (_t0, _t1)
            st.session_state["trim_confirmed_for"] = _files_key
            st.rerun()
    with c2:
        if st.button("Usar sesión completa"):
            st.session_state["trim_range"] = None
            st.session_state["trim_confirmed_for"] = _files_key
            st.rerun()


if st.session_state.get("trim_confirmed_for") != _files_key:
    _trim_dialog()
    # Por si se cierra el diálogo con la X: deja un botón para reabrirlo.
    st.info("Elige el tramo de la sesión que quieres analizar.")
    st.button("✂️ Seleccionar tramo", type="primary", on_click=_open_trim)
    st.stop()

_trim_range = st.session_state.get("trim_range")
if _trim_range:
    _t0, _t1 = _trim_range
    dfs = [d[d["time"].between(_t0, _t1)].copy() for d in dfs_full if "time" in d.columns]
    dfs = [d for d in dfs if not d.empty]
else:
    dfs = dfs_full

if not dfs:
    st.warning("No hay datos en el tramo seleccionado.")
    st.stop()

def _vkx_as_atlas_csv(data: bytes) -> pd.DataFrame:
    """Muestras de un .vkx con las columnas del CSV del Atlas (el binario no se puede recortar)."""
    raw = _read_atlas_raw(data, "track.vkx")
    out = raw.rename(columns={v: k for k, v in ATLAS_CSV_MAP.items()})
    out.insert(0, "timestamp", raw["time"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + "+0000")
    out.insert(3, "sog_kts", (raw["SOG"] * MS_TO_KNOTS).round(3))
    return out[["timestamp", "latitude", "longitude", "sog_kts", "cog", "hdg_true", "heel", "trim"]]


def _raw_trimmed_csv(file, trim_range) -> bytes:
    """Filas del archivo original (mismas columnas, hora en UTC) dentro del tramo, para que
    el recorte se pueda volver a subir como un export normal. Un .vkx sale como CSV del Atlas."""
    data = file.getvalue()
    if file.name.lower().endswith(".vkx"):
        raw = _vkx_as_atlas_csv(data)
    else:
        raw = pd.read_csv(io.BytesIO(data))
    if trim_range:
        time_col = next(c for c in raw.columns if c.strip().replace('"', "") in ("time", "timestamp"))
        utc_t = pd.to_datetime(raw[time_col], errors="coerce", utc=True).dt.tz_localize(None)
        raw = raw[(utc_t + LOCAL_UTC_OFFSET).between(*trim_range)]
    return raw.to_csv(index=False).encode("utf-8")


_bar_info, _bar_change, _bar_full = st.columns([3, 1, 1])
with _bar_info:
    if _trim_range:
        _mins = (_trim_range[1] - _trim_range[0]).total_seconds() / 60
        st.markdown(
            f"✂️ **Tramo:** {_trim_range[0].strftime('%H:%M:%S')} → "
            f"{_trim_range[1].strftime('%H:%M:%S')} ({_mins:.0f} min)"
        )
    else:
        st.markdown("✂️ **Tramo:** sesión completa")
with _bar_change:
    st.button("🔁 Cambiar tramo", on_click=_open_trim, use_container_width=True, key="bar_change_trim")
with _bar_full:
    if _trim_range:
        st.button("Usar sesión completa", on_click=_use_full_session,
                  use_container_width=True, key="bar_full_session")

with st.expander("⬇️ Descargar CSV recortado por regatista"):
    st.caption(
        "Mismo formato que el export original (los .vkx se descargan como CSV del Atlas): "
        "puedes volver a subirlo a la app tal cual."
    )
    for _dl_i, (f_dl, df_dl) in enumerate(zip(files_full, dfs_full)):
        _dl_name = df_dl["Regatista"].iloc[0]
        st.download_button(
            f"Descargar {_dl_name}_recortado.csv",
            data=_raw_trimmed_csv(f_dl, _trim_range),
            file_name=f"{_dl_name}_recortado.csv",
            mime="text/csv",
            key=f"dl_{_dl_i}",
        )

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 · KPIs
# ═══════════════════════════════════════════════════════════════════════════════
section("📊 Indicadores Clave por Regatista")

cols = st.columns(len(dfs))
for col, df in zip(cols, dfs):
    name = df["Regatista"].iloc[0]
    k = compute_kpis(df)
    _kts = lambda key: f"{k[key]:.1f} kts" if pd.notna(k.get(key)) else "—"
    with col:
        st.markdown(f"#### {name}")
        st.caption("Sin maniobras ni caídas.")
        c1, c2 = st.columns(2)
        c1.metric("🚀 Vel. Máx.", _kts("sog_max"))
        c2.metric("🎯 Vel. Media", _kts("sog_mean"))
        c3, c4 = st.columns(2)
        c3.metric("💨 VMG Máx.", _kts("vmg_max"))
        c4.metric("🧭 VMG Medio", _kts("vmg_mean"))
        if "Fase" in df.columns:
            st.markdown(
                f"🔴 Popa **{k.get('pct_popa', 0):.1f}%** &nbsp;·&nbsp; "
                f"🟢 Ceñida **{k.get('pct_cenida', 0):.1f}%** &nbsp;·&nbsp; "
                f"🔵 Través **{k.get('pct_traves', 0):.1f}%** &nbsp;·&nbsp; "
                f"🟠 Trans. **{k.get('pct_trans', 0):.1f}%**",
                unsafe_allow_html=True,
            )

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 · MAPA
# ═══════════════════════════════════════════════════════════════════════════════
has_gps = all("latitude" in d.columns and "longitude" in d.columns for d in dfs)

if has_gps:
    section("🗺️ Mapa de Tracks GPS")

    st.caption(
        "El mapa muestra solo el trazado por delante de la posición actual. "
        "Arrastra el cursor de la gráfica inferior o pulsa ▶ para avanzar. "
        "La brújula marca hacia dónde sopla el viento (TWD) y la línea perpendicular al viento "
        "marca al líder, con la distancia que saca a cada uno medida en el eje del viento (como Sailmon)."
    )
    _replay_racers, _replay_dur, _replay_clock = build_replay_racers(dfs)
    track_replay(
        _replay_racers, _replay_dur, _replay_clock,
        key=f"track_replay_{hash(_files_key)}_{hash(str(st.session_state.get('trim_range')))}",
    )

else:
    st.info("ℹ️ Los archivos no contienen columnas de GPS (latitude/longitude). El mapa no está disponible.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 · VELOCIDAD EN EL TIEMPO
# ═══════════════════════════════════════════════════════════════════════════════
section("📈 Velocidad en el Tiempo")
st.plotly_chart(build_speed_timeline(dfs), use_container_width=True)
st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 · ESCORA (HEEL) VS VELOCIDAD
# ═══════════════════════════════════════════════════════════════════════════════
has_heel = any("Heel" in d.columns for d in dfs)
if has_heel:
    section("⛵ Escora (Heel) vs Velocidad")
    st.caption(
        f"SOG media agrupada en bins de 2° de escora, filtrada solo cuando el foil vuela (≥ {FOIL_FLIGHT_KTS} kts). "
        "Permite identificar el **ángulo de Heel óptimo** para maximizar velocidad."
    )
    heel_fig = build_heel_analysis(dfs)
    if heel_fig:
        st.plotly_chart(heel_fig, use_container_width=True)
    else:
        st.info("Sin suficientes datos de Heel para el análisis.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 · VELOCIDAD PICO + TABLA DE BORDES
# ═══════════════════════════════════════════════════════════════════════════════

# ── Velocidad pico ────────────────────────────────────────────────────────────
section("⚡ Velocidad Pico (Ventana Deslizante)")
st.caption(
    "Mejor velocidad media sostenida en ventanas de 10 s, 30 s y 60 s. "
    "Más representativo que el máximo puntual, que puede ser un pico de ruido GPS. "
    "Un buen 60 s indica capacidad de mantener velocidad alta de forma consistente."
)
peak_cols = st.columns(len(dfs))
for col, df in zip(peak_cols, dfs):
    name  = df["Regatista"].iloc[0]
    peaks = compute_peak_speeds(df)
    with col:
        st.markdown(f"**{name}**")
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("10 s",  f"{peaks[10]} kts"  if peaks.get(10)  else "—")
        pc2.metric("30 s",  f"{peaks[30]} kts"  if peaks.get(30)  else "—")
        pc3.metric("60 s",  f"{peaks[60]} kts"  if peaks.get(60)  else "—")

st.divider()

# ── Tabla de bordes ───────────────────────────────────────────────────────────
section("🏁 Tabla de Bordes (Legs)")
st.caption(
    "Cada fila es una borda continua de Ceñida 🟢, Través 🔵 o Popa 🔴 (mínimo 5 segundos). "
    "Permite ver las ventajas tácticas: qué borda fue más rápida, cuánto se progresó "
    "y si el VMG fue consistente. Bordes cortos suelen indicar maniobras o cambios tácticos."
)

for df in dfs:
    name = df["Regatista"].iloc[0]
    legs = detect_legs(df)
    with st.expander(
        f"**{name}** — {len(legs)} borda{'s' if len(legs) != 1 else ''}",
        expanded=True,
    ):
        if legs.empty:
            st.info("Sin bordes detectados. Se necesitan columnas Fase y SOG_kts.")
            continue

        cenidas  = legs[legs["Fase"] == "Ceñida"]
        popas    = legs[legs["Fase"] == "Popa"]
        traveses = legs[legs["Fase"] == "Través"]
        lk1, lk2, lk3, lk4, lk5 = st.columns(5)
        lk1.metric("Bordes Ceñida", len(cenidas))
        lk2.metric("Bordes Través",  len(traveses))
        lk3.metric("Bordes Popa",   len(popas))
        lk4.metric(
            "SOG media Ceñida",
            f"{cenidas['SOG media (kts)'].mean():.1f} kts" if not cenidas.empty else "—",
        )
        lk5.metric(
            "SOG media Popa",
            f"{popas['SOG media (kts)'].mean():.1f} kts" if not popas.empty else "—",
        )

        col_cfg_legs = {
            "#":                st.column_config.NumberColumn("#", format="%d"),
            "Fase":             st.column_config.TextColumn("Fase"),
            "Inicio":           st.column_config.TextColumn("Inicio", help="Hora UTC de inicio de la borda."),
            "Duración":         st.column_config.TextColumn("Duración", help="Formato MM:SS"),
            "SOG media (kts)":  st.column_config.NumberColumn("SOG media (kts)", format="%.1f kts"),
            "VMG media (kts)":  st.column_config.NumberColumn("VMG media (kts)", format="%.1f kts"),
            "Distancia (m)":    st.column_config.NumberColumn("Distancia (m)", format="%.0f m",
                                    help="Distancia total recorrida en la borda (haversine GPS)."),
        }

        def _color_leg(row):
            f = row["Fase"]
            c = ("background-color: rgba(0,204,150,0.12)"  if f == "Ceñida"
                 else "background-color: rgba(239,85,59,0.12)"   if f == "Popa"
                 else "background-color: rgba(25,211,243,0.12)"  if f == "Través"
                 else "")
            return [c] * len(row)

        st.dataframe(
            legs.style.apply(_color_leg, axis=1),
            column_config=col_cfg_legs,
            use_container_width=True,
            hide_index=True,
        )

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6 · DETECCIÓN DE MANIOBRAS
# ═══════════════════════════════════════════════════════════════════════════════
section("🔄 Detección de Maniobras (Viradas & Trasluchadas)")

for df in dfs:
    name = df["Regatista"].iloc[0]
    man = detect_maneuvers(df)

    with st.expander(
        f"**{name}** — {len(man)} maniobra{'s' if len(man) != 1 else ''} detectada{'s' if len(man) != 1 else ''}",
        expanded=True,
    ):
        if man.empty:
            st.info(
                "No se detectaron maniobras. "
                "Verifica que la columna TWA contenga valores positivos (estribor) "
                "y negativos (babor) alternados."
            )
            continue

        # KPIs de maniobras
        tacks    = man[man["Tipo"] == "Virada"]
        gybes    = man[man["Tipo"] == "Trasluchada"]
        failed   = man[man["Estado"] == "🔴 Fallida"] if "Estado" in man.columns else pd.DataFrame()
        num_rec  = pd.to_numeric(man["Recovery (s)"], errors="coerce")

        _loss_mean = lambda sub: (f"{sub['Pérdida (m)'].mean():.0f} m"
                                  if sub["Pérdida (m)"].notna().any() else "—")
        cm1, cm2, cm3, cm4, cm5, cm6, cm7 = st.columns(7)
        cm1.metric("Viradas",         len(tacks))
        cm2.metric("Trasluchadas",    len(gybes))
        cm3.metric(
            "Recovery medio",
            f"{num_rec.mean():.0f} s" if not num_rec.isna().all() else "—",
        )
        cm4.metric("Caída SOG media", f"{man['Caída (kts)'].mean():.1f} kts")
        cm5.metric("Pérdida virada", _loss_mean(tacks),
                   help="Metros medios perdidos por virada frente a un ghost que sigue recto.")
        cm6.metric("Pérdida trasluchada", _loss_mean(gybes),
                   help="Metros medios perdidos por trasluchada frente a un ghost que sigue recto.")
        cm7.metric("🔴 Fallidas",       len(failed),
                   help=f"Maniobras en las que la velocidad bajó de {FALL_KTS} kts (caída al agua).")

        # Tabla de maniobras
        disp_cols = [
            c for c in
            ["Tiempo", "Tipo", "Estado", "SOG antes (kts)", "SOG mín (kts)", "Caída (kts)",
             "Recovery (s)", "Pérdida (m)"]
            if c in man.columns
        ]
        col_cfg = {
            "Tiempo": st.column_config.DatetimeColumn(
                "Tiempo",
                help="Marca de tiempo del momento en que se detecta la maniobra.",
            ),
            "Tipo": st.column_config.TextColumn(
                "Tipo",
                help="**Virada (Tack):** cambio de amura navegando de ceñida (viento de proa). "
                     "**Trasluchada (Gybe):** cambio de amura navegando de popa (viento en popa).",
            ),
            "Estado": st.column_config.TextColumn(
                "Estado",
                help=f"**✅ OK:** la maniobra se completó sin caída (SOG mín ≥ {FALL_KTS} kts). "
                     f"**🔴 Fallida:** la velocidad bajó de {FALL_KTS} kts, el regatista se cayó.",
            ),
            "SOG antes (kts)": st.column_config.NumberColumn(
                "SOG antes (kts)",
                format="%.1f kts",
                help="Velocidad de entrada: SOG media entre 15 y 5 segundos antes del cambio de amura, "
                     "antes de empezar a frenar.",
            ),
            "SOG mín (kts)": st.column_config.NumberColumn(
                "SOG mín (kts)",
                format="%.1f kts",
                help="Velocidad mínima registrada dentro de una ventana de ±15 s alrededor de la maniobra.",
            ),
            "Caída (kts)": st.column_config.NumberColumn(
                "Caída (kts)",
                format="%.1f kts",
                help="Pérdida de velocidad durante la maniobra: SOG antes − SOG mínima.",
            ),
            "Recovery (s)": st.column_config.TextColumn(
                "Recovery (s)",
                help="⏱️ **Tiempo de recuperación:** segundos desde que la velocidad empieza a caer "
                     "(por debajo del 95 % de la SOG de entrada) hasta recuperar el 90 % de la SOG "
                     "de entrada, o de la de salida si es menor (p. ej. trasluchada en boya que sale "
                     "a ceñida). "
                     "Un recovery bajo indica una maniobra más limpia y rápida.",
            ),
            "Pérdida (m)": st.column_config.NumberColumn(
                "Pérdida (m)",
                format="%.0f m",
                help="👻 **Metros perdidos frente a un ghost** que va contigo hasta que "
                     "empiezas la maniobra (empieza a caer la SOG) y desde ahí sigue recto con "
                     "tu rumbo de entrada, a un VMG igual a la media de tu VMG antes y después "
                     "de la maniobra. Se mide en el eje del viento (avance hacia barlovento en "
                     f"ceñida, hacia sotavento en popa) {GHOST_POST_S} s después del cambio de "
                     "amura. Negativo = ganas al ghost. "
                     "**—**: maniobra fallida, otra maniobra muy seguida, rodeo de baliza, rumbo irregular o VMG muy bajo "
                     "antes o después (entrada/salida no comparable), o sin viento. "
                     "Selecciona filas para ver el replay contra el ghost.",
            ),
        }

        # Colorear en rojo las filas de maniobras fallidas
        def _highlight_failed(row):
            color = "background-color: rgba(239, 85, 59, 0.18)" if row.get("Estado") == "🔴 Fallida" else ""
            return [color] * len(row)

        styled = man[disp_cols].style.apply(_highlight_failed, axis=1)
        man_sel = st.dataframe(
            styled,
            column_config=col_cfg,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row",
            key=f"man_sel_{name}",
        )

        # Gráficos de maniobras lado a lado
        ch1, ch2 = st.columns(2)
        with ch1:
            rec_chart = build_maneuver_recovery_chart(man)
            if rec_chart:
                st.plotly_chart(rec_chart, use_container_width=True)
        with ch2:
            drop_chart = build_maneuver_sog_chart(man)
            if drop_chart:
                st.plotly_chart(drop_chart, use_container_width=True)

        # Replay contra el ghost de cada maniobra seleccionada en la tabla
        sel_rows = man_sel.selection.rows if man_sel else []
        if not sel_rows:
            st.caption("👻 Selecciona maniobras en la tabla para ver el replay contra el ghost.")
        for r in sel_rows:
            m = man.iloc[r]
            hora = m["Tiempo"].strftime("%H:%M:%S") if "Tiempo" in m else ""
            if pd.isna(m["Pérdida (m)"]):
                st.info(f"👻 {m['Tipo']} {hora}: sin ghost ({m['_motivo']}).")
                continue
            st.markdown(f"##### 👻 {m['Tipo']} {hora} · {m['Estado']} · "
                        f"**{m['Pérdida (m)']:.0f} m** perdidos frente al ghost")
            ghost_replay = build_ghost_racers(df, m)
            if ghost_replay:
                track_replay(*ghost_replay, key=f"ghost_{name}_{int(m['_i'])}")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 7 · POLARES (SOG + VMG)
# ═══════════════════════════════════════════════════════════════════════════════
has_vmg  = any("VMG_kts" in d.columns for d in dfs)
has_fase = any("Fase" in d.columns for d in dfs)
has_twa  = any("TWA" in d.columns for d in dfs)

pl, pr = st.columns(2)
with pl:
    section("🧭 Polar Comparativa")
    if has_twa:
        st.caption(
            "Velocidad media (SOG) por ángulo al viento, en fases de vuelo (Ceñida, Través y Popa). "
            "El eje angular va de 0° (viento de frente) a 180° (viento de popa exacta). "
            "Los dos picos muestran los **ángulos donde se alcanza más velocidad**. "
            "En comparativas, la curva más exterior indica mayor velocidad en ese ángulo."
        )
        st.plotly_chart(build_polar(dfs), use_container_width=True)
    else:
        st.info("Sin datos de TWA para construir la polar.")
with pr:
    section("🎯 Polar de VMG")
    if has_vmg and has_twa:
        st.caption(
            "Muestra la velocidad real hacia el destino (VMG) según el ángulo al viento (TWA), "
            "solo en fases de vuelo (Ceñida, Través y Popa). "
            "El eje angular va de 0° (viento de frente) a 180° (viento de popa). "
            "Los dos picos de la curva indican los **ángulos óptimos** de ceñida y popa. "
            "Un pico más alto y exterior = más VMG a ese ángulo. "
            "En comparativas, la curva más exterior gana en avance real."
        )
        st.plotly_chart(build_vmg_polar(dfs), use_container_width=True)
    else:
        st.info("Sin datos de VMG o TWA.")

# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 8 · CONSISTENCIA POR FASE
# ═══════════════════════════════════════════════════════════════════════════════
if has_fase:
    st.divider()
    section("📦 Consistencia por Fase")
    st.caption(
        "Distribución de SOG en Popa, Través y Ceñida. "
        "La caja muestra el rango del 50% central de los datos (P25–P75); "
        "la línea central es la mediana y el rombo la media. "
        "Una caja estrecha y alta indica velocidad constante a ritmo elevado (consistente). "
        "Una caja ancha y baja indica velocidad irregular (inconsistente). "
        "En comparativas, el regatista con la caja más alta y estrecha domina esa fase."
    )
    st.plotly_chart(build_phase_boxplot(dfs), use_container_width=True)
