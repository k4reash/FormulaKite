"""
Formula Kite Analytics Dashboard
Telemetría Sailmon · Python 3 · Streamlit
"""

import math

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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
RECOVERY_KTS    = 15
FOIL_FLIGHT_KTS = 8   # por debajo de este umbral el foil no vuela (maniobra fallida)
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
def classify_phase(sog_kts: float, twa) -> str:
    """
    Clasifica la fase de navegación.
    - Caída y Transición solo dependen de SOG (TWA puede ser NaN).
    - Popa/Ceñida/Través requieren TWA; si no hay TWA a alta velocidad -> Transición.
    """
    if pd.isna(sog_kts):
        return "Caída"
    if sog_kts < 8:
        return "Caída"
    if 8 <= sog_kts < 17:
        return "Transición"
    # SOG >= 17 kts: usamos TWA para distinguir Popa / Través / Ceñida
    if pd.isna(twa):
        return "Transición"  # sin viento conocido
    a = abs(twa)
    if a > 110:
        return "Popa"
    if a < 70:
        return "Ceñida"
    return "Través"  # 70° ≤ |TWA| ≤ 110°


def load_csv(file, name: str) -> pd.DataFrame:
    """Carga y preprocesa un CSV de Sailmon."""
    df = pd.read_csv(file)
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

    # Convertir m/s → nudos
    for col in ("SOG", "VMG"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[f"{col}_kts"] = df[col] * MS_TO_KNOTS

    # Convertir TWA y Heel a numérico (pueden tener celdas vacías)
    for col in ("TWA", "Heel"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clasificar fases (TWA opcional: si falta, Caída/Transición se resuelven por SOG)
    if "SOG_kts" in df.columns:
        twa_col = df["TWA"] if "TWA" in df.columns else pd.Series([float("nan")] * len(df))
        df["Fase"] = [
            classify_phase(sog, twa)
            for sog, twa in zip(df["SOG_kts"], twa_col)
        ]

    # Parsear tiempo (CSV en UTC → convertir a hora local UTC+2)
    if "time" in df.columns:
        df["time"] = (
            pd.to_datetime(df["time"], errors="coerce") + pd.Timedelta(hours=2)
        )

    df["Regatista"] = name
    return df


def compute_kpis(df: pd.DataFrame) -> dict:
    k = {}
    if "SOG_kts" in df.columns:
        k["sog_max"]  = df["SOG_kts"].max()
        k["sog_mean"] = df["SOG_kts"].mean()
    if "VMG_kts" in df.columns:
        k["vmg_max"] = df["VMG_kts"].max()
    if "Fase" in df.columns:
        n = len(df)
        k["pct_flight"] = 100 * df["Fase"].isin(["Popa", "Ceñida", "Través", "Transición"]).sum() / n
        k["pct_popa"]    = 100 * (df["Fase"] == "Popa").sum() / n
        k["pct_cenida"]  = 100 * (df["Fase"] == "Ceñida").sum() / n
        k["pct_traves"]  = 100 * (df["Fase"] == "Través").sum() / n
        k["pct_trans"]   = 100 * (df["Fase"] == "Transición").sum() / n
    return k


def detect_maneuvers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta viradas y trasluchadas por cambio de signo en TWA.
    Para cada maniobra calcula:
      - SOG antes (media 5s previos)
      - SOG mínima (ventana ±15s)
      - Caída de SOG
      - Recovery time (segundos hasta volver a >= 15 kts)
    """
    if "TWA" not in df.columns or "SOG_kts" not in df.columns:
        return pd.DataFrame()

    df = df.reset_index(drop=True)
    twa = df["TWA"].values.astype(float)
    sog = df["SOG_kts"].values.astype(float)

    rows = []
    i = 1
    GAP = 8  # mínimo de segundos entre maniobras detectadas

    while i < len(twa) - 1:
        prev, curr = twa[i - 1], twa[i]
        if not (np.isnan(prev) or np.isnan(curr)) and prev * curr < 0:
            # SOG antes de la maniobra (media 5s previos)
            sog_before = float(np.nanmean(sog[max(0, i - 5):i]))

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

            # Recovery time: segundos desde min_sog hasta SOG >= RECOVERY_KTS
            rec = ">90"
            for j in range(min_idx, min(len(sog), min_idx + 90)):
                if sog[j] >= RECOVERY_KTS:
                    rec = j - min_idx
                    break

            row = {
                "Tipo":            mtype,
                "Estado":          "🔴 Fallida" if min_sog < FOIL_FLIGHT_KTS else "✅ OK",
                "SOG antes (kts)": round(sog_before, 1),
                "SOG mín (kts)":   round(min_sog, 1),
                "Caída (kts)":     round(sog_before - min_sog, 1),
                "Recovery (s)":    rec,
            }
            if "time" in df.columns:
                row["Tiempo"] = df["time"].iloc[i]
            if "latitude" in df.columns:
                row["lat"] = float(df["latitude"].iloc[i])
            if "longitude" in df.columns:
                row["lon"] = float(df["longitude"].iloc[i])

            rows.append(row)
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


def build_map(dfs, color_by="Velocidad"):
    fig = go.Figure()

    for idx, df in enumerate(dfs):
        name  = df["Regatista"].iloc[0]
        color = RACER_PALETTE[idx % len(RACER_PALETTE)]

        def _build_customdata(src):
            """Construye customdata con cols: [0]SOG [1]VMG [2]TWA [3]Fase [4]Hora."""
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
            return np.column_stack([sog.values, vmg.values, twa.values,
                                    fase.values, hora.values])

        if color_by == "Velocidad" and "SOG_kts" in df.columns:
            fig.add_trace(go.Scattermapbox(
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
                    "⏱ %{customdata[4]}<extra></extra>"
                ),
            ))
        elif color_by == "Fase" and "Fase" in df.columns:
            for fase, col in PHASE_COLORS.items():
                sub = df[df["Fase"] == fase]
                if sub.empty:
                    continue
                fig.add_trace(go.Scattermapbox(
                    lat=sub["latitude"], lon=sub["longitude"], mode="markers",
                    marker=dict(size=5, color=col),
                    name=f"{name} – {fase}",
                    customdata=_build_customdata(sub),
                    hovertemplate=(
                        f"<b>{name}</b> · {fase}<br>"
                        "SOG: %{customdata[0]} kts<br>"
                        "VMG: %{customdata[1]} kts<br>"
                        "TWA: %{customdata[2]}°<br>"
                        "⏱ %{customdata[4]}<extra></extra>"
                    ),
                ))
        else:
            fig.add_trace(go.Scattermapbox(
                lat=df["latitude"], lon=df["longitude"], mode="markers",
                marker=dict(size=5, color=color), name=name,
            ))

    all_lat = pd.concat([d["latitude"] for d in dfs])
    all_lon = pd.concat([d["longitude"] for d in dfs])
    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=float(all_lat.mean()), lon=float(all_lon.mean())),
            zoom=12,
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


def build_histogram(dfs):
    fig = go.Figure()
    for idx, df in enumerate(dfs):
        if "SOG_kts" not in df.columns:
            continue
        fig.add_trace(go.Histogram(
            x=df["SOG_kts"],
            name=df["Regatista"].iloc[0],
            nbinsx=40,
            marker_color=RACER_PALETTE[idx % len(RACER_PALETTE)],
            opacity=0.7,
            histnorm="percent",
        ))
    fig.update_layout(
        barmode="overlay",
        xaxis=dict(title="SOG (kts)", gridcolor="#2d3748", color="#a0aec0"),
        yaxis=dict(title="% del tiempo", gridcolor="#2d3748", color="#a0aec0"),
        **_dark_layout({
            "height": 400,
            "title": dict(text="Distribución de Velocidades", font=dict(color="#e2e8f0")),
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
    # Líneas de referencia de fases
    for kts, label, color in [
        (24, "Umbral Popa (24 kts)", "#EF553B"),
        (17, "Umbral Ceñida (17 kts)", "#00CC96"),
        (8,  "Umbral Vuelo (8 kts)",  "#FFA15A"),
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
    for kts, label, color in [
        (24, "Popa 24 kts", "#EF553B"),
        (17, "Ceñida 17 kts", "#00CC96"),
    ]:
        fig.add_hline(y=kts, line_dash="dot", line_color=color, opacity=0.4,
                      annotation_text=label, annotation_font_color=color)
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
                text="Escora vs Velocidad por Fase · SOG media (foil en vuelo ≥ 8 kts)",
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
        initial_data.append(go.Scattermapbox(
            lat=lats, lon=lons,
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=3, color=color),
            customdata=sogs,
            hovertemplate=f"<b>{name}</b><br>SOG: %{{customdata}} kts<extra></extra>",
            opacity=0.9,
            name=name, legendgroup=name, showlegend=True,
        ))
        initial_data.append(go.Scattermapbox(
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
            frame_data.append(go.Scattermapbox(
                lat=lats, lon=lons,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=3, color=color),
                customdata=sogs,
                hovertemplate=f"<b>{name}</b><br>SOG: %{{customdata}} kts<extra></extra>",
                opacity=0.9,
            ))
            # Barco — dot grande
            frame_data.append(go.Scattermapbox(
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
        mapbox=dict(
            style="carto-darkmatter",
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
            tr = go.Scattermapbox(
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
            bt = go.Scattermapbox(
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
        mapbox=dict(
            style="carto-darkmatter",
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


# ─── Legs, buoy detection & Race mode ────────────────────────────────────────
def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia en metros entre dos puntos GPS."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin(math.radians(lat2 - lat1) / 2) ** 2
         + math.cos(phi1) * math.cos(phi2)
         * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2.0 * R * math.asin(math.sqrt(min(a, 1.0)))


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


def detect_race_legs(df: pd.DataFrame,
                     buoys: list,
                     radius_m: float = 80.0) -> pd.DataFrame:
    """
    Segmenta el track en bordos delimitados por bordos de boya.
    Detecta pre-salida mediante heurística: bordos antes del primer
    par ceñida/popa de duración ≥ 30 s.
    """
    if not buoys or "latitude" not in df.columns or "longitude" not in df.columns:
        return pd.DataFrame()

    df = df.reset_index(drop=True)

    def _nearest_buoy(lat, lon):
        if pd.isna(lat) or pd.isna(lon):
            return None
        for i, b in enumerate(buoys):
            if _haversine_m(float(lat), float(lon), b["lat"], b["lon"]) < radius_m:
                return i
        return None

    near = [_nearest_buoy(r["latitude"], r["longitude"]) for _, r in df[["latitude", "longitude"]].iterrows()]

    # Detectar entradas y salidas de zona de boya → roundings
    roundings = []
    in_zone, zone_start = None, None
    for i, b_idx in enumerate(near):
        if b_idx is not None and in_zone is None:
            in_zone, zone_start = b_idx, i
        elif b_idx is None and in_zone is not None:
            roundings.append(((zone_start + i) // 2, in_zone))
            in_zone = None
    if in_zone is not None:
        roundings.append(((zone_start + len(near) - 1) // 2, in_zone))

    if len(roundings) < 2:
        return pd.DataFrame()

    rows = []
    bordo_num = 1
    for li in range(len(roundings) - 1):
        i0, b0 = roundings[li]
        i1, b1 = roundings[li + 1]
        # Descartar transiciones boya→misma boya o bordos con fase dominante Transición
        if b0 == b1:
            continue
        seg = df.iloc[i0:i1]
        if len(seg) < 5:
            continue

        dur_s    = len(seg)
        sog_mean = seg["SOG_kts"].mean()        if "SOG_kts" in seg.columns else float("nan")
        vmg_mean = seg["VMG_kts"].mean()        if "VMG_kts" in seg.columns else float("nan")
        fase_dom = (seg["Fase"].value_counts().idxmax()
                    if "Fase" in seg.columns and not seg["Fase"].isna().all() else "—")
        if fase_dom in ("Transición", "Caída"):
            continue
        dist_m   = (_haversine_total(seg["latitude"].ffill().values,
                                      seg["longitude"].ffill().values) * 1852
                    if not seg["latitude"].isna().all() else
                    sog_mean * dur_s / 3600.0 * 1852)
        t_inicio = "—"
        if "time" in seg.columns:
            t = seg["time"].dropna()
            if not t.empty:
                t_inicio = t.iloc[0].strftime("%H:%M:%S")

        rows.append({
            "Bordo":           bordo_num,
            "Boya inicio":     f"B{b0 + 1}",
            "Boya fin":        f"B{b1 + 1}",
            "Inicio":          t_inicio,
            "Duración":        f"{dur_s // 60}:{dur_s % 60:02d}",
            "Fase dominante":  fase_dom,
            "SOG media (kts)": round(sog_mean, 1) if not pd.isna(sog_mean) else None,
            "VMG media (kts)": round(vmg_mean, 1) if not pd.isna(vmg_mean) else None,
            "Distancia (m)":   round(dist_m, 0),
            "_dur_s":          dur_s,
        })
        bordo_num += 1

    if not rows:
        return pd.DataFrame()

    legs = pd.DataFrame(rows)

    # Heurística pre-salida: antes del primer borde ≥ 30 s en fase Ceñida, Popa o Través
    racing_phases = {"Ceñida", "Popa", "Través"}
    first_race = next(
        (i for i, r in legs.iterrows()
         if r["_dur_s"] >= 30 and r["Fase dominante"] in racing_phases),
        0,
    )
    legs["Pre-salida"] = legs.index < first_race
    return legs.drop(columns=["_dur_s"])


_BORDO_PALETTE = [
    "#00E5FF", "#FF4081", "#76FF03", "#FF6D00", "#D500F9",
    "#FFEA00", "#00E676", "#FF1744", "#2979FF", "#FF9100",
]


def _assign_bordo_col(df: pd.DataFrame, buoys: list, radius_m: float = 80.0) -> pd.Series:
    """Devuelve una Series con número de bordo (1-based) por fila, o 0 si no pertenece a ninguno."""
    bordo_col = pd.Series(0, index=range(len(df)), dtype=int)
    if not buoys or "latitude" not in df.columns or "longitude" not in df.columns:
        return bordo_col

    df = df.reset_index(drop=True)

    def _near(lat, lon):
        if pd.isna(lat) or pd.isna(lon):
            return None
        for i, b in enumerate(buoys):
            if _haversine_m(float(lat), float(lon), b["lat"], b["lon"]) < radius_m:
                return i
        return None

    near = [_near(r["latitude"], r["longitude"])
            for _, r in df[["latitude", "longitude"]].iterrows()]

    roundings, in_zone, zone_start = [], None, None
    for i, b_idx in enumerate(near):
        if b_idx is not None and in_zone is None:
            in_zone, zone_start = b_idx, i
        elif b_idx is None and in_zone is not None:
            roundings.append(((zone_start + i) // 2, in_zone))
            in_zone = None
    if in_zone is not None:
        roundings.append(((zone_start + len(near) - 1) // 2, in_zone))

    bordo_num = 1
    for li in range(len(roundings) - 1):
        i0, b0 = roundings[li]
        i1, b1 = roundings[li + 1]
        if b0 == b1:
            continue
        seg = df.iloc[i0:i1]
        if len(seg) < 5:
            continue
        fase_dom = (seg["Fase"].value_counts().idxmax()
                    if "Fase" in seg.columns and not seg["Fase"].isna().all() else "—")
        if fase_dom in ("Transición", "Caída"):
            continue
        bordo_col.iloc[i0:i1] = bordo_num
        bordo_num += 1

    return bordo_col


def build_race_map(dfs, buoys: list, color_by: str = "Velocidad", radius_m: float = 80.0) -> go.Figure:
    """
    Mapa de tracks con las boyas colocadas manualmente, marcadas en dorado.
    color_by: "Velocidad" | "Fase" | "Bordo"
    """
    if color_by != "Bordo":
        fig = build_map(dfs, color_by)
        for trace in fig.data:
            if hasattr(trace, "hovertemplate") and trace.hovertemplate:
                trace.hovertemplate = trace.hovertemplate.replace(
                    "<extra></extra>",
                    "📍 %{lat:.5f}, %{lon:.5f}<extra></extra>",
                )
    else:
        fig = go.Figure()
        all_lats, all_lons = [], []
        for df_r in dfs:
            r_name = df_r["Regatista"].iloc[0] if "Regatista" in df_r.columns else "?"
            df_r = df_r.reset_index(drop=True)
            bordo_col = _assign_bordo_col(df_r, buoys, radius_m)

            # Track gris de fondo (puntos sin bordo)
            mask0 = bordo_col == 0
            if mask0.any():
                fig.add_trace(go.Scattermapbox(
                    lat=df_r.loc[mask0, "latitude"],
                    lon=df_r.loc[mask0, "longitude"],
                    mode="lines",
                    line=dict(width=1, color="rgba(180,180,180,0.2)"),
                    showlegend=False,
                    hoverinfo="skip",
                ))
                all_lats += df_r.loc[mask0, "latitude"].dropna().tolist()
                all_lons += df_r.loc[mask0, "longitude"].dropna().tolist()

            # Un trace por bordo
            n_bordos = int(bordo_col.max())
            for bn in range(1, n_bordos + 1):
                mask_b = bordo_col == bn
                if not mask_b.any():
                    continue
                color = _BORDO_PALETTE[(bn - 1) % len(_BORDO_PALETTE)]
                seg = df_r.loc[mask_b]
                label = f"Bordo {bn}" + (f" – {r_name}" if len(dfs) > 1 else "")
                sog_vals = (seg["SOG_kts"].fillna(0).apply(lambda v: f"{v:.1f}").values
                            if "SOG_kts" in seg.columns else ["\u2014"] * len(seg))
                fase_vals = (seg["Fase"].fillna("—").values
                             if "Fase" in seg.columns else ["—"] * len(seg))
                hora_vals = (seg["time"].dt.strftime("%H:%M:%S").fillna("—").values
                             if "time" in seg.columns else ["—"] * len(seg))
                cd = np.column_stack([sog_vals, fase_vals, hora_vals])
                fig.add_trace(go.Scattermapbox(
                    lat=seg["latitude"],
                    lon=seg["longitude"],
                    mode="lines+markers",
                    line=dict(width=3, color=color),
                    marker=dict(size=4, color=color),
                    name=label,
                    customdata=cd,
                    hovertemplate=(
                        f"<b>{label}</b><br>"
                        "SOG: %{customdata[0]} kts<br>"
                        "Fase: %{customdata[1]}<br>"
                        "⏱ %{customdata[2]}<br>"
                        "📍 %{lat:.5f}, %{lon:.5f}<extra></extra>"
                    ),
                ))
                all_lats += seg["latitude"].dropna().tolist()
                all_lons += seg["longitude"].dropna().tolist()

        ctr_lat = float(np.mean(all_lats)) if all_lats else 0.0
        ctr_lon = float(np.mean(all_lons)) if all_lons else 0.0
        fig.update_layout(
            mapbox=dict(style="carto-darkmatter",
                        center=dict(lat=ctr_lat, lon=ctr_lon), zoom=12),
            margin=dict(l=0, r=0, t=0, b=0),
            height=520,
            paper_bgcolor="#0e1117",
            legend=dict(
                bgcolor="rgba(0,0,0,0.5)", font=dict(color="white"),
                x=0.01, y=0.99, xanchor="left", yanchor="top",
            ),
        )

    for b in buoys:
        label = b.get("name", "B?")
        fig.add_trace(go.Scattermapbox(
            lat=[b["lat"]],
            lon=[b["lon"]],
            mode="markers+text",
            marker=dict(size=18, color="#FFD700", opacity=0.95),
            text=[label],
            textposition="top right",
            textfont=dict(color="#FFD700", size=14),
            name=label,
            hovertemplate=(
                f"<b>{label}</b><br>"
                f"Lat: {b['lat']:.5f}<br>"
                f"Lon: {b['lon']:.5f}<extra></extra>"
            ),
        ))
    return fig


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


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🪁 Formula Kite")
    st.caption("Dashboard de Telemetría · v1.0")
    st.divider()

    uploaded_files = st.file_uploader(
        "Archivos CSV (Sailmon)",
        type=["csv"],
        accept_multiple_files=True,
        help="Un archivo .csv por regatista. Frecuencia esperada: 1 Hz.",
    )

    racer_names = []
    if uploaded_files:
        st.divider()
        st.subheader("Nombres de Regatistas")
        for i, f in enumerate(uploaded_files):
            default = f.name.replace(".csv", "").replace("_", " ").title()
            name = st.text_input(f"Regatista {i + 1}", value=default, key=f"n{i}")
            racer_names.append(name)

    st.divider()
    st.markdown(
        "**Fases de Navegación:**\n\n"
        "🔴 **Popa** · SOG ≥ 17 kts · TWA > 110°  \n"
        "🟢 **Ceñida** · SOG ≥ 17 kts · TWA < 70°  \n"
        "🔵 **Través** · SOG ≥ 17 kts · TWA 70–110°  \n"
        "🟠 **Transición** · 8–17 kts"
    )


# ─── Landing page (sin datos) ─────────────────────────────────────────────────
if not uploaded_files:
    st.title("🪁 Formula Kite Analytics")
    st.markdown(
        """
        Analiza y compara la telemetría de regatistas de **Formula Kite** con datos de **Sailmon**.

        **← Carga uno o más archivos CSV** desde el panel lateral para comenzar.

        ---

        ### Columnas esperadas en el CSV

        | Columna | Descripción | Unidad de entrada |
        |---------|-------------|-------------------|
        | `time` | Timestamp de la muestra | ISO 8601 / UTC |
        | `latitude` / `longitude` | Posición GPS | grados decimales |
        | `SOG` | Speed Over Ground | **m/s** (se convierte a kts) |
        | `VMG` | Velocity Made Good | **m/s** (se convierte a kts) |
        | `TWA` | True Wind Angle | grados (± babor/estribor) |
        | `HDT` | Heading True | grados |

        > Los datos de Sailmon tienen frecuencia de muestreo de **1 Hz** (1 fila/segundo).
        """
    )
    st.stop()


# ─── Carga de datos ───────────────────────────────────────────────────────────
dfs = []
for f, nm in zip(uploaded_files, racer_names):
    with st.spinner(f"Procesando {f.name}…"):
        try:
            dfs.append(load_csv(f, nm))
        except Exception as exc:
            st.error(f"❌ Error al cargar **{f.name}**: {exc}")

if not dfs:
    st.warning("No se pudieron cargar archivos válidos.")
    st.stop()


# ─── Helper UI ───────────────────────────────────────────────────────────────
def section(title):
    st.markdown(f'<p class="section-title">{title}</p>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 · KPIs
# ═══════════════════════════════════════════════════════════════════════════════
section("📊 Indicadores Clave por Regatista")

cols = st.columns(len(dfs))
for col, df in zip(cols, dfs):
    name = df["Regatista"].iloc[0]
    k = compute_kpis(df)
    with col:
        st.markdown(f"#### {name}")
        c1, c2 = st.columns(2)
        c1.metric("🚀 Vel. Máx.", f"{k.get('sog_max', 0):.1f} kts")
        c2.metric("🎯 Vel. Media", f"{k.get('sog_mean', 0):.1f} kts")
        c3, c4 = st.columns(2)
        c3.metric("💨 VMG Máx.", f"{k.get('vmg_max', 0):.1f} kts" if "vmg_max" in k else "—")
        c4.metric("✈️ % Vuelo", f"{k.get('pct_flight', 0):.1f}%")
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

    map_c1, map_c2 = st.columns([1, 3])
    with map_c1:
        color_by = st.radio("Colorear por:", ["Velocidad", "Fase"], horizontal=True)

    # ── Filtro de tiempo exacto ───────────────────────────────────────────────
    _time_dfs = [d for d in dfs if "time" in d.columns and d["time"].notna().any()]
    if _time_dfs:
        _all_t   = pd.concat([d["time"].dropna() for d in _time_dfs])
        _t_min   = _all_t.min()
        _t_max   = _all_t.max()
        _ref_date = _t_min.date()

        tc1, tc2 = st.columns(2)
        with tc1:
            t_start = st.time_input(
                "🕐 Hora inicio",
                value=_t_min.to_pydatetime().time(),
                step=60,
                key="map_t_start",
                help="Selecciona la hora de inicio con precisión de minutos.",
            )
        with tc2:
            t_end = st.time_input(
                "🕑 Hora fin",
                value=_t_max.to_pydatetime().time(),
                step=60,
                key="map_t_end",
                help="Selecciona la hora de fin con precisión de minutos.",
            )

        import datetime as _dt
        _t0 = pd.Timestamp.combine(_ref_date, t_start)
        _t1 = pd.Timestamp.combine(_ref_date, t_end)
        # Si la sesión cruza medianoche, ajustar
        if _t1 < _t0:
            _t1 += pd.Timedelta(days=1)

        dfs_map = [d[d["time"].between(_t0, _t1)].copy() for d in dfs
                   if "time" in d.columns]
        dfs_map = [d for d in dfs_map if not d.empty]
    else:
        dfs_map = dfs

    if dfs_map:
        st.plotly_chart(
            build_map(dfs_map, color_by),
            use_container_width=True,
            config={"scrollZoom": True},
        )
    else:
        st.info("Sin datos GPS en la ventana seleccionada.")

else:
    st.info("ℹ️ Los archivos no contienen columnas de GPS (latitude/longitude). El mapa no está disponible.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2B · MODO REGATA
# ═══════════════════════════════════════════════════════════════════════════════
if has_gps:
    section("🏁 Modo Regata")

    race_mode = st.radio(
        "Modo de sesión:",
        ["Entrenamiento", "Regata"],
        horizontal=True,
        key="race_mode_toggle",
        help="En modo Regata puedes colocar las boyas manualmente y analizar los bordos.",
    )

    if race_mode == "Regata":
        st.caption(
            "Coloca las boyas del recorrido usando las coordenadas GPS del mapa de tracks. "
            "💡 Pasa el ratón por el mapa superior — Plotly muestra la latitud y longitud "
            "en la barra inferior derecha del mapa."
        )

        # ── Configuración de boyas ───────────────────────────────────────────
        _all_lat = pd.concat([d["latitude"].dropna() for d in dfs if "latitude" in d.columns])
        _all_lon = pd.concat([d["longitude"].dropna() for d in dfs if "longitude" in d.columns])
        _center_lat = float(_all_lat.mean())
        _center_lon = float(_all_lon.mean())

        brc1, brc2 = st.columns([1, 3])
        with brc1:
            n_buoys = st.number_input(
                "Número de boyas", min_value=1, max_value=8, value=3, step=1,
                help="Boyas del recorrido (top mark, gate, línea de salida…).",
            )
            race_radius = st.slider(
                "Radio de rounding (m)", 20, 200, 60, step=5,
                help="Distancia a la que se considera que el barco ha pasado la boya.",
            )

        with brc2:
            st.markdown("**Posición de cada boya** — pega las coordenadas copiadas del mapa (`lat, lon`):")
            race_buoys = []
            buoy_cols = st.columns(int(n_buoys))
            _default_coords = f"{_center_lat:.5f}, {_center_lon:.5f}"
            for bi, bcol in enumerate(buoy_cols):
                with bcol:
                    st.markdown(f"🟡 **B{bi + 1}**")
                    raw = st.text_input(
                        "lat, lon",
                        value=_default_coords,
                        key=f"buoy_coords_{bi}",
                        placeholder="37.75223, -0.82061",
                    )
                    try:
                        parts = [p.strip() for p in raw.split(",")]
                        b_lat, b_lon = float(parts[0]), float(parts[1])
                    except Exception:
                        b_lat, b_lon = _center_lat, _center_lon
                        st.caption("⚠️ Formato inválido")
                    race_buoys.append({"lat": b_lat, "lon": b_lon, "name": f"B{bi + 1}"})

        # ── Mapa de regata ───────────────────────────────────────────────────
        race_color_by = st.radio(
            "Colorear por:", ["Velocidad", "Fase", "Bordo"],
            horizontal=True, key="race_color_by",
        )
        _race_dfs_src = dfs_map if ("dfs_map" in dir() and dfs_map) else dfs
        _race_map_event = st.plotly_chart(
            build_race_map(_race_dfs_src, race_buoys, race_color_by, race_radius),
            use_container_width=True,
            config={"scrollZoom": True},
            on_select="rerun",
            selection_mode="points",
            key="race_map_select",
        )
        # Mostrar coordenadas del punto pinchado con botón de copia
        _sel_pts = (
            _race_map_event.selection.points
            if _race_map_event and hasattr(_race_map_event, "selection")
            else []
        )
        if _sel_pts:
            _pt = _sel_pts[0]
            _pt_lat = _pt.get("lat")
            _pt_lon = _pt.get("lon")
            if _pt_lat is not None and _pt_lon is not None:
                st.info(
                    f"📍 Punto seleccionado — copia las coordenadas con el botón ▷"
                )
                st.code(f"{_pt_lat:.5f}, {_pt_lon:.5f}", language=None)

        st.divider()

        # ── Tabla de bordos ──────────────────────────────────────────────────
        section("📋 Tabla de Bordos")
        st.caption(
            "Cada fila es un bordo entre dos boyas."
        )

        col_cfg_race = {
            "Bordo":           st.column_config.NumberColumn("Bordo", format="%d"),
            "Boya inicio":     st.column_config.TextColumn("Boya inicio"),
            "Boya fin":        st.column_config.TextColumn("Boya fin"),
            "Inicio":          st.column_config.TextColumn("Inicio"),
            "Duración":        st.column_config.TextColumn("Duración", help="MM:SS"),
            "Fase dominante":  st.column_config.TextColumn("Fase dominante"),
            "SOG media (kts)": st.column_config.NumberColumn("SOG media (kts)", format="%.1f kts"),
            "VMG media (kts)": st.column_config.NumberColumn("VMG media (kts)", format="%.1f kts"),
            "Distancia (m)":   st.column_config.NumberColumn("Distancia (m)", format="%.0f m"),
        }

        def _color_race_leg(row):
            f = row["Fase dominante"]
            c = ("background-color: rgba(0,204,150,0.12)"  if f == "Ceñida"
                 else "background-color: rgba(239,85,59,0.12)"   if f == "Popa"
                 else "background-color: rgba(25,211,243,0.12)"  if f == "Través"
                 else "")
            return [c] * len(row)

        for df_r in dfs:
            r_name = df_r["Regatista"].iloc[0]
            race_legs = detect_race_legs(df_r, race_buoys, race_radius)

            legs_race = race_legs[~race_legs["Pre-salida"]].drop(columns=["Pre-salida"]) if not race_legs.empty else pd.DataFrame()

            summary = (f"{len(legs_race)} bordos"
                       if not race_legs.empty else "sin bordos detectados")

            with st.expander(f"**{r_name}** — {summary}", expanded=True):
                if race_legs.empty:
                    st.info(
                        "No se detectaron bordos. Verifica que las coordenadas de las boyas "
                        "sean correctas y ajusta el radio de rounding."
                    )
                    continue

                mk1, mk2, mk3 = st.columns(3)
                mk1.metric("Bordos regata", len(legs_race))
                mk2.metric(
                    "SOG media regata",
                    f"{legs_race['SOG media (kts)'].mean():.1f} kts"
                    if not legs_race.empty and legs_race["SOG media (kts)"].notna().any() else "—",
                )
                mk3.metric(
                    "Dist. total regata",
                    f"{legs_race['Distancia (m)'].sum():.0f} m"
                    if not legs_race.empty else "—",
                )

                if not legs_race.empty:
                    st.dataframe(
                        legs_race.style.apply(_color_race_leg, axis=1),
                        column_config=col_cfg_race,
                        use_container_width=True,
                        hide_index=True,
                    )

    st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 · VELOCIDAD EN EL TIEMPO
# ═══════════════════════════════════════════════════════════════════════════════
section("📈 Velocidad en el Tiempo")
st.plotly_chart(build_speed_timeline(dfs), use_container_width=True)
st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 · POLAR + HISTOGRAMA
# ═══════════════════════════════════════════════════════════════════════════════
cl, cr = st.columns(2)

with cl:
    section("🧭 Polar Comparativa")
    if any("TWA" in d.columns for d in dfs):
        st.caption(
            "Velocidad media (SOG) por ángulo al viento, en fases de vuelo (Ceñida, Través y Popa). "
            "El eje angular va de 0° (viento de frente) a 180° (viento de popa exacta). "
            "Los dos picos muestran los **ángulos donde se alcanza más velocidad**. "
            "En comparativas, la curva más exterior indica mayor velocidad en ese ángulo."
        )
        st.plotly_chart(build_polar(dfs), use_container_width=True)
    else:
        st.info("Sin datos de TWA para construir la polar.")

with cr:
    section("📊 Distribución de Velocidades")
    st.plotly_chart(build_histogram(dfs), use_container_width=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 · VMG POLAR + BOX PLOT
# ═══════════════════════════════════════════════════════════════════════════════
has_vmg  = any("VMG_kts" in d.columns for d in dfs)
has_fase = any("Fase" in d.columns for d in dfs)
has_twa  = any("TWA" in d.columns for d in dfs)

if has_vmg or has_fase:
    vl, vr = st.columns(2)
    with vl:
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
    with vr:
        section("📦 Consistencia por Fase")
        if has_fase:
            st.caption(
                "Distribución de SOG en Popa, Través y Ceñida. "
                "La caja muestra el rango del 50% central de los datos (P25–P75); "
                "la línea central es la mediana y el rombo la media. "
                "Una caja estrecha y alta indica velocidad constante a ritmo elevado (consistente). "
                "Una caja ancha y baja indica velocidad irregular (inconsistente). "
                "En comparativas, el regatista con la caja más alta y estrecha domina esa fase."
            )
            st.plotly_chart(build_phase_boxplot(dfs), use_container_width=True)
        else:
            st.info("Sin datos de fases.")

# ─── Heel Analysis ────────────────────────────────────────────────────────────
has_heel = any("Heel" in d.columns for d in dfs)
if has_heel:
    section("⛵ Escora (Heel) vs Velocidad")
    st.caption(
        "SOG media agrupada en bins de 2° de escora, filtrada solo cuando el foil vuela (≥ 8 kts). "
        "Permite identificar el **ángulo de Heel óptimo** para maximizar velocidad."
    )
    heel_fig = build_heel_analysis(dfs)
    if heel_fig:
        st.plotly_chart(heel_fig, use_container_width=True)
    else:
        st.info("Sin suficientes datos de Heel para el análisis.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6 · VELOCIDAD PICO + TABLA DE BORDES
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
# SECCIÓN 7 · DETECCIÓN DE MANIOBRAS
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

        cm1, cm2, cm3, cm4, cm5 = st.columns(5)
        cm1.metric("Viradas",         len(tacks))
        cm2.metric("Trasluchadas",    len(gybes))
        cm3.metric(
            "Recovery medio",
            f"{num_rec.mean():.0f} s" if not num_rec.isna().all() else "—",
        )
        cm4.metric("Caída SOG media", f"{man['Caída (kts)'].mean():.1f} kts")
        cm5.metric("🔴 Fallidas",       len(failed),
                   help="Maniobras en las que la velocidad bajó por debajo de 8 kts (foil sin vuelo).")

        # Tabla de maniobras
        disp_cols = [
            c for c in
            ["Tiempo", "Tipo", "Estado", "SOG antes (kts)", "SOG mín (kts)", "Caída (kts)", "Recovery (s)"]
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
                help="**✅ OK:** el foil mantuvo vuelo durante toda la maniobra (SOG mín ≥ 8 kts). "
                     "**🔴 Fallida:** la velocidad bajó por debajo de 8 kts, el foil tocó el agua.",
            ),
            "SOG antes (kts)": st.column_config.NumberColumn(
                "SOG antes (kts)",
                format="%.1f kts",
                help="Velocidad media (SOG) en los 5 segundos previos al inicio de la maniobra.",
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
                help="⏱️ **Tiempo de recuperación:** segundos que transcurren desde la velocidad mínima "
                     "hasta que el regatista vuelve a superar los 15 nudos. "
                     "Un recovery bajo indica una maniobra más limpia y rápida.",
            ),
        }

        # Colorear en rojo las filas de maniobras fallidas
        def _highlight_failed(row):
            color = "background-color: rgba(239, 85, 59, 0.18)" if row.get("Estado") == "🔴 Fallida" else ""
            return [color] * len(row)

        styled = man[disp_cols].style.apply(_highlight_failed, axis=1)
        st.dataframe(
            styled,
            column_config=col_cfg,
            use_container_width=True,
            hide_index=True,
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
