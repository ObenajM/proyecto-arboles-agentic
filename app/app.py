"""
app/app.py — TreeLens: Identificador de Especies Arbóreas
Interfaz Streamlit con ONNX Runtime para el Arboretum y Palmetum de la UNAL Medellín.
"""

from __future__ import annotations

import base64
import csv
import json
import os
import random
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import folium
import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image, ImageOps
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation

# =============================================================================
# Paths & constants
# =============================================================================

BASE_DIR     = Path(__file__).resolve().parent.parent
MODEL_PATH   = BASE_DIR / "models" / "modelo_arboles_best.onnx"
CLASSES_PATH = BASE_DIR / "models" / "clases.json"
INFO_PATH    = BASE_DIR / "data"   / "info.json"

CAMPUS_IMG_PATH = Path(__file__).parent / "assets" / "Campus.jpg"

# Derived from Campus.jgw (WGS84 / EPSG:4326) + image dimensions 1262×2052 px.
# JGW: pixel_size = 0.0000045°, upper-left pixel center at (-75.5792788353322, 6.2682597468666).
_PX = 0.0000045
_UL_LON, _UL_LAT = -75.5792788353322, 6.2682597468666
_IMG_W, _IMG_H   = 1262, 2052
CAMPUS_BOUNDS = [
    [_UL_LAT - _PX * _IMG_H + _PX / 2, _UL_LON - _PX / 2],   # [lat_south, lon_west]
    [_UL_LAT + _PX / 2,                 _UL_LON + _PX * _IMG_W - _PX / 2],  # [lat_north, lon_east]
]
CAMPUS_CENTER = [
    (_UL_LAT - _PX * _IMG_H / 2),
    (_UL_LON + _PX * _IMG_W / 2),
]

IMAGE_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TOP_K = 3

CSV_COLUMNS     = ["species_key", "common_name", "confidence", "latitude", "longitude",
                   "gps_accuracy", "datetime", "source"]
CSV_PATH        = BASE_DIR / "data" / "arboles_mapeados.csv"
GITHUB_REPO     = "ObenajM/proyecto-arboles-agentic"
GITHUB_CSV_PATH = "data/arboles_mapeados.csv"

# =============================================================================
# Page config  (must come before any other st.* call)
# =============================================================================

st.set_page_config(
    page_title="TreeLens | Árboles UNAL Medellín",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Global CSS
# =============================================================================

st.markdown(
    """
    <style>
    /* ── Variables ──────────────────────────────────────────────────── */
    :root {
        --g900: #071a0b;
        --g800: #0f2d18;
        --g700: #1B5E20;
        --g600: #2E7D32;
        --g500: #388E3C;
        --g400: #66BB6A;
        --g200: #C8E6C9;
        --g100: #E8F5E9;
        --g50:  #F1F8E9;
        --bg:   #F8FFF6;
        --text: #1C1C1C;
        --muted: #4B6050;
        --radius-lg: 20px;
        --radius-md: 14px;
        --radius-sm: 10px;
        --shadow-sm: 0 2px 8px rgba(27,94,32,.08);
        --shadow-md: 0 6px 20px rgba(27,94,32,.12);
    }

    /* ── Base ───────────────────────────────────────────────────────── */
    .stApp { background: var(--bg); }
    h1, h2, h3 { color: var(--g800); }

    /* ── Hero banner ────────────────────────────────────────────────── */
    .hero {
        background: linear-gradient(135deg, #071a0b 0%, #1B5E20 50%, #2E7D32 100%);
        border-radius: var(--radius-lg);
        padding: 2rem 2.5rem;
        margin-bottom: 1.75rem;
        box-shadow: 0 10px 36px rgba(7,26,11,.30);
    }
    .hero h1 { color: #ffffff !important; font-size: 1.95rem; margin-bottom: .35rem; }
    .hero p  { color: #A5D6A7; font-size: 1rem; margin: 0; line-height: 1.55; }

    /* ── Prediction card — main ─────────────────────────────────────── */
    .pred-main {
        background: linear-gradient(145deg, #dff3df, #f1f8e9);
        border: 2px solid var(--g400);
        border-radius: var(--radius-lg);
        padding: 1.25rem 1.5rem;
        margin-bottom: .9rem;
        box-shadow: var(--shadow-md);
    }
    .pred-badge {
        display: inline-block;
        background: var(--g600);
        color: #fff;
        font-size: .68rem;
        font-weight: 700;
        letter-spacing: .09em;
        text-transform: uppercase;
        padding: .2rem .65rem;
        border-radius: 999px;
        margin-bottom: .55rem;
    }
    .pred-name-main {
        font-size: 1.55rem;
        font-weight: 800;
        color: var(--g800);
        line-height: 1.2;
        margin-bottom: .2rem;
    }
    .pred-sci { font-style: italic; color: var(--g600); font-size: .95rem; margin-bottom: .4rem; }
    .pred-conf-main { font-size: .9rem; color: var(--muted); }

    /* ── Prediction card — alternative ─────────────────────────────── */
    .pred-alt {
        background: #ffffff;
        border: 1px solid var(--g200);
        border-radius: var(--radius-md);
        padding: .8rem 1.1rem;
        margin-bottom: .45rem;
        box-shadow: var(--shadow-sm);
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .pred-alt-rank {
        font-size: .7rem;
        font-weight: 700;
        color: var(--g500);
        text-transform: uppercase;
        letter-spacing: .07em;
        min-width: 1.5rem;
    }
    .pred-alt-name { font-size: 1rem; font-weight: 700; color: var(--g700); flex: 1; }
    .pred-alt-conf { font-size: .9rem; font-weight: 600; color: var(--muted); white-space: nowrap; }

    /* ── Species info card ──────────────────────────────────────────── */
    .info-card {
        background: #ffffff;
        border: 1px solid var(--g200);
        border-radius: var(--radius-lg);
        padding: 1.5rem 1.75rem;
        box-shadow: var(--shadow-md);
        margin-top: .5rem;
    }
    .info-card h3 { color: var(--g800); font-size: 1.3rem; margin-bottom: .1rem; }
    .info-sci { font-style: italic; color: var(--g600); font-size: .97rem; margin-bottom: .85rem; }

    .pills { display: flex; flex-wrap: wrap; gap: .45rem; margin-bottom: .9rem; }
    .pill {
        background: var(--g50);
        border: 1px solid var(--g200);
        border-radius: 999px;
        padding: .22rem .75rem;
        font-size: .82rem;
        color: var(--g700);
    }
    .pill b { color: var(--g800); }

    .section-lbl {
        font-size: .72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .09em;
        color: var(--g500);
        margin: .85rem 0 .25rem;
    }

    /* ── Banners ────────────────────────────────────────────────────── */
    .banner-warn {
        background: #fffce5;
        border-left: 4px solid #f9a825;
        border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
        padding: .65rem .95rem;
        font-size: .88rem;
        color: #5d4037;
        margin: .5rem 0;
    }
    .banner-tip {
        background: #e3f2fd;
        border-left: 4px solid #1e88e5;
        border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
        padding: .65rem .95rem;
        font-size: .88rem;
        color: #1a237e;
        margin: .5rem 0;
    }

    /* ── Quiz card ──────────────────────────────────────────────────── */
    .quiz-card {
        background: #ffffff;
        border: 1px solid var(--g200);
        border-radius: var(--radius-lg);
        padding: 1.5rem 1.75rem;
        box-shadow: var(--shadow-md);
    }
    .quiz-q {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--g800);
        line-height: 1.45;
        margin-bottom: .9rem;
    }
    .score-bar {
        display: inline-flex;
        align-items: center;
        gap: .6rem;
        background: var(--g50);
        border: 1px solid var(--g200);
        border-radius: var(--radius-md);
        padding: .45rem 1rem;
        font-size: .9rem;
        color: var(--g700);
        margin-top: 1rem;
    }
    .score-bar b { color: var(--g800); }

    /* ── Streamlit overrides ────────────────────────────────────────── */
    div.stButton > button {
        border-radius: 999px;
        background: var(--g600);
        border: none;
        color: #ffffff;
        font-weight: 700;
        padding: .45rem 1.15rem;
        transition: background .15s;
    }
    div.stButton > button:hover { background: var(--g700) !important; color: #fff; }
    .stProgress > div > div > div > div { background: var(--g500); }
    section[data-testid="stSidebar"] { background: var(--g50); }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Cached loaders
# =============================================================================


@st.cache_data
def get_campus_img_b64() -> str:
    """Return Campus.jpg as a base64-encoded data URI for folium ImageOverlay."""
    with open(CAMPUS_IMG_PATH, "rb") as f:
        return base64.b64encode(f.read()).decode()


@st.cache_data(show_spinner=False)
def load_mapped_trees() -> list[dict]:
    """Read arboles_mapeados.csv and return rows as map-point dicts."""
    if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
        return []
    rows: list[dict] = []
    with open(CSV_PATH, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                rows.append({
                    "species_key":  row.get("species_key", ""),
                    "name":         row.get("common_name", ""),
                    "sci":          "",
                    "lat":          float(row["latitude"]),
                    "lon":          float(row["longitude"]),
                    "confidence":   float(row.get("confidence") or 0),
                    "gps_accuracy": row.get("gps_accuracy") or None,
                    "datetime":     row.get("datetime", ""),
                    "source":       row.get("source", ""),
                })
            except (ValueError, KeyError):
                continue
    return rows


@st.cache_data
def load_classes() -> list[str]:
    if not CLASSES_PATH.exists():
        st.error(f"Archivo de clases no encontrado: `{CLASSES_PATH}`")
        st.stop()
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        return [raw[str(i)] for i in range(len(raw))]
    return list(raw)


@st.cache_data
def load_species_info(path_str: str, mtime: float) -> dict:
    """Load info.json. Both path and file mtime are cache-key args so stale
    data is evicted automatically whenever the file is updated on disk."""
    _ = mtime  # used only as cache-key discriminator; not read inside the function
    path = Path(path_str)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Modelo ONNX no encontrado: `{MODEL_PATH}`")
        st.stop()
    sess = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name, sess.get_outputs()[0].name


class_names  = load_classes()
_info_mtime  = INFO_PATH.stat().st_mtime if INFO_PATH.exists() else 0.0
species_info = load_species_info(str(INFO_PATH), _info_mtime)  # mtime busts cache on file change
ort_sess, inp_name, out_name = load_model()

# =============================================================================
# Core helpers
# =============================================================================


def clean_name(key: str) -> str:
    """ceiba_roja → Ceiba Roja"""
    return key.replace("_", " ").title()


def get_info(key: str) -> dict:
    """Return species info dict, tolerating case/key mismatches."""
    return species_info.get(key) or species_info.get(key.lower()) or {}


def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB")
    img = ImageOps.fit(img, (IMAGE_SIZE, IMAGE_SIZE), method=Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return np.transpose(arr, (2, 0, 1))[np.newaxis].astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def build_campus_map(tree_points: list[dict]) -> folium.Map:
    """Return a folium map with Campus.jpg georeferenced overlay and tree markers."""
    m = folium.Map(location=CAMPUS_CENTER, zoom_start=17, tiles="CartoDB positron")

    folium.raster_layers.ImageOverlay(
        image=f"data:image/jpeg;base64,{get_campus_img_b64()}",
        bounds=CAMPUS_BOUNDS,
        opacity=0.85,
        name="Plano del campus",
        interactive=True,
        zindex=1,
    ).add_to(m)

    for pt in tree_points:
        folium.Marker(
            location=[pt["lat"], pt["lon"]],
            popup=folium.Popup(
                f"<b>{pt['name']}</b><br>"
                f"<i>{pt.get('sci', '')}</i><br>"
                f"<small>{pt['lat']:.6f}, {pt['lon']:.6f}</small>",
                max_width=220,
            ),
            tooltip=pt["name"],
            icon=folium.Icon(color="green", icon="tree", prefix="fa"),
        ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def _ensure_csv_headers() -> None:
    """Write column headers if the CSV is absent or empty."""
    if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=CSV_COLUMNS).writeheader()


def _get_github_token() -> str | None:
    try:
        return st.secrets["GITHUB_TOKEN"]
    except (KeyError, FileNotFoundError, AttributeError):
        return os.environ.get("GITHUB_TOKEN")


def _push_csv_to_github() -> tuple[bool, str]:
    """Upload the current CSV to GitHub via the Contents API."""
    token = _get_github_token()
    if not token:
        return False, "GITHUB_TOKEN no configurado — árbol guardado solo localmente."

    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_CSV_PATH}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept":        "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # 1. Fetch current SHA (required for updates; empty for a new file)
    sha = ""
    try:
        req = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            sha = json.loads(resp.read()).get("sha", "")
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            return False, f"Error al leer el archivo en GitHub: HTTP {exc.code}"
    except OSError as exc:
        return False, f"Error de red al leer SHA: {exc}"

    # 2. Encode local file
    with open(CSV_PATH, "rb") as fh:
        content_b64 = base64.b64encode(fh.read()).decode()

    # 3. PUT updated file
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    body: dict = {
        "message": f"TreeLens: actualizar arboles_mapeados.csv [{now_str}]",
        "content": content_b64,
    }
    if sha:
        body["sha"] = sha

    payload = json.dumps(body).encode()
    put_req = urllib.request.Request(
        api_url,
        data=payload,
        headers={**headers, "Content-Type": "application/json"},
        method="PUT",
    )
    try:
        with urllib.request.urlopen(put_req, timeout=15):
            return True, "✅ CSV sincronizado con GitHub correctamente."
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        return False, f"Error al actualizar GitHub: HTTP {exc.code} — {detail}"
    except OSError as exc:
        return False, f"Error de red al actualizar GitHub: {exc}"


def save_tree_to_csv(row: dict) -> None:
    """Append *row* to the CSV (creates headers if file is empty) and bust cache."""
    _ensure_csv_headers()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore").writerow(row)
    load_mapped_trees.clear()


def _init_map_points() -> None:
    """Populate session_state['map_points'] from the CSV on the first visit."""
    if "map_initialized" not in st.session_state:
        st.session_state["map_points"]    = load_mapped_trees()
        st.session_state["map_initialized"] = True


def run_inference(image: Image.Image, top_k: int = TOP_K) -> list[dict]:
    logits = ort_sess.run([out_name], {inp_name: preprocess(image)})[0][0]
    probs  = softmax(logits)
    idx    = probs.argsort()[::-1][:top_k]
    return [
        {"key": class_names[int(i)], "name": clean_name(class_names[int(i)]), "prob": float(probs[int(i)])}
        for i in idx
    ]


# =============================================================================
# UI — prediction results
# =============================================================================


def render_predictions(results: list[dict]) -> None:
    if not results:
        st.warning("No se obtuvieron predicciones.")
        return

    top  = results[0]
    conf = top["prob"]
    sci  = get_info(top["key"]).get("nombre_cientifico", "")

    st.markdown(
        f"""
        <div class="pred-main">
            <div class="pred-badge">🥇 Especie más probable</div>
            <div class="pred-name-main">{top['name']}</div>
            {'<div class="pred-sci">' + sci + '</div>' if sci else ''}
            <div class="pred-conf-main">Confianza: <strong>{conf:.1%}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(conf)

    if conf < 0.50:
        st.markdown(
            '<div class="banner-warn">⚠️ Confianza baja. Intenta con una foto más nítida '
            "donde se vean bien las hojas, flores, frutos o la silueta completa del árbol.</div>",
            unsafe_allow_html=True,
        )
    elif conf < 0.75:
        st.markdown(
            '<div class="banner-tip">💡 Confianza moderada. Verifica los rasgos botánicos '
            "de la especie antes de concluir.</div>",
            unsafe_allow_html=True,
        )

    if len(results) > 1:
        st.markdown("**Otras posibilidades:**")
        medals = ["🥈", "🥉"]
        for i, item in enumerate(results[1:], start=0):
            medal = medals[i] if i < len(medals) else f"#{i + 2}"
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(
                    f'<div class="pred-alt"><span class="pred-alt-rank">{medal}</span>'
                    f'<span class="pred-alt-name">{item["name"]}</span></div>',
                    unsafe_allow_html=True,
                )
                st.progress(item["prob"])
            with c2:
                st.markdown(
                    f"<div style='padding-top:.3rem;font-size:.92rem;"
                    f"font-weight:600;color:var(--muted,#4B6050);'>{item['prob']:.1%}</div>",
                    unsafe_allow_html=True,
                )


# =============================================================================
# UI — species info card
# =============================================================================


def _esc(text: str) -> str:
    """Escape text before embedding in HTML to prevent injection."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_species_card(key: str) -> None:
    """Render a full botanical info card for *key* using data from info.json.

    The entire card is built as a single HTML string and emitted in one
    st.markdown call. This is essential: Streamlit renders each st.*() call
    as an independent React component, so mixing an opening <div> in one call
    with st.write() calls afterwards leaves those widgets *outside* the div
    in the DOM — CSS wrapping and the card background never apply.
    Generating everything as one HTML block solves this completely.
    """
    info = get_info(key)

    if not info:
        st.info(
            f"No se encontró información adicional para **{clean_name(key)}** "
            "en `data/info.json`."
        )
        return

    common_name = info.get("nombre_comun") or clean_name(key)
    scientific  = info.get("nombre_cientifico", "")

    p: list[str] = ['<div class="info-card">']

    # ── Header ────────────────────────────────────────────────────────────
    p.append(
        f'<h3 style="color:#0f2d18;margin:0 0 .2rem">🌿 {_esc(common_name)}</h3>'
    )
    if scientific:
        p.append(f'<div class="info-sci">{_esc(scientific)}</div>')

    # ── Quick-fact pills ──────────────────────────────────────────────────
    pills: list[str] = []
    if info.get("familia"):
        pills.append(
            f"<span class='pill'>🏷️ Familia: <b>{_esc(info['familia'])}</b></span>"
        )
    if info.get("altura_aproximada"):
        pills.append(
            f"<span class='pill'>📏 Altura: <b>{_esc(info['altura_aproximada'])}</b></span>"
        )
    if pills:
        p.append(f"<div class='pills' style='margin:.65rem 0 .9rem'>{''.join(pills)}</div>")

    # ── Full-width narrative sections ─────────────────────────────────────
    for field, icon_label in [
        ("descripcion",        "📖 Descripción"),
        ("como_identificarlo", "🔎 Cómo identificarlo"),
    ]:
        val = info.get(field)
        if val:
            p.append(
                f'<div class="section-lbl">{icon_label}</div>'
                f'<p style="margin:.15rem 0 .7rem;line-height:1.55;color:#1C1C1C">'
                f'{_esc(val)}</p>'
            )

    # ── Two-column morphological details ──────────────────────────────────
    for (f1, l1), (f2, l2) in [
        (("hojas",  "🍃 Hojas"),  ("flores",      "🌸 Flores")),
        (("frutos", "🍑 Frutos"), ("distribucion", "🌍 Distribución")),
    ]:
        v1, v2 = info.get(f1), info.get(f2)
        if v1 or v2:
            p.append('<div style="display:flex;gap:1.5rem;flex-wrap:wrap;margin:.3rem 0 .4rem">')
            for v, lbl in ((v1, l1), (v2, l2)):
                if v:
                    p.append(
                        f'<div style="flex:1;min-width:180px">'
                        f'<div class="section-lbl">{lbl}</div>'
                        f'<p style="margin:.1rem 0;line-height:1.55;color:#1C1C1C">{_esc(v)}</p>'
                        f'</div>'
                    )
            p.append('</div>')

    # ── Uses ──────────────────────────────────────────────────────────────
    if info.get("usos"):
        p.append(
            f'<div class="section-lbl">🛠️ Usos</div>'
            f'<p style="margin:.15rem 0 .7rem;line-height:1.55;color:#1C1C1C">'
            f'{_esc(info["usos"])}</p>'
        )

    # ── Fun fact ──────────────────────────────────────────────────────────
    if info.get("dato_curioso"):
        p.append(
            f'<div class="banner-tip" style="margin-top:.5rem">'
            f'💡 <strong>Dato curioso:</strong> {_esc(info["dato_curioso"])}'
            f'</div>'
        )

    # ── Origin, history in Colombia and uses ──────────────────────────────
    historia = info.get("historia_origen_colombia_usos")
    p.append('<div class="section-lbl" style="margin-top:.85rem">🌎 Origen, historia en Colombia y usos</div>')
    if historia:
        p.append(
            f'<p style="margin:.15rem 0 .7rem;line-height:1.55;color:#1C1C1C">'
            f'{_esc(historia)}</p>'
        )
    else:
        p.append(
            '<p style="margin:.15rem 0 .7rem;font-style:italic;color:#4B6050">'
            'Información sobre el origen, historia en Colombia y usos aún no disponible para esta especie.'
            '</p>'
        )

    p.append('</div>')
    st.markdown('\n'.join(p), unsafe_allow_html=True)


# =============================================================================
# Trivia engine
# =============================================================================

_TRIVIA_TEMPLATES: list[tuple[str, str]] = [
    ("nombre_cientifico",  "¿Cuál es el nombre científico de {n}?"),
    ("familia",            "¿A qué familia botánica pertenece {n}?"),
    ("altura_aproximada",  "¿Cuánto puede llegar a medir {n}?"),
    ("usos",               "¿Para qué se usa principalmente {n}?"),
    ("hojas",              "¿Cómo son las hojas de {n}?"),
    ("flores",             "¿Cómo son las flores de {n}?"),
    ("frutos",             "¿Cómo son los frutos de {n}?"),
    ("distribucion",       "¿En qué zonas se encuentra {n}?"),
    ("dato_curioso",       "¿Cuál es un dato curioso sobre {n}?"),
]


def _pick_distractors(field: str, exclude_key: str, n: int = 3) -> list[str]:
    seen: set[str] = set()
    candidates: list[str] = []
    for sp_key, sp_info in species_info.items():
        if sp_key == exclude_key:
            continue
        val = sp_info.get(field)
        if val and val not in seen:
            seen.add(val)
            candidates.append(val)
    random.shuffle(candidates)
    return candidates[:n]


def build_question(species_key: str) -> dict | None:
    info = get_info(species_key)
    if not info:
        return None
    nombre = info.get("nombre_comun") or clean_name(species_key)

    templates = _TRIVIA_TEMPLATES.copy()
    random.shuffle(templates)

    for field, template in templates:
        answer = info.get(field)
        if not answer:
            continue
        distractors = _pick_distractors(field, species_key, n=3)
        if len(distractors) < 2:
            continue
        options = list(dict.fromkeys([answer] + distractors))
        random.shuffle(options)
        return {
            "question": template.format(n=nombre),
            "options":  options,
            "answer":   answer,
        }
    return None


MAX_TRIVIA_QUESTIONS = 5


def render_trivia(species_key: str) -> None:
    """
    Stateful trivia widget limited to MAX_TRIVIA_QUESTIONS per round.
    Uses a counter (tv_ctr) as part of widget keys so that every new question
    creates fresh radio/button widgets, resetting any prior selection.
    """
    # Session-state keys (short, collision-safe)
    QK    = "tv_q"       # current question dict
    SPK   = "tv_sp"      # species the question belongs to
    STK   = "tv_st"      # "ask" | "fb" (feedback) | "done"
    SELK  = "tv_sel"     # user's selected answer
    SCRK  = "tv_score"
    TOTK  = "tv_total"   # questions answered this round
    CTRK  = "tv_ctr"     # monotonic counter → unique widget keys per question

    for key, default in [(SCRK, 0), (TOTK, 0), (CTRK, 0), (STK, "ask")]:
        if key not in st.session_state:
            st.session_state[key] = default

    # Auto-generate question when species changes or on first visit
    if QK not in st.session_state or st.session_state.get(SPK) != species_key:
        st.session_state[SPK]  = species_key
        st.session_state[QK]   = build_question(species_key)
        st.session_state[STK]  = "ask"
        st.session_state[SELK] = None
        st.session_state[SCRK] = 0
        st.session_state[TOTK] = 0
        st.session_state[CTRK] += 1

    score = st.session_state[SCRK]
    total = st.session_state[TOTK]

    # ── Round complete ─────────────────────────────────────────────────────
    if st.session_state[STK] == "done" or total >= MAX_TRIVIA_QUESTIONS:
        pct = score / total if total else 0
        st.markdown('<div class="quiz-card">', unsafe_allow_html=True)
        st.markdown(
            f"<div class='quiz-q'>🎉 ¡Trivia completada!</div>"
            f"<p style='color:#1C1C1C'>Respondiste <strong>{total}</strong> preguntas sobre "
            f"<strong>{get_info(species_key).get('nombre_comun') or clean_name(species_key)}</strong>.</p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='score-bar'>🏆 Resultado final: <b>{score}/{total}</b>&nbsp;—&nbsp;{pct:.0%}</div>",
            unsafe_allow_html=True,
        )
        if st.button("🔄 Jugar de nuevo", key="tv_reset"):
            st.session_state[SCRK] = 0
            st.session_state[TOTK] = 0
            st.session_state[STK]  = "ask"
            st.session_state[QK]   = build_question(species_key)
            st.session_state[SELK] = None
            st.session_state[CTRK] += 1
            st.rerun()
        return

    q = st.session_state[QK]
    if q is None:
        st.warning("No hay suficientes datos para generar preguntas de esta especie.")
        return

    ctr = st.session_state[CTRK]
    q_num = total + 1 if st.session_state[STK] == "ask" else total

    st.markdown('<div class="quiz-card">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="quiz-q">🧠 Pregunta {q_num}/{MAX_TRIVIA_QUESTIONS} — {q["question"]}</div>',
        unsafe_allow_html=True,
    )

    if st.session_state[STK] == "ask":
        choice = st.radio(
            "Elige tu respuesta:",
            q["options"],
            key=f"tv_radio_{ctr}",
            label_visibility="collapsed",
        )
        if st.button("✅ Confirmar respuesta", key=f"tv_confirm_{ctr}"):
            st.session_state[SELK] = choice
            st.session_state[TOTK] += 1
            if choice == q["answer"]:
                st.session_state[SCRK] += 1
            st.session_state[STK] = "fb"
            st.rerun()

    else:  # feedback state
        chosen  = st.session_state.get(SELK)
        correct = (chosen == q["answer"])
        for opt in q["options"]:
            if opt == q["answer"]:
                st.success(f"✅ {opt}")
            elif opt == chosen:
                st.error(f"❌ {opt}")
            else:
                st.write(f"   {opt}")

        if correct:
            st.success("**¡Correcto!**")
        else:
            st.error(f"**Incorrecto.** La respuesta era: **{q['answer']}**")

        answered = st.session_state[TOTK]
        if answered >= MAX_TRIVIA_QUESTIONS:
            if st.button("🏁 Ver resultados", key=f"tv_finish_{ctr}"):
                st.session_state[STK] = "done"
                st.rerun()
        else:
            if st.button("➡️ Siguiente pregunta", key=f"tv_next_{ctr}"):
                st.session_state[QK]   = build_question(species_key)
                st.session_state[STK]  = "ask"
                st.session_state[SELK] = None
                st.session_state[CTRK] += 1
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # Score progress during round
    cur_score = st.session_state[SCRK]
    cur_total = st.session_state[TOTK]
    if cur_total > 0:
        pct = cur_score / cur_total
        st.markdown(
            f"<div class='score-bar'>🏆 Puntaje: <b>{cur_score}/{cur_total}</b>&nbsp;—&nbsp;{pct:.0%}"
            f"&nbsp;|&nbsp;Pregunta {cur_total}/{MAX_TRIVIA_QUESTIONS}</div>",
            unsafe_allow_html=True,
        )


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.markdown("## 🌳 TreeLens")
    st.write("Identificador de especies arbóreas para el Arboretum y Palmetum de la UNAL Medellín.")
    st.markdown(f"**Modelo:** {len(class_names)} especies")
    st.markdown("---")

    if species_info:
        st.markdown("### Especies del modelo")
        for k in class_names[:18]:
            label = get_info(k).get("nombre_comun") or clean_name(k)
            st.caption(f"• {label}")
        if len(class_names) > 18:
            st.caption(f"… y {len(class_names) - 18} más — ver en **Explorar**")
    else:
        st.caption("Información botánica no disponible (`data/info.json`).")

    st.markdown("---")
    st.caption(
        "Este clasificador es una herramienta de apoyo. "
        "Para identificaciones definitivas consulta un especialista en botánica."
    )


# =============================================================================
# Hero banner
# =============================================================================

st.markdown(
    """
    <div class="hero">
        <h1>🌳 TreeLens: Identificador de Especies Arbóreas</h1>
        <p>
        Sistema de identificación para el <strong>Arboretum y Palmetum de la Universidad Nacional
        de Colombia, sede Medellín</strong>. Carga una imagen o toma una foto con la cámara y
        el modelo reconocerá la especie más probable junto con su información botánica.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Tabs
# =============================================================================

tab_classify, tab_catalog, tab_trivia, tab_map = st.tabs(
    ["📷 Clasificar", "📚 Explorar especies", "🧠 Trivia botánica", "🗺️ Mapa del campus"]
)

# ── Tab 1: Classify ───────────────────────────────────────────────────────────

with tab_classify:
    st.markdown("## Identificar una imagen")
    st.markdown(
        "Sube una fotografía o usa la cámara. Para mejores resultados elige imágenes nítidas "
        "donde se aprecien bien las hojas, flores, frutos o la silueta completa del árbol."
    )

    mode = st.radio(
        "Fuente de imagen:",
        ["📁 Subir desde el computador", "📷 Tomar foto con la cámara"],
        horizontal=True,
    )

    uploaded_file = None
    if mode == "📁 Subir desde el computador":
        uploaded_file = st.file_uploader(
            "Selecciona una imagen (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
        )
    else:
        uploaded_file = st.camera_input("Captura una foto")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col_img, col_res = st.columns([1, 1.4], gap="large")

        with col_img:
            st.image(image, caption="Imagen cargada", use_container_width=True)

        with col_res:
            with st.spinner("🔍 Analizando imagen…"):
                results = run_inference(image, top_k=min(TOP_K, len(class_names)))
            st.session_state["detected_species"]    = results[0]["key"]
            st.session_state["detected_confidence"] = results[0]["prob"]
            render_predictions(results)

        st.markdown("---")
        st.markdown("### Información de la especie identificada")
        render_species_card(results[0]["key"])

    else:
        st.info("📸 Carga una imagen para iniciar la identificación.")

# ── Tab 2: Catalog ────────────────────────────────────────────────────────────

with tab_catalog:
    st.markdown("## Catálogo de especies")

    if not species_info:
        st.warning(
            "No se encontró el archivo `data/info.json`. "
            "El catálogo botánico no está disponible en este momento."
        )
    else:
        st.markdown(
            "Consulta la ficha botánica de cualquier especie del modelo. "
            "Escribe el nombre en el selector para buscar rápidamente."
        )
        selected = st.selectbox(
            "Selecciona una especie:",
            class_names,
            format_func=lambda k: (get_info(k).get("nombre_comun") or clean_name(k))
            + f"  ({clean_name(k)})",
        )
        render_species_card(selected)

# ── Tab 3: Trivia ─────────────────────────────────────────────────────────────

with tab_trivia:
    st.markdown("## Trivia botánica")
    st.markdown(
        "Pon a prueba tu conocimiento sobre las especies del Arboretum y Palmetum. "
        "Cada ronda tiene hasta 5 preguntas. Cuando el modelo identifica una especie, "
        "la trivia cambia automáticamente a esa especie."
    )

    if not species_info:
        st.warning(
            "No se encontró el archivo `data/info.json`. "
            "La trivia no está disponible sin información botánica."
        )
    else:
        valid_keys = [k for k in class_names if get_info(k)]

        if not valid_keys:
            st.warning("No hay especies con información suficiente para la trivia.")
        else:
            # Auto-switch to the last detected species when it changes
            detected = st.session_state.get("detected_species")
            last_auto = st.session_state.get("trivia_last_auto")
            if detected and detected in valid_keys and detected != last_auto:
                st.session_state["trivia_selector"] = detected
                st.session_state["trivia_last_auto"] = detected

            trivia_key = st.selectbox(
                "Especie para practicar:",
                valid_keys,
                format_func=lambda k: get_info(k).get("nombre_comun") or clean_name(k),
                key="trivia_selector",
            )

            if detected and detected in valid_keys:
                detected_name = get_info(detected).get("nombre_comun") or clean_name(detected)
                st.caption(f"🔍 Especie detectada: **{detected_name}**")

            st.markdown("---")
            render_trivia(trivia_key)

# ── Tab 4: Campus map ─────────────────────────────────────────────────────────

with tab_map:
    _init_map_points()   # load CSV into session state on first visit

    st.markdown("## 🗺️ Mapa del campus")
    st.markdown(
        "Activa la ubicación GPS de tu dispositivo para marcar en el mapa dónde se encuentra "
        "el árbol que acabas de identificar. Los puntos se guardan en el repositorio."
    )

    # ── GitHub token status (sidebar-style note) ───────────────────────────────
    if not _get_github_token():
        st.warning(
            "⚠️ **GITHUB_TOKEN no configurado.** Los árboles se guardarán en el CSV local "
            "pero no se sincronizarán con el repositorio. "
            "Copia `.streamlit/secrets.toml.example` a `.streamlit/secrets.toml` y añade tu token.",
            icon=None,
        )

    # ── Geolocation widget ─────────────────────────────────────────────────────
    location = streamlit_geolocation()

    col_ctrl, col_map_view = st.columns([1, 2.5], gap="large")

    with col_ctrl:
        # Show flash message from previous save action
        if flash := st.session_state.pop("_map_flash", None):
            ftype, ftext = flash
            if ftype == "success":
                st.success(ftext)
            elif ftype == "warning":
                st.warning(ftext)
            else:
                st.info(ftext)

        lat = lon = acc = None
        if location and location.get("latitude") is not None:
            lat = float(location["latitude"])
            lon = float(location["longitude"])
            acc = location.get("accuracy")

            acc_str = f"\n\nPrecisión: ±{acc:.0f} m" if acc is not None else ""
            st.success(
                f"**📍 Ubicación actual**\n\nLat: `{lat:.6f}`\n\nLon: `{lon:.6f}`{acc_str}"
            )

            detected = st.session_state.get("detected_species")
            if detected:
                sp_name = get_info(detected).get("nombre_comun") or clean_name(detected)
                sp_sci  = get_info(detected).get("nombre_cientifico", "")
                conf    = float(st.session_state.get("detected_confidence", 0.0))
                st.markdown(
                    f"**Especie detectada:** {sp_name}  \n"
                    f"Confianza: **{conf:.1%}**"
                )

                if st.button("💾 Guardar árbol en el mapa", key="map_save"):
                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # 1 — Persist to CSV
                    csv_row = {
                        "species_key":  detected,
                        "common_name":  sp_name,
                        "confidence":   f"{conf:.4f}",
                        "latitude":     f"{lat:.7f}",
                        "longitude":    f"{lon:.7f}",
                        "gps_accuracy": f"{acc:.1f}" if acc is not None else "",
                        "datetime":     now_str,
                        "source":       "gps",
                    }
                    save_tree_to_csv(csv_row)

                    # 2 — Update session state
                    st.session_state["map_points"].append({
                        "species_key": detected,
                        "name":        sp_name,
                        "sci":         sp_sci,
                        "lat":         lat,
                        "lon":         lon,
                        "confidence":  conf,
                        "datetime":    now_str,
                        "source":      "gps",
                    })

                    # 3 — Push to GitHub
                    ok, msg = _push_csv_to_github()
                    flash_type = "success" if ok else "warning"
                    flash_text = (
                        f"✅ **{sp_name}** guardado en el mapa y sincronizado con GitHub."
                        if ok
                        else f"🌳 **{sp_name}** guardado localmente. {msg}"
                    )
                    st.session_state["_map_flash"] = (flash_type, flash_text)
                    st.rerun()
            else:
                st.info(
                    "Identifica una especie en la pestaña **📷 Clasificar** "
                    "y luego guárdala aquí con su coordenada GPS."
                )
        else:
            st.info(
                "Haz clic en **Get Location** para activar tu GPS "
                "y marcar un árbol en el mapa."
            )

        # ── Saved points list ──────────────────────────────────────────────────
        saved_pts: list[dict] = st.session_state.get("map_points", [])
        if saved_pts:
            st.markdown("---")
            st.markdown(f"**Árboles guardados ({len(saved_pts)})**")
            for i, pt in enumerate(saved_pts):
                c1, c2 = st.columns([5, 1])
                with c1:
                    conf_str = f" · {pt['confidence']:.0%}" if pt.get("confidence") else ""
                    dt_str   = f"\n{pt['datetime']}" if pt.get("datetime") else ""
                    st.caption(
                        f"🌳 **{pt['name']}**{conf_str}\n"
                        f"{pt['lat']:.5f}, {pt['lon']:.5f}{dt_str}"
                    )
                with c2:
                    if st.button("🗑️", key=f"map_del_{i}", help="Eliminar de la vista"):
                        st.session_state["map_points"].pop(i)
                        st.rerun()

    with col_map_view:
        tree_points = st.session_state.get("map_points", [])
        campus_map  = build_campus_map(tree_points)
        st_folium(campus_map, use_container_width=True, height=580, returned_objects=[])
