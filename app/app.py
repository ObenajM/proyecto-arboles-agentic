"""
app/app.py — TreeLens: Identificador de Especies Arbóreas
Interfaz Streamlit con ONNX Runtime para el Arboretum y Palmetum de la UNAL Medellín.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image, ImageOps

# =============================================================================
# Paths & constants
# =============================================================================

BASE_DIR     = Path(__file__).resolve().parent.parent
MODEL_PATH   = BASE_DIR / "models" / "modelo_arboles_best.onnx"
CLASSES_PATH = BASE_DIR / "models" / "clases.json"
INFO_PATH    = BASE_DIR / "data"   / "info.json"

IMAGE_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TOP_K = 3

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
def load_species_info() -> dict:
    if not INFO_PATH.exists():
        return {}
    with open(INFO_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Modelo ONNX no encontrado: `{MODEL_PATH}`")
        st.stop()
    sess = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name, sess.get_outputs()[0].name


class_names  = load_classes()
species_info = load_species_info()
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


def render_species_card(key: str) -> None:
    info        = get_info(key)
    common_name = info.get("nombre_comun") or clean_name(key)
    scientific  = info.get("nombre_cientifico", "")

    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown(f"### 🌿 {common_name}")
    if scientific:
        st.markdown(f'<div class="info-sci">{scientific}</div>', unsafe_allow_html=True)

    if not info:
        st.markdown(
            '<div class="banner-tip">ℹ️ No hay información botánica registrada para esta especie.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Quick-fact pills: familia + altura_aproximada
    pill_defs = [
        ("familia",            "🏷️ Familia"),
        ("altura_aproximada",  "📏 Altura"),
    ]
    pills = [
        f"<span class='pill'>{label}: <b>{info[field]}</b></span>"
        for field, label in pill_defs
        if info.get(field)
    ]
    if pills:
        st.markdown(f"<div class='pills'>{''.join(pills)}</div>", unsafe_allow_html=True)

    # Description
    if info.get("descripcion"):
        st.markdown('<div class="section-lbl">📖 Descripción</div>', unsafe_allow_html=True)
        st.write(info["descripcion"])

    # How to identify
    if info.get("como_identificarlo"):
        st.markdown('<div class="section-lbl">🔎 Cómo identificarlo</div>', unsafe_allow_html=True)
        st.write(info["como_identificarlo"])

    # Morphological details in two columns
    morfo_pairs = [
        ("hojas",  "🍃 Hojas",  "flores", "🌸 Flores"),
        ("frutos", "🍑 Frutos", "distribucion", "🌍 Distribución"),
    ]
    for f1, l1, f2, l2 in morfo_pairs:
        v1, v2 = info.get(f1), info.get(f2)
        if v1 or v2:
            c1, c2 = st.columns(2)
            with c1:
                if v1:
                    st.markdown(f'<div class="section-lbl">{l1}</div>', unsafe_allow_html=True)
                    st.write(v1)
            with c2:
                if v2:
                    st.markdown(f'<div class="section-lbl">{l2}</div>', unsafe_allow_html=True)
                    st.write(v2)

    # Uses
    if info.get("usos"):
        st.markdown('<div class="section-lbl">🛠️ Usos</div>', unsafe_allow_html=True)
        st.write(info["usos"])

    # Fun fact
    if info.get("dato_curioso"):
        st.markdown(
            f'<div class="banner-tip">💡 <strong>Dato curioso:</strong> {info["dato_curioso"]}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


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


def render_trivia(species_key: str) -> None:
    """
    Stateful single-question trivia widget.
    Uses a counter (tv_ctr) as part of widget keys so that every new question
    creates fresh radio/button widgets, resetting any prior selection.
    """
    # Session-state keys (short, collision-safe)
    QK    = "tv_q"       # current question dict
    SPK   = "tv_sp"      # species the question belongs to
    STK   = "tv_st"      # "ask" | "fb" (feedback)
    SELK  = "tv_sel"     # user's selected answer
    SCRK  = "tv_score"
    TOTK  = "tv_total"
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
        st.session_state[CTRK] += 1

    q = st.session_state[QK]
    if q is None:
        st.warning("No hay suficientes datos para generar preguntas de esta especie.")
        return

    ctr = st.session_state[CTRK]

    st.markdown('<div class="quiz-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="quiz-q">🧠 {q["question"]}</div>', unsafe_allow_html=True)

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

        if st.button("➡️ Siguiente pregunta", key=f"tv_next_{ctr}"):
            st.session_state[QK]   = build_question(species_key)
            st.session_state[STK]  = "ask"
            st.session_state[SELK] = None
            st.session_state[CTRK] += 1
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # Score display
    score = st.session_state[SCRK]
    total = st.session_state[TOTK]
    if total > 0:
        pct = score / total
        st.markdown(
            f"<div class='score-bar'>🏆 Puntaje: <b>{score}/{total}</b>&nbsp;—&nbsp;{pct:.0%}</div>",
            unsafe_allow_html=True,
        )
        if st.button("🔄 Reiniciar puntaje", key="tv_reset"):
            st.session_state[SCRK] = 0
            st.session_state[TOTK] = 0
            st.rerun()


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

tab_classify, tab_catalog, tab_trivia = st.tabs(
    ["📷 Clasificar", "📚 Explorar especies", "🧠 Trivia botánica"]
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
        "Selecciona una especie o elige una al azar, responde la pregunta y avanza "
        "acumulando puntaje."
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
            col_sel, col_rand = st.columns([3, 1], gap="medium")

            with col_sel:
                trivia_key = st.selectbox(
                    "Especie para practicar:",
                    valid_keys,
                    format_func=lambda k: get_info(k).get("nombre_comun") or clean_name(k),
                    key="trivia_selector",
                )

            with col_rand:
                # Vertical alignment trick
                st.markdown(
                    "<div style='padding-top:1.75rem;'>",
                    unsafe_allow_html=True,
                )
                if st.button("🎲 Aleatoria", key="trivia_random"):
                    st.session_state["trivia_selector"] = random.choice(valid_keys)
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("---")
            render_trivia(trivia_key)
