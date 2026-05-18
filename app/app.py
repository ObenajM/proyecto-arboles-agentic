from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps


# =====================================================
# Rutas del proyecto
# Funciona si el archivo está en:
# - app/app.py  -> usa la raíz del repositorio
# - app.py      -> usa la carpeta actual
# =====================================================

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent if (APP_DIR.parent / "models").exists() else APP_DIR

MODEL_PATH = BASE_DIR / "models" / "modelo_arboles.onnx"
CLASSES_PATH = BASE_DIR / "models" / "clases.json"
INFO_PATH = BASE_DIR / "data" / "species_info.json"

IMAGE_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# =====================================================
# Configuración visual
# =====================================================

st.set_page_config(
    page_title="TreeLens | Clasificador de árboles",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
.main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
.hero {
    padding: 1.5rem 1.8rem;
    border-radius: 1.25rem;
    background: linear-gradient(135deg, #E8F5E9 0%, #FFFFFF 60%, #F1F8E9 100%);
    border: 1px solid #D7EAD7;
    margin-bottom: 1.1rem;
}
.hero h1 { margin-bottom: .25rem; }
.small-muted { color: #5f6f64; font-size: .95rem; }
.pred-card {
    padding: .9rem 1rem;
    border-radius: .9rem;
    background: #ffffff;
    border: 1px solid #E2E8E2;
    box-shadow: 0 1px 8px rgba(0,0,0,.04);
    margin-bottom: .7rem;
}
.info-card {
    padding: 1.1rem 1.25rem;
    border-radius: 1rem;
    background: #ffffff;
    border: 1px solid #E1EAE1;
    margin-bottom: 1rem;
}
.badge {
    display: inline-block;
    padding: .2rem .55rem;
    border-radius: 999px;
    background: #E8F5E9;
    color: #1B5E20;
    font-size: .85rem;
    border: 1px solid #CFE8D1;
}
.warning-box {
    padding: .85rem 1rem;
    border-radius: .8rem;
    background: #FFF8E1;
    border: 1px solid #FFE082;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =====================================================
# Carga de archivos
# =====================================================

@st.cache_resource(show_spinner="Cargando modelo ONNX...")
def load_session() -> ort.InferenceSession:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}")

    return ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )


@st.cache_data(show_spinner=False)
def load_labels() -> list[str]:
    if not CLASSES_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo de clases en: {CLASSES_PATH}")

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    if isinstance(labels, list):
        return labels

    # Soporta también formato {"0": "aguacate", "1": "caucho", ...}
    if isinstance(labels, dict):
        if all(str(k).isdigit() for k in labels.keys()):
            return [labels[str(i)] for i in range(len(labels))]

        # Soporta formato {"aguacate": 0, "caucho": 1, ...}
        inverted = {int(v): k for k, v in labels.items()}
        return [inverted[i] for i in range(len(inverted))]

    raise ValueError("Formato de clases.json no reconocido.")


@st.cache_data(show_spinner=False)
def load_species_info() -> dict[str, dict[str, Any]]:
    if not INFO_PATH.exists():
        return {}

    with open(INFO_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =====================================================
# Inferencia
# =====================================================

def normalize_label(label: str) -> str:
    return label.replace("_", " ").title()


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Mismo preprocesamiento usado en entrenamiento: RGB, Resize, ToTensor y Normalize."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)

    array = np.asarray(image).astype(np.float32) / 255.0
    array = (array - MEAN) / STD
    array = np.transpose(array, (2, 0, 1))
    array = np.expand_dims(array, axis=0)

    return array.astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float32)
    logits = logits - np.max(logits)
    exp_values = np.exp(logits)
    return exp_values / np.sum(exp_values)


def predict(image: Image.Image, top_k: int = 5) -> pd.DataFrame:
    session = load_session()
    labels = load_labels()
    input_array = preprocess_image(image)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    logits = session.run([output_name], {input_name: input_array})[0][0]
    probabilities = softmax(logits)

    top_indices = probabilities.argsort()[::-1][:top_k]

    rows = []
    for idx in top_indices:
        idx = int(idx)
        rows.append(
            {
                "clase": labels[idx],
                "especie": normalize_label(labels[idx]),
                "probabilidad": float(probabilities[idx]),
                "porcentaje": float(probabilities[idx] * 100),
            }
        )

    return pd.DataFrame(rows)


def confidence_message(confidence: float) -> tuple[str, str]:
    if confidence >= 0.80:
        return "Alta", "Predicción fuerte. Aun así, valida con rasgos botánicos visibles."
    if confidence >= 0.50:
        return "Media", "Predicción razonable. Conviene tomar otra foto con mejor ángulo o iluminación."
    return "Baja", "La imagen puede no mostrar rasgos suficientes o estar fuera del conjunto de entrenamiento."


# =====================================================
# Componentes visuales
# =====================================================

def show_species_card(class_key: str, confidence: float | None = None) -> None:
    info = load_species_info().get(class_key, {})

    display_name = info.get("nombre_comun", normalize_label(class_key))
    scientific = info.get("nombre_cientifico", "No disponible")
    family = info.get("familia", "No disponible")
    origin = info.get("origen", "No disponible")
    description = info.get("descripcion", "Información no disponible.")
    traits = info.get("rasgos", [])
    uses = info.get("usos", [])
    recommendation = info.get(
        "recomendacion_foto",
        "Toma una foto nítida de hojas, flores, frutos o tronco.",
    )
    note = info.get("nota", "")

    conf_html = ""
    if confidence is not None:
        conf_html = f"<span class='badge'>Confianza: {confidence * 100:.2f}%</span>"

    st.markdown(
        f"""
        <div class="info-card">
            <h3>🌿 {display_name}</h3>
            {conf_html}
            <p><b>Nombre científico:</b> <i>{scientific}</i></p>
            <p><b>Familia:</b> {family}</p>
            <p><b>Origen/distribución:</b> {origin}</p>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if traits:
        st.markdown("**Rasgos útiles para reconocerla:**")
        for trait in traits:
            st.markdown(f"- {trait}")

    if uses:
        st.markdown("**Usos o importancia:**")
        for use in uses:
            st.markdown(f"- {use}")

    st.info(f"📷 {recommendation}")

    if note:
        st.caption(f"Nota: {note}")


def build_report(predictions: pd.DataFrame) -> dict[str, Any]:
    best = predictions.iloc[0].to_dict()
    return {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "modelo": "modelo_arboles.onnx",
        "prediccion_principal": best,
        "top_predicciones": predictions.to_dict(orient="records"),
        "nota": "Resultado generado por un prototipo de clasificación de imágenes. Validar con criterio botánico.",
    }


# =====================================================
# Sidebar
# =====================================================

try:
    labels = load_labels()
except Exception as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.title("🌳 TreeLens")
    st.caption("Clasificador de especies de árboles")

    top_k = st.slider(
        "Número de predicciones",
        min_value=3,
        max_value=min(10, len(labels)),
        value=min(5, len(labels)),
    )

    confidence_threshold = st.slider(
        "Umbral de advertencia",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05,
    )

    st.divider()
    st.markdown("**Modelo**")
    st.write("ONNX Runtime · CPU")
    st.write(f"Entrada: {IMAGE_SIZE} × {IMAGE_SIZE}")
    st.write(f"Clases: {len(labels)}")

    st.divider()
    st.caption(
        "Recomendación: usa fotos nítidas, con buena luz y donde se vean hojas, flores, frutos o tronco."
    )


# =====================================================
# App principal
# =====================================================

st.markdown(
    """
    <div class="hero">
        <h1>🌳 TreeLens: identificación visual de árboles</h1>
        <p class="small-muted">
        Sube una imagen o toma una foto desde la cámara. La app usa tu modelo ONNX para estimar la especie y mostrar información botánica básica.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

main_tab, catalog_tab, about_tab = st.tabs(
    ["🔎 Clasificar", "📚 Catálogo", "ℹ️ Acerca del modelo"]
)

with main_tab:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("1. Ingresa una imagen")
        input_mode = st.radio(
            "Modo de entrada",
            ["Subir imagen", "Tomar foto"],
            horizontal=True,
            label_visibility="collapsed",
        )

        image_file = None
        if input_mode == "Subir imagen":
            image_file = st.file_uploader(
                "Sube una imagen JPG, JPEG o PNG",
                type=["jpg", "jpeg", "png"],
            )
        else:
            image_file = st.camera_input("Toma una foto del árbol, hoja, flor o fruto")

        image = None
        if image_file is not None:
            image = Image.open(image_file)
            image = ImageOps.exif_transpose(image).convert("RGB")
            st.image(image, caption="Imagen analizada", use_container_width=True)

    with right:
        st.subheader("2. Resultado")

        if image is None:
            st.markdown(
                """
                <div class="warning-box">
                Sube una imagen o toma una foto para ejecutar la clasificación.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            try:
                with st.spinner("Analizando imagen..."):
                    predictions = predict(image, top_k=top_k)
            except Exception as exc:
                st.error(f"No fue posible ejecutar la predicción: {exc}")
                st.stop()

            best = predictions.iloc[0]
            level, message = confidence_message(float(best["probabilidad"]))

            st.metric(
                label="Predicción principal",
                value=str(best["especie"]),
                delta=f"{best['porcentaje']:.2f}% · Confianza {level}",
            )
            st.caption(message)

            if float(best["probabilidad"]) < confidence_threshold:
                st.warning(
                    "La confianza está por debajo del umbral seleccionado. Prueba con otra foto que muestre mejor hojas, flores, frutos o tronco."
                )

            st.markdown("### Top predicciones")
            for i, row in predictions.iterrows():
                st.markdown(
                    f"""
                    <div class="pred-card">
                        <b>{i + 1}. {row['especie']}</b><br>
                        <span class="small-muted">Probabilidad: {row['porcentaje']:.2f}%</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(float(row["probabilidad"]))

            st.markdown("### Información de la especie más probable")
            show_species_card(str(best["clase"]), float(best["probabilidad"]))

            report = build_report(predictions)
            st.download_button(
                "Descargar resultado en JSON",
                data=json.dumps(report, ensure_ascii=False, indent=2),
                file_name="resultado_treelens.json",
                mime="application/json",
            )

with catalog_tab:
    st.subheader("Catálogo de especies incluidas")
    species_info = load_species_info()

    selected = st.selectbox(
        "Selecciona una especie",
        labels,
        format_func=normalize_label,
    )
    show_species_card(selected)

    st.markdown("### Tabla resumen")
    table_rows = []
    for label in labels:
        info = species_info.get(label, {})
        table_rows.append(
            {
                "Clase": label,
                "Nombre común": info.get("nombre_comun", normalize_label(label)),
                "Nombre científico": info.get("nombre_cientifico", ""),
                "Familia": info.get("familia", ""),
                "Origen/distribución": info.get("origen", ""),
            }
        )

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

with about_tab:
    st.subheader("Acerca de esta app")
    st.markdown(
        """
        Esta aplicación ejecuta un modelo de clasificación de imágenes convertido a ONNX.  
        El flujo de inferencia es:

        1. Convertir la imagen a RGB.
        2. Redimensionar a 224 × 224 píxeles.
        3. Normalizar con media y desviación estándar de ImageNet.
        4. Ejecutar el modelo ONNX con ONNX Runtime.
        5. Aplicar softmax y mostrar las clases más probables.
        """
    )

    st.warning(
        "Esta app es una ayuda académica y de prototipo. No debe usarse como identificación botánica definitiva sin validación experta."
    )

    st.markdown("### Estructura esperada para despliegue")
    st.code(
        """.
├── app/
│   └── app.py
├── models/
│   ├── modelo_arboles.onnx
│   └── clases.json
├── data/
│   └── species_info.json
├── requirements.txt
└── .streamlit/
    └── config.toml
""",
        language="text",
    )
