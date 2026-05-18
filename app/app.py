from pathlib import Path
import json
import random

import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image, ImageOps


# =====================================================
# Rutas del proyecto
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "modelo_arboles.onnx"
CLASSES_PATH = BASE_DIR / "models" / "clases.json"
INFO_PATH = BASE_DIR / "data" / "species_info.json"


# =====================================================
# Configuración general
# =====================================================

IMAGE_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

st.set_page_config(
    page_title="TreeLens | Clasificador de árboles",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================
# Estilos visuales
# =====================================================

def aplicar_estilos():
    st.markdown(
        """
        <style>
        :root {
            --green-dark: #123524;
            --green-main: #1B5E20;
            --green-soft: #E8F5E9;
            --green-card: #F1F8E9;
            --green-line: #A5D6A7;
            --text-main: #1B1B1B;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(129, 199, 132, 0.28), transparent 28%),
                linear-gradient(180deg, #FBFFF9 0%, #F4FBF2 100%);
        }

        h1, h2, h3 {
            color: var(--green-dark);
        }

        .hero-card {
            padding: 1.4rem 1.6rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 52%, #66BB6A 100%);
            color: white;
            box-shadow: 0 12px 30px rgba(27, 94, 32, 0.20);
            margin-bottom: 1rem;
        }

        .hero-card h1 {
            color: white;
            margin-bottom: 0.2rem;
        }

        .hero-card p {
            color: #F1F8E9;
            font-size: 1.05rem;
            margin-bottom: 0;
        }

        .green-card {
            padding: 1.15rem 1.25rem;
            border-radius: 20px;
            border: 1px solid var(--green-line);
            background: rgba(241, 248, 233, 0.90);
            box-shadow: 0 8px 22px rgba(27, 94, 32, 0.08);
            margin-bottom: 1rem;
        }

        .metric-card {
            padding: 1rem;
            border-radius: 18px;
            background: white;
            border: 1px solid #C8E6C9;
            text-align: center;
            box-shadow: 0 4px 14px rgba(0,0,0,0.04);
        }

        .metric-label {
            font-size: 0.82rem;
            color: #4B604D;
            margin-bottom: 0.25rem;
        }

        .metric-value {
            font-size: 1.15rem;
            font-weight: 700;
            color: #1B5E20;
        }

        .species-title {
            font-size: 1.6rem;
            font-weight: 800;
            color: #123524;
            margin-bottom: 0.1rem;
        }

        .scientific-name {
            font-style: italic;
            color: #2E7D32;
            font-size: 1.05rem;
            margin-bottom: 1rem;
        }

        .pill {
            display: inline-block;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            background: #C8E6C9;
            color: #1B5E20;
            font-weight: 600;
            font-size: 0.82rem;
            margin: 0.15rem 0.25rem 0.15rem 0;
        }

        div.stButton > button {
            border-radius: 999px;
            border: 1px solid #2E7D32;
            background: #2E7D32;
            color: white;
            font-weight: 700;
            padding: 0.55rem 1rem;
        }

        div.stButton > button:hover {
            border: 1px solid #1B5E20;
            background: #1B5E20;
            color: white;
        }

        .stProgress > div > div > div > div {
            background-color: #2E7D32;
        }

        section[data-testid="stSidebar"] {
            background: #F1F8E9;
        }

        .small-note {
            color: #4B604D;
            font-size: 0.9rem;
        }

        .quiz-question {
            padding: 1rem;
            border-radius: 16px;
            border: 1px solid #C8E6C9;
            background: white;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


aplicar_estilos()


# =====================================================
# Utilidades de carga
# =====================================================

@st.cache_data
def cargar_clases():
    if not CLASSES_PATH.exists():
        st.error(f"No se encontró el archivo de clases en: {CLASSES_PATH}")
        st.stop()

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        raw_classes = json.load(f)

    if isinstance(raw_classes, dict):
        return [raw_classes[str(i)] for i in range(len(raw_classes))]

    return raw_classes


@st.cache_data
def cargar_info_especies():
    if not INFO_PATH.exists():
        return {}

    with open(INFO_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def cargar_modelo():
    if not MODEL_PATH.exists():
        st.error(f"No se encontró el modelo ONNX en: {MODEL_PATH}")
        st.stop()

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session, input_name, output_name


class_names = cargar_clases()
species_info = cargar_info_especies()
session, input_name, output_name = cargar_modelo()


# =====================================================
# Procesamiento e inferencia
# =====================================================

def nombre_limpio(nombre: str) -> str:
    return nombre.replace("_", " ").title()


def obtener_info(nombre: str) -> dict:
    return species_info.get(nombre, species_info.get(nombre.lower(), {}))


def preprocesar_imagen(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")

    # Se usa Resize directo porque el entrenamiento reportado usaba Resize((224, 224)).
    image = ImageOps.fit(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        method=Image.Resampling.BILINEAR,
    )

    array = np.asarray(image).astype(np.float32) / 255.0
    array = (array - MEAN) / STD
    array = np.transpose(array, (2, 0, 1))
    array = np.expand_dims(array, axis=0)

    return array.astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits)
    exp_values = np.exp(logits)
    return exp_values / np.sum(exp_values)


def predecir(image: Image.Image, top_k: int = 5):
    input_array = preprocesar_imagen(image)
    outputs = session.run([output_name], {input_name: input_array})
    logits = outputs[0][0]
    probabilities = softmax(logits)

    top_indices = probabilities.argsort()[::-1][:top_k]

    resultados = []
    for idx in top_indices:
        resultados.append(
            {
                "id": int(idx),
                "clase": class_names[int(idx)],
                "nombre": nombre_limpio(class_names[int(idx)]),
                "probabilidad": float(probabilities[int(idx)]),
            }
        )

    return resultados


# =====================================================
# Componentes visuales
# =====================================================

def mostrar_info_especie(nombre_modelo: str, mostrar_boton_quiz: bool = True):
    info = obtener_info(nombre_modelo)

    if not info:
        st.warning("No hay información botánica registrada para esta especie.")
        return

    st.markdown('<div class="green-card">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="species-title">{info.get("nombre_comun", nombre_limpio(nombre_modelo))}</div>
        <div class="scientific-name">{info.get("nombre_cientifico", "Nombre científico no registrado")}</div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Familia</div>
                <div class="metric-value">{info.get("familia", "No registrada")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Origen</div>
                <div class="metric-value">{info.get("origen", "No registrado")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Altura</div>
                <div class="metric-value">{info.get("altura_max_m", "No registrada")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### 🌱 Hábitat")
    st.write(info.get("habitat", "No registrado."))

    st.markdown("#### 🐦 Importancia para el ecosistema")
    st.write(info.get("importancia_ecosistemica", "No registrada."))

    st.markdown("#### 🔎 Rasgos para identificarla")
    st.write(info.get("rasgos_identificacion", "No registrados."))

    datos_clave = info.get("datos_clave", [])
    if datos_clave:
        st.markdown("#### 📌 Datos clave")
        for dato in datos_clave:
            st.markdown(f"- {dato}")

    recomendacion = info.get("recomendacion_foto")
    if recomendacion:
        st.info(f"📷 Consejo para mejorar la predicción: {recomendacion}")

    if mostrar_boton_quiz:
        if st.button("🧠 Hacer quiz sobre esta especie", key=f"quiz_btn_{nombre_modelo}"):
            st.session_state["quiz_species"] = nombre_modelo
            st.session_state["quiz_items"] = generar_quiz(nombre_modelo)
            st.session_state["show_quiz_after_prediction"] = True

    st.markdown("</div>", unsafe_allow_html=True)


def mostrar_resultados(resultados):
    principal = resultados[0]
    especie = principal["clase"]
    confianza = principal["probabilidad"]

    st.markdown("### Resultado principal")

    c1, c2 = st.columns([1.25, 1])
    with c1:
        st.success(f"🌳 Especie predicha: **{principal['nombre']}**")
        st.write(f"Confianza del modelo: **{confianza:.2%}**")
        st.progress(confianza)

        if confianza < 0.50:
            st.warning(
                "La confianza es baja. Intenta con una foto más clara de hojas, flores, frutos "
                "o una vista completa del árbol."
            )
        elif confianza < 0.75:
            st.info(
                "La predicción es razonable, pero conviene validar con rasgos botánicos de la especie."
            )

    with c2:
        info = obtener_info(especie)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Nombre científico</div>
                <div class="metric-value"><i>{info.get("nombre_cientifico", "No registrado")}</i></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Top 5 predicciones")
    for item in resultados:
        st.write(f"**{item['nombre']}** — {item['probabilidad']:.2%}")
        st.progress(item["probabilidad"])

    st.markdown("### Información de la especie")
    mostrar_info_especie(especie, mostrar_boton_quiz=True)

    resultado_json = {
        "prediccion_principal": principal,
        "top_predicciones": resultados,
        "info_especie": obtener_info(especie),
    }

    st.download_button(
        label="⬇️ Descargar resultado en JSON",
        data=json.dumps(resultado_json, ensure_ascii=False, indent=2),
        file_name=f"resultado_{especie}.json",
        mime="application/json",
    )


# =====================================================
# Quiz
# =====================================================

def tomar_distractores(campo: str, especie_correcta: str, n: int = 3):
    valores = []
    for especie, info in species_info.items():
        if especie == especie_correcta:
            continue
        valor = info.get(campo)
        if valor and valor not in valores:
            valores.append(valor)

    random.shuffle(valores)
    return valores[:n]


def crear_pregunta(especie: str, texto: str, campo: str):
    info = obtener_info(especie)
    respuesta = info.get(campo)

    distractores = tomar_distractores(campo, especie, n=3)
    opciones = [respuesta] + distractores
    opciones = [op for op in opciones if op]

    # Evita preguntas incompletas si faltara un dato.
    if len(opciones) < 2 or not respuesta:
        return None

    opciones = list(dict.fromkeys(opciones))
    random.shuffle(opciones)

    return {
        "pregunta": texto,
        "opciones": opciones,
        "respuesta": respuesta,
    }


def generar_quiz(especie: str):
    nombre = obtener_info(especie).get("nombre_comun", nombre_limpio(especie))

    plantillas = [
        (
            f"¿Cuál es el nombre científico de {nombre}?",
            "nombre_cientifico",
        ),
        (
            f"¿De dónde es originaria la especie {nombre}?",
            "origen",
        ),
        (
            f"¿Cuánto puede crecer aproximadamente {nombre}?",
            "altura_max_m",
        ),
        (
            f"¿Cuál es una importancia ecológica de {nombre}?",
            "importancia_ecosistemica",
        ),
        (
            f"¿Qué rasgo ayuda a identificar {nombre}?",
            "rasgos_identificacion",
        ),
    ]

    preguntas = []
    for texto, campo in plantillas:
        pregunta = crear_pregunta(especie, texto, campo)
        if pregunta:
            preguntas.append(pregunta)

    return preguntas


def render_quiz(especie: str):
    info = obtener_info(especie)
    nombre = info.get("nombre_comun", nombre_limpio(especie))

    st.markdown(f"## 🧠 Quiz: {nombre}")
    st.write(
        "Responde con base en la información de la especie. "
        "El objetivo es practicar identificación, origen, crecimiento e importancia ecológica."
    )

    if "quiz_items" not in st.session_state or st.session_state.get("quiz_species") != especie:
        st.session_state["quiz_species"] = especie
        st.session_state["quiz_items"] = generar_quiz(especie)

    preguntas = st.session_state["quiz_items"]

    if not preguntas:
        st.warning("No hay suficientes datos para generar el quiz de esta especie.")
        return

    with st.form(key=f"quiz_form_{especie}"):
        respuestas_usuario = []
        for i, item in enumerate(preguntas, 1):
            st.markdown('<div class="quiz-question">', unsafe_allow_html=True)
            st.markdown(f"**Pregunta {i}. {item['pregunta']}**")
            respuesta = st.radio(
                "Selecciona una respuesta:",
                item["opciones"],
                key=f"quiz_{especie}_{i}",
                label_visibility="collapsed",
            )
            respuestas_usuario.append(respuesta)
            st.markdown("</div>", unsafe_allow_html=True)

        enviado = st.form_submit_button("✅ Calificar quiz")

    if enviado:
        correctas = 0
        st.markdown("### Retroalimentación")

        for i, (item, respuesta_usuario) in enumerate(zip(preguntas, respuestas_usuario), 1):
            es_correcta = respuesta_usuario == item["respuesta"]
            correctas += int(es_correcta)

            if es_correcta:
                st.success(f"Pregunta {i}: correcta ✅")
            else:
                st.error(
                    f"Pregunta {i}: incorrecta ❌\n\n"
                    f"Tu respuesta: {respuesta_usuario}\n\n"
                    f"Respuesta correcta: {item['respuesta']}"
                )

        puntaje = correctas / len(preguntas)
        st.markdown(f"## Puntaje: {correctas}/{len(preguntas)} — {puntaje:.0%}")
        st.progress(puntaje)

        if puntaje >= 0.8:
            st.balloons()
            st.success("Muy bien. Ya reconoces los datos principales de esta especie.")
        elif puntaje >= 0.6:
            st.info("Buen avance. Revisa nuevamente los datos de origen, altura e importancia ecológica.")
        else:
            st.warning("Conviene leer otra vez la ficha de la especie y repetir el quiz.")

    if st.button("🔄 Generar nuevo quiz", key=f"new_quiz_{especie}"):
        st.session_state["quiz_items"] = generar_quiz(especie)
        st.rerun()


# =====================================================
# Sidebar
# =====================================================

with st.sidebar:
    st.markdown("## 🌳 TreeLens")
    st.write("Clasificador de especies de árboles con modelo ONNX.")
    st.markdown("---")
    st.markdown("### Especies del modelo")
    for clase in class_names:
        info = obtener_info(clase)
        etiqueta = info.get("nombre_comun", nombre_limpio(clase))
        st.markdown(f"- {etiqueta}")

    st.markdown("---")
    st.caption(
        "La predicción es una ayuda automática. Para decisiones académicas o técnicas, "
        "valida con rasgos botánicos y fuentes especializadas."
    )


# =====================================================
# Interfaz principal
# =====================================================

st.markdown(
    """
    <div class="hero-card">
        <h1>🌳 TreeLens: clasificador de árboles</h1>
        <p>
        Sube una imagen o toma una foto. La app identifica la especie probable,
        muestra información botánica y genera un quiz para practicar lo aprendido.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_clasificar, tab_catalogo, tab_quiz = st.tabs(
    ["📷 Clasificar", "📚 Catálogo de especies", "🧠 Quiz"]
)

with tab_clasificar:
    st.markdown("## 📷 Clasificar una imagen")

    modo = st.radio(
        "Selecciona la entrada de imagen:",
        ["Subir imagen", "Tomar foto"],
        horizontal=True,
    )

    archivo = None
    if modo == "Subir imagen":
        archivo = st.file_uploader(
            "Sube una imagen del árbol, hoja, flor, fruto o tronco",
            type=["jpg", "jpeg", "png"],
        )
    else:
        archivo = st.camera_input("Toma una foto")

    if archivo is not None:
        image = Image.open(archivo).convert("RGB")

        col_img, col_info = st.columns([1, 1.15])
        with col_img:
            st.image(image, caption="Imagen analizada", use_container_width=True)

        with col_info:
            with st.spinner("Analizando imagen..."):
                resultados = predecir(image, top_k=min(5, len(class_names)))
            mostrar_resultados(resultados)

        if st.session_state.get("show_quiz_after_prediction"):
            st.markdown("---")
            render_quiz(st.session_state["quiz_species"])

    else:
        st.info(
            "Carga una imagen para iniciar. Para mejores resultados usa fotos claras, "
            "con buena luz y donde se vean hojas, flores, frutos o la forma general del árbol."
        )

with tab_catalogo:
    st.markdown("## 📚 Catálogo de especies")

    especie_catalogo = st.selectbox(
        "Selecciona una especie para consultar su ficha:",
        class_names,
        format_func=lambda x: obtener_info(x).get("nombre_comun", nombre_limpio(x)),
    )

    mostrar_info_especie(especie_catalogo, mostrar_boton_quiz=False)

with tab_quiz:
    st.markdown("## 🧠 Practica con un quiz")

    especie_quiz = st.selectbox(
        "Escoge la especie que quieres estudiar:",
        class_names,
        format_func=lambda x: obtener_info(x).get("nombre_comun", nombre_limpio(x)),
        key="quiz_selector",
    )

    col_a, col_b = st.columns([1, 2])
    with col_a:
        if st.button("🧪 Crear quiz", key="crear_quiz_tab"):
            st.session_state["quiz_species"] = especie_quiz
            st.session_state["quiz_items"] = generar_quiz(especie_quiz)

    with col_b:
        st.markdown(
            '<p class="small-note">El quiz usa la ficha botánica: nombre científico, origen, altura, rasgos e importancia ecológica.</p>',
            unsafe_allow_html=True,
        )

    render_quiz(especie_quiz)
