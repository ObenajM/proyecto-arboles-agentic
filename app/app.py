from pathlib import Path
import json

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# =====================================================
# Rutas del proyecto
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "modelo_arboles.pth"
CLASSES_PATH = BASE_DIR / "models" / "clases.json"


# =====================================================
# Configuración
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="Clasificador de árboles",
    page_icon="🌳",
    layout="centered"
)


# =====================================================
# Verificar archivos
# =====================================================

if not MODEL_PATH.exists():
    st.error(f"No se encontró el modelo en: {MODEL_PATH}")
    st.stop()

if not CLASSES_PATH.exists():
    st.error(f"No se encontró el archivo de clases en: {CLASSES_PATH}")
    st.stop()


# =====================================================
# Cargar clases
# =====================================================

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    raw_classes = json.load(f)

# clases.json viene como diccionario: {"0": "aguacate", ...}
if isinstance(raw_classes, dict):
    class_names = [raw_classes[str(i)] for i in range(len(raw_classes))]
else:
    class_names = raw_classes

num_classes = len(class_names)


# =====================================================
# Transformación de imagen
# =====================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =====================================================
# Arquitectura usada en entrenamiento
# =====================================================

class TreeResNet18(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        # weights=None para evitar descargar pesos en Streamlit Cloud
        self.model = models.resnet18(weights=None)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# =====================================================
# Cargar modelo
# =====================================================

@st.cache_resource
def load_model():
    model = TreeResNet18(num_classes)

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Solo quitar "module." si fue entrenado con DataParallel.
    # NO quitar "model.", porque tu arquitectura lo necesita.
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    return model


model = load_model()


# =====================================================
# Interfaz web
# =====================================================

st.title("🌳 Clasificador de especies de árboles")

st.write(
    "Sube una imagen de un árbol, hoja, flor, fruto o tronco. "
    "El modelo intentará predecir la especie."
)

uploaded_file = st.file_uploader(
    "Selecciona una imagen",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Imagen cargada",
        use_container_width=True
    )

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        top_probs, top_indices = torch.topk(
            probabilities,
            k=min(5, num_classes),
            dim=1
        )

    predicted_index = top_indices[0][0].item()
    predicted_class = class_names[predicted_index]
    confidence = top_probs[0][0].item() * 100

    st.subheader("Resultado principal")

    st.success(f"Especie predicha: **{predicted_class}**")
    st.write(f"Confianza: **{confidence:.2f}%**")

    st.subheader("Top predicciones")

    for prob, idx in zip(top_probs[0], top_indices[0]):
        species = class_names[idx.item()]
        percentage = prob.item() * 100
        st.write(f"{species}: {percentage:.2f}%")