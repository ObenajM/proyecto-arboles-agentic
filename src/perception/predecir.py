import argparse
import json
import os
import sys
import importlib

import torch
from PIL import Image
from torchvision import transforms

import modelo
importlib.reload(modelo)
from modelo import crear_modelo


MODELO_PATH = "/content/drive/MyDrive/modelo_arboles_best.pth"
INFO_PATH = "info.json"
IMG_SIZE = 224


def cargar_modelo(modelo_path, device):
    if not os.path.isfile(modelo_path):
        raise FileNotFoundError(
            f"No se encontró '{modelo_path}'. Ejecuta primero entrenar.py."
        )

    checkpoint = torch.load(modelo_path, map_location=device)

    clases = checkpoint["clases"]
    img_size = checkpoint.get("img_size", IMG_SIZE)
    num_clases = checkpoint["num_clases"]

    modelo = crear_modelo(num_clases, device)
    modelo.load_state_dict(checkpoint["model_state_dict"])
    modelo.eval()

    return modelo, clases, img_size


def preprocesar_imagen(ruta, img_size):
    media = [0.485, 0.456, 0.406]
    std_dev = [0.229, 0.224, 0.225]

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(media, std_dev),
    ])

    imagen = Image.open(ruta).convert("RGB")
    return tf(imagen).unsqueeze(0)


def cargar_info(info_path):
    if not os.path.isfile(info_path):
        return {}

    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def mostrar_resultado(especie, confianza, info, top_k_list):
    print("\n" + "─" * 60)
    print(f"🌳 ESPECIE IDENTIFICADA: {especie.upper()}")
    print(f"📈 Confianza: {confianza * 100:.2f}%")
    print("─" * 60)

    datos = info.get(especie, info.get(especie.lower(), {}))

    if datos:
        print("\n📋 INFORMACIÓN BOTÁNICA:")
        for clave, valor in datos.items():
            print(f"   • {clave}: {valor}")
    else:
        print("\nℹ️ No se encontró información botánica en info.json.")

    print(f"\n📊 TOP {len(top_k_list)} PREDICCIONES:")
    for i, (nombre, prob) in enumerate(top_k_list, 1):
        marca = "👉" if i == 1 else "  "
        print(f"{marca} {i}. {nombre:25s} {prob * 100:.2f}%")

    print("─" * 60 + "\n")


def predecir(ruta_imagen, modelo, clases, img_size, info, device, top_k=3):
    if not os.path.isfile(ruta_imagen):
        print(f"❌ No se encontró la imagen: {ruta_imagen}")
        return

    tensor = preprocesar_imagen(ruta_imagen, img_size).to(device)

    with torch.no_grad():
        logits = modelo(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top_indices = probs.argsort(descending=True)[:top_k]
    top_k_list = [(clases[i], probs[i].item()) for i in top_indices]

    especie_pred, confianza = top_k_list[0]
    mostrar_resultado(especie_pred, confianza, info, top_k_list)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n🌿 Sistema de Clasificación de Árboles")
    print(f"🖥️ Dispositivo: {device}")
    print("Cargando mejor modelo...")

    modelo, clases, img_size = cargar_modelo(MODELO_PATH, device)
    info = cargar_info(INFO_PATH)

    print(f"✅ Modelo cargado con {len(clases)} especies")

    if args.imagen:
        ruta_imagen = args.imagen
    else:
        ruta_imagen = input("\n📷 Ingresa la ruta de la imagen: ").strip()
        if not ruta_imagen:
            print("❌ No ingresaste ninguna imagen.")
            sys.exit(1)

    predecir(
        ruta_imagen,
        modelo,
        clases,
        img_size,
        info,
        device,
        top_k=args.top
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagen", type=str, default=None)
    parser.add_argument("--top", type=int, default=3)

    args = parser.parse_args([])  # Colab/Jupyter
    main(args)