import argparse
import json
import os
import time
import importlib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import modelo
importlib.reload(modelo)
from modelo import crear_modelo


DATASET_DIR = "/content/drive/MyDrive/dataset"

BEST_MODEL = "/content/drive/MyDrive/modelo_arboles_best.pth"
LAST_CHECKPOINT = "/content/drive/MyDrive/checkpoint_last.pth"
CLASES_JSON = "/content/drive/MyDrive/clases.json"

IMG_SIZE = 224
VAL_SPLIT = 0.2


def get_transforms(img_size):
    media = [0.485, 0.456, 0.406]
    std_dev = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
        transforms.ToTensor(),
        transforms.Normalize(media, std_dev),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(media, std_dev),
    ])

    return train_tf, val_tf


def cargar_datasets(dataset_dir, img_size, val_split, batch_size):
    train_tf, val_tf = get_transforms(img_size)

    dataset_train_full = datasets.ImageFolder(root=dataset_dir, transform=train_tf)
    dataset_val_full = datasets.ImageFolder(root=dataset_dir, transform=val_tf)

    clases = dataset_train_full.classes
    n_total = len(dataset_train_full)

    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    indices = torch.randperm(
        n_total,
        generator=torch.Generator().manual_seed(42)
    ).tolist()

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_set = Subset(dataset_train_full, train_indices)
    val_set = Subset(dataset_val_full, val_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader, clases


def contar_imagenes_por_clase(dataset_dir):
    total = 0
    print("\n📊 Imágenes por clase:")

    for clase in sorted(os.listdir(dataset_dir)):
        ruta = os.path.join(dataset_dir, clase)

        if os.path.isdir(ruta):
            imagenes = [
                f for f in os.listdir(ruta)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            ]
            total += len(imagenes)
            print(f"   {clase}: {len(imagenes)}")

    print(f"\n🖼️ Total de imágenes: {total}")


def entrenar_una_epoca(modelo, loader, criterio, optimizador, device):
    modelo.train()

    total_loss = 0.0
    correctos = 0
    total = 0

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    for imagenes, etiquetas in loader:
        imagenes = imagenes.to(device, non_blocking=True)
        etiquetas = etiquetas.to(device, non_blocking=True)

        optimizador.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            salidas = modelo(imagenes)
            loss = criterio(salidas, etiquetas)

        scaler.scale(loss).backward()
        scaler.step(optimizador)
        scaler.update()

        total_loss += loss.item() * imagenes.size(0)
        preds = salidas.argmax(dim=1)
        correctos += (preds == etiquetas).sum().item()
        total += imagenes.size(0)

    return total_loss / total, correctos / total


def validar(modelo, loader, criterio, device):
    modelo.eval()

    total_loss = 0.0
    correctos = 0
    total = 0

    with torch.no_grad():
        for imagenes, etiquetas in loader:
            imagenes = imagenes.to(device, non_blocking=True)
            etiquetas = etiquetas.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                salidas = modelo(imagenes)
                loss = criterio(salidas, etiquetas)

            total_loss += loss.item() * imagenes.size(0)
            preds = salidas.argmax(dim=1)
            correctos += (preds == etiquetas).sum().item()
            total += imagenes.size(0)

    return total_loss / total, correctos / total


def guardar_checkpoint(epoca, modelo, optimizador, scheduler, clases, num_clases, val_acc):
    torch.save({
        "epoch": epoca,
        "model_state_dict": modelo.state_dict(),
        "optimizer_state_dict": optimizador.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "clases": clases,
        "num_clases": num_clases,
        "img_size": IMG_SIZE,
        "val_acc": val_acc,
        "arquitectura": "ConvNeXt-Tiny",
    }, LAST_CHECKPOINT)


def guardar_mejor_modelo(epoca, modelo, clases, num_clases, val_acc):
    torch.save({
        "epoch": epoca,
        "model_state_dict": modelo.state_dict(),
        "clases": clases,
        "num_clases": num_clases,
        "img_size": IMG_SIZE,
        "val_acc": val_acc,
        "arquitectura": "ConvNeXt-Tiny",
    }, BEST_MODEL)


def main(args):
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Dispositivo: {device}")

    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(
            f"No existe la carpeta '{DATASET_DIR}'. "
            "Monta Google Drive antes con drive.mount('/content/drive')."
        )

    contar_imagenes_por_clase(DATASET_DIR)

    train_loader, val_loader, clases = cargar_datasets(
        DATASET_DIR,
        IMG_SIZE,
        VAL_SPLIT,
        args.batch
    )

    num_clases = len(clases)

    print(f"\n✅ {num_clases} especies encontradas")
    print(f"Train: {len(train_loader.dataset)} imágenes")
    print(f"Val:   {len(val_loader.dataset)} imágenes")

    with open(CLASES_JSON, "w", encoding="utf-8") as f:
        json.dump({str(i): c for i, c in enumerate(clases)}, f, ensure_ascii=False, indent=2)

    modelo = crear_modelo(num_clases, device)

    criterio = nn.CrossEntropyLoss()

    optimizador = optim.AdamW(
        modelo.parameters(),
        lr=args.lr,
        weight_decay=1e-3
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizador,
        patience=4,
        factor=0.5
    )

    start_epoch = 1
    mejor_val_acc = 0.0

    if os.path.isfile(LAST_CHECKPOINT):
        print(f"\n🔄 Checkpoint encontrado: {LAST_CHECKPOINT}")

        checkpoint = torch.load(LAST_CHECKPOINT, map_location=device)

        modelo.load_state_dict(checkpoint["model_state_dict"])
        optimizador.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        mejor_val_acc = checkpoint.get("val_acc", 0.0)

        print(f"✅ Continuando desde época {start_epoch}")

    print(f"\n🚀 Entrenando desde época {start_epoch} hasta {args.epocas}\n")

    for epoca in range(start_epoch, args.epocas + 1):
        t0 = time.time()

        train_loss, train_acc = entrenar_una_epoca(
            modelo, train_loader, criterio, optimizador, device
        )

        val_loss, val_acc = validar(
            modelo, val_loader, criterio, device
        )

        scheduler.step(val_loss)

        duracion = time.time() - t0

        print(
            f"Época {epoca:02d}/{args.epocas} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} | "
            f"⏱ {duracion:.1f}s"
        )

        guardar_checkpoint(
            epoca, modelo, optimizador, scheduler, clases, num_clases, val_acc
        )

        print("   💾 checkpoint_last.pth actualizado")

        if val_acc > mejor_val_acc:
            mejor_val_acc = val_acc

            guardar_mejor_modelo(
                epoca, modelo, clases, num_clases, val_acc
            )

            print(f"   ⭐ Mejor modelo guardado con val_acc={val_acc:.4f}")

    print("\n✅ Entrenamiento completo.")
    print(f"Mejor val_acc: {mejor_val_acc:.4f}")
    print(f"Mejor modelo: {BEST_MODEL}")
    print(f"Último checkpoint: {LAST_CHECKPOINT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epocas", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=32)

    args = parser.parse_args([])  # Colab/Jupyter
    main(args)