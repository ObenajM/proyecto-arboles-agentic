"""
main.py — Campus Tree Explorer (FastAPI + PWA)
Migración desde Streamlit. Misma lógica ONNX, nueva interfaz web progresiva.
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
from io import BytesIO
import tempfile
from pathlib import Path
from typing import Optional

# Cargar variables de entorno desde .env (PLANTNET_KEY, ANTHROPIC_API_KEY, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except Exception:
    pass

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageOps
from pydantic import BaseModel

try:
    from agent_validation import ejecutar_agente
except Exception as exc:  # El backend puede seguir funcionando solo con ONNX
    ejecutar_agente = None
    AGENT_IMPORT_ERROR = str(exc)
else:
    AGENT_IMPORT_ERROR = None

# =============================================================================
# Paths & constants  (mismo layout que el proyecto original)
# =============================================================================

BASE_DIR     = Path(__file__).resolve().parent
MODEL_PATH   = BASE_DIR / "models" / "modelo_arboles_best.onnx"
CLASSES_PATH = BASE_DIR / "models" / "clases.json"
INFO_PATH    = BASE_DIR / "data"   / "info.json"
CSV_PATH     = BASE_DIR / "data"   / "arboles_mapeados.csv"

CAMPUS_CENTER   = [6.2636427, -75.5764393]
IMAGE_SIZE      = 224
MEAN            = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD             = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TOP_K           = 3
CSV_COLUMNS     = ["species_key", "common_name", "confidence", "latitude",
                   "longitude", "gps_accuracy", "datetime", "source"]
GITHUB_REPO     = "ObenajM/proyecto-arboles-agentic"
GITHUB_CSV_PATH = "data/arboles_mapeados.csv"

# =============================================================================
# App setup
# =============================================================================

app = FastAPI(title="Campus Tree Explorer")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# =============================================================================
# Loaders (ejecutados una vez al arrancar)
# =============================================================================

def load_classes() -> list[str]:
    if not CLASSES_PATH.exists():
        raise RuntimeError(f"Archivo de clases no encontrado: {CLASSES_PATH}")
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [raw[str(i)] for i in range(len(raw))] if isinstance(raw, dict) else list(raw)


def load_species_info() -> dict:
    if not INFO_PATH.exists():
        return {}
    with open(INFO_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Modelo ONNX no encontrado: {MODEL_PATH}")
    sess = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name, sess.get_outputs()[0].name


def load_mapped_trees() -> list[dict]:
    if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
        return []
    rows: list[dict] = []
    with open(CSV_PATH, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                rows.append({
                    "species_key": row.get("species_key", ""),
                    "name":        row.get("common_name", ""),
                    "lat":         float(row["latitude"]),
                    "lon":         float(row["longitude"]),
                    "confidence":  float(row.get("confidence") or 0),
                    "datetime":    row.get("datetime", ""),
                    "source":      row.get("source", ""),
                })
            except (ValueError, KeyError):
                continue
    return rows


class_names  = load_classes()
species_info = load_species_info()
ort_sess, inp_name, out_name = load_model()

# =============================================================================
# Helpers  (lógica idéntica al app.py original)
# =============================================================================

def clean_name(key: str) -> str:
    return key.replace("_", " ").title()


def get_info(key: str) -> dict:
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
        {
            "key":  class_names[int(i)],
            "name": clean_name(class_names[int(i)]),
            "prob": float(probs[int(i)]),
            "info": get_info(class_names[int(i)]),
        }
        for i in idx
    ]


def _ensure_csv_headers() -> None:
    if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=CSV_COLUMNS).writeheader()


def _get_github_config() -> dict | None:
    token    = os.environ.get("GITHUB_TOKEN")
    repo     = os.environ.get("GITHUB_REPO",     GITHUB_REPO)
    branch   = os.environ.get("GITHUB_BRANCH",   "main")
    csv_path = os.environ.get("GITHUB_CSV_PATH", GITHUB_CSV_PATH)
    return {"token": token, "repo": repo, "branch": branch, "csv_path": csv_path} if token else None


def _push_csv_to_github() -> tuple[bool, str]:
    cfg = _get_github_config()
    if not cfg:
        return False, "token_missing"
    api_url = (
        f"https://api.github.com/repos/{cfg['repo']}/contents/{cfg['csv_path']}"
    )
    headers = {
        "Authorization": f"Bearer {cfg['token']}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    sha = ""
    try:
        req = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            sha = json.loads(resp.read()).get("sha", "")
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            return False, f"HTTP {exc.code}"
    except OSError as exc:
        return False, str(exc)

    with open(CSV_PATH, "rb") as fh:
        content_b64 = base64.b64encode(fh.read()).decode()

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    body: dict = {
        "message": f"TreeLens: actualizar CSV [{now_str}]",
        "content": content_b64,
    }
    if sha:
        body["sha"] = sha

    payload = json.dumps(body).encode()
    put_req = urllib.request.Request(
        api_url, data=payload,
        headers={**headers, "Content-Type": "application/json"},
        method="PUT",
    )
    try:
        with urllib.request.urlopen(put_req, timeout=15):
            return True, "CSV sincronizado con GitHub."
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except OSError as exc:
        return False, str(exc)


# ── Trivia engine (idéntico al original) ─────────────────────────────────────

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
    tmpl   = _TRIVIA_TEMPLATES.copy()
    random.shuffle(tmpl)
    for field, template in tmpl:
        answer = info.get(field)
        if not answer:
            continue
        distractors = _pick_distractors(field, species_key, n=3)
        if len(distractors) < 2:
            continue
        options = list(dict.fromkeys([answer] + distractors))
        random.shuffle(options)
        return {"question": template.format(n=nombre), "options": options, "answer": answer}
    return None


# =============================================================================
# API routes
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "species_count": len(class_names),
            "campus_center": json.dumps(CAMPUS_CENTER),
        }
    )


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen.")

    data = await file.read()
    image = Image.open(BytesIO(data)).convert("RGB")

    # 1) Predicción local ONNX: rápida y siempre disponible.
    results = run_inference(image, top_k=min(TOP_K, len(class_names)))

    agent_decision: dict | None = None
    plantnet_result: dict | None = None
    comparison: dict | None = None
    agent_error: str | None = None

    # 2) Validación agente: ONNX top-k + Pl@ntNet + reglas de decisión.
    #    Si falla, la API conserva la predicción local y reporta el error.
    if ejecutar_agente is None:
        agent_error = AGENT_IMPORT_ERROR or "agent_validation.py no disponible"
    elif results:
        top = results[0]
        top_k_list = [(r["key"], float(r["prob"])) for r in results]

        suffix = Path(file.filename or "imagen.jpg").suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".jpg"

        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            estado_final = ejecutar_agente(
                especie_pred=top["key"],
                confianza=float(top["prob"]),
                top_k_list=top_k_list,
                info_especie=get_info(top["key"]),
                ruta_imagen=tmp_path,
                info_global=species_info,
            )
            agent_decision = estado_final.get("decision_final", {})
            plantnet_result = estado_final.get("plantnet_resultado", {})
            comparison = estado_final.get("comparacion", {})
        except Exception as exc:
            agent_error = str(exc)
        finally:
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    return {
        "results": results,
        "agent_decision": agent_decision,
        "plantnet": plantnet_result,
        "comparison": comparison,
        "agent_error": agent_error,
    }


@app.get("/api/classes")
async def classes_list():
    return {
        "classes": [
            {
                "key":        k,
                "name":       get_info(k).get("nombre_comun") or clean_name(k),
                "scientific": get_info(k).get("nombre_cientifico", ""),
                "family":     get_info(k).get("familia", ""),
            }
            for k in class_names
        ]
    }


@app.get("/api/catalog/{key}")
async def species_detail(key: str):
    info = get_info(key)
    if not info:
        raise HTTPException(status_code=404, detail="Especie no encontrada.")
    return {"key": key, "display_name": clean_name(key), **info}


@app.get("/api/map-data")
async def map_data():
    return {"points": load_mapped_trees(), "center": CAMPUS_CENTER}


class SaveTreeRequest(BaseModel):
    species_key: str
    common_name: str
    confidence:  float
    latitude:    float
    longitude:   float
    gps_accuracy: Optional[float] = None
    source: str = "gps"


@app.post("/api/save-tree")
async def save_tree(body: SaveTreeRequest):
    _ensure_csv_headers()
    row = {
        "species_key":  body.species_key,
        "common_name":  body.common_name,
        "confidence":   f"{body.confidence:.4f}",
        "latitude":     f"{body.latitude:.7f}",
        "longitude":    f"{body.longitude:.7f}",
        "gps_accuracy": f"{body.gps_accuracy:.1f}" if body.gps_accuracy else "",
        "datetime":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source":       body.source,
    }
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore").writerow(row)
    return {"ok": True, "row": row}


@app.post("/api/sync-github")
async def sync_github():
    ok, msg = _push_csv_to_github()
    return {"ok": ok, "message": msg}


@app.get("/api/trivia/{key}")
async def trivia(key: str):
    q = build_question(key)
    if not q:
        raise HTTPException(status_code=404, detail="Sin datos suficientes para trivia.")
    return q
