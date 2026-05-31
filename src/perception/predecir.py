# predecir.py — Sistema de Clasificación de Árboles
# Flujo: Imagen → ConvNeXt (modelo local) → Pl@ntNet (validación visual) → Decisión final
#
# Prioridad de decisión (score ponderado):
#   PESO_MODELO   = 1.00
#   PESO_PLANTNET = 1.20  ← Pl@ntNet tiene ligera ventaja por ser especializado
#   Excepción: si modelo >= 99.9% → modelo gana siempre

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import argparse
import json
import os
import re
import sys
import importlib
import unicodedata
from typing import TypedDict, List, Tuple, Optional, Set

# ── Cargar secrets desde Colab ─────────────────────────────────────────────
try:
    from google.colab import userdata
    os.environ["PLANTNET_KEY"]      = (userdata.get("PLANTNET_KEY") or
                                       userdata.get("planet")       or "")
    os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY") or ""
    os.environ["OPENAI_API_KEY"]    = userdata.get("OPENAI_API_KEY")    or ""
except Exception:
    pass
# ───────────────────────────────────────────────────────────────────────────

_CARPETA_PROYECTO = "/content/"
if _CARPETA_PROYECTO not in sys.path:
    sys.path.insert(0, _CARPETA_PROYECTO)

import torch
from PIL import Image
from torchvision import transforms

import modelo
importlib.reload(modelo)
from modelo import crear_modelo

MODELO_PATH = "/content/drive/MyDrive/Archivos entrenamiento/modelo_arboles_best.pth"
INFO_PATH   = "info.json"
IMG_SIZE    = 224

# ── Umbrales y pesos ────────────────────────────────────────────────────────
CONFIANZA_MODELO_PRIORIDAD = 0.999
PESO_MODELO                = 1.00
PESO_PLANTNET              = 1.20

# ── Separadores de salida ───────────────────────────────────────────────────
SEP  = "─" * 60
SEP2 = "═" * 60


# ───────────────────────────────────────────────────────────────────────────
#  DICCIONARIO DE SINÓNIMOS
# ───────────────────────────────────────────────────────────────────────────

SINONIMOS_ESPECIES = {
    "guayacan_amarillo": [
        "guayacan amarillo", "guayacán amarillo", "lapacho amarillo",
        "araguaney", "roble amarillo",
        "handroanthus chrysanthus", "tabebuia chrysantha",
    ],
    "guayacan_rosado": [
        "guayacan rosado", "guayacán rosado", "roble rosado",
        "apamate", "tabebuia rosea", "handroanthus roseus",
    ],
    "ceiba": [
        "ceiba", "ceiba pentandra", "kapok tree", "silk cotton tree",
    ],
    "mango": ["mango", "mangifera indica"],
    "aguacate": ["aguacate", "palta", "persea americana"],
    "platymiscium_pinnatum": [
        "platymiscium pinnatum", "platymiscium",
        "granadillo", "cristobal", "cristóbal",
    ],
    "pinus_patula": [
        "pinus patula", "pino patula", "mexican weeping pine",
    ],
    "piptadenia_flava": ["piptadenia flava", "piptadenia"],
    "balso": [
        "balso", "balsa", "balsa kapok", "balzovec",
        "ochroma pyramidale", "ochroma lagopus",
        "balsa wood", "madera balsa",
    ],
    "palma_abanico": [
        "palma abanico", "palmera abanico", "palma de abanico",
        "washingtonia filifera", "washingtonia robusta",
        "palmera abanico mexicana", "fan palm",
    ],
    "saman": [
        "saman", "samán", "samanea saman", "albizia saman",
        "rain tree", "monkey pod",
    ],
    "bala_de_canon": [
        "bala de canon", "bala de cañon", "bala de cañón",
        "couroupita guianensis", "cannonball tree",
    ],
    "roble"  : ["roble", "quercus", "oak"],
    "acacia" : ["acacia", "acacia mangium", "acacia", "wattle"],
}

# Especies cuyo color amarillo/rosado/naranja es floral, no señal de estrés
ESPECIES_COLOR_FLORAL = {
    "guayacan_amarillo", "guayacan_rosado", "cambulo", "flamboyan",
    "tulipan_africano", "acacia", "araguaney",
}


# ───────────────────────────────────────────────────────────────────────────
#  ESTADO DEL GRAFO LANGGRAPH
# ───────────────────────────────────────────────────────────────────────────

class EstadoArbol(TypedDict):
    especie_pred       : str
    confianza          : float
    top_k_list         : List[Tuple[str, float]]
    info_especie       : dict
    info_global        : dict
    ruta_imagen        : str
    plantnet_resultado : dict
    comparacion        : dict
    decision_final     : dict
    salud_arbol        : dict


# ───────────────────────────────────────────────────────────────────────────
#  NORMALIZACIÓN DE NOMBRES
# ───────────────────────────────────────────────────────────────────────────

def normalizar_nombre(nombre: str) -> str:
    if not nombre:
        return ""
    texto = nombre.lower().replace("_", " ").replace("-", " ")
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = re.sub(r"[^a-z0-9 ]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def obtener_nombre_comun_modelo(especie_modelo: str, info: dict) -> str:
    claves_nombre = [
        "nombre_comun", "nombre_común", "nombre comun", "nombre común",
        "common_name", "commonNames", "nombres_comunes", "nombre",
    ]
    for clave in claves_nombre:
        valor = info.get(clave) or info.get(especie_modelo, {}).get(clave)
        if valor:
            if isinstance(valor, list):
                valor = valor[0]
            return normalizar_nombre(str(valor))
    return normalizar_nombre(especie_modelo)


def obtener_alias_modelo(especie_modelo: str, info: dict) -> Set[str]:
    alias = set()
    alias.add(normalizar_nombre(especie_modelo))
    alias.add(especie_modelo.lower())

    nombre_comun = obtener_nombre_comun_modelo(especie_modelo, info)
    if nombre_comun:
        alias.add(nombre_comun)

    for clave in ["nombre_cientifico", "scientific_name", "scientificName"]:
        val = info.get(clave)
        if val:
            alias.add(normalizar_nombre(str(val)))

    for clave in ["sinonimos", "synonyms", "alias"]:
        val = info.get(clave, [])
        if isinstance(val, list):
            for s in val:
                alias.add(normalizar_nombre(str(s)))

    for sin in SINONIMOS_ESPECIES.get(especie_modelo, []):
        alias.add(normalizar_nombre(sin))

    alias.discard("")
    return alias


# ───────────────────────────────────────────────────────────────────────────
#  CONSULTA A Pl@ntNet
# ───────────────────────────────────────────────────────────────────────────

def consultar_plantnet(ruta_imagen: str) -> dict:
    plantnet_key = os.environ.get("PLANTNET_KEY", "").strip()
    if not plantnet_key:
        return {"es_planta": None, "resultados": [],
                "razon": "PLANTNET_KEY no configurada"}

    if not ruta_imagen or not os.path.isfile(ruta_imagen):
        return {"es_planta": None, "resultados": [],
                "razon": "imagen no encontrada"}

    try:
        import requests
        with open(ruta_imagen, "rb") as f:
            resp = requests.post(
                "https://my-api.plantnet.org/v2/identify/all",
                params={"api-key": plantnet_key, "lang": "es", "nb-results": 5},
                files ={"images": f},
                timeout=20,
            )

        if resp.status_code == 404:
            return {"es_planta": False, "resultados": [],
                    "razon": "Pl@ntNet no reconoció ninguna planta"}

        if resp.status_code in (500, 502, 503, 504):
            return {"es_planta": None, "resultados": [], "razon": None}

        if resp.status_code != 200:
            return {"es_planta": None, "resultados": [], "razon": None}

        datos      = resp.json()
        resultados = []
        for rank, r in enumerate(datos.get("results", []), 1):
            sp = r.get("species", {})
            nombres_comunes = [n for n in sp.get("commonNames", []) if n]
            resultados.append({
                "rank"             : rank,
                "nombre_cientifico": sp.get("scientificNameWithoutAuthor", ""),
                "nombres_comunes"  : nombres_comunes,
                "score"            : round(r.get("score", 0), 4),
            })

        if not resultados:
            return {"es_planta": None, "resultados": [],
                    "razon": "Pl@ntNet respondió pero sin resultados"}

        top = resultados[0]
        nombre_mostrar = (top["nombres_comunes"][0]
                          if top["nombres_comunes"]
                          else top["nombre_cientifico"])
        return {
            "es_planta"  : True,
            "resultados" : resultados,
            "top_especie": top["nombre_cientifico"],
            "top_nombre" : nombre_mostrar,
            "top_score"  : top["score"],
            "razon"      : (f"Pl@ntNet identificó '{nombre_mostrar}' "
                            f"({top['nombre_cientifico']}) "
                            f"score {top['score']:.2f}"),
        }

    except Exception:
        return {"es_planta": None, "resultados": [], "razon": None}


# ───────────────────────────────────────────────────────────────────────────
#  COMPARACIÓN MODELO vs Pl@ntNet
# ───────────────────────────────────────────────────────────────────────────

def buscar_coincidencia_nombre_comun(
    top_k_list: list,
    plantnet_resultado: dict,
    info_global: dict,
) -> dict:
    score_modelo_top1   = top_k_list[0][1] if top_k_list else 0
    especie_modelo_top1 = top_k_list[0][0] if top_k_list else ""

    if not plantnet_resultado.get("resultados"):
        return {
            "coinciden"                 : False,
            "nombre_comun_compartido"   : None,
            "clase_modelo_coincidente"  : None,
            "rank_modelo"               : None,
            "rank_plantnet"             : None,
            "score_modelo"              : score_modelo_top1,
            "score_plantnet"            : 0,
            "nombre_cientifico_plantnet": None,
            "nombres_comunes_plantnet"  : [],
            "razon_comparacion"         : "Pl@ntNet no disponible.",
        }

    pn_resultados = plantnet_resultado["resultados"]
    top_pn        = pn_resultados[0]

    info_especie = info_global.get(especie_modelo_top1,
                   info_global.get(especie_modelo_top1.lower(), {}))
    alias_modelo = obtener_alias_modelo(especie_modelo_top1, info_especie)

    cientifico_pn = normalizar_nombre(top_pn["nombre_cientifico"])
    comunes_pn    = [normalizar_nombre(n) for n in top_pn["nombres_comunes"]]
    todos_pn      = set(comunes_pn) | {cientifico_pn}

    # 1. Coincidencia exacta nombre común
    match_comun = alias_modelo & set(comunes_pn)
    if match_comun:
        nombre_compartido = sorted(match_comun)[0]
        return {
            "coinciden"                 : True,
            "nombre_comun_compartido"   : nombre_compartido,
            "clase_modelo_coincidente"  : especie_modelo_top1,
            "rank_modelo"               : 1,
            "rank_plantnet"             : 1,
            "score_modelo"              : score_modelo_top1,
            "score_plantnet"            : top_pn["score"],
            "nombre_cientifico_plantnet": top_pn["nombre_cientifico"],
            "nombres_comunes_plantnet"  : top_pn["nombres_comunes"],
            "razon_comparacion"         : f"Top-1 coincide por nombre común: '{nombre_compartido}'",
        }

    # 2. Coincidencia exacta nombre científico
    if alias_modelo & {cientifico_pn}:
        return {
            "coinciden"                 : True,
            "nombre_comun_compartido"   : cientifico_pn,
            "clase_modelo_coincidente"  : especie_modelo_top1,
            "rank_modelo"               : 1,
            "rank_plantnet"             : 1,
            "score_modelo"              : score_modelo_top1,
            "score_plantnet"            : top_pn["score"],
            "nombre_cientifico_plantnet": top_pn["nombre_cientifico"],
            "nombres_comunes_plantnet"  : top_pn["nombres_comunes"],
            "razon_comparacion"         : f"Top-1 coincide por nombre científico: '{top_pn['nombre_cientifico']}'",
        }

    # 3. Verificar si top-2 es variante más específica del top-1 y Pl@ntNet la confirma
    if len(top_k_list) >= 2:
        especie_top2_early = top_k_list[1][0]
        score_top2_early   = top_k_list[1][1]
        nombre_top1_e      = normalizar_nombre(especie_modelo_top1)
        nombre_top2_e      = normalizar_nombre(especie_top2_early)
        son_variantes_early = (nombre_top1_e in nombre_top2_e or
                               nombre_top2_e in nombre_top1_e)

        if son_variantes_early:
            info_top2_e  = info_global.get(especie_top2_early,
                           info_global.get(especie_top2_early.lower(), {}))
            alias_top2_e = obtener_alias_modelo(especie_top2_early, info_top2_e)
            comunes_pn_e = [normalizar_nombre(n) for n in top_pn["nombres_comunes"]]
            cient_pn_e   = normalizar_nombre(top_pn["nombre_cientifico"])
            todos_pn_e   = set(comunes_pn_e) | {cient_pn_e}

            match_c2 = alias_top2_e & set(comunes_pn_e)
            match_s2 = alias_top2_e & {cient_pn_e}
            match_p2 = any(
                any(p in pn_n for p in [w for w in a.split() if len(w) > 4])
                for a in alias_top2_e for pn_n in todos_pn_e
            )

            if match_c2 or match_s2 or match_p2:
                nombre_comp_e   = (sorted(match_c2)[0] if match_c2
                                   else cient_pn_e if match_s2
                                   else nombre_top2_e)
                nombre_comun_t2 = obtener_nombre_comun_modelo(
                    especie_top2_early,
                    info_global.get(especie_top2_early,
                    info_global.get(especie_top2_early.lower(), {}))
                )
                return {
                    "coinciden"                 : True,
                    "nombre_comun_compartido"   : nombre_comp_e,
                    "clase_modelo_coincidente"  : especie_top2_early,
                    "rank_modelo"               : 2,
                    "rank_plantnet"             : 1,
                    "score_modelo"              : score_top2_early,
                    "score_plantnet"            : top_pn["score"],
                    "nombre_cientifico_plantnet": top_pn["nombre_cientifico"],
                    "nombres_comunes_plantnet"  : top_pn["nombres_comunes"],
                    "es_especificacion_top2"    : True,
                    "nombre_comun_top2"         : nombre_comun_t2 or nombre_top2_e,
                    "razon_comparacion"         : (
                        f"Top-1 ('{especie_modelo_top1}') y top-2 ('{especie_top2_early}') "
                        f"son variantes del mismo nombre. "
                        f"Pl@ntNet confirma el más específico: '{top_pn['nombre_cientifico']}'"
                    ),
                }

    # 4. Coincidencia parcial (top-1 vs top-1)
    for alias in alias_modelo:
        palabras = [p for p in alias.split() if len(p) > 4]
        for pn_nombre in todos_pn:
            if any(p in pn_nombre for p in palabras):
                return {
                    "coinciden"                 : True,
                    "nombre_comun_compartido"   : alias,
                    "clase_modelo_coincidente"  : especie_modelo_top1,
                    "rank_modelo"               : 1,
                    "rank_plantnet"             : 1,
                    "score_modelo"              : score_modelo_top1,
                    "score_plantnet"            : top_pn["score"],
                    "nombre_cientifico_plantnet": top_pn["nombre_cientifico"],
                    "nombres_comunes_plantnet"  : top_pn["nombres_comunes"],
                    "razon_comparacion"         : (
                        f"Top-1 coincide parcialmente: '{alias}' → "
                        f"'{top_pn['nombre_cientifico']}'"
                    ),
                }

    # 5. Fallback: top-2 variante sin coincidencia previa
    if len(top_k_list) >= 2:
        especie_top2  = top_k_list[1][0]
        score_top2    = top_k_list[1][1]
        nombre_top1   = normalizar_nombre(especie_modelo_top1)
        nombre_top2   = normalizar_nombre(especie_top2)
        son_variantes = nombre_top1 in nombre_top2 or nombre_top2 in nombre_top1

        if son_variantes:
            info_top2  = info_global.get(especie_top2,
                         info_global.get(especie_top2.lower(), {}))
            alias_top2 = obtener_alias_modelo(especie_top2, info_top2)
            todos_pn2  = set([normalizar_nombre(n) for n in top_pn["nombres_comunes"]]) | {cientifico_pn}

            match_comun2   = alias_top2 & set([normalizar_nombre(n) for n in top_pn["nombres_comunes"]])
            match_cient2   = alias_top2 & {cientifico_pn}
            match_parcial2 = any(
                any(p in pn_n for p in [w for w in a.split() if len(w) > 4])
                for a in alias_top2 for pn_n in todos_pn2
            )

            if match_comun2 or match_cient2 or match_parcial2:
                nombre_compartido = (sorted(match_comun2)[0] if match_comun2
                                     else cientifico_pn if match_cient2
                                     else nombre_top2)
                nombre_comun_top2 = obtener_nombre_comun_modelo(
                    especie_top2,
                    info_global.get(especie_top2, info_global.get(especie_top2.lower(), {}))
                )
                return {
                    "coinciden"                 : True,
                    "nombre_comun_compartido"   : nombre_compartido,
                    "clase_modelo_coincidente"  : especie_top2,
                    "rank_modelo"               : 2,
                    "rank_plantnet"             : 1,
                    "score_modelo"              : score_top2,
                    "score_plantnet"            : top_pn["score"],
                    "nombre_cientifico_plantnet": top_pn["nombre_cientifico"],
                    "nombres_comunes_plantnet"  : top_pn["nombres_comunes"],
                    "es_especificacion_top2"    : True,
                    "nombre_comun_top2"         : nombre_comun_top2 or nombre_top2,
                    "razon_comparacion"         : (
                        f"Top-1 ('{especie_modelo_top1}') y top-2 ('{especie_top2}') "
                        f"son variantes del mismo nombre. "
                        f"Pl@ntNet coincide con el top-2 más específico: "
                        f"'{top_pn['nombre_cientifico']}'"
                    ),
                }

    # Sin coincidencia
    return {
        "coinciden"                 : False,
        "nombre_comun_compartido"   : None,
        "clase_modelo_coincidente"  : None,
        "rank_modelo"               : None,
        "rank_plantnet"             : None,
        "score_modelo"              : score_modelo_top1,
        "score_plantnet"            : top_pn["score"],
        "nombre_cientifico_plantnet": top_pn["nombre_cientifico"],
        "nombres_comunes_plantnet"  : top_pn["nombres_comunes"],
        "razon_comparacion"         : (
            f"Top-1 modelo ('{especie_modelo_top1}' {score_modelo_top1*100:.1f}%) "
            f"no coincide con top-1 Pl@ntNet "
            f"('{top_pn['nombre_cientifico']}' score {top_pn['score']:.2f})."
        ),
    }


# ───────────────────────────────────────────────────────────────────────────
#  REGLAS DE DECISIÓN
# ───────────────────────────────────────────────────────────────────────────

def _reglas_decision(estado: EstadoArbol) -> dict:
    confianza    = estado["confianza"]
    especie_pred = estado["especie_pred"]
    top_k_list   = estado["top_k_list"]
    info_global  = estado.get("info_global", {})
    plantnet     = estado.get("plantnet_resultado", {})
    comparacion  = estado.get("comparacion", {})

    info_especie   = info_global.get(especie_pred, info_global.get(especie_pred.lower(), {}))
    nombre_comun_m = obtener_nombre_comun_modelo(especie_pred, info_especie)
    alt            = top_k_list[1][0] if len(top_k_list) > 1 else "otra especie"

    es_planta    = plantnet.get("es_planta")
    plantnet_ok  = bool(plantnet.get("resultados"))
    top_score_pn = comparacion.get("score_plantnet", 0)
    _comunes_pn  = comparacion.get("nombres_comunes_plantnet", [])
    top_nombre_pn  = (_comunes_pn[0] if _comunes_pn
                      else comparacion.get("nombre_cientifico_plantnet"))
    top_cient_pn   = comparacion.get("nombre_cientifico_plantnet")

    score_m_pond  = confianza    * PESO_MODELO
    score_pn_pond = top_score_pn * PESO_PLANTNET

    base = {
        "model_prediction_raw"           : especie_pred,
        "model_prediction_common"        : nombre_comun_m,
        "model_confidence"               : round(confianza, 4),
        "plantnet_prediction_scientific" : top_cient_pn,
        "plantnet_prediction_common"     : top_nombre_pn,
        "plantnet_common_names"          : comparacion.get("nombres_comunes_plantnet", []),
        "plantnet_score"                 : round(top_score_pn, 4),
        "model_weighted_score"           : round(score_m_pond, 4),
        "plantnet_weighted_score"        : round(score_pn_pond, 4),
        "model_plantnet_match"           : comparacion.get("coinciden", False),
        "matching_reason"                : comparacion.get("razon_comparacion", ""),
        "web_evidence_used"              : plantnet_ok,
    }

    # CASO 1: Pl@ntNet no disponible
    if not plantnet_ok and es_planta is None:
        sin_pn = " (Pl@ntNet no disponible)."
        if confianza >= 0.75:
            return {**base,
                "decision"          : "aceptar_prediccion",
                "species_selected"  : nombre_comun_m,
                "source_priority"   : "fallback_modelo",
                "contradicts_model" : False,
                "reasoning"         : f"Confianza alta ({confianza*100:.1f}%).{sin_pn}",
                "recommended_action": f"Registrar como '{nombre_comun_m}'.",
            }
        if confianza >= 0.50:
            return {**base,
                "decision"          : "mostrar_alternativas",
                "species_selected"  : nombre_comun_m,
                "source_priority"   : "fallback_modelo",
                "contradicts_model" : False,
                "reasoning"         : f"Confianza moderada ({confianza*100:.1f}%).{sin_pn}",
                "recommended_action": f"Compare '{nombre_comun_m}' y '{alt}' físicamente.",
            }
        return {**base,
            "decision"          : "pedir_nueva_foto",
            "species_selected"  : nombre_comun_m,
            "source_priority"   : "fallback_modelo",
            "contradicts_model" : False,
            "reasoning"         : f"Confianza baja ({confianza*100:.1f}%).{sin_pn}",
            "recommended_action": "Tome una nueva foto enfocando hojas, tronco y copa.",
        }

    # CASO 2: imagen no es planta
    if es_planta is False:
        return {**base,
            "decision"          : "imagen_incorrecta",
            "species_selected"  : "ninguna",
            "source_priority"   : "plantnet_no_planta",
            "contradicts_model" : True,
            "reasoning"         : (
                f"Pl@ntNet no reconoció ninguna planta en la imagen. "
                f"El modelo predijo '{nombre_comun_m}' ({confianza*100:.1f}%) "
                f"pero Pl@ntNet tiene prioridad."
            ),
            "recommended_action": (
                "La imagen no parece ser de un árbol. "
                "Tome una foto enfocando hojas, tronco o copa."
            ),
        }

    # CASO 3: coinciden por nombre común
    if comparacion.get("coinciden"):
        nombre_compartido = comparacion["nombre_comun_compartido"]

        if comparacion.get("es_especificacion_top2"):
            especie_top2   = comparacion["clase_modelo_coincidente"]
            nombre_mostrar = comparacion.get("nombre_comun_top2") or normalizar_nombre(especie_top2)
            score_top2     = comparacion["score_modelo"]
            return {**base,
                "decision"               : "aceptar_prediccion",
                "species_selected"       : nombre_mostrar,
                "source_priority"        : "modelo_y_plantnet",
                "model_prediction_common": nombre_mostrar,
                "contradicts_model"      : False,
                "reasoning"              : (
                    f"Pl@ntNet confirma la especie más específica del modelo: "
                    f"'{nombre_mostrar}' (top-2 con {score_top2*100:.1f}%). "
                    f"Pl@ntNet: {top_score_pn:.2f} ({top_cient_pn})."
                ),
                "recommended_action": f"Registrar la observación como '{nombre_mostrar}'.",
            }

        nombre_mostrar = nombre_comun_m or normalizar_nombre(especie_pred)
        return {**base,
            "decision"          : "aceptar_prediccion",
            "species_selected"  : nombre_mostrar,
            "source_priority"   : "modelo_y_plantnet",
            "contradicts_model" : False,
            "reasoning"         : (
                f"Modelo y Pl@ntNet coinciden en '{nombre_compartido}'. "
                f"Modelo: '{nombre_mostrar}' {confianza*100:.1f}% | "
                f"Pl@ntNet: {top_score_pn:.2f} ({top_cient_pn})."
            ),
            "recommended_action": f"Registrar la observación como '{nombre_mostrar}'.",
        }

    # CASO 4: modelo >= 99.9% → modelo gana siempre
    if confianza >= CONFIANZA_MODELO_PRIORIDAD:
        return {**base,
            "decision"          : "aceptar_prediccion",
            "species_selected"  : nombre_comun_m,
            "source_priority"   : "modelo",
            "contradicts_model" : False,
            "reasoning"         : (
                f"El modelo tiene confianza prácticamente perfecta "
                f"({confianza*100:.2f}%), por lo que tiene prioridad. "
                f"Pl@ntNet sugirió '{top_nombre_pn}' (score {top_score_pn:.2f})."
            ),
            "recommended_action": f"Registrar como '{nombre_comun_m}' (modelo ~100%).",
        }

    # CASO 7: ambos con baja confianza
    if confianza < 0.50 and top_score_pn < 0.50:
        return {**base,
            "decision"          : "pedir_nueva_foto",
            "species_selected"  : nombre_comun_m,
            "source_priority"   : "incertidumbre",
            "contradicts_model" : False,
            "reasoning"         : (
                f"Tanto el modelo ({confianza*100:.1f}%) como Pl@ntNet "
                f"(score {top_score_pn:.2f}) tienen baja confianza."
            ),
            "recommended_action": (
                "Tome una nueva foto con buena iluminación, "
                "enfocando hojas, tronco y copa."
            ),
        }

    # CASOS 5 y 6: no coinciden → score ponderado decide
    if score_pn_pond > score_m_pond:
        return {**base,
            "decision"          : "revision_manual",
            "species_selected"  : top_nombre_pn or top_cient_pn or nombre_comun_m,
            "source_priority"   : "plantnet",
            "contradicts_model" : True,
            "reasoning"         : (
                f"Pl@ntNet sugiere '{top_nombre_pn}' (score ponderado "
                f"{score_pn_pond:.3f}) vs modelo '{nombre_comun_m}' "
                f"(score ponderado {score_m_pond:.3f}). "
                f"Pl@ntNet tiene mayor score ponderado."
            ),
            "recommended_action": (
                f"Revisión manual recomendada. "
                f"Pl@ntNet sugiere '{top_nombre_pn}' ({top_cient_pn}). "
                f"Consulte un experto botánico."
            ),
        }
    else:
        return {**base,
            "decision"          : "revision_manual",
            "species_selected"  : nombre_comun_m,
            "source_priority"   : "modelo",
            "contradicts_model" : True,
            "reasoning"         : (
                f"El modelo sugiere '{nombre_comun_m}' (score ponderado "
                f"{score_m_pond:.3f}) vs Pl@ntNet '{top_nombre_pn}' "
                f"(score ponderado {score_pn_pond:.3f}). "
                f"El modelo tiene mayor score ponderado."
            ),
            "recommended_action": (
                f"Revisión manual recomendada. "
                f"El modelo predijo '{nombre_comun_m}', "
                f"Pl@ntNet sugiere '{top_nombre_pn}'. "
                f"Compare físicamente."
            ),
        }


# ───────────────────────────────────────────────────────────────────────────
#  ENRIQUECIMIENTO CON LLM
# ───────────────────────────────────────────────────────────────────────────

def _enriquecer_con_llm(decision: dict, estado: EstadoArbol) -> dict:
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    openai_key    = os.environ.get("OPENAI_API_KEY",    "").strip()
    if not anthropic_key and not openai_key:
        return decision

    plantnet    = estado.get("plantnet_resultado", {})
    comparacion = estado.get("comparacion", {})
    pn_lista    = "\n".join(
        f"  {r['rank']}. {r['nombre_cientifico']} "
        f"(comunes: {', '.join(r['nombres_comunes'][:3])}) score {r['score']:.2f}"
        for r in plantnet.get("resultados", [])[:3]
    ) or "  sin resultados"

    prompt = f"""Eres un botánico experto. Se tomó una decisión automática sobre la especie de un árbol.
Tu tarea es mejorar la redacción de la explicación y la recomendación.

DECISIÓN TOMADA (NO LA CAMBIES):
  Decisión         : {decision['decision']}
  Especie elegida  : {decision['species_selected']}
  Fuente prioridad : {decision['source_priority']}
  Coinciden        : {decision['model_plantnet_match']}

DATOS:
  Modelo predijo   : {decision['model_prediction_common']} ({decision['model_confidence']*100:.1f}%)
  Pl@ntNet top-3   :
{pn_lista}
  Razón comparación: {decision.get('matching_reason','')}

Responde ÚNICAMENTE con JSON:
{{
  "reasoning": "<explicación clara y útil en español, máximo 2 oraciones>",
  "recommended_action": "<acción concreta y específica para el usuario>"
}}"""

    texto_raw = None
    try:
        if anthropic_key:
            import anthropic
            msg = anthropic.Anthropic(api_key=anthropic_key).messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            texto_raw = msg.content[0].text
        elif openai_key:
            from openai import OpenAI
            resp = OpenAI(api_key=openai_key).chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            texto_raw = resp.choices[0].message.content
    except Exception as e:
        print(f"   ⚠️  LLM: {e}")
        return decision

    if texto_raw:
        try:
            texto = texto_raw.strip()
            if "```" in texto:
                texto = texto.split("```")[1]
                if texto.startswith("json"):
                    texto = texto[4:]
            mejoras = json.loads(texto.strip())
            if "reasoning" in mejoras:
                decision["reasoning"] = mejoras["reasoning"]
            if "recommended_action" in mejoras:
                decision["recommended_action"] = mejoras["recommended_action"]
        except Exception:
            pass

    return decision


# ───────────────────────────────────────────────────────────────────────────
#  AGENTE DE MANTENIMIENTO — Evaluación visual de salud del árbol
# ───────────────────────────────────────────────────────────────────────────

def _evaluar_salud_con_numpy(ruta_imagen: str, especie_pred: str = "") -> dict:
    """Fallback: análisis de color con numpy."""
    especie_normalizada = especie_pred.lower().replace(" ", "_")
    ignorar_amarillo    = especie_normalizada in ESPECIES_COLOR_FLORAL

    try:
        import numpy as np
        from PIL import Image as PILImage

        img = PILImage.open(ruta_imagen).convert("RGB")
        img = img.resize((300, 300))
        arr = np.array(img, dtype=np.float32) / 255.0
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        total   = r.size

        mask_cielo        = (b > 0.45) & (b > g * 1.20) & (b > r * 1.20)
        mask_verde_claro  = (g > 0.25) & (g > r * 1.08) & (g > b * 0.88) & ~mask_cielo
        mask_verde_oscuro = (g > r * 1.12) & (g > b * 1.10) & (g > 0.08) & (g < 0.28) & ~mask_cielo & ~mask_verde_claro
        mask_verde        = mask_verde_claro | mask_verde_oscuro
        mask_seco         = (np.zeros_like(r, dtype=bool) if ignorar_amarillo
                             else (r > 0.45) & (g > 0.35) & (b < 0.32) & ~mask_verde & ~mask_cielo)
        mask_marron       = (r > 0.28) & (g < r * 0.80) & (b < r * 0.70) & ~mask_verde & ~mask_seco & ~mask_cielo

        pct_verde  = int(mask_verde.sum()  / total * 100)
        pct_seco   = int(mask_seco.sum()   / total * 100)
        pct_marron = int(mask_marron.sum() / total * 100)

        if pct_verde >= 25 and pct_seco < 20:
            estado        = "aparentemente_sano"
            recomendacion = "El árbol parece visualmente saludable (análisis de color)."
        elif pct_verde >= 12 and 20 <= pct_seco < 40:
            estado        = "estres_moderado"
            recomendacion = "Se observan áreas secas. Verificar riego y suelo."
        elif pct_seco >= 40 or pct_verde < 12:
            estado        = "posible_enfermedad"
            recomendacion = "Alto porcentaje de follaje seco. Revisión fitosanitaria recomendada."
        else:
            estado        = "indeterminado"
            recomendacion = "No se pudo determinar el estado con certeza."

        nota = (f" (colores florales de '{especie_pred}' excluidos)"
                if ignorar_amarillo else "")
        return {
            "estado"       : estado,
            "pct_verde"    : pct_verde,
            "pct_seco"     : pct_seco,
            "pct_marron"   : pct_marron,
            "recomendacion": recomendacion,
            "metodo"       : "analisis_color_numpy",
            "limitacion"   : (
                "Diagnóstico por análisis de color (Pl@ntNet no fue concluyente)."
                + nota +
                " No reemplaza revisión fitosanitaria profesional."
            ),
        }
    except ImportError:
        return {"estado": "no_determinado", "metodo": "numpy_no_disponible",
                "razon": "Instala numpy: pip install numpy"}
    except Exception as e:
        return {"estado": "no_determinado", "metodo": "error", "razon": str(e)}


def evaluar_salud_visual(ruta_imagen: str, especie_pred: str = "") -> dict:
    """Evalúa el estado visual de salud del árbol con Pl@ntNet + numpy como fallback."""
    UMBRAL_SCORE_CONFIABLE = 0.15
    plantnet_key = os.environ.get("PLANTNET_KEY", "").strip()

    if plantnet_key and ruta_imagen and os.path.isfile(ruta_imagen):
        try:
            import requests
            print("   🔬 Analizando hojas con Pl@ntNet...")
            with open(ruta_imagen, "rb") as f:
                resp = requests.post(
                    "https://my-api.plantnet.org/v2/identify/all",
                    params={
                        "api-key"   : plantnet_key,
                        "lang"      : "es",
                        "nb-results": 3,
                        "organs"    : "leaf",
                    },
                    files  ={"images": f},
                    timeout=20,
                )

            if resp.status_code == 200:
                datos    = resp.json()
                top_r    = datos.get("results", [])
                score_pn = top_r[0].get("score", 0) if top_r else 0

                if score_pn >= 0.30:
                    return {
                        "estado"       : "aparentemente_sano",
                        "score_hoja"   : round(score_pn, 3),
                        "recomendacion": (
                            f"Pl@ntNet reconoció las hojas con score {score_pn:.2f}. "
                            "El follaje parece estar en buen estado."
                        ),
                        "metodo"       : "plantnet_leaf",
                        "limitacion"   : (
                            "Basado en reconocimiento visual de hojas por Pl@ntNet. "
                            "No reemplaza revisión fitosanitaria profesional."
                        ),
                    }
                elif score_pn >= UMBRAL_SCORE_CONFIABLE:
                    return {
                        "estado"       : "estres_moderado",
                        "score_hoja"   : round(score_pn, 3),
                        "recomendacion": (
                            f"Pl@ntNet reconoció las hojas con score moderado {score_pn:.2f}. "
                            "Posible estrés o imagen poco clara. Verificar en campo."
                        ),
                        "metodo"       : "plantnet_leaf",
                        "limitacion"   : (
                            "Score moderado puede indicar hojas deterioradas o ángulo de foto. "
                            "No reemplaza revisión fitosanitaria profesional."
                        ),
                    }
                else:
                    print(f"   ⚠️  Pl@ntNet leaf score bajo ({score_pn:.2f}) — usando análisis de color.")

            elif resp.status_code == 404:
                print("   ⚠️  Pl@ntNet no detectó hojas — usando análisis de color.")

        except Exception as e:
            print(f"   ⚠️  Pl@ntNet leaf: {e} — usando análisis de color.")

    return _evaluar_salud_con_numpy(ruta_imagen, especie_pred)


def health_assessment_agent(estado: EstadoArbol) -> EstadoArbol:
    """Agente de mantenimiento: evalúa el estado visual de salud del árbol."""
    ruta_imagen  = estado.get("ruta_imagen", "")
    especie_pred = estado.get("especie_pred", "")
    plantnet     = estado.get("plantnet_resultado", {})
    top_score_pn = plantnet.get("top_score", 0)
    es_planta    = plantnet.get("es_planta")

    if es_planta is True and top_score_pn > 0:
        top_nombre = (plantnet.get("top_nombre") or
                      plantnet.get("top_especie") or especie_pred)

        if top_score_pn >= 0.25:
            estado_salud  = "aparentemente_sano"
            recomendacion = (
                f"Pl@ntNet reconoció '{top_nombre}' con score {top_score_pn:.2f}. "
                "El árbol parece visualmente saludable."
            )
        elif top_score_pn >= 0.10:
            estado_salud  = "estres_moderado"
            recomendacion = (
                f"Pl@ntNet reconoció '{top_nombre}' con score moderado "
                f"({top_score_pn:.2f}). Verifique el estado del follaje en campo."
            )
        else:
            estado_salud  = "indeterminado"
            recomendacion = (
                f"Score muy bajo ({top_score_pn:.2f}). "
                "La imagen puede no mostrar el follaje claramente. "
                "Se recomienda tomar una nueva foto enfocando las hojas."
            )

        numpy_datos = _evaluar_salud_con_numpy(ruta_imagen, especie_pred)
        salud = {
            "estado"       : estado_salud,
            "score_hoja"   : round(top_score_pn, 3),
            "recomendacion": recomendacion,
            "metodo"       : "plantnet_score",
            "limitacion"   : (
                "Basado en el score de Pl@ntNet complementado con análisis de color. "
                "No reemplaza revisión fitosanitaria profesional."
            ),
            "pct_verde"    : numpy_datos.get("pct_verde",  0),
            "pct_seco"     : numpy_datos.get("pct_seco",   0),
            "pct_marron"   : numpy_datos.get("pct_marron", 0),
        }

    elif es_planta is False:
        salud = {
            "estado"       : "no_es_planta",
            "score_hoja"   : 0,
            "recomendacion": "Pl@ntNet no reconoció ninguna planta. La imagen no parece ser de un árbol.",
            "metodo"       : "plantnet_score",
            "limitacion"   : "No aplica evaluación de salud.",
            "pct_verde"    : 0,
            "pct_seco"     : 0,
            "pct_marron"   : 0,
        }
    else:
        salud = evaluar_salud_visual(ruta_imagen, especie_pred)

    return {**estado, "salud_arbol": salud}


# ───────────────────────────────────────────────────────────────────────────
#  NODOS DEL GRAFO LANGGRAPH
# ───────────────────────────────────────────────────────────────────────────

def prediction_validator_agent(estado: EstadoArbol) -> EstadoArbol:
    return {**estado}


def web_species_research_agent(estado: EstadoArbol) -> EstadoArbol:
    ruta_imagen = estado.get("ruta_imagen", "")
    top_k_list  = estado["top_k_list"]
    info_global = estado.get("info_global", {})

    print("   🌐 Pl@ntNet: analizando imagen...")

    plantnet = (
        consultar_plantnet(ruta_imagen)
        if ruta_imagen and os.path.isfile(ruta_imagen)
        else {"es_planta": None, "resultados": [], "razon": "sin imagen"}
    )
    razon_pn = plantnet.get("razon")
    if razon_pn:
        print(f"   📋 Pl@ntNet: {razon_pn}")

    comparacion = buscar_coincidencia_nombre_comun(top_k_list, plantnet, info_global)

    return {**estado, "plantnet_resultado": plantnet, "comparacion": comparacion}


def final_decision_agent(estado: EstadoArbol) -> EstadoArbol:
    decision = _reglas_decision(estado)
    decision = _enriquecer_con_llm(decision, estado)
    return {**estado, "decision_final": decision}


# ───────────────────────────────────────────────────────────────────────────
#  GRAFO LANGGRAPH
# ───────────────────────────────────────────────────────────────────────────

def construir_grafo():
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        return None
    g = StateGraph(EstadoArbol)
    g.add_node("validator", prediction_validator_agent)
    g.add_node("plantnet",  web_species_research_agent)
    g.add_node("final",     final_decision_agent)
    g.add_node("health",    health_assessment_agent)
    g.set_entry_point("validator")
    g.add_edge("validator", "plantnet")
    g.add_edge("plantnet",  "final")
    g.add_edge("final",     "health")
    g.add_edge("health",    END)
    return g.compile()


def ejecutar_agente(especie_pred, confianza, top_k_list,
                    info_especie, ruta_imagen=None,
                    info_global=None) -> dict:
    estado_inicial: EstadoArbol = {
        "especie_pred"      : especie_pred,
        "confianza"         : confianza,
        "top_k_list"        : top_k_list,
        "info_especie"      : info_especie,
        "info_global"       : info_global or {},
        "ruta_imagen"       : ruta_imagen or "",
        "plantnet_resultado": {},
        "comparacion"       : {},
        "decision_final"    : {},
        "salud_arbol"       : {},
    }
    grafo = construir_grafo()
    if grafo:
        return grafo.invoke(estado_inicial)
    print("   ℹ️  LangGraph no instalado. Ejecutando en secuencia.")
    e = prediction_validator_agent(estado_inicial)
    e = web_species_research_agent(e)
    e = final_decision_agent(e)
    return health_assessment_agent(e)


# ───────────────────────────────────────────────────────────────────────────
#  VISUALIZACIÓN — Agente de Validación
# ───────────────────────────────────────────────────────────────────────────

def mostrar_resultado_completo(estado_final: dict):
    """Agente de Validación: modelo local + verificación + decisión."""
    d           = estado_final.get("decision_final", {})
    plantnet    = estado_final.get("plantnet_resultado", {})
    comparacion = estado_final.get("comparacion", {})

    if not d:
        return

    nombre_m   = d.get("model_prediction_common", d.get("model_prediction_raw", ""))
    confianza  = d.get("model_confidence", 0)
    nombre_pn  = d.get("plantnet_prediction_common", "")
    cient_pn   = d.get("plantnet_prediction_scientific", "")
    comunes_pn = d.get("plantnet_common_names", [])
    score_pn   = d.get("plantnet_score", 0)
    coinciden  = d.get("model_plantnet_match", False)
    match_r    = d.get("matching_reason", "")
    decision   = d.get("decision", "")
    especie_sel= d.get("species_selected", "")
    razon      = d.get("reasoning", "")

    iconos_dec = {
        "aceptar_prediccion"  : "✅",
        "mostrar_alternativas": "⚠️ ",
        "pedir_nueva_foto"    : "📷",
        "revision_manual"     : "🔬",
        "imagen_incorrecta"   : "🚫",
    }

    print()
    print(SEP)

    # ── Modelo local ─────────────────────────────────────────────────────────
    print("🌳 Modelo local")
    print(f"   Predicción  : {nombre_m}")
    print(f"   Confianza   : {confianza*100:.1f}%")

    # ── Agente de Verificación ────────────────────────────────────────────────
    print()
    if plantnet.get("resultados"):
        print("   Agente de Verificación")
        print(f"   Predicción  : {nombre_pn}")
        if cient_pn and normalizar_nombre(cient_pn) != normalizar_nombre(nombre_pn):
            print(f"   Científico  : {cient_pn}")
        print(f"   Score       : {score_pn:.3f}")
        if comunes_pn:
            print(f"   Nombres comunes: {', '.join(comunes_pn[:4])}")
    elif plantnet.get("es_planta") is False:
        print("   Agente de Verificación")
        print("   No se reconoció ninguna planta en la imagen.")
    else:
        print("   Agente de Verificación")
        print("   No disponible — decisión basada solo en el modelo.")

    print()
    print(SEP)

    # ── Comparación ──────────────────────────────────────────────────────────
    print("Comparación")
    if coinciden:
        print(f"   Coinciden por: {match_r}")
        print(f"   Nombre común compartido: '{especie_sel}'")
    else:
        source     = d.get("source_priority", "")
        score_m_p  = d.get("model_weighted_score", 0)
        score_pn_p = d.get("plantnet_weighted_score", 0)
        print("   No coinciden.")
        if source == "plantnet":
            print(f"   Prioridad: Pl@ntNet  (score ponderado {score_pn_p:.3f} > {score_m_p:.3f})")
        elif source == "modelo":
            print(f"   Prioridad: Modelo    (score ponderado {score_m_p:.3f} > {score_pn_p:.3f})")

    print()

    # ── Decisión ─────────────────────────────────────────────────────────────
    print("🤖 Decisión")
    print(f"   {iconos_dec.get(decision, '🤔')} {decision}")
    print(f"   Especie     : {especie_sel}")
    print(f"   Razonamiento: {razon}")

    if d.get("contradicts_model"):
        print()
        print("   ⚠️  Pl@ntNet contradice la predicción del modelo.")

    print(SEP)
    print()


# ───────────────────────────────────────────────────────────────────────────
#  VISUALIZACIÓN — Agente de Mantenimiento
# ───────────────────────────────────────────────────────────────────────────

def mostrar_salud_arbol(estado_final: dict):
    """Agente de Mantenimiento: evaluación visual de salud del árbol."""
    salud = estado_final.get("salud_arbol", {})
    if not salud:
        return

    estado   = salud.get("estado", "no_determinado")
    metodo   = salud.get("metodo", "")
    usa_plantnet = metodo in ("plantnet_leaf", "plantnet_score")

    iconos_estado = {
        "aparentemente_sano" : "🟢",
        "estres_moderado"    : "🟡",
        "posible_enfermedad" : "🔴",
        "indeterminado"      : "⚪",
        "no_determinado"     : "⚪",
        "no_es_planta"       : "🚫",
    }

    print("🌿 EVALUACIÓN VISUAL DE SALUD")
    print(SEP)
    print(f"   Estado            : {iconos_estado.get(estado, '⚪')} {estado}")

    if estado in ("no_determinado", "no_es_planta"):
        razon = salud.get("razon", salud.get("recomendacion", ""))
        if razon:
            print(f"   Razón             : {razon}")
    else:
        if usa_plantnet:
            print(f"   Score (Pl@ntNet)  : {salud.get('score_hoja', 0):.3f}")
        print(f"   Porcentaje verde  : {salud.get('pct_verde',  0)}%")
        print(f"   Porcentaje seco   : {salud.get('pct_seco',   0)}%")
        print(f"   Porcentaje marrón : {salud.get('pct_marron', 0)}%")
        metodo_str = ("Pl@ntNet + análisis de color" if usa_plantnet
                      else "análisis de color (numpy)")
        print(f"   Método            : {metodo_str}")
        print(f"   Recomendación     : {salud.get('recomendacion', '')}")
        print(f"   Limitación        : {salud.get('limitacion', '')}")

    print(SEP)
    print()


# ───────────────────────────────────────────────────────────────────────────
#  FUNCIONES PRINCIPALES
# ───────────────────────────────────────────────────────────────────────────

def cargar_modelo(modelo_path, device):
    if not os.path.isfile(modelo_path):
        raise FileNotFoundError(f"No se encontró '{modelo_path}'.")
    checkpoint = torch.load(modelo_path, map_location=device)
    clases     = checkpoint["clases"]
    img_size   = checkpoint.get("img_size", IMG_SIZE)
    num_clases = checkpoint["num_clases"]
    m = crear_modelo(num_clases, device)
    m.load_state_dict(checkpoint["model_state_dict"])
    m.eval()
    return m, clases, img_size


def preprocesar_imagen(ruta, img_size):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tf(Image.open(ruta).convert("RGB")).unsqueeze(0)


def cargar_info(info_path):
    if os.path.isfile(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def predecir(ruta_imagen, modelo, clases, img_size, info, device, top_k=3):
    if not os.path.isfile(ruta_imagen):
        print(f"✗ No se encontró la imagen: {ruta_imagen}")
        return

    tensor = preprocesar_imagen(ruta_imagen, img_size).to(device)
    with torch.no_grad():
        logits = modelo(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    top_indices  = probs.argsort(descending=True)[:top_k]
    top_k_list   = [(clases[i], probs[i].item()) for i in top_indices]
    especie_pred, confianza = top_k_list[0]

    print(f"\n   Analizando con agentes de verificación y mantenimiento...\n")

    info_especie = info.get(especie_pred, info.get(especie_pred.lower(), {}))
    estado_final = ejecutar_agente(
        especie_pred, confianza, top_k_list,
        info_especie, ruta_imagen,
        info_global=info,
    )

    mostrar_resultado_completo(estado_final)
    mostrar_salud_arbol(estado_final)
    return estado_final


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n🌿 Sistema de Clasificación de Árboles")
    print(f"💻 Dispositivo: {device}")
    print("Cargando modelo...")
    m, clases, img_size = cargar_modelo(MODELO_PATH, device)
    info = cargar_info(INFO_PATH)
    print(f"✓ Modelo cargado con {len(clases)} especies")
    ruta_imagen = args.imagen or input("\n📷 Ingresa la ruta de la imagen: ").strip()
    if not ruta_imagen:
        print("✗ No ingresaste ninguna imagen.")
        sys.exit(1)
    predecir(ruta_imagen, m, clases, img_size, info, device, top_k=args.top)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagen", type=str, default=None)
    parser.add_argument("--top",    type=int, default=3)
    args = parser.parse_args([])
    main(args)
