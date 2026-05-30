"""
agent_validation.py — Agente de validación para Campus Tree Explorer

Este módulo NO carga el modelo local. El modelo ONNX vive en main.py.
Aquí solo está la capa agente:
Imagen + resultado ONNX → Pl@ntNet → comparación → decisión final.

Variables opcionales:
- PLANTNET_KEY: activa validación visual con Pl@ntNet.
- ANTHROPIC_API_KEY u OPENAI_API_KEY: mejora la redacción de la explicación.

Si LangGraph no está instalado, el flujo se ejecuta en secuencia normal.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import List, Optional, Set, Tuple, TypedDict

# ─────────────────────────────────────────────────────────────────────────────
# Configuración de decisión
# ─────────────────────────────────────────────────────────────────────────────

CONFIANZA_MODELO_PRIORIDAD = 0.999
PESO_MODELO = 1.00
PESO_PLANTNET = 1.20

SINONIMOS_ESPECIES = {
    "guayacan_amarillo": [
        "guayacan amarillo", "guayacán amarillo", "lapacho amarillo",
        "araguaney", "roble amarillo", "handroanthus chrysanthus",
        "tabebuia chrysantha",
    ],
    "guayacan_rosado": [
        "guayacan rosado", "guayacán rosado", "roble rosado", "apamate",
        "tabebuia rosea", "handroanthus roseus",
    ],
    "ceiba": ["ceiba", "ceiba pentandra", "kapok tree", "silk cotton tree"],
    "mango": ["mango", "mangifera indica"],
    "aguacate": ["aguacate", "palta", "persea americana"],
    "platymiscium_pinnatum": [
        "platymiscium pinnatum", "platymiscium", "granadillo",
        "cristobal", "cristóbal",
    ],
    "pinus_patula": ["pinus patula", "pino patula", "mexican weeping pine"],
    "piptadenia_flava": ["piptadenia flava", "piptadenia"],
    "balso": [
        "balso", "balsa", "balsa kapok", "balzovec",
        "ochroma pyramidale", "ochroma lagopus", "balsa wood", "madera balsa",
    ],
    "palma_abanico": [
        "palma abanico", "palmera abanico", "palma de abanico",
        "washingtonia filifera", "washingtonia robusta",
        "palmera abanico mexicana", "fan palm",
    ],
    "saman": ["saman", "samán", "samanea saman", "albizia saman", "rain tree", "monkey pod"],
    "bala_de_canon": [
        "bala de canon", "bala de cañon", "bala de cañón",
        "couroupita guianensis", "cannonball tree",
    ],
    "roble": ["roble", "quercus", "oak"],
    "acacia": ["acacia", "acacia mangium", "wattle"],
}


class EstadoArbol(TypedDict):
    especie_pred: str
    confianza: float
    top_k_list: List[Tuple[str, float]]
    info_especie: dict
    info_global: dict
    ruta_imagen: str
    plantnet_resultado: dict
    comparacion: dict
    decision_final: dict


# ─────────────────────────────────────────────────────────────────────────────
# Normalización y alias
# ─────────────────────────────────────────────────────────────────────────────

def normalizar_nombre(nombre: str) -> str:
    """Normaliza nombres comunes/científicos para comparación robusta."""
    if not nombre:
        return ""
    texto = str(nombre).lower().replace("_", " ").replace("-", " ")
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = re.sub(r"[^a-z0-9 ]", "", texto)
    return re.sub(r"\s+", " ", texto).strip()


def _info_value(info: dict, keys: list[str]) -> Optional[str]:
    for key in keys:
        value = info.get(key) if isinstance(info, dict) else None
        if value:
            if isinstance(value, list):
                return str(value[0]) if value else None
            return str(value)
    return None


def obtener_nombre_comun_modelo(especie_modelo: str, info: dict) -> str:
    value = _info_value(info, [
        "nombre_comun", "nombre_común", "nombre comun", "nombre común",
        "common_name", "commonNames", "nombres_comunes", "nombre",
    ])
    return normalizar_nombre(value or especie_modelo)


def obtener_alias_modelo(especie_modelo: str, info: dict) -> Set[str]:
    alias: Set[str] = {
        normalizar_nombre(especie_modelo),
        especie_modelo.lower(),
    }

    nombre_comun = obtener_nombre_comun_modelo(especie_modelo, info)
    if nombre_comun:
        alias.add(nombre_comun)

    for key in ["nombre_cientifico", "scientific_name", "scientificName"]:
        value = info.get(key) if isinstance(info, dict) else None
        if value:
            alias.add(normalizar_nombre(str(value)))

    for key in ["sinonimos", "synonyms", "alias"]:
        value = info.get(key, []) if isinstance(info, dict) else []
        if isinstance(value, list):
            for item in value:
                alias.add(normalizar_nombre(str(item)))
        elif value:
            alias.add(normalizar_nombre(str(value)))

    for item in SINONIMOS_ESPECIES.get(especie_modelo, []):
        alias.add(normalizar_nombre(item))

    alias.discard("")
    return alias


def _display_name_from_info(especie_key: str, info: dict) -> str:
    value = _info_value(info, ["nombre_comun", "nombre", "common_name"])
    return value or especie_key.replace("_", " ").title()


def _get_info(info_global: dict, key: str) -> dict:
    if not isinstance(info_global, dict):
        return {}
    return info_global.get(key) or info_global.get(key.lower()) or {}


# ─────────────────────────────────────────────────────────────────────────────
# Pl@ntNet
# ─────────────────────────────────────────────────────────────────────────────

def consultar_plantnet(ruta_imagen: str) -> dict:
    """Consulta Pl@ntNet con la imagen y devuelve los 5 mejores resultados."""
    plantnet_key = os.environ.get("PLANTNET_KEY", "").strip()
    if not plantnet_key:
        return {
            "es_planta": None,
            "resultados": [],
            "razon": "PLANTNET_KEY no configurada",
        }

    if not ruta_imagen or not os.path.isfile(ruta_imagen):
        return {"es_planta": None, "resultados": [], "razon": "imagen no encontrada"}

    try:
        import requests
    except Exception:
        return {"es_planta": None, "resultados": [], "razon": "requests no instalado"}

    try:
        with open(ruta_imagen, "rb") as fh:
            resp = requests.post(
                "https://my-api.plantnet.org/v2/identify/all",
                params={"api-key": plantnet_key, "lang": "es", "nb-results": 5},
                files={"images": fh},
                timeout=20,
            )

        if resp.status_code == 404:
            return {
                "es_planta": False,
                "resultados": [],
                "razon": "Pl@ntNet no reconoció ninguna planta",
            }

        if resp.status_code != 200:
            return {
                "es_planta": None,
                "resultados": [],
                "razon": f"Pl@ntNet no disponible HTTP {resp.status_code}",
            }

        datos = resp.json()
        resultados = []
        for rank, item in enumerate(datos.get("results", []), 1):
            species = item.get("species", {})
            comunes = [n for n in species.get("commonNames", []) if n]
            resultados.append({
                "rank": rank,
                "nombre_cientifico": species.get("scientificNameWithoutAuthor", ""),
                "nombres_comunes": comunes,
                "score": round(float(item.get("score", 0.0)), 4),
            })

        if not resultados:
            return {
                "es_planta": None,
                "resultados": [],
                "razon": "Pl@ntNet respondió sin resultados",
            }

        top = resultados[0]
        nombre_mostrar = top["nombres_comunes"][0] if top["nombres_comunes"] else top["nombre_cientifico"]
        return {
            "es_planta": True,
            "resultados": resultados,
            "top_especie": top["nombre_cientifico"],
            "top_nombre": nombre_mostrar,
            "top_score": top["score"],
            "razon": (
                f"Pl@ntNet identificó '{nombre_mostrar}' "
                f"({top['nombre_cientifico']}) score {top['score']:.2f}"
            ),
        }
    except Exception as exc:
        return {"es_planta": None, "resultados": [], "razon": f"Pl@ntNet error: {exc}"}


# ─────────────────────────────────────────────────────────────────────────────
# Comparación modelo vs Pl@ntNet
# ─────────────────────────────────────────────────────────────────────────────

def buscar_coincidencia_nombre_comun(top_k_list: list, plantnet_resultado: dict, info_global: dict) -> dict:
    """
    Compara el top-1 del modelo contra el top-1 de Pl@ntNet.
    Si top-1 y top-2 del modelo son variantes y Pl@ntNet confirma top-2,
    permite seleccionar el top-2 más específico.
    """
    score_modelo_top1 = float(top_k_list[0][1]) if top_k_list else 0.0
    especie_modelo_top1 = top_k_list[0][0] if top_k_list else ""

    if not plantnet_resultado.get("resultados"):
        return {
            "coinciden": False,
            "nombre_comun_compartido": None,
            "clase_modelo_coincidente": None,
            "rank_modelo": None,
            "rank_plantnet": None,
            "score_modelo": score_modelo_top1,
            "score_plantnet": 0.0,
            "nombre_cientifico_plantnet": None,
            "nombres_comunes_plantnet": [],
            "razon_comparacion": "Pl@ntNet no disponible.",
        }

    top_pn = plantnet_resultado["resultados"][0]
    info_top1 = _get_info(info_global, especie_modelo_top1)
    alias_modelo = obtener_alias_modelo(especie_modelo_top1, info_top1)

    cientifico_pn = normalizar_nombre(top_pn.get("nombre_cientifico", ""))
    comunes_pn = [normalizar_nombre(n) for n in top_pn.get("nombres_comunes", [])]
    todos_pn = set(comunes_pn) | {cientifico_pn}

    match_comun = alias_modelo & set(comunes_pn)
    if match_comun:
        nombre_compartido = sorted(match_comun)[0]
        return {
            "coinciden": True,
            "nombre_comun_compartido": nombre_compartido,
            "clase_modelo_coincidente": especie_modelo_top1,
            "rank_modelo": 1,
            "rank_plantnet": 1,
            "score_modelo": score_modelo_top1,
            "score_plantnet": float(top_pn.get("score", 0.0)),
            "nombre_cientifico_plantnet": top_pn.get("nombre_cientifico"),
            "nombres_comunes_plantnet": top_pn.get("nombres_comunes", []),
            "razon_comparacion": f"Top-1 coincide por nombre común: '{nombre_compartido}'",
        }

    if alias_modelo & {cientifico_pn}:
        return {
            "coinciden": True,
            "nombre_comun_compartido": cientifico_pn,
            "clase_modelo_coincidente": especie_modelo_top1,
            "rank_modelo": 1,
            "rank_plantnet": 1,
            "score_modelo": score_modelo_top1,
            "score_plantnet": float(top_pn.get("score", 0.0)),
            "nombre_cientifico_plantnet": top_pn.get("nombre_cientifico"),
            "nombres_comunes_plantnet": top_pn.get("nombres_comunes", []),
            "razon_comparacion": f"Top-1 coincide por nombre científico: '{top_pn.get('nombre_cientifico', '')}'",
        }

    # Coincidencia parcial prudente: palabras informativas largas.
    for alias in alias_modelo:
        palabras = [p for p in alias.split() if len(p) > 4]
        if not palabras:
            continue
        for pn_nombre in todos_pn:
            if any(p in pn_nombre for p in palabras):
                return {
                    "coinciden": True,
                    "nombre_comun_compartido": alias,
                    "clase_modelo_coincidente": especie_modelo_top1,
                    "rank_modelo": 1,
                    "rank_plantnet": 1,
                    "score_modelo": score_modelo_top1,
                    "score_plantnet": float(top_pn.get("score", 0.0)),
                    "nombre_cientifico_plantnet": top_pn.get("nombre_cientifico"),
                    "nombres_comunes_plantnet": top_pn.get("nombres_comunes", []),
                    "razon_comparacion": f"Top-1 coincide parcialmente: '{alias}' ↔ '{top_pn.get('nombre_cientifico', '')}'",
                }

    # Caso especial: top-1 genérico y top-2 específico.
    if len(top_k_list) >= 2:
        especie_top2 = top_k_list[1][0]
        score_top2 = float(top_k_list[1][1])
        nombre_top1 = normalizar_nombre(especie_modelo_top1)
        nombre_top2 = normalizar_nombre(especie_top2)
        son_variantes = bool(nombre_top1 and nombre_top2 and (nombre_top1 in nombre_top2 or nombre_top2 in nombre_top1))

        if son_variantes:
            info_top2 = _get_info(info_global, especie_top2)
            alias_top2 = obtener_alias_modelo(especie_top2, info_top2)
            match_comun2 = alias_top2 & set(comunes_pn)
            match_cient2 = alias_top2 & {cientifico_pn}
            match_parcial2 = any(
                any(p in pn_n for p in [w for w in alias.split() if len(w) > 4])
                for alias in alias_top2
                for pn_n in todos_pn
            )
            if match_comun2 or match_cient2 or match_parcial2:
                nombre_compartido = sorted(match_comun2)[0] if match_comun2 else (cientifico_pn if match_cient2 else nombre_top2)
                nombre_comun_top2 = obtener_nombre_comun_modelo(especie_top2, info_top2)
                return {
                    "coinciden": True,
                    "nombre_comun_compartido": nombre_compartido,
                    "clase_modelo_coincidente": especie_top2,
                    "rank_modelo": 2,
                    "rank_plantnet": 1,
                    "score_modelo": score_top2,
                    "score_plantnet": float(top_pn.get("score", 0.0)),
                    "nombre_cientifico_plantnet": top_pn.get("nombre_cientifico"),
                    "nombres_comunes_plantnet": top_pn.get("nombres_comunes", []),
                    "es_especificacion_top2": True,
                    "nombre_comun_top2": nombre_comun_top2 or nombre_top2,
                    "razon_comparacion": (
                        f"Top-1 ('{especie_modelo_top1}') y top-2 ('{especie_top2}') son variantes. "
                        f"Pl@ntNet coincide con el top-2 más específico."
                    ),
                }

    return {
        "coinciden": False,
        "nombre_comun_compartido": None,
        "clase_modelo_coincidente": None,
        "rank_modelo": None,
        "rank_plantnet": None,
        "score_modelo": score_modelo_top1,
        "score_plantnet": float(top_pn.get("score", 0.0)),
        "nombre_cientifico_plantnet": top_pn.get("nombre_cientifico"),
        "nombres_comunes_plantnet": top_pn.get("nombres_comunes", []),
        "razon_comparacion": (
            f"Top-1 modelo ('{especie_modelo_top1}' {score_modelo_top1*100:.1f}%) "
            f"no coincide con top-1 Pl@ntNet ('{top_pn.get('nombre_cientifico', '')}' "
            f"score {float(top_pn.get('score', 0.0)):.2f})."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Decisión final
# ─────────────────────────────────────────────────────────────────────────────

def _reglas_decision(estado: EstadoArbol) -> dict:
    confianza = float(estado.get("confianza", 0.0))
    especie_pred = estado.get("especie_pred", "")
    top_k_list = estado.get("top_k_list", [])
    info_global = estado.get("info_global", {})
    plantnet = estado.get("plantnet_resultado", {})
    comparacion = estado.get("comparacion", {})

    info_especie = _get_info(info_global, especie_pred) or estado.get("info_especie", {})
    nombre_comun_m = _display_name_from_info(especie_pred, info_especie)
    alt = top_k_list[1][0] if len(top_k_list) > 1 else "otra especie"

    es_planta = plantnet.get("es_planta")
    plantnet_ok = bool(plantnet.get("resultados"))
    top_score_pn = float(comparacion.get("score_plantnet", 0.0) or 0.0)
    comunes_pn = comparacion.get("nombres_comunes_plantnet", []) or []
    top_nombre_pn = comunes_pn[0] if comunes_pn else comparacion.get("nombre_cientifico_plantnet")
    top_cient_pn = comparacion.get("nombre_cientifico_plantnet")

    score_m_pond = confianza * PESO_MODELO
    score_pn_pond = top_score_pn * PESO_PLANTNET

    base = {
        "model_prediction_raw": especie_pred,
        "model_prediction_common": nombre_comun_m,
        "model_confidence": round(confianza, 4),
        "plantnet_prediction_scientific": top_cient_pn,
        "plantnet_prediction_common": top_nombre_pn,
        "plantnet_common_names": comunes_pn,
        "plantnet_score": round(top_score_pn, 4),
        "model_weighted_score": round(score_m_pond, 4),
        "plantnet_weighted_score": round(score_pn_pond, 4),
        "model_plantnet_match": bool(comparacion.get("coinciden", False)),
        "matching_reason": comparacion.get("razon_comparacion", ""),
        "web_evidence_used": plantnet_ok,
        "species_selected_key": especie_pred,
    }

    if not plantnet_ok and es_planta is None:
        if confianza >= 0.75:
            return {**base,
                "decision": "aceptar_prediccion",
                "species_selected": nombre_comun_m,
                "source_priority": "fallback_modelo",
                "contradicts_model": False,
                "reasoning": f"Confianza alta del modelo ({confianza*100:.1f}%). Pl@ntNet no está disponible.",
                "recommended_action": f"Registrar como '{nombre_comun_m}' si la imagen corresponde claramente al árbol.",
            }
        if confianza >= 0.50:
            return {**base,
                "decision": "mostrar_alternativas",
                "species_selected": nombre_comun_m,
                "source_priority": "fallback_modelo",
                "contradicts_model": False,
                "reasoning": f"Confianza moderada del modelo ({confianza*100:.1f}%). Pl@ntNet no está disponible.",
                "recommended_action": f"Compare físicamente '{nombre_comun_m}' con '{alt.replace('_', ' ')}' antes de registrar.",
            }
        return {**base,
            "decision": "pedir_nueva_foto",
            "species_selected": nombre_comun_m,
            "source_priority": "fallback_modelo",
            "contradicts_model": False,
            "reasoning": f"Confianza baja del modelo ({confianza*100:.1f}%). Pl@ntNet no está disponible.",
            "recommended_action": "Tome una nueva foto con buena luz, enfocando hojas, flores, frutos o corteza.",
        }

    if es_planta is False:
        return {**base,
            "decision": "imagen_incorrecta",
            "species_selected": "ninguna",
            "species_selected_key": None,
            "source_priority": "plantnet_no_planta",
            "contradicts_model": True,
            "reasoning": "Pl@ntNet no reconoció una planta en la imagen, por eso se invalida la predicción local.",
            "recommended_action": "Tome otra foto donde se vea claramente un árbol, preferiblemente hojas o flores.",
        }

    if comparacion.get("coinciden"):
        if comparacion.get("es_especificacion_top2"):
            especie_top2 = comparacion.get("clase_modelo_coincidente") or especie_pred
            info_top2 = _get_info(info_global, especie_top2)
            nombre_mostrar = _display_name_from_info(especie_top2, info_top2)
            score_top2 = float(comparacion.get("score_modelo", 0.0))
            return {**base,
                "decision": "aceptar_prediccion",
                "species_selected": nombre_mostrar,
                "species_selected_key": especie_top2,
                "source_priority": "modelo_y_plantnet",
                "model_prediction_common": nombre_mostrar,
                "contradicts_model": False,
                "reasoning": (
                    f"Pl@ntNet confirma la alternativa más específica del modelo: "
                    f"'{nombre_mostrar}' (top-2 con {score_top2*100:.1f}%)."
                ),
                "recommended_action": f"Registrar la observación como '{nombre_mostrar}'.",
            }

        return {**base,
            "decision": "aceptar_prediccion",
            "species_selected": nombre_comun_m,
            "source_priority": "modelo_y_plantnet",
            "contradicts_model": False,
            "reasoning": (
                f"Modelo y Pl@ntNet coinciden. Modelo: {confianza*100:.1f}%; "
                f"Pl@ntNet: {top_score_pn:.2f}."
            ),
            "recommended_action": f"Registrar la observación como '{nombre_comun_m}'.",
        }

    if confianza >= CONFIANZA_MODELO_PRIORIDAD:
        return {**base,
            "decision": "aceptar_prediccion",
            "species_selected": nombre_comun_m,
            "source_priority": "modelo",
            "contradicts_model": False,
            "reasoning": f"El modelo tiene confianza prácticamente perfecta ({confianza*100:.2f}%).",
            "recommended_action": f"Registrar como '{nombre_comun_m}', aunque Pl@ntNet sugiera otra alternativa.",
        }

    if confianza < 0.50 and top_score_pn < 0.50:
        return {**base,
            "decision": "pedir_nueva_foto",
            "species_selected": nombre_comun_m,
            "source_priority": "incertidumbre",
            "contradicts_model": False,
            "reasoning": f"Modelo ({confianza*100:.1f}%) y Pl@ntNet ({top_score_pn:.2f}) tienen baja confianza.",
            "recommended_action": "Tome una nueva foto con mejor enfoque y buena iluminación.",
        }

    if score_pn_pond > score_m_pond:
        return {**base,
            "decision": "revision_manual",
            "species_selected": top_nombre_pn or top_cient_pn or nombre_comun_m,
            "species_selected_key": None,
            "source_priority": "plantnet",
            "contradicts_model": True,
            "reasoning": (
                f"Pl@ntNet tiene mayor score ponderado ({score_pn_pond:.3f}) que el modelo "
                f"({score_m_pond:.3f})."
            ),
            "recommended_action": (
                f"Revise manualmente. Pl@ntNet sugiere '{top_nombre_pn or top_cient_pn}', "
                f"mientras el modelo sugiere '{nombre_comun_m}'."
            ),
        }

    return {**base,
        "decision": "revision_manual",
        "species_selected": nombre_comun_m,
        "source_priority": "modelo",
        "contradicts_model": True,
        "reasoning": (
            f"El modelo tiene mayor score ponderado ({score_m_pond:.3f}) que Pl@ntNet "
            f"({score_pn_pond:.3f}), pero no coinciden."
        ),
        "recommended_action": (
            f"Revise manualmente. Compare '{nombre_comun_m}' con la sugerencia de Pl@ntNet "
            f"'{top_nombre_pn or top_cient_pn}'."
        ),
    }


def _extraer_json(texto: str) -> Optional[dict]:
    if not texto:
        return None
    texto = texto.strip()
    if "```" in texto:
        partes = texto.split("```")
        for parte in partes:
            parte = parte.strip()
            if parte.startswith("json"):
                parte = parte[4:].strip()
            if parte.startswith("{") and parte.endswith("}"):
                texto = parte
                break
    try:
        return json.loads(texto)
    except Exception:
        return None


def _enriquecer_con_llm(decision: dict, estado: EstadoArbol) -> dict:
    """Opcional: mejora solo la redacción, nunca cambia la decisión."""
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not anthropic_key and not openai_key:
        return decision

    plantnet = estado.get("plantnet_resultado", {})
    pn_lista = "\n".join(
        f"  {r['rank']}. {r['nombre_cientifico']} "
        f"(comunes: {', '.join(r['nombres_comunes'][:3])}) score {r['score']:.2f}"
        for r in plantnet.get("resultados", [])[:3]
    ) or "  sin resultados"

    prompt = f"""Eres un botánico experto. Mejora la explicación para un usuario final.

NO cambies estos campos:
- decision: {decision.get('decision')}
- species_selected: {decision.get('species_selected')}
- source_priority: {decision.get('source_priority')}
- model_plantnet_match: {decision.get('model_plantnet_match')}

Datos:
- Modelo: {decision.get('model_prediction_common')} ({decision.get('model_confidence', 0)*100:.1f}%)
- Pl@ntNet top-3:
{pn_lista}
- Comparación: {decision.get('matching_reason', '')}

Responde solo JSON válido:
{{
  "reasoning": "máximo 2 oraciones claras en español",
  "recommended_action": "acción concreta para el usuario"
}}"""

    texto_raw = None
    try:
        if anthropic_key:
            import anthropic
            msg = anthropic.Anthropic(api_key=anthropic_key).messages.create(
                model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
                max_tokens=350,
                messages=[{"role": "user", "content": prompt}],
            )
            texto_raw = msg.content[0].text
        elif openai_key:
            from openai import OpenAI
            resp = OpenAI(api_key=openai_key).chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                max_tokens=350,
                messages=[{"role": "user", "content": prompt}],
            )
            texto_raw = resp.choices[0].message.content
    except Exception:
        return decision

    data = _extraer_json(texto_raw or "")
    if isinstance(data, dict):
        if data.get("reasoning"):
            decision["reasoning"] = str(data["reasoning"])
        if data.get("recommended_action"):
            decision["recommended_action"] = str(data["recommended_action"])
    return decision


# ─────────────────────────────────────────────────────────────────────────────
# Nodos del agente
# ─────────────────────────────────────────────────────────────────────────────

def prediction_validator_agent(estado: EstadoArbol) -> EstadoArbol:
    return {**estado}


def web_species_research_agent(estado: EstadoArbol) -> EstadoArbol:
    ruta_imagen = estado.get("ruta_imagen", "")
    top_k_list = estado.get("top_k_list", [])
    info_global = estado.get("info_global", {})

    plantnet = consultar_plantnet(ruta_imagen)
    comparacion = buscar_coincidencia_nombre_comun(top_k_list, plantnet, info_global)
    return {**estado, "plantnet_resultado": plantnet, "comparacion": comparacion}


def final_decision_agent(estado: EstadoArbol) -> EstadoArbol:
    decision = _reglas_decision(estado)
    decision = _enriquecer_con_llm(decision, estado)
    return {**estado, "decision_final": decision}


def construir_grafo():
    try:
        from langgraph.graph import END, StateGraph
    except Exception:
        return None

    graph = StateGraph(EstadoArbol)
    graph.add_node("validator", prediction_validator_agent)
    graph.add_node("plantnet", web_species_research_agent)
    graph.add_node("final", final_decision_agent)
    graph.set_entry_point("validator")
    graph.add_edge("validator", "plantnet")
    graph.add_edge("plantnet", "final")
    graph.add_edge("final", END)
    return graph.compile()


def ejecutar_agente(
    especie_pred: str,
    confianza: float,
    top_k_list: list,
    info_especie: dict,
    ruta_imagen: Optional[str] = None,
    info_global: Optional[dict] = None,
) -> dict:
    """Punto de entrada que llama main.py después de run_inference()."""
    estado_inicial: EstadoArbol = {
        "especie_pred": especie_pred,
        "confianza": float(confianza),
        "top_k_list": [(str(k), float(v)) for k, v in top_k_list],
        "info_especie": info_especie or {},
        "info_global": info_global or {},
        "ruta_imagen": ruta_imagen or "",
        "plantnet_resultado": {},
        "comparacion": {},
        "decision_final": {},
    }

    graph = construir_grafo()
    if graph:
        return graph.invoke(estado_inicial)

    estado = prediction_validator_agent(estado_inicial)
    estado = web_species_research_agent(estado)
    return final_decision_agent(estado)
