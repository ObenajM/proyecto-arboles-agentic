"""
app/app.py — Campus Tree Explorer
Streamlit interface with ONNX Runtime for the Arboretum and Palmetum of UNAL Medellín.
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

# Geographic centre of the UNAL Medellín campus (WGS84)
CAMPUS_CENTER = [6.2636427, -75.5764393]

# Esri World Imagery tile layer
_ESRI_SATELLITE_URL  = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
_ESRI_SATELLITE_ATTR = (
    "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, "
    "Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"
)

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
    page_title="Campus Tree Explorer | UNAL Medellín",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Design system — CSS
# =============================================================================

def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        /* ── Google Font ──────────────────────────────────────────────────── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        /* ── Design tokens ────────────────────────────────────────────────── */
        :root {
            /* Forest spectrum */
            --f-950:  #040E07;
            --f-900:  #071510;
            --f-800:  #0C1E17;
            --f-700:  #112A20;
            --f-600:  #173829;
            --f-500:  #1F4A35;
            --f-400:  #285E42;
            --f-300:  #337352;

            /* Sage spectrum */
            --s-700:  #2E6645;
            --s-600:  #3A7D55;
            --s-500:  #499468;
            --s-400:  #5DAB7D;
            --s-300:  #7DC095;
            --s-200:  #A3D4B5;
            --s-100:  #C8E8D4;
            --s-50:   #E5F4EC;

            /* Neutral green-tinted */
            --n-900:  #111E16;
            --n-700:  #2C4034;
            --n-500:  #506B5C;
            --n-400:  #6B8A79;
            --n-300:  #9CB5A8;
            --n-200:  #C5D9CE;
            --n-100:  #E2EDEA;
            --n-50:   #F2F7F4;
            --n-0:    #FAFCF9;

            /* Accent — warm terra */
            --terra:    #B3521A;
            --terra-lt: #D46828;
            --gold:     #C49230;

            /* Semantic aliases (backward compat for existing HTML classes) */
            --forest:      var(--f-600);
            --forest-mid:  var(--f-500);
            --sage:        var(--s-600);
            --sage-lt:     var(--s-400);
            --mint:        var(--s-200);
            --mint-lt:     var(--s-50);
            --cream:       var(--n-50);
            --cream-lt:    var(--n-0);
            --terra-lt:    #D46828;
            --text:        var(--n-900);
            --muted:       var(--n-500);
            --white:       #FFFFFF;

            /* Radius scale */
            --r-xs:  4px;
            --r-sm:  8px;
            --r-md:  12px;
            --r-lg:  18px;
            --r-xl:  24px;
            --r-2xl: 32px;

            /* Shadow scale (green-tinted) */
            --s-sm: 0 1px 4px rgba(7,21,16,.07), 0 2px 8px rgba(7,21,16,.05);
            --s-md: 0 2px 10px rgba(7,21,16,.09), 0 6px 24px rgba(7,21,16,.07);
            --s-lg: 0 6px 28px rgba(7,21,16,.13), 0 16px 48px rgba(7,21,16,.09);
            --s-xl: 0 12px 48px rgba(7,21,16,.18), 0 32px 80px rgba(7,21,16,.12);
        }

        /* ── Base reset ───────────────────────────────────────────────────── */
        .stApp {
            background: linear-gradient(160deg, var(--n-0) 0%, var(--s-50) 45%, var(--n-0) 100%) !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            color: var(--text);
            -webkit-font-smoothing: antialiased;
        }
        h1,h2,h3,h4,h5 {
            font-family: 'Inter', sans-serif;
            color: var(--f-600);
            letter-spacing: -.02em;
        }
        p { color: var(--n-700); line-height: 1.7; }

        /* ── Hero ─────────────────────────────────────────────────────────── */
        .hero-wrap {
            background:
                radial-gradient(ellipse 80% 60% at 90% -10%, rgba(61,125,85,.18) 0%, transparent 60%),
                radial-gradient(ellipse 50% 70% at -5% 110%, rgba(35,79,53,.22) 0%, transparent 55%),
                linear-gradient(148deg, var(--f-950) 0%, var(--f-900) 30%, var(--f-800) 60%, var(--f-700) 100%);
            border-radius: var(--r-2xl);
            padding: 3.6rem 4rem 3.2rem;
            margin-bottom: 2.5rem;
            position: relative;
            overflow: hidden;
            box-shadow: var(--s-xl), inset 0 1px 0 rgba(163,212,181,.08);
            border: 1px solid rgba(163,212,181,.06);
        }
        .hero-wrap::before {
            content: '';
            position: absolute;
            inset: 0;
            background: repeating-linear-gradient(
                -45deg,
                transparent,
                transparent 80px,
                rgba(163,212,181,.018) 80px,
                rgba(163,212,181,.018) 81px
            );
            pointer-events: none;
        }
        .hero-eyebrow {
            display: inline-flex;
            align-items: center;
            gap: .45rem;
            font-size: .65rem;
            font-weight: 700;
            letter-spacing: .2em;
            text-transform: uppercase;
            color: var(--s-300);
            margin-bottom: 1.1rem;
            background: rgba(125,192,149,.08);
            border: 1px solid rgba(125,192,149,.2);
            border-radius: 999px;
            padding: .32rem 1rem;
        }
        .hero-title {
            font-size: 3.2rem;
            font-weight: 900;
            color: #fff !important;
            line-height: 1.0;
            margin: 0 0 1rem;
            letter-spacing: -.04em;
            text-shadow: 0 2px 20px rgba(0,0,0,.25);
        }
        .hero-sub {
            font-size: 1.05rem;
            color: rgba(163,212,181,.80);
            line-height: 1.72;
            max-width: 560px;
            margin: 0;
            font-weight: 400;
        }
        .hero-stat-row {
            display: flex;
            gap: 2.8rem;
            margin-top: 2.2rem;
            padding-top: 1.9rem;
            border-top: 1px solid rgba(125,192,149,.12);
            flex-wrap: wrap;
        }
        .hero-stat { display: flex; flex-direction: column; }
        .hero-stat-val {
            font-size: 2rem;
            font-weight: 900;
            color: #fff;
            line-height: 1;
            letter-spacing: -.03em;
        }
        .hero-stat-lbl {
            font-size: .62rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: .14em;
            color: var(--s-300);
            margin-top: .35rem;
        }

        /* ── Section headers ──────────────────────────────────────────────── */
        .section-head {
            display: flex;
            align-items: flex-start;
            gap: .9rem;
            margin-bottom: 1.25rem;
            padding-bottom: 1.1rem;
            border-bottom: 1px solid var(--n-100);
        }
        .section-head-icon {
            width: 42px; height: 42px;
            background: linear-gradient(135deg, var(--f-500) 0%, var(--f-600) 100%);
            border-radius: var(--r-md);
            display: flex; align-items: center; justify-content: center;
            font-size: 1.15rem;
            flex-shrink: 0;
            box-shadow: var(--s-sm);
        }
        .section-head-title {
            font-size: 1.3rem;
            font-weight: 800;
            color: var(--f-600);
            letter-spacing: -.025em;
            margin: .1rem 0 0;
            line-height: 1.2;
        }
        .section-head-sub {
            font-size: .82rem;
            color: var(--n-500);
            margin: .3rem 0 0;
            line-height: 1.5;
        }

        /* ── Prediction — top result ──────────────────────────────────────── */
        .pred-card {
            background: var(--white);
            border: 1px solid var(--n-100);
            border-radius: var(--r-xl);
            padding: 1.8rem 2.1rem;
            margin-bottom: .9rem;
            box-shadow: var(--s-md);
            position: relative;
            overflow: hidden;
        }
        .pred-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0;
            width: 4px; height: 100%;
            background: linear-gradient(180deg, var(--terra) 0%, var(--gold) 100%);
            border-radius: 4px 0 0 4px;
        }
        .pred-card::after {
            content: '';
            position: absolute;
            top: 0; right: 0;
            width: 160px; height: 100%;
            background: radial-gradient(ellipse at 100% 50%, rgba(93,171,125,.06) 0%, transparent 70%);
            pointer-events: none;
        }
        .pred-rank-label {
            font-size: .63rem;
            font-weight: 700;
            letter-spacing: .16em;
            text-transform: uppercase;
            color: var(--terra);
            margin-bottom: .55rem;
        }
        .pred-name {
            font-size: 1.9rem;
            font-weight: 900;
            color: var(--f-600);
            line-height: 1.05;
            margin-bottom: .3rem;
            letter-spacing: -.035em;
        }
        .pred-sci {
            font-style: italic;
            font-size: .93rem;
            color: var(--s-600);
            margin-bottom: .65rem;
            font-weight: 400;
        }
        .pred-conf-row {
            display: flex;
            align-items: center;
            gap: .7rem;
            margin-top: .35rem;
        }
        .pred-conf-pill {
            font-size: .74rem;
            font-weight: 700;
            padding: .3rem .9rem;
            border-radius: 999px;
            letter-spacing: .02em;
        }
        .pill-green { background: #E2F5EA; color: #145E32; border: 1px solid #9FD8B5; }
        .pill-amber { background: #FEF3E2; color: #7A4500; border: 1px solid #EFC470; }
        .pill-red   { background: #FDEAEA; color: #8A2020; border: 1px solid #EEAAAA; }
        .pred-conf-text { font-size: .87rem; color: var(--n-500); }

        /* ── Prediction — alternatives ────────────────────────────────────── */
        .pred-alts-label {
            font-size: .67rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: .13em;
            color: var(--s-500);
            margin: 1.1rem 0 .55rem;
        }
        .pred-alt-row {
            background: var(--n-50);
            border: 1px solid var(--n-100);
            border-radius: var(--r-md);
            padding: .75rem 1.1rem;
            margin-bottom: .4rem;
            display: flex;
            align-items: center;
            gap: 1rem;
            transition: background .12s;
        }
        .pred-alt-row:hover { background: var(--s-50); }
        .pred-alt-medal { font-size: 1rem; flex-shrink: 0; }
        .pred-alt-name  { font-size: .94rem; font-weight: 600; color: var(--f-600); flex: 1; }
        .pred-alt-pct   { font-size: .9rem; font-weight: 800; color: var(--terra); white-space: nowrap; }

        /* ── Alert banners ────────────────────────────────────────────────── */
        .alert-warn {
            background: #FFF8EE;
            border: 1px solid #F5CC80;
            border-left: 4px solid var(--terra);
            border-radius: 0 var(--r-md) var(--r-md) 0;
            padding: .9rem 1.1rem;
            font-size: .87rem;
            color: #5A3200;
            margin: .65rem 0;
            line-height: 1.55;
        }
        .alert-tip {
            background: #EFF6FF;
            border: 1px solid #BFDBFE;
            border-left: 4px solid #3B82F6;
            border-radius: 0 var(--r-md) var(--r-md) 0;
            padding: .9rem 1.1rem;
            font-size: .87rem;
            color: #1E3A70;
            margin: .65rem 0;
            line-height: 1.55;
        }

        /* ── Species info card ────────────────────────────────────────────── */
        .info-card {
            background: var(--white);
            border: 1px solid var(--n-100);
            border-radius: var(--r-xl);
            padding: 2rem 2.3rem;
            box-shadow: var(--s-md);
        }
        .info-title {
            font-size: 1.65rem;
            font-weight: 800;
            color: var(--f-600);
            margin: 0 0 .3rem;
            letter-spacing: -.03em;
        }
        .info-sci {
            font-style: italic;
            font-size: .95rem;
            color: var(--s-600);
            margin-bottom: 1.1rem;
        }

        /* ── Pills ────────────────────────────────────────────────────────── */
        .pills { display: flex; flex-wrap: wrap; gap: .45rem; margin: .65rem 0 1.1rem; }
        .pill {
            background: var(--n-50);
            border: 1px solid var(--n-200);
            border-radius: 999px;
            padding: .26rem .88rem;
            font-size: .79rem;
            color: var(--n-700);
            font-weight: 500;
        }
        .pill b { color: var(--f-600); font-weight: 700; }

        /* ── Section label ────────────────────────────────────────────────── */
        .sec-lbl {
            font-size: .64rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: .14em;
            color: var(--s-500);
            margin: 1.2rem 0 .38rem;
        }

        /* ── Historia card ────────────────────────────────────────────────── */
        .historia-card {
            background: linear-gradient(135deg, var(--s-50) 0%, rgba(163,212,181,.18) 100%);
            border: 1px solid var(--s-100);
            border-radius: var(--r-lg);
            padding: 1.3rem 1.55rem;
            margin-top: 1rem;
        }
        .historia-title {
            font-size: .64rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: .14em;
            color: var(--s-600);
            margin-bottom: .5rem;
        }

        /* ── Fun-fact card ────────────────────────────────────────────────── */
        .fact-card {
            background: linear-gradient(135deg, #FFFAF2 0%, #FFF4E2 100%);
            border: 1px solid #F5CC80;
            border-left: 4px solid var(--terra);
            border-radius: 0 var(--r-md) var(--r-md) 0;
            padding: .9rem 1.15rem;
            font-size: .88rem;
            color: #3E2100;
            margin: .85rem 0;
            line-height: 1.62;
        }

        /* ── Quiz / trivia ────────────────────────────────────────────────── */
        .quiz-wrap {
            background: var(--white);
            border: 1px solid var(--n-100);
            border-radius: var(--r-xl);
            padding: 2rem 2.3rem;
            box-shadow: var(--s-md);
        }
        .quiz-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.1rem;
        }
        .quiz-counter {
            font-size: .68rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: .13em;
            color: var(--terra);
            background: #FEF5E8;
            border: 1px solid #F5CC80;
            border-radius: 999px;
            padding: .28rem .85rem;
        }
        .quiz-q {
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--f-600);
            line-height: 1.55;
            margin-bottom: 1.1rem;
        }
        .score-wrap {
            display: inline-flex;
            align-items: center;
            gap: .7rem;
            background: var(--s-50);
            border: 1px solid var(--s-100);
            border-radius: var(--r-lg);
            padding: .6rem 1.3rem;
            font-size: .92rem;
            color: var(--n-700);
            margin-top: 1.1rem;
        }
        .score-wrap b { color: var(--f-600); font-weight: 800; }

        /* ── Sidebar ──────────────────────────────────────────────────────── */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, var(--f-950) 0%, var(--f-900) 60%, var(--f-800) 100%) !important;
            border-right: 1px solid rgba(125,192,149,.06) !important;
        }
        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] .stMarkdown li,
        section[data-testid="stSidebar"] .stMarkdown span,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] .stCaption,
        section[data-testid="stSidebar"] small {
            color: rgba(163,212,181,.65) !important;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] strong {
            color: rgba(255,255,255,.92) !important;
        }
        section[data-testid="stSidebar"] hr {
            border-color: rgba(125,192,149,.10) !important;
        }
        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: .7rem;
            margin-bottom: .4rem;
        }
        .sidebar-brand-icon {
            width: 38px; height: 38px;
            background: linear-gradient(135deg, var(--s-700), var(--f-500));
            border-radius: var(--r-sm);
            display: flex; align-items: center; justify-content: center;
            font-size: 1.2rem;
            flex-shrink: 0;
            box-shadow: 0 2px 8px rgba(0,0,0,.3);
        }
        .sidebar-brand-name {
            font-size: 1.05rem;
            font-weight: 800;
            color: #fff !important;
            letter-spacing: -.02em;
            line-height: 1.2;
        }
        .sidebar-section {
            font-size: .6rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: .18em;
            color: var(--s-300) !important;
            margin: 1.3rem 0 .5rem;
        }
        .sidebar-stat {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: .38rem 0;
            border-bottom: 1px solid rgba(125,192,149,.07);
        }
        .sidebar-stat-key { font-size: .81rem; color: rgba(163,212,181,.52) !important; }
        .sidebar-stat-val { font-size: .81rem; font-weight: 700; color: rgba(255,255,255,.88) !important; }
        .sidebar-sync-off {
            background: rgba(125,192,149,.06);
            border: 1px solid rgba(125,192,149,.12);
            border-radius: var(--r-md);
            padding: .55rem .9rem;
            font-size: .77rem;
            color: rgba(163,212,181,.52) !important;
            margin-top: .5rem;
            line-height: 1.5;
        }
        .sidebar-sync-on {
            background: rgba(61,125,85,.18);
            border: 1px solid rgba(93,171,125,.28);
            border-radius: var(--r-md);
            padding: .55rem .9rem;
            font-size: .77rem;
            color: var(--s-300) !important;
            margin-top: .5rem;
            line-height: 1.5;
        }

        /* ── Tabs ─────────────────────────────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {
            background: transparent;
            gap: .25rem;
            border-bottom: 2px solid var(--n-100);
            padding-bottom: 0;
            margin-bottom: 1.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            font-weight: 600;
            font-size: .87rem;
            color: var(--n-500);
            padding: .72rem 1.3rem;
            background: transparent;
            border: none;
            border-radius: var(--r-md) var(--r-md) 0 0;
            transition: color .15s, background .15s;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: var(--f-600);
            background: var(--n-50);
        }
        .stTabs [aria-selected="true"] {
            color: var(--f-600) !important;
            font-weight: 800 !important;
            background: var(--white) !important;
            border: 2px solid var(--n-100) !important;
            border-bottom: 2px solid var(--white) !important;
            margin-bottom: -2px;
        }

        /* ── Buttons ──────────────────────────────────────────────────────── */
        div.stButton > button {
            border-radius: var(--r-md);
            background: linear-gradient(135deg, var(--f-400) 0%, var(--f-500) 100%);
            border: 1px solid rgba(255,255,255,.08);
            color: #fff;
            font-weight: 700;
            font-size: .86rem;
            padding: .58rem 1.55rem;
            letter-spacing: .015em;
            transition: all .18s ease;
            box-shadow: 0 2px 6px rgba(7,21,16,.25), 0 1px 2px rgba(7,21,16,.12),
                        inset 0 1px 0 rgba(255,255,255,.09);
        }
        div.stButton > button:hover {
            background: linear-gradient(135deg, var(--f-300) 0%, var(--f-400) 100%) !important;
            color: #fff !important;
            transform: translateY(-1.5px);
            box-shadow: 0 6px 20px rgba(7,21,16,.30), 0 2px 6px rgba(7,21,16,.15),
                        inset 0 1px 0 rgba(255,255,255,.12) !important;
        }
        div.stButton > button:active {
            transform: translateY(0) !important;
            box-shadow: 0 2px 6px rgba(7,21,16,.20) !important;
        }

        /* ── Progress bar ─────────────────────────────────────────────────── */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, var(--terra) 0%, var(--gold) 100%);
            border-radius: 999px;
        }

        /* ── Upload zone ──────────────────────────────────────────────────── */
        div[data-testid="stFileUploader"] {
            border: 2px dashed var(--s-200) !important;
            border-radius: var(--r-xl) !important;
            background: var(--s-50) !important;
            transition: border-color .15s, background .15s;
        }
        div[data-testid="stFileUploader"]:hover {
            border-color: var(--s-400) !important;
            background: rgba(163,212,181,.15) !important;
        }

        /* ── Map container ────────────────────────────────────────────────── */
        .map-wrap {
            background: var(--white);
            border: 1px solid var(--n-100);
            border-radius: var(--r-xl);
            padding: 1.1rem;
            box-shadow: var(--s-md);
            overflow: hidden;
        }

        /* ── Empty state ──────────────────────────────────────────────────── */
        .empty-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 4.5rem 2rem;
            text-align: center;
        }
        .empty-state-icon {
            font-size: 3.2rem;
            margin-bottom: .9rem;
            opacity: .45;
            filter: saturate(0.6);
        }
        .empty-state-text {
            font-size: .95rem;
            line-height: 1.65;
            max-width: 300px;
            color: var(--n-500);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_custom_css()

# =============================================================================
# Cached loaders
# =============================================================================


@st.cache_data(show_spinner=False)
def load_mapped_trees() -> list[dict]:
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
    _ = mtime
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
species_info = load_species_info(str(INFO_PATH), _info_mtime)
ort_sess, inp_name, out_name = load_model()

# =============================================================================
# Core helpers
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


def build_campus_map(
    tree_points: list[dict],
    style: str = "satellite",
    gps_point: tuple[float, float] | None = None,
    click_point: tuple[float, float] | None = None,
) -> folium.Map:
    """Build a folium map with the chosen tile style and all markers."""
    if style == "satellite":
        m = folium.Map(
            location=CAMPUS_CENTER,
            zoom_start=18,
            tiles=_ESRI_SATELLITE_URL,
            attr=_ESRI_SATELLITE_ATTR,
            max_zoom=20,
        )
    else:
        m = folium.Map(location=CAMPUS_CENTER, zoom_start=18, tiles="CartoDB positron")

    # Saved tree markers
    for pt in tree_points:
        conf_str = f"{float(pt.get('confidence', 0)):.0%}" if pt.get("confidence") else "—"
        dt_str   = pt.get("datetime", "")
        src_str  = pt.get("source", "")
        popup_html = (
            f"<div style='font-family:sans-serif;min-width:160px'>"
            f"<b style='font-size:1rem'>{pt['name']}</b><br>"
            f"<i style='color:#555'>{pt.get('sci','')}</i><br>"
            f"<hr style='margin:.4rem 0'>"
            f"<table style='font-size:.82rem;border-collapse:collapse'>"
            f"<tr><td style='color:#777;padding-right:.5rem'>Confianza</td><td><b>{conf_str}</b></td></tr>"
            f"<tr><td style='color:#777'>Fecha</td><td>{dt_str}</td></tr>"
            f"<tr><td style='color:#777'>Fuente</td><td>{src_str}</td></tr>"
            f"<tr><td style='color:#777'>Lat</td><td>{pt['lat']:.6f}</td></tr>"
            f"<tr><td style='color:#777'>Lon</td><td>{pt['lon']:.6f}</td></tr>"
            f"</table></div>"
        )
        folium.Marker(
            location=[pt["lat"], pt["lon"]],
            popup=folium.Popup(popup_html, max_width=240),
            tooltip=pt["name"],
            icon=folium.Icon(color="green", icon="tree", prefix="fa"),
        ).add_to(m)

    # GPS location marker (blue pulsing circle)
    if gps_point:
        folium.CircleMarker(
            location=gps_point,
            radius=10,
            color="#1976D2",
            fill=True,
            fill_color="#42A5F5",
            fill_opacity=0.75,
            tooltip="Tu ubicación GPS",
            popup=folium.Popup(
                f"<b>Ubicación GPS</b><br>{gps_point[0]:.6f}, {gps_point[1]:.6f}",
                max_width=180,
            ),
        ).add_to(m)

    # Manual click marker (orange pin)
    if click_point:
        folium.Marker(
            location=click_point,
            tooltip="Punto seleccionado manualmente",
            popup=folium.Popup(
                f"<b>Punto seleccionado</b><br>{click_point[0]:.6f}, {click_point[1]:.6f}",
                max_width=180,
            ),
            icon=folium.Icon(color="orange", icon="map-marker", prefix="fa"),
        ).add_to(m)

    return m


def _ensure_csv_headers() -> None:
    if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=CSV_COLUMNS).writeheader()


def _get_github_config() -> dict | None:
    token    = None
    repo     = GITHUB_REPO
    branch   = "main"
    csv_path = GITHUB_CSV_PATH

    try:
        gh       = st.secrets["github"]
        token    = gh.get("token")
        repo     = gh.get("repo",     repo)
        branch   = gh.get("branch",   branch)
        csv_path = gh.get("csv_path", csv_path)
    except (KeyError, FileNotFoundError, AttributeError):
        pass

    if not token:
        try:
            token = st.secrets["GITHUB_TOKEN"]
        except (KeyError, FileNotFoundError, AttributeError):
            pass
    try:
        repo     = st.secrets.get("GITHUB_REPO",     repo)
        branch   = st.secrets.get("GITHUB_BRANCH",   branch)
        csv_path = st.secrets.get("GITHUB_CSV_PATH", csv_path)
    except (AttributeError, FileNotFoundError):
        pass

    token    = token    or os.environ.get("GITHUB_TOKEN")
    repo     = os.environ.get("GITHUB_REPO",     repo)
    branch   = os.environ.get("GITHUB_BRANCH",   branch)
    csv_path = os.environ.get("GITHUB_CSV_PATH", csv_path)

    if not token:
        return None
    return {"token": token, "repo": repo, "branch": branch, "csv_path": csv_path}


def _push_csv_to_github() -> tuple[bool, str]:
    cfg = _get_github_config()
    if not cfg:
        return False, "token_missing"

    api_url = f"https://api.github.com/repos/{cfg['repo']}/contents/{cfg['csv_path']}"
    headers = {
        "Authorization": f"Bearer {cfg['token']}",
        "Accept":        "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

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

    with open(CSV_PATH, "rb") as fh:
        content_b64 = base64.b64encode(fh.read()).decode()

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
            return True, "CSV sincronizado con GitHub correctamente."
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        return False, f"Error al actualizar GitHub: HTTP {exc.code} — {detail}"
    except OSError as exc:
        return False, f"Error de red al actualizar GitHub: {exc}"


def save_tree_to_csv(row: dict) -> None:
    _ensure_csv_headers()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore").writerow(row)
    load_mapped_trees.clear()


def _init_map_points() -> None:
    if "map_initialized" not in st.session_state:
        st.session_state["map_points"]      = load_mapped_trees()
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
# UI helpers
# =============================================================================


def _esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _conf_pill(conf: float) -> str:
    if conf >= 0.75:
        cls, label = "pill-green", "Alta"
    elif conf >= 0.50:
        cls, label = "pill-amber", "Moderada"
    else:
        cls, label = "pill-red", "Baja"
    return f'<span class="pred-conf-pill {cls}">{label} {conf:.0%}</span>'


def render_predictions(results: list[dict]) -> None:
    if not results:
        st.warning("No se obtuvieron predicciones.")
        return

    top  = results[0]
    conf = top["prob"]
    sci  = get_info(top["key"]).get("nombre_cientifico", "")

    sci_html = f'<div class="pred-sci">{_esc(sci)}</div>' if sci else ""

    st.markdown(
        f"""
        <div class="pred-card">
            <div class="pred-rank-label">🌿 Especie más probable</div>
            <div class="pred-name">{_esc(top['name'])}</div>
            {sci_html}
            <div class="pred-conf-row">
                {_conf_pill(conf)}
                <span class="pred-conf-text">Confianza: <strong>{conf:.1%}</strong></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(conf)

    if conf < 0.50:
        st.markdown(
            '<div class="alert-warn">⚠️ Confianza baja — intenta con una foto más nítida '
            "donde se aprecien bien las hojas, flores, frutos o la silueta completa.</div>",
            unsafe_allow_html=True,
        )
    elif conf < 0.75:
        st.markdown(
            '<div class="alert-tip">💡 Confianza moderada — verifica los rasgos botánicos '
            "antes de concluir la identificación.</div>",
            unsafe_allow_html=True,
        )

    if len(results) > 1:
        st.markdown('<div class="pred-alts-label">Otras posibilidades</div>', unsafe_allow_html=True)
        medals = ["🥈", "🥉"]
        rows_html: list[str] = []
        for i, item in enumerate(results[1:]):
            medal = medals[i] if i < len(medals) else f"#{i + 2}"
            rows_html.append(
                f'<div class="pred-alt-row">'
                f'<span class="pred-alt-medal">{medal}</span>'
                f'<span class="pred-alt-name">{_esc(item["name"])}</span>'
                f'<span class="pred-alt-pct">{item["prob"]:.1%}</span>'
                f'</div>'
            )
        st.markdown("".join(rows_html), unsafe_allow_html=True)


def render_species_card(key: str) -> None:
    info = get_info(key)

    if not info:
        st.markdown(
            f'<div class="empty-state">'
            f'<div class="empty-state-icon">🔍</div>'
            f'<div class="empty-state-text">No se encontró información adicional para '
            f'<strong>{_esc(clean_name(key))}</strong> en el catálogo botánico.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    common_name = info.get("nombre_comun") or clean_name(key)
    scientific  = info.get("nombre_cientifico", "")

    p: list[str] = ['<div class="info-card">']

    # Header
    p.append(f'<div class="info-title">🌿 {_esc(common_name)}</div>')
    if scientific:
        p.append(f'<div class="info-sci">{_esc(scientific)}</div>')

    # Quick-fact pills
    pills: list[str] = []
    if info.get("familia"):
        pills.append(f"<span class='pill'>🏷 Familia: <b>{_esc(info['familia'])}</b></span>")
    if info.get("altura_aproximada"):
        pills.append(f"<span class='pill'>📏 Altura: <b>{_esc(info['altura_aproximada'])}</b></span>")
    if pills:
        p.append(f"<div class='pills'>{''.join(pills)}</div>")

    # Narrative sections
    for field, icon_label in [
        ("descripcion",        "📖 Descripción"),
        ("como_identificarlo", "🔎 Cómo identificarlo"),
    ]:
        val = info.get(field)
        if val:
            p.append(
                f'<div class="sec-lbl">{icon_label}</div>'
                f'<p style="margin:.2rem 0 .8rem;line-height:1.6;color:var(--text)">'
                f'{_esc(val)}</p>'
            )

    # Two-column morphological details
    for (f1, l1), (f2, l2) in [
        (("hojas",  "🍃 Hojas"),  ("flores",      "🌸 Flores")),
        (("frutos", "🍑 Frutos"), ("distribucion", "🌍 Distribución")),
    ]:
        v1, v2 = info.get(f1), info.get(f2)
        if v1 or v2:
            p.append('<div style="display:flex;gap:1.5rem;flex-wrap:wrap;margin:.2rem 0 .5rem">')
            for v, lbl in ((v1, l1), (v2, l2)):
                if v:
                    p.append(
                        f'<div style="flex:1;min-width:180px">'
                        f'<div class="sec-lbl">{lbl}</div>'
                        f'<p style="margin:.1rem 0;line-height:1.55;color:var(--text)">{_esc(v)}</p>'
                        f'</div>'
                    )
            p.append('</div>')

    # Uses
    if info.get("usos"):
        p.append(
            f'<div class="sec-lbl">🛠 Usos</div>'
            f'<p style="margin:.2rem 0 .8rem;line-height:1.6;color:var(--text)">{_esc(info["usos"])}</p>'
        )

    # Fun fact
    if info.get("dato_curioso"):
        p.append(
            f'<div class="fact-card">'
            f'💡 <strong>Dato curioso:</strong> {_esc(info["dato_curioso"])}'
            f'</div>'
        )

    # Historia / origin highlighted card
    historia = info.get("historia_origen_colombia_usos")
    p.append('<div class="historia-card">')
    p.append('<div class="historia-title">🌎 Origen, historia en Colombia y usos</div>')
    if historia:
        p.append(
            f'<p style="margin:0;line-height:1.6;color:var(--forest-mid);font-size:.92rem">'
            f'{_esc(historia)}</p>'
        )
    else:
        p.append(
            '<p style="margin:0;font-style:italic;color:var(--muted);font-size:.9rem">'
            'Información sobre el origen, historia en Colombia y usos aún no disponible para esta especie.'
            '</p>'
        )
    p.append('</div>')

    p.append('</div>')
    st.markdown('\n'.join(p), unsafe_allow_html=True)


# =============================================================================
# Trivia engine  (logic unchanged, CSS classes updated)
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
        return {"question": template.format(n=nombre), "options": options, "answer": answer}
    return None


MAX_TRIVIA_QUESTIONS = 5


def render_trivia(species_key: str) -> None:
    QK   = "tv_q";   SPK  = "tv_sp";  STK  = "tv_st"
    SELK = "tv_sel"; SCRK = "tv_score"; TOTK = "tv_total"; CTRK = "tv_ctr"

    for key, default in [(SCRK, 0), (TOTK, 0), (CTRK, 0), (STK, "ask")]:
        if key not in st.session_state:
            st.session_state[key] = default

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

    # Round complete
    if st.session_state[STK] == "done" or total >= MAX_TRIVIA_QUESTIONS:
        pct = score / total if total else 0
        sp_label = get_info(species_key).get("nombre_comun") or clean_name(species_key)
        st.markdown(
            f'<div class="quiz-wrap">'
            f'<div class="quiz-q">🎉 ¡Ronda completada!</div>'
            f'<p style="color:var(--muted);margin:0">Respondiste <strong>{total}</strong> '
            f'preguntas sobre <strong>{_esc(sp_label)}</strong>.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="score-wrap">🏆 Resultado: <b>{score}/{total}</b> · {pct:.0%}</div>',
            unsafe_allow_html=True,
        )
        if st.button("Jugar de nuevo", key="tv_reset"):
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
        st.info("No hay suficientes datos para generar preguntas de esta especie.")
        return

    ctr   = st.session_state[CTRK]
    q_num = total + 1 if st.session_state[STK] == "ask" else total

    st.markdown(
        f'<div class="quiz-wrap">'
        f'<div class="quiz-header">'
        f'<span class="quiz-counter">Pregunta {q_num} / {MAX_TRIVIA_QUESTIONS}</span>'
        f'</div>'
        f'<div class="quiz-q">{_esc(q["question"])}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if st.session_state[STK] == "ask":
        choice = st.radio(
            "Elige tu respuesta:",
            q["options"],
            key=f"tv_radio_{ctr}",
            label_visibility="collapsed",
        )
        if st.button("Confirmar respuesta", key=f"tv_confirm_{ctr}"):
            st.session_state[SELK] = choice
            st.session_state[TOTK] += 1
            if choice == q["answer"]:
                st.session_state[SCRK] += 1
            st.session_state[STK] = "fb"
            st.rerun()

    else:
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
            st.success("**¡Correcto!** 🎉")
        else:
            st.error(f"**Incorrecto.** La respuesta era: **{q['answer']}**")

        answered = st.session_state[TOTK]
        if answered >= MAX_TRIVIA_QUESTIONS:
            if st.button("Ver resultados", key=f"tv_finish_{ctr}"):
                st.session_state[STK] = "done"
                st.rerun()
        else:
            if st.button("Siguiente pregunta →", key=f"tv_next_{ctr}"):
                st.session_state[QK]   = build_question(species_key)
                st.session_state[STK]  = "ask"
                st.session_state[SELK] = None
                st.session_state[CTRK] += 1
                st.rerun()

    cur_score = st.session_state[SCRK]
    cur_total = st.session_state[TOTK]
    if cur_total > 0:
        pct = cur_score / cur_total
        st.markdown(
            f'<div class="score-wrap">🏆 Puntaje: <b>{cur_score}/{cur_total}</b>'
            f' &nbsp;·&nbsp; {pct:.0%}'
            f' &nbsp;·&nbsp; Pregunta {cur_total}/{MAX_TRIVIA_QUESTIONS}</div>',
            unsafe_allow_html=True,
        )


# =============================================================================
# Sidebar
# =============================================================================

_github_active = _get_github_config() is not None
_sync_cls      = "sidebar-sync-on" if _github_active else "sidebar-sync-off"
_sync_icon     = "☁️" if _github_active else "💾"
_sync_text     = "Sincronización con GitHub activa" if _github_active else "Guardado local · sin sync con GitHub"

with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">'
        '<span class="sidebar-brand-icon">🌿</span>'
        '<span class="sidebar-brand-name">Campus Tree Explorer</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size:.82rem;color:rgba(212,222,201,.65);margin:.15rem 0 0">'
        'Arboretum & Palmetum · UNAL Medellín</p>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown('<div class="sidebar-section">Modelo</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sidebar-stat">'
        f'<span class="sidebar-stat-key">Especies</span>'
        f'<span class="sidebar-stat-val">{len(class_names)}</span>'
        f'</div>'
        f'<div class="sidebar-stat">'
        f'<span class="sidebar-stat-key">Formato</span>'
        f'<span class="sidebar-stat-val">ONNX · CPU</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if species_info:
        st.markdown('<div class="sidebar-section">Algunas especies</div>', unsafe_allow_html=True)
        for k in class_names[:14]:
            label = get_info(k).get("nombre_comun") or clean_name(k)
            st.markdown(
                f'<p style="font-size:.8rem;color:rgba(212,222,201,.7);margin:.15rem 0">· {_esc(label)}</p>',
                unsafe_allow_html=True,
            )
        if len(class_names) > 14:
            st.markdown(
                f'<p style="font-size:.78rem;color:rgba(173,188,159,.55);margin:.3rem 0">'
                f'… y {len(class_names) - 14} más — ver en Catálogo</p>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown('<div class="sidebar-section">Repositorio</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="{_sync_cls}">{_sync_icon} {_sync_text}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        '<p style="font-size:.77rem;color:rgba(173,188,159,.5);line-height:1.5;margin:0">'
        'Este clasificador es una herramienta de apoyo. '
        'Para identificaciones definitivas, consulta un especialista en botánica.</p>',
        unsafe_allow_html=True,
    )


# =============================================================================
# Hero banner
# =============================================================================

_tree_count = len(st.session_state.get("map_points", load_mapped_trees()))

st.markdown(
    f"""
    <div class="hero-wrap">
        <div class="hero-eyebrow">
            <span style="opacity:.7">🌿</span>
            Arboretum &amp; Palmetum &nbsp;·&nbsp; UNAL Medellín
        </div>
        <h1 class="hero-title">Campus Tree<br>Explorer</h1>
        <p class="hero-sub">
            Identifica, aprende y mapea las especies arbóreas del campus
            con inteligencia artificial. Carga una foto y el modelo
            reconocerá la especie en segundos.
        </p>
        <div class="hero-stat-row">
            <div class="hero-stat">
                <span class="hero-stat-val">{len(class_names)}</span>
                <span class="hero-stat-lbl">Especies</span>
            </div>
            <div style="width:1px;background:rgba(125,192,149,.15);align-self:stretch"></div>
            <div class="hero-stat">
                <span class="hero-stat-val">{_tree_count}</span>
                <span class="hero-stat-lbl">Árboles mapeados</span>
            </div>
            <div style="width:1px;background:rgba(125,192,149,.15);align-self:stretch"></div>
            <div class="hero-stat">
                <span class="hero-stat-val">ConvNeXt</span>
                <span class="hero-stat-lbl">Modelo · ONNX CPU</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Tabs
# =============================================================================

tab_classify, tab_catalog, tab_trivia, tab_map = st.tabs(
    ["📷  Identificar", "📚  Catálogo", "🧠  Trivia", "🗺  Mapa del campus"]
)

# ── Tab 1: Classify ───────────────────────────────────────────────────────────

with tab_classify:
    st.markdown(
        '<div class="section-head">'
        '<div class="section-head-icon">📷</div>'
        '<div><div class="section-head-title">Identificar especie</div>'
        '<div class="section-head-sub">Sube una foto o usa la cámara — el modelo detectará la especie</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Fuente:",
        ["📁 Subir imagen", "📷 Usar cámara"],
        horizontal=True,
        label_visibility="collapsed",
    )

    uploaded_file = None
    if mode == "📁 Subir imagen":
        uploaded_file = st.file_uploader(
            "Selecciona una imagen (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
    else:
        uploaded_file = st.camera_input("Captura una foto", label_visibility="collapsed")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col_img, col_res = st.columns([1, 1.4], gap="large")

        with col_img:
            st.image(image, caption="Imagen cargada", use_container_width=True)

        with col_res:
            with st.spinner("Analizando imagen…"):
                results = run_inference(image, top_k=min(TOP_K, len(class_names)))
            st.session_state["detected_species"]    = results[0]["key"]
            st.session_state["detected_confidence"] = results[0]["prob"]
            render_predictions(results)

        st.markdown("---")
        st.markdown(
            '<div class="section-head">'
            '<div class="section-head-icon">🌿</div>'
            '<div><div class="section-head-title">Información de la especie</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )
        render_species_card(results[0]["key"])

    else:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-state-icon">🌿</div>'
            '<div class="empty-state-text">Carga una fotografía o usa la cámara para identificar '
            'una especie arbórea del campus.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

# ── Tab 2: Catalog ────────────────────────────────────────────────────────────

with tab_catalog:
    st.markdown(
        '<div class="section-head">'
        '<div class="section-head-icon">📚</div>'
        '<div><div class="section-head-title">Catálogo de especies</div>'
        '<div class="section-head-sub">Consulta la ficha botánica completa de cualquier especie del modelo</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    if not species_info:
        st.info("El catálogo botánico no está disponible (`data/info.json` no encontrado).")
    else:
        selected = st.selectbox(
            "Especie:",
            class_names,
            format_func=lambda k: (get_info(k).get("nombre_comun") or clean_name(k))
                                   + f"  —  {clean_name(k)}",
            label_visibility="collapsed",
        )
        render_species_card(selected)

# ── Tab 3: Trivia ─────────────────────────────────────────────────────────────

with tab_trivia:
    st.markdown(
        '<div class="section-head">'
        '<div class="section-head-icon">🧠</div>'
        '<div><div class="section-head-title">Trivia botánica</div>'
        '<div class="section-head-sub">Pon a prueba tu conocimiento · 5 preguntas por ronda</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    if not species_info:
        st.info("La trivia no está disponible sin información botánica (`data/info.json`).")
    else:
        valid_keys = [k for k in class_names if get_info(k)]

        if not valid_keys:
            st.info("No hay especies con información suficiente para la trivia.")
        else:
            detected  = st.session_state.get("detected_species")
            last_auto = st.session_state.get("trivia_last_auto")
            if detected and detected in valid_keys and detected != last_auto:
                st.session_state["trivia_selector"] = detected
                st.session_state["trivia_last_auto"] = detected

            col_sel, col_det = st.columns([2, 1], gap="small")
            with col_sel:
                trivia_key = st.selectbox(
                    "Especie:",
                    valid_keys,
                    format_func=lambda k: get_info(k).get("nombre_comun") or clean_name(k),
                    key="trivia_selector",
                    label_visibility="collapsed",
                )
            with col_det:
                if detected and detected in valid_keys:
                    det_name = get_info(detected).get("nombre_comun") or clean_name(detected)
                    st.markdown(
                        f'<div style="padding:.45rem .8rem;background:var(--cream);'
                        f'border:1px solid var(--mint);border-radius:var(--r-sm);'
                        f'font-size:.8rem;color:var(--muted);margin-top:.15rem">'
                        f'🔍 Detectada: <strong style="color:var(--forest)">{_esc(det_name)}</strong></div>',
                        unsafe_allow_html=True,
                    )

            st.markdown('<div style="margin-top:1rem"></div>', unsafe_allow_html=True)
            render_trivia(trivia_key)

# ── Tab 4: Campus map ─────────────────────────────────────────────────────────

with tab_map:
    _init_map_points()

    st.markdown(
        '<div class="section-head">'
        '<div class="section-head-icon">🗺</div>'
        '<div><div class="section-head-title">Mapa del campus</div>'
        '<div class="section-head-sub">'
        'Activa tu GPS, identifica un árbol y márcalo en el mapa — o haz clic para fijar un punto manualmente'
        '</div></div></div>',
        unsafe_allow_html=True,
    )

    # ── Map style selector ─────────────────────────────────────────────────────
    _style_label = st.radio(
        "Estilo de mapa:",
        ["🛰 Vista satelital", "🗺 Mapa de calles"],
        horizontal=True,
        label_visibility="collapsed",
    )
    _map_style = "satellite" if _style_label.startswith("🛰") else "streets"

    # ── GPS widget ─────────────────────────────────────────────────────────────
    location = streamlit_geolocation()

    col_ctrl, col_map_view = st.columns([1, 2.4], gap="large")

    with col_ctrl:
        # Flash message from previous save
        if flash := st.session_state.pop("_map_flash", None):
            ftype, ftext = flash
            if ftype == "success":
                st.success(ftext)
            elif ftype == "warning":
                st.warning(ftext)
            else:
                st.info(ftext)

        # ── GPS location card ──────────────────────────────────────────────────
        lat = lon = acc = None
        if location and location.get("latitude") is not None:
            lat = float(location["latitude"])
            lon = float(location["longitude"])
            acc = location.get("accuracy")
            acc_str = f" · ±{acc:.0f} m" if acc is not None else ""
            st.markdown(
                f'<div style="background:var(--cream);border:1.5px solid var(--mint);'
                f'border-radius:var(--r-md);padding:.8rem 1rem;margin-bottom:.75rem">'
                f'<div style="font-size:.68rem;font-weight:700;text-transform:uppercase;'
                f'letter-spacing:.1em;color:var(--sage);margin-bottom:.3rem">📍 Ubicación GPS</div>'
                f'<div style="font-size:.92rem;color:var(--forest);font-weight:600">'
                f'{lat:.6f}, {lon:.6f}</div>'
                f'<div style="font-size:.78rem;color:var(--muted)">{acc_str}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="empty-state" style="padding:1.5rem 1rem">'
                '<div class="empty-state-icon" style="font-size:2rem">📍</div>'
                '<div class="empty-state-text" style="font-size:.85rem">'
                'Pulsa <strong>Get Location</strong> para activar el GPS.</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        # ── Manual click location card ─────────────────────────────────────────
        click_pt = st.session_state.get("map_click_point")
        if click_pt:
            st.markdown(
                f'<div style="background:var(--cream);border:1.5px solid var(--terra-lt);'
                f'border-radius:var(--r-md);padding:.8rem 1rem;margin-bottom:.75rem">'
                f'<div style="font-size:.68rem;font-weight:700;text-transform:uppercase;'
                f'letter-spacing:.1em;color:var(--terra);margin-bottom:.3rem">🖱 Punto seleccionado</div>'
                f'<div style="font-size:.92rem;color:var(--forest);font-weight:600">'
                f'{click_pt[0]:.6f}, {click_pt[1]:.6f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button("✕ Borrar punto seleccionado", key="map_clear_click"):
                st.session_state.pop("map_click_point", None)
                st.rerun()

        # ── Save button — GPS or manual click ─────────────────────────────────
        detected = st.session_state.get("detected_species")
        save_lat = save_lon = save_source = None

        if lat is not None:
            save_lat, save_lon, save_source = lat, lon, "gps"
        elif click_pt:
            save_lat, save_lon, save_source = click_pt[0], click_pt[1], "manual_map_click"

        if detected and save_lat is not None:
            sp_name = get_info(detected).get("nombre_comun") or clean_name(detected)
            sp_sci  = get_info(detected).get("nombre_cientifico", "")
            conf    = float(st.session_state.get("detected_confidence", 0.0))
            src_icon = "📍" if save_source == "gps" else "🖱"
            st.markdown(
                f'<div style="font-size:.88rem;color:var(--muted);margin-bottom:.5rem">'
                f'Especie: <strong style="color:var(--forest)">{_esc(sp_name)}</strong>'
                f' &nbsp;·&nbsp; <span style="color:var(--terra)">{conf:.0%}</span></div>',
                unsafe_allow_html=True,
            )
            if st.button(f"💾 Guardar {src_icon} en el mapa", key="map_save"):
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                csv_row = {
                    "species_key":  detected,
                    "common_name":  sp_name,
                    "confidence":   f"{conf:.4f}",
                    "latitude":     f"{save_lat:.7f}",
                    "longitude":    f"{save_lon:.7f}",
                    "gps_accuracy": f"{acc:.1f}" if (acc is not None and save_source == "gps") else "",
                    "datetime":     now_str,
                    "source":       save_source,
                }
                save_tree_to_csv(csv_row)
                st.session_state["map_points"].append({
                    "species_key": detected,
                    "name":        sp_name,
                    "sci":         sp_sci,
                    "lat":         save_lat,
                    "lon":         save_lon,
                    "confidence":  conf,
                    "datetime":    now_str,
                    "source":      save_source,
                })
                if save_source == "manual_map_click":
                    st.session_state.pop("map_click_point", None)
                ok, msg = _push_csv_to_github()
                if ok:
                    flash_text, flash_type = f"✅ {sp_name} guardado y sincronizado.", "success"
                elif msg == "token_missing":
                    flash_text, flash_type = f"✅ {sp_name} guardado localmente.", "success"
                else:
                    flash_text, flash_type = f"✅ {sp_name} guardado. (GitHub: {msg})", "warning"
                st.session_state["_map_flash"] = (flash_type, flash_text)
                st.rerun()
        elif not detected and save_lat is not None:
            st.markdown(
                '<div class="alert-tip">Identifica una especie en '
                '<strong>Identificar</strong> y luego guárdala aquí.</div>',
                unsafe_allow_html=True,
            )

        # ── Saved points list ──────────────────────────────────────────────────
        saved_pts: list[dict] = st.session_state.get("map_points", [])
        if saved_pts:
            st.markdown("---")
            st.markdown(
                f'<div style="font-size:.75rem;font-weight:700;text-transform:uppercase;'
                f'letter-spacing:.1em;color:var(--sage-lt);margin-bottom:.5rem">'
                f'Árboles guardados ({len(saved_pts)})</div>',
                unsafe_allow_html=True,
            )
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
                    if st.button("✕", key=f"map_del_{i}", help="Eliminar de la vista"):
                        st.session_state["map_points"].pop(i)
                        st.rerun()

            st.markdown("---")
            if st.button("☁️ Sincronizar con GitHub", key="map_sync"):
                if not _get_github_config():
                    st.info(
                        "Requiere GITHUB_TOKEN. Copia `.streamlit/secrets.toml.example` "
                        "→ `.streamlit/secrets.toml` y añade tu token."
                    )
                else:
                    ok, msg = _push_csv_to_github()
                    st.success(msg) if ok else st.error(msg)

    # ── Map view ───────────────────────────────────────────────────────────────
    with col_map_view:
        st.markdown('<div class="map-wrap">', unsafe_allow_html=True)
        gps_pt  = (lat, lon) if lat is not None else None
        map_obj = build_campus_map(
            tree_points=st.session_state.get("map_points", []),
            style=_map_style,
            gps_point=gps_pt,
            click_point=st.session_state.get("map_click_point"),
        )
        map_data = st_folium(
            map_obj,
            use_container_width=True,
            height=560,
            returned_objects=["last_clicked"],
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Persist map click into session state (ignore clicks on existing markers)
        last_clicked = map_data.get("last_clicked") if map_data else None
        if last_clicked and isinstance(last_clicked, dict):
            clat = last_clicked.get("lat")
            clng = last_clicked.get("lng")
            if clat is not None and clng is not None:
                st.session_state["map_click_point"] = (float(clat), float(clng))
                st.rerun()
