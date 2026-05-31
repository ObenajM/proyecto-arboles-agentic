/* app.js — Campus Tree Explorer PWA frontend */

// =============================================================================
// State
// =============================================================================

const state = {
  section: "home",
  detectedSpecies: null,   // especie final seleccionada para UI/mapa
  modelTopSpecies: null,    // top-1 original del modelo local
  agentDecision: null,
  plantnetResult: null,
  comparison: null,
  mapPoints: [],
  gpsCoords: null,
  clickCoords: null,
  leafletMap: null,
  miniMap: null,
  triviaScore: 0,
  triviaTotal: 0,
  triviaAnswer: null,
  triviaKey: null,
  mapStyle: "satellite",
  classes: [],
  installPrompt: null,
};

// =============================================================================
// Navigation
// =============================================================================

function navigate(id, pushHistory = true) {
  state.section = id;

  // Hero
  const hero = document.getElementById("hero");
  const feats = document.getElementById("feature-cards");
  const isHome = id === "home";
  hero.style.display  = isHome ? "" : "none";
  feats.style.display = isHome ? "" : "none";

  // Sections
  document.querySelectorAll(".section").forEach(s => {
    s.classList.toggle("active", s.id === `sec-${id}`);
  });

  // Nav links
  document.querySelectorAll(".nav-link").forEach(l => {
    l.classList.toggle("active", l.dataset.nav === id);
  });

  if (pushHistory) history.pushState({ section: id }, "", id === "home" ? "/" : `#${id}`);

  // Lazy init
  if (id === "map")     initMap();
  if (id === "trivia")  initTrivia();
  if (id === "catalog") initCatalog();
}

window.addEventListener("popstate", e => {
  navigate(e.state?.section || "home", false);
});

// =============================================================================
// Toast
// =============================================================================

function toast(msg, type = "success", ms = 3500) {
  const c = document.getElementById("toast-container");
  const t = document.createElement("div");
  t.className = `toast ${type}`;
  t.textContent = msg;
  c.appendChild(t);
  setTimeout(() => t.remove(), ms);
}

let dropTarget;

// =============================================================================
// Camera modal (getUserMedia)
// =============================================================================

let cameraStream = null;
let cameraFacingMode = "environment"; // rear camera by default

function openCameraFallback() {
  document.getElementById("camera-fallback").click();
}

async function openCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    openCameraFallback();
    return;
  }

  const modal    = document.getElementById("camera-modal");
  const video    = document.getElementById("camera-video");
  const capture  = document.getElementById("camera-capture");
  const closeBtn = document.getElementById("camera-close");
  const switchBtn = document.getElementById("camera-switch");

  const startStream = async (facingMode) => {
    if (cameraStream) cameraStream.getTracks().forEach(t => t.stop());
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode, width: { ideal: 1280 }, height: { ideal: 960 } },
        audio: false
      });
      video.srcObject = cameraStream;
    } catch (err) {
      closeCamera();
      if (err.name === "NotAllowedError") {
        toast("Permiso de cámara denegado. Habilítalo en la configuración del navegador.", "error", 5000);
      } else {
        openCameraFallback();
      }
    }
  };

  await startStream(cameraFacingMode);
  if (!cameraStream) return;

  modal.style.display = "flex";

  capture.onclick = () => {
    const canvas = document.getElementById("camera-canvas");
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    canvas.toBlob(blob => {
      closeCamera();
      runPredict(new File([blob], "captura.jpg", { type: "image/jpeg" }));
    }, "image/jpeg", 0.92);
  };

  switchBtn.onclick = async () => {
    cameraFacingMode = cameraFacingMode === "environment" ? "user" : "environment";
    await startStream(cameraFacingMode);
  };

  closeBtn.onclick = closeCamera;

  modal.onclick = e => { if (e.target === modal) closeCamera(); };
}

function closeCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
  }
  const modal = document.getElementById("camera-modal");
  modal.style.display = "none";
  document.getElementById("camera-video").srcObject = null;
}

// =============================================================================
// IDENTIFY section
// =============================================================================

function initIdentify() {
  const zone  = document.getElementById("upload-zone");
  const input = document.getElementById("file-input");
  const camBtn = document.getElementById("camera-btn");

  zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("drag-over"); });
  zone.addEventListener("dragleave", e => { if (!zone.contains(e.relatedTarget)) zone.classList.remove("drag-over"); });
  zone.addEventListener("drop", e => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    const f = e.dataTransfer.files[0];
    if (f) runPredict(f);
  });

  input.addEventListener("change", () => { if (input.files[0]) runPredict(input.files[0]); });
  camBtn.addEventListener("click", openCamera);

  const fallback = document.getElementById("camera-fallback");
  fallback.addEventListener("change", () => { if (fallback.files[0]) runPredict(fallback.files[0]); });

  const identifyCamBtn = document.getElementById("identify-camera");
  identifyCamBtn.addEventListener("click", openCamera);
}

async function runPredict(file) {
  const resultBox = document.getElementById("identify-result");
  resultBox.innerHTML = `<div class="spinner-wrap"><div class="spinner"></div><p style="color:var(--neutral-500);font-size:.9rem">Analizando imagen…</p></div>`;
  resultBox.style.display = "block";
  document.getElementById("upload-zone").style.display = "none";
  document.getElementById("identify-reset").style.display = "";
  document.getElementById("identify-camera").style.display = "";

  // Preview
  const reader = new FileReader();
  reader.onload = e => {
    document.getElementById("preview-img").src = e.target.result;
    document.getElementById("preview-wrap").style.display = "flex";
    document.getElementById("identify-grid").style.display = "grid";
  };
  reader.readAsDataURL(file);

  const fd = new FormData();
  fd.append("file", file);

  try {
    const resp = await fetch("/api/predict", { method: "POST", body: fd });
    if (!resp.ok) throw new Error(await resp.text());
    const payload = await resp.json();
    const { results, agent_decision, plantnet, comparison, health, agent_error } = payload;

    if (!results || !results.length) throw new Error("No se obtuvieron predicciones.");

    state.modelTopSpecies = results[0];
    state.agentDecision = agent_decision || null;
    state.plantnetResult = plantnet || null;
    state.comparison = comparison || null;

    const selectedKey = getSelectedSpeciesKey(results, agent_decision);
    const selectedResult = results.find(r => r.key === selectedKey) || results[0];
    state.detectedSpecies = selectedResult;

    renderPredictions(results, resultBox, agent_decision, plantnet, comparison, agent_error, health);
    renderSpeciesInfo(selectedResult.key, document.getElementById("species-info-box"), selectedResult.info || null);
    document.getElementById("species-info-box").style.display = "block";

    // Update trivia / map detected badge
    const badgeName = getFinalSpeciesName();
    document.querySelectorAll(".detected-name").forEach(el => {
      el.textContent = badgeName;
    });
    document.querySelectorAll(".detected-badge-wrap").forEach(el => el.style.display = "flex");

    // Si el select de trivia ya está renderizado, apuntarlo a la especie detectada
    const trivSel = document.getElementById("trivia-species-select");
    if (trivSel && trivSel.options.length > 0 && state.detectedSpecies?.key) {
      trivSel.value = state.detectedSpecies.key;
    }
  } catch (err) {
    resultBox.innerHTML = `<div class="alert alert-warn">⚠️ Error al analizar: ${err.message}</div>`;
  }
}

function renderPredictions(results, container, agentDecision = null, plantnet = null, comparison = null, agentError = null, health = null) {
  const top  = results[0];
  const conf = top.prob;
  const sci  = top.info?.nombre_cientifico || "";
  const confCls = conf >= .75 ? "pill-green" : conf >= .50 ? "pill-amber" : "pill-red";
  const confLbl = conf >= .75 ? "Alta" : conf >= .50 ? "Moderada" : "Baja";

  let altHtml = "";
  if (results.length > 1) {
    const medals = ["🥈", "🥉"];
    altHtml = `<div class="alts-label">Otras posibilidades</div>` +
      results.slice(1).map((r, i) => `
        <div class="alt-row" onclick="loadSpeciesDetail('${r.key}')">
          <span class="alt-medal">${medals[i] || `#${i+2}`}</span>
          <span class="alt-name">${esc(r.name)}</span>
          <span class="alt-pct">${(r.prob*100).toFixed(1)}%</span>
        </div>`).join("");
  }

  const warnHtml = conf < .50
    ? `<div class="alert alert-warn">Confianza baja — intenta con foto más nítida mostrando hojas, flores o frutos.</div>`
    : conf < .75
    ? `<div class="alert alert-tip">Confianza moderada — verifica rasgos botánicos antes de concluir.</div>`
    : "";

  container.innerHTML = `
    <div class="pred-card">
      <div class="pred-label">Especie detectada por el modelo</div>
      <div class="pred-name">${esc(top.name)}</div>
      ${sci ? `<div class="pred-sci">${esc(sci)}</div>` : ""}
      <div class="conf-bar-wrap"><div class="conf-bar" style="width:${(conf*100).toFixed(1)}%"></div></div>
      <div class="conf-row">
        <span class="conf-pill ${confCls}">${confLbl} ${(conf*100).toFixed(0)}%</span>
        <span class="conf-text">Confianza: <strong>${(conf*100).toFixed(1)}%</strong></span>
      </div>
    </div>
    ${warnHtml}
    ${renderAgentDecision(agentDecision, plantnet, comparison, agentError)}
    ${altHtml}
    ${renderSaludArbol(health)}
  `;
}

function renderAgentDecision(decision, plantnet, comparison, agentError) {
  if (agentError && !decision) {
    return `<div class="alert alert-tip">Agente validador no disponible — se muestra solo la predicción local.<br><small>${esc(agentError)}</small></div>`;
  }
  if (!decision) return "";

  const decisionMap = {
    aceptar_prediccion:   ["✅", "Predicción aceptada",           "decision-ok"],
    mostrar_alternativas: ["⚠️",  "Revisar alternativas",          "decision-warn"],
    pedir_nueva_foto:     ["📷", "Se recomienda nueva foto",       "decision-warn"],
    revision_manual:      ["🔬", "Revisión manual recomendada",    "decision-warn"],
    imagen_incorrecta:    ["🚫", "Imagen no válida",               "decision-err"],
  };
  const [decIcon, decTitle, decCls] = decisionMap[decision.decision] || ["🤖", "Decisión del agente", "decision-warn"];

  const modelName    = decision.model_prediction_common || "—";
  const modelConf    = Number(decision.model_confidence || 0);
  const modelPond    = Number(decision.model_weighted_score || 0);
  const pnName       = decision.plantnet_prediction_common || "";
  const pnSci        = decision.plantnet_prediction_scientific || "";
  const pnScore      = Number(decision.plantnet_score || 0);
  const pnPond       = Number(decision.plantnet_weighted_score || 0);
  const pnComunes    = decision.plantnet_common_names || [];
  const coinciden    = decision.model_plantnet_match;
  const matchReason  = decision.matching_reason || "";
  const source       = decision.source_priority || "";
  const selected     = decision.species_selected || modelName;
  const webUsed      = decision.web_evidence_used;

  // ── Bloque agente validador ──────────────────────────────────────────────
  const pnBlock = webUsed
    ? `<div class="agent-source">
         <div class="agent-source-hdr">🔬 Agente validador</div>
         <div class="agent-source-name">${esc(pnName || pnSci)}</div>
         ${pnSci && pnSci.toLowerCase() !== (pnName||"").toLowerCase()
           ? `<div class="agent-source-sci">${esc(pnSci)}</div>` : ""}
         <div class="agent-source-meta">Score: <strong>${(pnScore).toFixed(3)}</strong></div>
         ${!coinciden ? `<div class="agent-source-meta">Score pond.: <strong>${pnPond.toFixed(3)}</strong></div>` : ""}
         ${pnComunes.length ? `<div class="agent-source-meta muted">Comunes: ${esc(pnComunes.slice(0,3).join(", "))}</div>` : ""}
       </div>`
    : `<div class="agent-source agent-source-na">
         <div class="agent-source-hdr">🔬 Agente validador</div>
         <div class="agent-source-na-msg">No disponible${plantnet?.razon ? ` — ${esc(plantnet.razon)}` : ""}</div>
       </div>`;

  // ── Bloque comparación ───────────────────────────────────────────────────
  let compBody = "";
  if (webUsed) {
    if (coinciden) {
      compBody = `<span class="comp-badge comp-ok">✅ Coinciden</span>
        <span class="comp-reason">${esc(matchReason)}</span>
        <span class="comp-shared">Nombre compartido: <strong>${esc(selected)}</strong></span>`;
    } else {
      const prioLabel = source === "plantnet" ? "Agente validador" : source === "modelo" ? "Modelo local" : source;
      const prioScore = source === "plantnet"
        ? `${pnPond.toFixed(3)} > ${modelPond.toFixed(3)}`
        : `${modelPond.toFixed(3)} > ${pnPond.toFixed(3)}`;
      compBody = `<span class="comp-badge comp-no">❌ No coinciden</span>
        ${matchReason ? `<span class="comp-reason">${esc(matchReason)}</span>` : ""}
        <span class="comp-prio">Prioridad: <strong>${esc(prioLabel)}</strong> (score ponderado ${prioScore})</span>`;
    }
  } else {
    compBody = `<span class="comp-badge comp-na">— Agente validador no disponible</span>`;
  }

  return `
    <div class="agent-block">
      <div class="agent-block-title">Validación por agente</div>

      <div class="agent-sources">
        <div class="agent-source">
          <div class="agent-source-hdr">🌳 Modelo local</div>
          <div class="agent-source-name">${esc(modelName)}</div>
          <div class="agent-source-meta">Confianza: <strong>${(modelConf*100).toFixed(1)}%</strong></div>
          ${!coinciden && webUsed ? `<div class="agent-source-meta">Score pond.: <strong>${modelPond.toFixed(3)}</strong></div>` : ""}
        </div>
        <div class="agent-source-sep"></div>
        ${pnBlock}
      </div>

      <div class="agent-comparison">${compBody}</div>

      <div class="agent-decision ${decCls}">
        <div class="agent-dec-title">${decIcon} ${decTitle}</div>
        <div class="agent-dec-species">Especie final: <strong>${esc(selected)}</strong></div>
        ${decision.reasoning        ? `<div class="agent-dec-row"><strong>Razonamiento:</strong> ${esc(decision.reasoning)}</div>` : ""}
        ${decision.recommended_action ? `<div class="agent-dec-row"><strong>Acción:</strong> ${esc(decision.recommended_action)}</div>` : ""}
      </div>
    </div>`;
}

function renderSaludArbol(health) {
  if (!health || !health.estado || health.estado === "no_determinado") return "";

  const estadoMap = {
    aparentemente_sano: ["🟢", "Aparentemente sano"],
    estres_moderado:    ["🟡", "Estrés moderado"],
    posible_enfermedad: ["🔴", "Posible enfermedad"],
    no_es_planta:       ["⚫", "No es planta"],
    indeterminado:      ["⚪", "Indeterminado"],
  };
  const [estIcon, estLabel] = estadoMap[health.estado] || ["⚪", health.estado];
  const usaPlantnet = ["plantnet_leaf", "plantnet_score"].includes(health.metodo);

  const scoreRow = usaPlantnet && health.score_hoja != null
    ? `<div class="health-metric"><span class="hm-label">Score (agente validador)</span><span class="hm-val">${Number(health.score_hoja).toFixed(3)}</span></div>`
    : "";

  const colorRows = (health.pct_verde != null)
    ? `<div class="health-metric"><span class="hm-label hm-verde">Verde</span><span class="hm-val">${health.pct_verde}%</span></div>
       <div class="health-metric"><span class="hm-label hm-seco">Seco</span><span class="hm-val">${health.pct_seco}%</span></div>
       <div class="health-metric"><span class="hm-label hm-marron">Marrón</span><span class="hm-val">${health.pct_marron}%</span></div>`
    : "";

  return `
    <div class="health-block">
      <div class="health-block-title">🌿 Evaluación visual de salud</div>
      <div class="health-estado">
        <span class="health-estado-icon">${estIcon}</span>
        <span class="health-estado-label">${esc(estLabel)}</span>
        <span class="health-metodo">${esc(health.metodo || "")}</span>
      </div>
      <div class="health-metrics">
        ${scoreRow}
        ${colorRows}
      </div>
      ${health.recomendacion ? `<div class="health-row"><strong>Recomendación:</strong> ${esc(health.recomendacion)}</div>` : ""}
      ${health.limitacion    ? `<div class="health-row health-limit">${esc(health.limitacion)}</div>` : ""}
    </div>`;
}

function resetIdentify() {
  document.getElementById("upload-zone").style.display = "";
  document.getElementById("identify-result").style.display = "none";
  document.getElementById("species-info-box").style.display = "none";
  document.getElementById("identify-grid").style.display = "none";
  document.getElementById("preview-wrap").style.display = "none";
  document.getElementById("identify-reset").style.display = "none";
  document.getElementById("identify-camera").style.display = "none";
  document.querySelectorAll(".detected-badge-wrap").forEach(el => el.style.display = "none");
}

// =============================================================================
// CATALOG section
// =============================================================================

async function initCatalog() {
  if (state.classes.length) { renderCatalogGrid(state.classes); return; }
  const res = await fetch("/api/classes");
  const { classes } = await res.json();
  state.classes = classes;
  renderCatalogGrid(classes);

  document.getElementById("catalog-search").addEventListener("input", e => {
    const q = e.target.value.toLowerCase();
    renderCatalogGrid(classes.filter(c =>
      c.name.toLowerCase().includes(q) || c.scientific.toLowerCase().includes(q)
    ));
  });
}

function renderCatalogGrid(items) {
  const grid = document.getElementById("catalog-grid");
  if (!items.length) {
    grid.innerHTML = `<div class="empty-state" style="grid-column:1/-1">
      <div class="empty-icon">🔍</div>
      <div class="empty-title">Sin resultados</div>
      <div class="empty-sub">Prueba con otro nombre o nombre científico.</div>
    </div>`;
    return;
  }
  grid.innerHTML = items.map(c => `
    <div class="species-card" onclick="loadSpeciesDetail('${c.key}')">
      <div class="species-card-name">${esc(c.name)}</div>
      ${c.scientific ? `<div class="species-card-sci">${esc(c.scientific)}</div>` : ""}
      ${c.family ? `<div class="species-card-fam">${esc(c.family)}</div>` : ""}
    </div>`).join("");
}

async function loadSpeciesDetail(key) {
  document.getElementById("catalog-list-view").style.display = "none";
  const detail = document.getElementById("catalog-detail-view");
  detail.style.display = "block";
  detail.innerHTML = `<div class="spinner-wrap"><div class="spinner"></div></div>`;

  try {
    const res = await fetch(`/api/catalog/${key}`);
    const info = await res.json();
    renderSpeciesInfo(key, detail, info);
  } catch {
    detail.innerHTML = `<div class="alert alert-warn">Error cargando información.</div>`;
  }
}

function showCatalogList() {
  document.getElementById("catalog-list-view").style.display = "";
  document.getElementById("catalog-detail-view").style.display = "none";
}

function renderSpeciesInfo(key, container, infoData = null) {
  const info = infoData || state.detectedSpecies?.info || {};
  const name = info.nombre_comun || info.display_name || key?.replace(/_/g, " ");
  const sci  = info.nombre_cientifico || "";

  const pills = [
    info.familia ? `<span class="pill-tag">Familia: <strong>${esc(info.familia)}</strong></span>` : "",
    info.altura_aproximada ? `<span class="pill-tag">Altura: <strong>${esc(info.altura_aproximada)}</strong></span>` : "",
  ].filter(Boolean).join("");

  const row = (label, val) => val
    ? `<div><div class="sec-label">${label}</div><div class="info-text">${esc(val)}</div></div>`
    : "";

  const backBtn = infoData
    ? `<button class="back-btn" onclick="showCatalogList()">← Volver al catálogo</button>`
    : "";

  container.innerHTML = `
    ${backBtn}
    <div class="info-card">
      <div class="info-title">${esc(name)}</div>
      ${sci ? `<div class="info-sci">${esc(sci)}</div>` : ""}
      ${pills ? `<div class="pills">${pills}</div>` : ""}
      ${row("Descripción", info.descripcion)}
      ${row("Cómo identificarlo", info.como_identificarlo)}
      <div class="grid-2">
        ${row("Hojas", info.hojas)}
        ${row("Flores", info.flores)}
      </div>
      <div class="grid-2">
        ${row("Frutos", info.frutos)}
        ${row("Distribución", info.distribucion)}
      </div>
      ${row("Usos", info.usos)}
      ${info.dato_curioso ? `<div class="fact-card"><strong>Dato curioso:</strong> ${esc(info.dato_curioso)}</div>` : ""}
      <div class="historia-card">
        <div class="historia-label">Origen, historia en Colombia y usos</div>
        <div class="historia-text">${info.historia_origen_colombia_usos
          ? esc(info.historia_origen_colombia_usos)
          : '<em style="color:var(--neutral-400)">Información no disponible para esta especie.</em>'
        }</div>
      </div>
    </div>`;
}

// =============================================================================
// MAP section
// =============================================================================

function initMap() {
  if (state.leafletMap) return;
  const center = window.CAMPUS_CENTER || [6.2636427, -75.5764393];

  const satUrl  = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
  const satAttr = "Tiles © Esri";
  const satLayer   = L.tileLayer(satUrl, { attribution: satAttr, maxZoom: 20 });
  const streetLayer = L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
    attribution: "© CartoDB", maxZoom: 20
  });

  const map = L.map("campus-map", { center, zoom: 18, layers: [satLayer] });
  state.leafletMap = map;
  state.mapLayers  = { satellite: satLayer, streets: streetLayer };

  // Markers layer group
  state.markersLayer = L.layerGroup().addTo(map);
  state.gpsMarker    = null;
  state.clickMarker  = null;

  // Click handler
  map.on("click", e => {
    state.clickCoords = [e.latlng.lat, e.latlng.lng];
    if (state.clickMarker) state.clickMarker.remove();
    state.clickMarker = L.marker([e.latlng.lat, e.latlng.lng], {
      icon: L.divIcon({ className: "", html: `<div style="background:#D97706;width:14px;height:14px;border-radius:50%;border:2.5px solid #fff;box-shadow:0 2px 6px rgba(0,0,0,.4)"></div>`, iconSize: [14,14], iconAnchor: [7,7] })
    }).addTo(map).bindPopup(`Punto seleccionado<br>${e.latlng.lat.toFixed(6)}, ${e.latlng.lng.toFixed(6)}`);
    updateClickDisplay();
  });

  // Load persisted points
  loadMapPoints();
  refreshMapMarkers();

  // Style toggle
  document.getElementById("btn-satellite").addEventListener("click", () => setMapStyle("satellite"));
  document.getElementById("btn-streets").addEventListener("click",   () => setMapStyle("streets"));

  // GPS
  document.getElementById("btn-gps").addEventListener("click", requestGPS);

  // Save
  document.getElementById("btn-save-tree").addEventListener("click", saveTree);
  document.getElementById("btn-sync").addEventListener("click", syncGithub);
}

function setMapStyle(style) {
  if (!state.leafletMap) return;
  state.mapStyle = style;
  const map = state.leafletMap;
  const { satellite, streets } = state.mapLayers;
  if (style === "satellite") { map.removeLayer(streets); map.addLayer(satellite); }
  else { map.removeLayer(satellite); map.addLayer(streets); }
  document.getElementById("btn-satellite").classList.toggle("active", style === "satellite");
  document.getElementById("btn-streets").classList.toggle("active",   style === "streets");
}

function requestGPS() {
  if (!navigator.geolocation) { toast("GPS no disponible en este dispositivo.", "error"); return; }
  navigator.geolocation.getCurrentPosition(
    pos => {
      state.gpsCoords = [pos.coords.latitude, pos.coords.longitude];
      const acc = pos.coords.accuracy;
      if (state.gpsMarker) state.gpsMarker.remove();
      state.gpsMarker = L.circleMarker(state.gpsCoords, {
        radius: 10, color: "#1976D2", fillColor: "#42A5F5", fillOpacity: .75,
        weight: 2
      }).addTo(state.leafletMap).bindPopup("Tu ubicación GPS");
      state.leafletMap.setView(state.gpsCoords, 19);
      document.getElementById("gps-display").textContent =
        `${state.gpsCoords[0].toFixed(6)}, ${state.gpsCoords[1].toFixed(6)} · ±${acc.toFixed(0)} m`;
      document.getElementById("gps-display-wrap").style.display = "";
    },
    err => toast(`Error GPS: ${err.message}`, "error")
  );
}

function updateClickDisplay() {
  if (!state.clickCoords) return;
  document.getElementById("click-display").textContent =
    `${state.clickCoords[0].toFixed(6)}, ${state.clickCoords[1].toFixed(6)}`;
  document.getElementById("click-display-wrap").style.display = "";
}

function refreshMapMarkers() {
  if (!state.markersLayer) return;
  state.markersLayer.clearLayers();
  state.mapPoints.forEach(pt => {
    const popup = `<b>${pt.name}</b><br><em>${pt.sci||""}</em><br>
      Confianza: ${pt.confidence ? (pt.confidence*100).toFixed(0)+"%" : "—"}<br>
      ${pt.datetime || ""}`;
    L.marker([pt.lat, pt.lon], {
      icon: L.divIcon({
        className: "",
        html: `<div style="background:var(--forest-600,#1B4332);color:#fff;border-radius:50%;width:28px;height:28px;display:flex;align-items:center;justify-content:center;font-size:14px;box-shadow:0 2px 8px rgba(0,0,0,.3);border:2px solid #fff">🌳</div>`,
        iconSize: [28,28], iconAnchor: [14,14]
      })
    }).addTo(state.markersLayer).bindPopup(popup).bindTooltip(pt.name);
  });
}

async function saveTree() {
  if (!state.detectedSpecies) { toast("Identifica una especie primero en la sección Identificar.", "error"); return; }
  const coords = state.gpsCoords || state.clickCoords;
  if (!coords) { toast("Activa el GPS o haz clic en el mapa para fijar ubicación.", "error"); return; }

  const source = state.gpsCoords ? "gps" : "manual_map_click";
  const body = {
    species_key: getSelectedSpeciesKey([state.detectedSpecies], state.agentDecision) || state.detectedSpecies.key,
    common_name: getFinalSpeciesName(),
    confidence:  state.modelTopSpecies?.prob ?? state.detectedSpecies.prob,
    latitude:    coords[0],
    longitude:   coords[1],
    source,
  };

  try {
    const res = await fetch("/api/save-tree", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    state.mapPoints.push({ ...body, name: body.common_name, lat: coords[0], lon: coords[1] });
    refreshMapMarkers();
    renderSavedList();
    toast(`${body.common_name} guardado en el mapa.`);
  } catch (err) {
    toast(`Error al guardar: ${err.message}`, "error");
  }
}

async function syncGithub() {
  toast("Sincronizando con GitHub…");
  try {
    const res = await fetch("/api/sync-github", { method: "POST" });
    const { ok, message } = await res.json();
    toast(message, ok ? "success" : "error");
  } catch {
    toast("Error de red.", "error");
  }
}

function loadMapPoints() {
  fetch("/api/map-data").then(r => r.json()).then(({ points }) => {
    state.mapPoints = points;
    refreshMapMarkers();
    renderSavedList();
  });
}

function renderSavedList() {
  const list = document.getElementById("saved-list");
  if (!state.mapPoints.length) {
    list.innerHTML = `<div style="font-size:.8rem;color:var(--neutral-400);text-align:center;padding:.5rem">Sin árboles guardados aún.</div>`;
    return;
  }
  list.innerHTML = state.mapPoints.map((pt, i) => `
    <div class="saved-item">
      <span class="saved-item-name">${esc(pt.name)}</span>
      <button class="saved-item-del" onclick="removeSavedPoint(${i})">✕</button>
    </div>`).join("");
}

function removeSavedPoint(i) {
  state.mapPoints.splice(i, 1);
  refreshMapMarkers();
  renderSavedList();
}

// =============================================================================
// TRIVIA section
// =============================================================================

const TV = { score: 0, total: 0, max: 5, state: "ask", answer: null };

async function initTrivia() {
  if (state.classes.length) {
    applyDetectedToTrivia();
    return;
  }
  const res = await fetch("/api/classes");
  const { classes } = await res.json();
  state.classes = classes;
  populateTriviaSelect();
}

function populateTriviaSelect() {
  const sel = document.getElementById("trivia-species-select");
  sel.innerHTML = state.classes.map(c =>
    `<option value="${c.key}">${esc(c.name)}</option>`
  ).join("");
  applyDetectedToTrivia();
}

function applyDetectedToTrivia() {
  const sel = document.getElementById("trivia-species-select");
  if (!sel || !sel.options.length) return;
  const detectedKey = state.detectedSpecies?.key;
  if (detectedKey) sel.value = detectedKey;
  startTriviaRound();
}

function startTriviaRound() {
  TV.score = 0; TV.total = 0; TV.state = "ask"; TV.answer = null;
  loadNextQuestion();
}

async function loadNextQuestion() {
  const key = document.getElementById("trivia-species-select").value;
  state.triviaKey = key;
  document.getElementById("trivia-result-view").style.display = "none";
  const card = document.getElementById("quiz-card");
  card.style.display = "";
  card.innerHTML = `<div class="spinner-wrap"><div class="spinner"></div></div>`;

  try {
    const res = await fetch(`/api/trivia/${key}`);
    if (!res.ok) throw new Error("Sin datos suficientes.");
    const q = await res.json();
    TV.answer = q.answer;
    renderQuestion(q);
  } catch (err) {
    card.innerHTML = `<div class="alert alert-tip">${err.message}</div>`;
  }
}

function renderQuestion(q) {
  const card = document.getElementById("quiz-card");
  const pct  = TV.max > 0 ? ((TV.total / TV.max) * 100).toFixed(0) : 0;

  card.innerHTML = `
    <div class="quiz-progress">
      <span class="quiz-counter">Pregunta ${TV.total + 1} / ${TV.max}</span>
      <span class="quiz-score-badge">🏆 ${TV.score} / ${TV.total}</span>
    </div>
    <div class="quiz-prog-bar"><div class="quiz-prog-fill" style="width:${pct}%"></div></div>
    <div class="quiz-question">${esc(q.question)}</div>
    <div class="quiz-options">
      ${q.options.map(opt => {
        const arg = JSON.stringify(String(opt)).replace(/"/g, "&quot;");
        return `<button class="quiz-option" onclick="checkAnswer(${arg})">${esc(opt)}</button>`;
      }).join("")}
    </div>
  `;
}

function checkAnswer(chosen) {
  TV.total += 1;
  const correct = chosen === TV.answer;
  if (correct) TV.score += 1;

  // Freeze buttons + highlight
  document.querySelectorAll(".quiz-option").forEach(btn => {
    btn.disabled = true;
    if (btn.textContent === TV.answer)  btn.classList.add("correct");
    if (btn.textContent === chosen && !correct) btn.classList.add("wrong");
  });

  // Add feedback + next button
  const card = document.getElementById("quiz-card");
  const fb   = document.createElement("div");
  fb.style.marginTop = "1rem";
  const isLast = TV.total >= TV.max;
  fb.innerHTML = `
    <div class="alert ${correct ? "alert-ok" : "alert-warn"}" style="margin-bottom:.8rem">
      ${correct ? "✅ ¡Correcto!" : `❌ La respuesta era: <strong>${esc(TV.answer)}</strong>`}
    </div>
    <button class="btn-primary" style="width:100%;justify-content:center" onclick="${isLast ? "showTriviaResult()" : "loadNextQuestion()"}">
      ${isLast ? "Ver resultados" : "Siguiente"}
    </button>`;
  card.appendChild(fb);
}

function showTriviaResult() {
  document.getElementById("quiz-card").style.display = "none";
  const rv = document.getElementById("trivia-result-view");
  rv.style.display = "";
  const pct = TV.total > 0 ? Math.round((TV.score / TV.total) * 100) : 0;
  const emoji = pct >= 80 ? "🏆" : pct >= 50 ? "🌿" : "🌱";
  rv.innerHTML = `
    <div class="quiz-result-card">
      <div class="quiz-result-emoji">${emoji}</div>
      <div class="quiz-result-title">¡Ronda completada!</div>
      <div class="quiz-result-score">${TV.score} / ${TV.total}</div>
      <div class="quiz-result-sub">${pct}% de respuestas correctas</div>
      <button class="btn-restart" onclick="startTriviaRound()">Jugar de nuevo</button>
    </div>`;
}

// =============================================================================
// Mini hero map
// =============================================================================

function initMiniMap() {
  const center = window.CAMPUS_CENTER || [6.2636427, -75.5764393];
  const m = L.map("hero-mini-map", { center, zoom: 17, zoomControl: false,
    scrollWheelZoom: false, dragging: false, touchZoom: false });
  L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
    attribution: ""
  }).addTo(m);
  state.miniMap = m;
}

// =============================================================================
// PWA install
// =============================================================================

window.addEventListener("beforeinstallprompt", e => {
  e.preventDefault();
  state.installPrompt = e;
  document.getElementById("install-banner").style.display = "flex";
});

function installApp() {
  if (!state.installPrompt) return;
  state.installPrompt.prompt();
  state.installPrompt.userChoice.then(() => {
    state.installPrompt = null;
    document.getElementById("install-banner").style.display = "none";
  });
}

function dismissInstall() {
  document.getElementById("install-banner").style.display = "none";
}

// =============================================================================
// Agent helpers
// =============================================================================

function getSelectedSpeciesKey(results = [], decision = null) {
  const key = decision?.species_selected_key;
  if (key && results.some(r => r.key === key)) return key;
  return results[0]?.key || state.detectedSpecies?.key || null;
}

function getFinalSpeciesName() {
  const d = state.agentDecision;
  if (d && d.species_selected && d.species_selected !== "ninguna") {
    return d.species_selected;
  }
  return state.detectedSpecies?.info?.nombre_comun || state.detectedSpecies?.name || "Especie detectada";
}

// =============================================================================
// Utility
// =============================================================================

function esc(s) {
  return String(s ?? "")
    .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

// =============================================================================
// Boot
// =============================================================================

document.addEventListener("DOMContentLoaded", () => {
  // Nav links
  document.querySelectorAll(".nav-link").forEach(l => {
    l.addEventListener("click", () => navigate(l.dataset.nav));
  });
  document.getElementById("nav-cta").addEventListener("click", () => navigate("identify"));
  document.querySelectorAll("[data-nav-to]").forEach(el => {
    el.addEventListener("click", () => navigate(el.dataset.navTo));
  });

  // Init identify
  initIdentify();
  document.getElementById("identify-reset").addEventListener("click", resetIdentify);

  // Init mini map on hero
  if (document.getElementById("hero-mini-map")) initMiniMap();

  // Service worker
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("/static/sw.js").catch(() => {});
  }
});
