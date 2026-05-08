from flask import Flask, request, jsonify
from collections import deque
 
app = Flask(__name__)
 
# ── Cola de comandos para el programa principal ───────────────────────
comandos_pendientes = deque()
 
# ── Estado visible del sistema (lo actualiza el programa principal) ───
estado_sistema = {
    "categoria":     "abdomen",
    "exploracion":   "hepatica",
    "pitch_actual":  90.0,
    "roll_actual":   0.0,
    "estado_angulo": "ok",
    "ajuste_manual": False,
    "msg_pitch":     "",
    "msg_roll":      ""
}
 
# =============================================================
# ENDPOINTS DE CONTROL REMOTO
# =============================================================
 
@app.route('/comando', methods=['POST'])
def recibir_comando():
    data = request.json
    tecla = data.get('tecla', '')
    if tecla:
        comandos_pendientes.append(tecla)
    return jsonify({"ok": True})
 
@app.route('/estado')
def obtener_estado():
    return jsonify(estado_sistema)
 
@app.route('/actualizar_estado', methods=['POST'])
def actualizar_estado():
    estado_sistema.update(request.json)
    return jsonify({"ok": True})
 
@app.route('/leer_comando')
def leer_comando():
    if comandos_pendientes:
        return jsonify({"comando": comandos_pendientes.popleft()})
    return jsonify({"comando": None})
 
# =============================================================
# PANEL DE CONTROL — interfaz web del médico
# =============================================================
 
@app.route('/')
def panel_medico():
    return r'''<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Panel Médico — Guía Ecográfica</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;600;700;800&display=swap');
 
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
 
  :root {
    --bg:       #080c10;
    --surface:  #0d1318;
    --surface2: #111820;
    --border:   #1e2d3d;
    --accent:   #00d4ff;
    --accent2:  #0099cc;
    --ok:       #00e676;
    --warn:     #ffab00;
    --err:      #ff1744;
    --text:     #e0eaf5;
    --muted:    #4a6070;
    --mono:     'JetBrains Mono', monospace;
    --sans:     'Syne', sans-serif;
  }
 
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
    padding: 24px;
    background-image:
      radial-gradient(ellipse 80% 40% at 50% -10%, rgba(0,212,255,0.08) 0%, transparent 60%),
      repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(30,45,61,0.3) 40px),
      repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(30,45,61,0.3) 40px);
  }
 
  header {
    display: flex; align-items: center; gap: 16px;
    margin-bottom: 28px; padding-bottom: 20px;
    border-bottom: 1px solid var(--border);
  }
  .logo {
    width: 40px; height: 40px; border-radius: 10px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    display: flex; align-items: center; justify-content: center; font-size: 20px;
  }
  header h1 { font-size: 18px; font-weight: 800; }
  header p  { font-size: 12px; color: var(--muted); font-family: var(--mono); margin-top: 2px; }
 
  .dot {
    margin-left: auto; display: flex; align-items: center; gap: 8px;
    font-family: var(--mono); font-size: 11px; color: var(--muted);
  }
  .dot span {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--ok); box-shadow: 0 0 8px var(--ok);
    animation: pulse 2s infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
 
  .grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 16px; max-width: 920px; margin: 0 auto;
  }
 
  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; padding: 20px; position: relative; overflow: hidden;
  }
  .card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    opacity: .4;
  }
  .card.full { grid-column: 1 / -1; }
 
  .card-title {
    font-size: 10px; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; color: var(--muted);
    margin-bottom: 14px; font-family: var(--mono);
  }
 
  .status-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .status-item label {
    display: block; font-size: 10px; color: var(--muted);
    font-family: var(--mono); letter-spacing: 1px; text-transform: uppercase; margin-bottom: 4px;
  }
  .status-item .value { font-size: 18px; font-weight: 700; color: var(--accent); font-family: var(--mono); text-transform: uppercase; }
 
  .badge {
    display: inline-block; padding: 4px 10px; border-radius: 6px;
    font-size: 11px; font-weight: 700; font-family: var(--mono);
    letter-spacing: 1px; text-transform: uppercase;
  }
  .badge.ok      { background: rgba(0,230,118,.15); color: var(--ok);  border: 1px solid rgba(0,230,118,.3); }
  .badge.warning { background: rgba(255,171,0,.15); color: var(--warn); border: 1px solid rgba(255,171,0,.3); }
  .badge.error   { background: rgba(255,23,68,.15);  color: var(--err); border: 1px solid rgba(255,23,68,.3); }
 
  .angle-row { display: flex; gap: 12px; margin-top: 12px; }
  .angle-box {
    flex: 1; background: var(--surface2); border-radius: 10px;
    padding: 12px; border: 1px solid var(--border);
  }
  .angle-box label { font-size: 10px; color: var(--muted); font-family: var(--mono); }
  .angle-box .val  { font-size: 22px; font-weight: 700; font-family: var(--mono); margin: 4px 0 2px; }
  .angle-box .msg  { font-size: 11px; color: var(--muted); font-family: var(--mono); }
 
  .btn-group { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; }
  .btn {
    flex: 1; min-width: 100px; padding: 12px 10px;
    border-radius: 10px; border: 1px solid var(--border);
    background: var(--surface2); color: var(--text);
    font-family: var(--sans); font-size: 13px; font-weight: 600;
    cursor: pointer; transition: all .15s;
    display: flex; align-items: center; justify-content: center; gap: 6px;
  }
  .btn:hover  { border-color: var(--accent); color: var(--accent); background: rgba(0,212,255,.06); transform: translateY(-1px); }
  .btn:active { transform: translateY(0); }
  .btn.primary { background: rgba(0,212,255,.12); border-color: var(--accent); color: var(--accent); }
 
  .explo-list { display: flex; flex-direction: column; gap: 6px; }
  .explo-item {
    padding: 10px 14px; background: var(--surface2); border: 1px solid var(--border);
    border-radius: 10px; cursor: pointer;
    display: flex; align-items: center; justify-content: space-between;
    transition: all .15s; font-size: 13px;
  }
  .explo-item:hover  { border-color: var(--accent); color: var(--accent); }
  .explo-item.active { border-color: var(--accent); background: rgba(0,212,255,.08); color: var(--accent); }
  .explo-item .tag   { font-size: 10px; font-family: var(--mono); color: var(--muted); }
  .explo-item.active .tag { color: var(--accent2); }
 
  .section-label {
    font-size: 10px; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; color: var(--muted);
    margin: 14px 0 8px; font-family: var(--mono);
  }
 
  /* D-pad */
  .dpad {
    display: grid; grid-template-columns: repeat(3, 52px);
    grid-template-rows: repeat(3, 52px); gap: 6px;
    justify-content: center; margin: 0 auto 14px;
  }
  .dpad-btn {
    width: 52px; height: 52px; background: var(--surface2);
    border: 1px solid var(--border); border-radius: 10px;
    color: var(--text); font-size: 18px; cursor: pointer;
    transition: all .15s; display: flex; align-items: center; justify-content: center;
  }
  .dpad-btn:hover  { border-color: var(--accent); color: var(--accent); background: rgba(0,212,255,.08); }
  .dpad-btn:active { transform: scale(.95); }
  .dpad-btn.center {
    background: rgba(0,212,255,.1); border-color: var(--accent);
    font-size: 11px; color: var(--accent); font-family: var(--mono); font-weight: 700;
  }
  .dpad-empty { visibility: hidden; }
 
  .scale-row { display: flex; gap: 8px; }
  .scale-btn {
    flex: 1; padding: 10px; background: var(--surface2);
    border: 1px solid var(--border); border-radius: 10px; color: var(--text);
    font-size: 11px; font-family: var(--mono); cursor: pointer;
    transition: all .15s; text-align: center; line-height: 1.5;
  }
  .scale-btn:hover { border-color: var(--accent); color: var(--accent); }
 
  .toggle-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 16px; background: var(--surface2); border-radius: 10px;
    border: 1px solid var(--border); margin-bottom: 16px;
    cursor: pointer; transition: border-color .2s;
  }
  .toggle-row:hover { border-color: var(--accent); }
  .toggle-row span  { font-size: 13px; font-weight: 600; }
  .toggle-row small { font-size: 11px; color: var(--muted); display: block; font-family: var(--mono); }
  .toggle {
    width: 44px; height: 24px; background: var(--border);
    border-radius: 12px; position: relative; transition: background .2s; flex-shrink: 0;
  }
  .toggle.on { background: var(--accent); }
  .toggle::after {
    content: ''; position: absolute; top: 3px; left: 3px;
    width: 18px; height: 18px; background: white; border-radius: 50%; transition: left .2s;
  }
  .toggle.on::after { left: 23px; }
 
  .feedback {
    position: fixed; bottom: 24px; right: 24px;
    padding: 10px 18px; border-radius: 10px;
    background: rgba(0,212,255,.15); border: 1px solid var(--accent);
    color: var(--accent); font-family: var(--mono); font-size: 12px;
    opacity: 0; transition: opacity .3s; pointer-events: none;
  }
  .feedback.show { opacity: 1; }
 
  @media (max-width: 620px) {
    .grid { grid-template-columns: 1fr; }
    .card.full { grid-column: 1; }
  }
</style>
</head>
<body>
 
<header>
  <div class="logo">🔊</div>
  <div>
    <h1>Panel de Control Médico</h1>
    <p>Sistema de Guía de Sonda Ecográfica — Control Remoto</p>
  </div>
  <div class="dot"><span></span> CONECTADO</div>
</header>
 
<div class="grid">
 
  <!-- ── ESTADO ACTUAL ── -->
  <div class="card full">
    <div class="card-title">Estado actual del sistema</div>
    <div class="status-grid">
      <div class="status-item">
        <label>Categoría</label>
        <div class="value" id="st-categoria">—</div>
      </div>
      <div class="status-item">
        <label>Exploración</label>
        <div class="value" id="st-exploracion">—</div>
      </div>
      <div class="status-item">
        <label>Estado ángulo</label>
        <div id="st-badge"><span class="badge ok">—</span></div>
      </div>
      <div class="status-item">
        <label>Ajuste manual</label>
        <div class="value" id="st-ajuste">—</div>
      </div>
    </div>
    <div class="angle-row">
      <div class="angle-box">
        <label>PITCH</label>
        <div class="val" id="st-pitch">—°</div>
        <div class="msg" id="st-msg-pitch"></div>
      </div>
      <div class="angle-box">
        <label>ROLL</label>
        <div class="val" id="st-roll">—°</div>
        <div class="msg" id="st-msg-roll"></div>
      </div>
    </div>
  </div>
 
  <!-- ── NAVEGACIÓN ── -->
  <div class="card">
    <div class="card-title">Navegación de exploraciones</div>
 
    <div class="btn-group">
      <button class="btn primary" onclick="cmd('c')">⇄ Cambiar categoría</button>
    </div>
    <div class="btn-group">
      <button class="btn" onclick="cmd('b')">← Anterior</button>
      <button class="btn" onclick="cmd('n')">Siguiente →</button>
    </div>
 
    <div class="section-label">Abdomen</div>
    <div class="explo-list">
      <div class="explo-item" data-explo="hepatica"            onclick="irA('hepatica')">           <span>Hepática</span>            <span class="tag">HIPOCONDRIO DER</span></div>
      <div class="explo-item" data-explo="epigastrica"         onclick="irA('epigastrica')">        <span>Epigástrica</span>         <span class="tag">EPIGASTRIO</span></div>
      <div class="explo-item" data-explo="esplenica"           onclick="irA('esplenica')">          <span>Esplénica</span>           <span class="tag">HIPOCONDRIO IZQ</span></div>
      <div class="explo-item" data-explo="flanco_derecho"      onclick="irA('flanco_derecho')">     <span>Flanco derecho</span>      <span class="tag">LATERAL DER</span></div>
      <div class="explo-item" data-explo="flanco_izquierdo"    onclick="irA('flanco_izquierdo')">   <span>Flanco izquierdo</span>    <span class="tag">LATERAL IZQ</span></div>
      <div class="explo-item" data-explo="suprapubica"         onclick="irA('suprapubica')">        <span>Suprapúbica</span>         <span class="tag">HIPOGASTRIO</span></div>
    </div>
 
    <div class="section-label">Tiroides</div>
    <div class="explo-list">
      <div class="explo-item" data-explo="transversal"            onclick="irA('transversal')">           <span>Transversal</span>            <span class="tag">AXIAL</span></div>
      <div class="explo-item" data-explo="longitudinal_derecha"   onclick="irA('longitudinal_derecha')">  <span>Longitudinal derecha</span>   <span class="tag">SAGITAL DER</span></div>
      <div class="explo-item" data-explo="longitudinal_izquierda" onclick="irA('longitudinal_izquierda')"><span>Longitudinal izquierda</span> <span class="tag">SAGITAL IZQ</span></div>
    </div>
  </div>
 
  <!-- ── AJUSTE MANUAL ROI ── -->
  <div class="card">
    <div class="card-title">Ajuste manual de ROI</div>
 
    <div class="toggle-row" onclick="cmd('m')">
      <div>
        <span>Ajuste manual</span>
        <small>Activa para mover y escalar la ROI</small>
      </div>
      <div class="toggle" id="toggle-indicator"></div>
    </div>
 
    <div class="dpad">
      <div class="dpad-empty"></div>
      <button class="dpad-btn" onclick="cmd('w')">▲</button>
      <div class="dpad-empty"></div>
      <button class="dpad-btn" onclick="cmd('a')">◀</button>
      <div class="dpad-btn center">ROI</div>
      <button class="dpad-btn" onclick="cmd('d')">▶</button>
      <div class="dpad-empty"></div>
      <button class="dpad-btn" onclick="cmd('s')">▼</button>
      <div class="dpad-empty"></div>
    </div>
 
    <div class="scale-row">
      <button class="scale-btn" onclick="cmd('j')">↔ ancho<br>−</button>
      <button class="scale-btn" onclick="cmd('l')">↔ ancho<br>+</button>
      <button class="scale-btn" onclick="cmd('k')">↕ alto<br>−</button>
      <button class="scale-btn" onclick="cmd('i')">↕ alto<br>+</button>
    </div>
  </div>
 
</div>
 
<div class="feedback" id="feedback"></div>
 
<script>
  async function cmd(tecla) {
    try {
      await fetch('/comando', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({tecla})
      });
      flash('Comando: ' + tecla.toUpperCase());
    } catch(e) { flash('Error de conexión'); }
  }
 
  async function irA(nombre) {
    await fetch('/comando', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({tecla: 'goto:' + nombre})
    });
    flash('→ ' + nombre);
  }
 
  async function poll() {
    try {
      const r = await fetch('/estado');
      const d = await r.json();
 
      document.getElementById('st-categoria').textContent   = d.categoria   || '—';
      document.getElementById('st-exploracion').textContent = d.exploracion || '—';
      document.getElementById('st-pitch').textContent = (d.pitch_actual != null ? (+d.pitch_actual).toFixed(1) : '—') + '°';
      document.getElementById('st-roll').textContent  = (d.roll_actual  != null ? (+d.roll_actual).toFixed(1)  : '—') + '°';
      document.getElementById('st-msg-pitch').textContent = d.msg_pitch || '';
      document.getElementById('st-msg-roll').textContent  = d.msg_roll  || '';
      document.getElementById('st-ajuste').textContent    = d.ajuste_manual ? 'ACTIVO' : 'DESACTIVADO';
 
      const labels = {ok:'CORRECTO ✓', warning:'AJUSTE LEVE', error:'CORREGIR !'};
      document.getElementById('st-badge').innerHTML =
        '<span class="badge ' + d.estado_angulo + '">' + (labels[d.estado_angulo]||'—') + '</span>';
 
      document.getElementById('toggle-indicator').className = 'toggle' + (d.ajuste_manual ? ' on' : '');
 
      document.querySelectorAll('.explo-item').forEach(el => {
        el.classList.toggle('active', el.dataset.explo === d.exploracion);
      });
    } catch(e) {}
  }
 
  let fbT;
  function flash(msg) {
    const el = document.getElementById('feedback');
    el.textContent = msg; el.classList.add('show');
    clearTimeout(fbT); fbT = setTimeout(() => el.classList.remove('show'), 1800);
  }
 
  setInterval(poll, 800);
  poll();
</script>
</body>
</html>'''
 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)