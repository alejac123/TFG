import cv2
import numpy as np
import time
import math
import threading
import requests
 
# =============================================================
# CONFIGURACIÓN
# =============================================================
USAR_SENSOR_REAL = False
SERVIDOR_URL     = "http://127.0.0.1:5000"
PUERTO_SERIAL    = "COM3"
BAUDRATE         = 9600
 
# ---------- Inicializar serial si se usa sensor real ----------
ser = None
if USAR_SENSOR_REAL:
    import serial
    try:
        ser = serial.Serial(PUERTO_SERIAL, BAUDRATE, timeout=0.1)
        time.sleep(2)
        print(f"BNO055 conectado en {PUERTO_SERIAL}")
    except Exception as e:
        print(f"Error abriendo serial: {e}")
        USAR_SENSOR_REAL = False
 
# ---------- Config ArUco ----------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
params = cv2.aruco.DetectorParameters()
params.adaptiveThreshWinSizeMin        = 3
params.adaptiveThreshWinSizeMax        = 23
params.adaptiveThreshWinSizeStep       = 10
params.adaptiveThreshConstant          = 7
params.minMarkerPerimeterRate          = 0.03
params.maxMarkerPerimeterRate          = 4.0
params.polygonalApproxAccuracyRate     = 0.05
params.cornerRefinementMethod          = cv2.aruco.CORNER_REFINE_SUBPIX
params.cornerRefinementWinSize         = 5
params.cornerRefinementMaxIterations   = 30
params.cornerRefinementMinAccuracy     = 0.01
detector = cv2.aruco.ArucoDetector(aruco_dict, params)
 
# ---------- Cámara ----------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    print("No se pudo abrir la cámara")
    raise SystemExit
 
required_ids   = {0, 1, 2, 3}
last_centers   = {}
last_seen_time = {}
HOLD_SECONDS   = 2.0
clahe          = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
 
# =============================================================
# EXPLORACIONES
# =============================================================
exploraciones = {
    "abdomen": {
        "hipocondrio_derecho": {
            "roi": np.array([[58,12],[90,12],[90,38],[58,38]], dtype=np.float32),
            "target": np.array([[[74,25]]], dtype=np.float32),
            "orientacion": "vertical",
            "ancho_sonda": 70, "alto_sonda": 20,
            "pitch_ideal": 90, "roll_ideal": 0,
            "tolerancia_pitch": 15, "tolerancia_roll": 10
        },
        "epigastrica": {
            "roi": np.array([[35,12],[65,12],[65,38],[35,38]], dtype=np.float32),
            "target": np.array([[[50,25]]], dtype=np.float32),
            "orientacion": "horizontal",
            "ancho_sonda": 90, "alto_sonda": 25,
            "pitch_ideal": 90, "roll_ideal": 0,
            "tolerancia_pitch": 15, "tolerancia_roll": 10
        },
        "Hipocondrio_izquierdo": {
            "roi": np.array([[10,12],[42,12],[42,38],[10,38]], dtype=np.float32),
            "target": np.array([[[26,25]]], dtype=np.float32),
            "orientacion": "vertical",
            "ancho_sonda": 70, "alto_sonda": 20,
            "pitch_ideal": 90, "roll_ideal": 0,
            "tolerancia_pitch": 15, "tolerancia_roll": 10
        },
        "flanco_derecho": {
            "roi": np.array([[60,35],[88,35],[88,62],[60,62]], dtype=np.float32),
            "target": np.array([[[74,48]]], dtype=np.float32),
            "orientacion": "vertical",
            "ancho_sonda": 75, "alto_sonda": 22,
            "pitch_ideal": 75, "roll_ideal": 0,
            "tolerancia_pitch": 15, "tolerancia_roll": 10
        },
        "flanco_izquierdo": {
            "roi": np.array([[12,35],[40,35],[40,62],[12,62]], dtype=np.float32),
            "target": np.array([[[26,48]]], dtype=np.float32),
            "orientacion": "vertical",
            "ancho_sonda": 75, "alto_sonda": 22,
            "pitch_ideal": 75, "roll_ideal": 0,
            "tolerancia_pitch": 15, "tolerancia_roll": 10
        },
        "Hipogastrio_suprapúbica": {
            "roi": np.array([[35,70],[65,70],[65,94],[35,94]], dtype=np.float32),
            "target": np.array([[[50,82]]], dtype=np.float32),
            "orientacion": "horizontal",
            "ancho_sonda": 80, "alto_sonda": 25,
            "pitch_ideal": 90, "roll_ideal": 0,
            "tolerancia_pitch": 15, "tolerancia_roll": 10
        }
    },
    "tiroides": {
        "transversal": {
            "roi": np.array([[30,35],[70,35],[70,50],[30,50]], dtype=np.float32),
            "target": np.array([[[50,42]]], dtype=np.float32),
            "orientacion": "horizontal",
            "ancho_sonda": 70, "alto_sonda": 18,
            "pitch_ideal": 90, "roll_ideal": 0,
            "tolerancia_pitch": 10, "tolerancia_roll": 8
        },
        "longitudinal_derecha": {
            "roi": np.array([[55,25],[70,25],[70,60],[55,60]], dtype=np.float32),
            "target": np.array([[[62,42]]], dtype=np.float32),
            "orientacion": "vertical",
            "ancho_sonda": 60, "alto_sonda": 18,
            "pitch_ideal": 90, "roll_ideal": 0,
            "tolerancia_pitch": 10, "tolerancia_roll": 8
        },
        "longitudinal_izquierda": {
            "roi": np.array([[30,25],[45,25],[45,60],[30,60]], dtype=np.float32),
            "target": np.array([[[38,42]]], dtype=np.float32),
            "orientacion": "vertical",
            "ancho_sonda": 60, "alto_sonda": 18,
            "pitch_ideal": 90, "roll_ideal": 0,
            "tolerancia_pitch": 10, "tolerancia_roll": 8
        },
        
    }
}
 
# ---------- Estado ----------
categoria_actual    = "abdomen"
lista_exploraciones = list(exploraciones[categoria_actual].keys())
indice_exploracion  = 0
ajuste_manual       = False
offset_x = 0.0; offset_y = 0.0
scale_x  = 1.0; scale_y  = 1.0
MOVE_STEP  = 2.0; SCALE_STEP = 0.05
MIN_SCALE  = 0.4; MAX_SCALE  = 1.8
 
# ---------- Ángulos ----------
pitch_actual = 90.0
roll_actual  = 0.0
 
# =============================================================
# COMUNICACIÓN CON EL SERVIDOR (panel del médico)
# =============================================================
def leer_comando_remoto():
    """Devuelve el siguiente comando pendiente del panel del médico, o None."""
    try:
        r = requests.get(f"{SERVIDOR_URL}/leer_comando", timeout=0.3)
        return r.json().get("comando")
    except:
        return None
 
def publicar_estado(cat, explo, pitch, roll, estado_ang, ajuste, msg_p, msg_r):
    """Envía el estado actual al servidor para que el médico lo vea en tiempo real."""
    try:
        requests.post(f"{SERVIDOR_URL}/actualizar_estado", json={
            "categoria":     cat,
            "exploracion":   explo,
            "pitch_actual":  round(float(pitch), 1),
            "roll_actual":   round(float(roll),  1),
            "estado_angulo": estado_ang,
            "ajuste_manual": ajuste,
            "msg_pitch":     msg_p,
            "msg_roll":      msg_r
        }, timeout=0.3)
    except:
        pass
 
# Publicar estado en hilo separado para no bloquear el bucle de vídeo
_estado_cache = {}
def _hilo_publicar():
    while True:
        if _estado_cache:
            publicar_estado(**_estado_cache)
        time.sleep(0.5)
 
threading.Thread(target=_hilo_publicar, daemon=True).start()
 
# =============================================================
# FUNCIONES DE ÁNGULO
# =============================================================
def leer_angulos_serial():
    global pitch_actual, roll_actual
    if ser and ser.in_waiting:
        try:
            linea = ser.readline().decode("utf-8", errors="ignore").strip()
            partes = linea.replace("Yaw:", "").replace("Pitch:", "").replace("Roll:", "").split()
            if len(partes) >= 3:
                pitch_actual = float(partes[1])
                roll_actual  = float(partes[2])
        except:
            pass
 
def evaluar_angulo(pitch, roll, pitch_ideal, roll_ideal, tol_pitch, tol_roll):
    diff_pitch = pitch - pitch_ideal
    diff_roll  = roll  - roll_ideal
    errores = []; avisos = []
 
    if abs(diff_pitch) <= tol_pitch:
        msg_pitch = f"Pitch {pitch:.0f}deg  OK"
    elif abs(diff_pitch) <= tol_pitch * 2:
        msg_pitch = f"Pitch {pitch:.0f}deg  {'Inclina mas hacia ti' if diff_pitch > 0 else 'Inclina mas hacia el paciente'}"
        avisos.append("pitch")
    else:
        msg_pitch = f"Pitch {pitch:.0f}deg  {'DEMASIADO VERTICAL' if diff_pitch > 0 else 'DEMASIADO INCLINADO'}"
        errores.append("pitch")
 
    if abs(diff_roll) <= tol_roll:
        msg_roll = f"Roll  {roll:.0f}deg  OK"
    elif abs(diff_roll) <= tol_roll * 2:
        msg_roll = f"Roll  {roll:.0f}deg  {'Gira sonda izquierda' if diff_roll > 0 else 'Gira sonda derecha'}"
        avisos.append("roll")
    else:
        msg_roll = f"Roll  {roll:.0f}deg  {'MUY GIRADA IZQ' if diff_roll > 0 else 'MUY GIRADA DER'}"
        errores.append("roll")
 
    if errores:
        estado = "error";   color = (0, 0, 220)
    elif avisos:
        estado = "warning"; color = (0, 180, 255)
    else:
        estado = "ok";      color = (0, 200, 80)
 
    return estado, msg_pitch, msg_roll, color
 
def dibujar_panel_angulo(img, pitch, roll, estado, msg_pitch, msg_roll, color, x0=20, y0=200):
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0+380, y0+130), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.rectangle(img, (x0, y0), (x0+380, y0+130), color, 3)
    titulo = {"ok": "ANGULO CORRECTO  OK", "warning": "AJUSTE LEVE NECESARIO"}.get(estado, "CORREGIR ANGULO!")
    cv2.putText(img, titulo,   (x0+10, y0+25),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.putText(img, msg_pitch,(x0+10, y0+60),  cv2.FONT_HERSHEY_SIMPLEX, 0.6,  (255,255,255), 1)
    cv2.putText(img, msg_roll, (x0+10, y0+90),  cv2.FONT_HERSHEY_SIMPLEX, 0.6,  (255,255,255), 1)
    _dibujar_barra(img, pitch, 0, 180, x0+10, y0+108, 360, 12, color)
 
def _dibujar_barra(img, valor, vmin, vmax, x, y, ancho, alto, color):
    cv2.rectangle(img, (x, y), (x+ancho, y+alto), (80,80,80), -1)
    pct  = np.clip((valor - vmin) / (vmax - vmin), 0, 1)
    fill = int(ancho * pct)
    cv2.rectangle(img, (x, y), (x+fill, y+alto), color, -1)
    cx = x + ancho // 2
    cv2.line(img, (cx, y-3), (cx, y+alto+3), (200,200,200), 1)
 
def dibujar_indicador_sonda(img, pitch, roll, cx, cy, radio=45):
    overlay = img.copy()
    cv2.circle(overlay, (cx, cy), radio, (40,40,40), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
    cv2.circle(img, (cx, cy), radio, (150,150,150), 1)
    ang_rad = math.radians(pitch)
    dx = int(radio * 0.85 * math.cos(math.radians(90) - ang_rad))
    dy = int(radio * 0.85 * math.sin(math.radians(90) - ang_rad))
    cv2.line(img, (cx, cy), (cx+dx, cy-dy), (0,255,180), 3)
    cv2.circle(img, (cx+dx, cy-dy), 5, (0,255,180), -1)
    cv2.putText(img, "SONDA", (cx-25, cy+radio+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)
 
# =============================================================
# ARRANQUE
# =============================================================
print("=" * 55)
print("  SISTEMA DE GUIA DE SONDA ECOGRAFICA")
print("=" * 55)
print("q      : salir")
print("c      : cambiar categoria (abdomen / tiroides)")
print("n / b  : siguiente / anterior exploracion")
print("m      : activar/desactivar ajuste manual ROI")
print("w/a/s/d: mover ROI  |  i/k/j/l: escalar ROI")
print("flechas: simular angulo (sin sensor)")
print("-" * 55)
print("Panel del medico:  http://TU_IP_LOCAL:5000")
print("(Ejecuta primero servidor_sensor.py)")
print("=" * 55)
 
# =============================================================
# BUCLE PRINCIPAL
# =============================================================
while True:
    ok, frame = cap.read()
    if not ok:
        break
 
    proyector = np.zeros_like(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
 
    corners, ids, _ = detector.detectMarkers(gray)
    now = time.time()
 
    if ids is not None:
        ids_flat = ids.flatten()
        cv2.aruco.drawDetectedMarkers(frame, corners, ids_flat)
        for i, marker_id in enumerate(ids_flat):
            marker_id = int(marker_id)
            if marker_id in required_ids:
                pts = corners[i][0]
                c   = pts.mean(axis=0)
                last_centers[marker_id]   = c
                last_seen_time[marker_id] = now
 
    centers = {}
    for mid in required_ids:
        if mid in last_centers and (now - last_seen_time.get(mid, 0)) <= HOLD_SECONDS:
            centers[mid] = last_centers[mid]
 
    nombre_exploracion = lista_exploraciones[indice_exploracion]
    config             = exploraciones[categoria_actual][nombre_exploracion]
 
    if USAR_SENSOR_REAL:
        leer_angulos_serial()
 
    # --- Evaluar ángulo ---
    estado, msg_pitch, msg_roll, color_angulo = evaluar_angulo(
        pitch_actual, roll_actual,
        config["pitch_ideal"], config["roll_ideal"],
        config["tolerancia_pitch"], config["tolerancia_roll"]
    )
 
    # --- Publicar estado para el panel del médico ---
    _estado_cache.update({
        "cat":       categoria_actual,
        "explo":     nombre_exploracion,
        "pitch":     pitch_actual,
        "roll":      roll_actual,
        "estado_ang": estado,
        "ajuste":    ajuste_manual,
        "msg_p":     msg_pitch,
        "msg_r":     msg_roll
    })
 
    # --- Leer comando remoto del médico ---
    comando = leer_comando_remoto()
    if comando:
        if comando.startswith("goto:"):
            # Ir directamente a una exploración por nombre
            destino = comando.split(":", 1)[1]
            # Cambiar categoría si hace falta
            for cat, explos in exploraciones.items():
                if destino in explos:
                    if cat != categoria_actual:
                        categoria_actual    = cat
                        lista_exploraciones = list(exploraciones[categoria_actual].keys())
                    indice_exploracion = lista_exploraciones.index(destino)
                    offset_x = 0.0; offset_y = 0.0
                    scale_x  = 1.0; scale_y  = 1.0
                    break
        else:
            # Simular pulsación de tecla local
            if len(comando) == 1:
                tecla_remota = ord(comando)
                # Navegación
                if tecla_remota == ord("c"):
                    categoria_actual    = "tiroides" if categoria_actual == "abdomen" else "abdomen"
                    lista_exploraciones = list(exploraciones[categoria_actual].keys())
                    indice_exploracion  = 0
                    offset_x = 0.0; offset_y = 0.0; scale_x = 1.0; scale_y = 1.0
                elif tecla_remota == ord("n"):
                    indice_exploracion = (indice_exploracion + 1) % len(lista_exploraciones)
                    offset_x = 0.0; offset_y = 0.0; scale_x = 1.0; scale_y = 1.0
                elif tecla_remota == ord("b"):
                    indice_exploracion = (indice_exploracion - 1) % len(lista_exploraciones)
                    offset_x = 0.0; offset_y = 0.0; scale_x = 1.0; scale_y = 1.0
                elif tecla_remota == ord("m"):
                    ajuste_manual = not ajuste_manual
                # Ajuste ROI remoto
                elif ajuste_manual:
                    if tecla_remota == ord("w"): offset_y -= MOVE_STEP
                    elif tecla_remota == ord("s"): offset_y += MOVE_STEP
                    elif tecla_remota == ord("a"): offset_x -= MOVE_STEP
                    elif tecla_remota == ord("d"): offset_x += MOVE_STEP
                    elif tecla_remota == ord("j"): scale_x = max(MIN_SCALE, scale_x - SCALE_STEP)
                    elif tecla_remota == ord("l"): scale_x = min(MAX_SCALE, scale_x + SCALE_STEP)
                    elif tecla_remota == ord("i"): scale_y = min(MAX_SCALE, scale_y + SCALE_STEP)
                    elif tecla_remota == ord("k"): scale_y = max(MIN_SCALE, scale_y - SCALE_STEP)
 
    # --- Recalcular nombre después de posibles cambios remotos ---
    nombre_exploracion = lista_exploraciones[indice_exploracion]
    config             = exploraciones[categoria_actual][nombre_exploracion]
 
    # ── Dibujar escena ───────────────────────────────────────────────
    if required_ids.issubset(centers.keys()):
        pts      = np.array([centers[i] for i in [0,1,2,3]], dtype=np.float32)
        idx_by_y = np.argsort(pts[:, 1])
        top      = pts[idx_by_y[:2]]; bottom = pts[idx_by_y[2:]]
        top      = top[np.argsort(top[:, 0])]; bottom = bottom[np.argsort(bottom[:, 0])]
        tl, tr   = top[0], top[1]; bl, br = bottom[0], bottom[1]
        image_pts   = np.array([tl, tr, br, bl], dtype=np.float32)
        virtual_pts = np.array([[0,0],[100,0],[100,100],[0,100]], dtype=np.float32)
        H, _        = cv2.findHomography(virtual_pts, image_pts)
 
        roi_base    = config["roi"].copy()
        target_base = config["target"].copy()
        centro_roi  = np.mean(roi_base, axis=0)
        roi_ajustada = (roi_base - centro_roi) * np.array([scale_x, scale_y]) + centro_roi
        roi_ajustada[:, 0] += offset_x; roi_ajustada[:, 1] += offset_y
        target_ajustada = target_base.copy()
        target_ajustada[:,:,0] += offset_x; target_ajustada[:,:,1] += offset_y
        roi_ajustada[:, 0]     = np.clip(roi_ajustada[:, 0], 0, 100)
        roi_ajustada[:, 1]     = np.clip(roi_ajustada[:, 1], 0, 100)
        target_ajustada[:,:,0] = np.clip(target_ajustada[:,:,0], 0, 100)
        target_ajustada[:,:,1] = np.clip(target_ajustada[:,:,1], 0, 100)
 
        roi_virtual    = roi_ajustada.reshape(-1,1,2).astype(np.float32)
        target_virtual = target_ajustada.astype(np.float32)
        roi_real       = cv2.perspectiveTransform(roi_virtual, H)
        target_real    = cv2.perspectiveTransform(target_virtual, H)
        tx, ty         = target_real[0][0]
 
        cv2.polylines(frame,     [image_pts.astype(int)], True, (0,255,0),   2)
        cv2.polylines(frame,     [roi_real.astype(int)],  True, (0,255,255), 2)
        cv2.polylines(proyector, [roi_real.astype(int)],  True, (0,255,255), 3)
 
        centro_x    = int(tx); centro_y = int(ty)
        orientacion = config["orientacion"]
        ancho_sonda = config["ancho_sonda"]; alto_sonda = config["alto_sonda"]
 
        if orientacion == "horizontal":
            x1=centro_x-ancho_sonda//2; y1=centro_y-alto_sonda//2
            x2=centro_x+ancho_sonda//2; y2=centro_y+alto_sonda//2
            cv2.rectangle(frame,     (x1,y1),(x2,y2),(255,0,0),2)
            cv2.line(frame,          (x1,centro_y),(x2,centro_y),(255,0,0),2)
            cv2.rectangle(proyector, (x1,y1),(x2,y2),(255,0,0),3)
            cv2.line(proyector,      (x1,centro_y),(x2,centro_y),(255,0,0),3)
        else:
            x1=centro_x-alto_sonda//2; y1=centro_y-ancho_sonda//2
            x2=centro_x+alto_sonda//2; y2=centro_y+ancho_sonda//2
            cv2.rectangle(frame,     (x1,y1),(x2,y2),(255,0,0),2)
            cv2.line(frame,          (centro_x,y1),(centro_x,y2),(255,0,0),2)
            cv2.rectangle(proyector, (x1,y1),(x2,y2),(255,0,0),3)
            cv2.line(proyector,      (centro_x,y1),(centro_x,y2),(255,0,0),3)
 
        cv2.putText(frame, "Huella objetivo", (centro_x-60, centro_y-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        cv2.putText(proyector, "OBJETIVO", (int(tx)+15, int(ty)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
 
        missing = [mid for mid in required_ids if mid not in (ids.flatten().tolist() if ids is not None else [])]
        if missing:
            cv2.putText(frame, f"Usando HOLD: {missing}", (20,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
 
        dibujar_indicador_sonda(frame,     pitch_actual, roll_actual, frame.shape[1]-80,     280)
        dibujar_indicador_sonda(proyector, pitch_actual, roll_actual, proyector.shape[1]-80, 280)
 
    else:
        faltan = [mid for mid in required_ids if mid not in centers]
        cv2.putText(frame, f"Faltan IDs: {faltan}", (20,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        # Mensaje de instruccion para el operador
        if categoria_actual == "abdomen":
            msg_instruccion = "Coloca los 4 marcadores en las esquinas del abdomen"
        else:
            msg_instruccion = "Coloca los 4 marcadores alrededor del cuello"
        overlay = frame.copy()
        h_frame, w_frame = frame.shape[:2]
        cv2.rectangle(overlay, (0, h_frame//2 - 50), (w_frame, h_frame//2 + 50), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, msg_instruccion,
                    (w_frame//2 - len(msg_instruccion)*7, h_frame//2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
 
    dibujar_panel_angulo(frame,     pitch_actual, roll_actual, estado, msg_pitch, msg_roll, color_angulo, x0=20, y0=340)
    dibujar_panel_angulo(proyector, pitch_actual, roll_actual, estado, msg_pitch, msg_roll, color_angulo, x0=20, y0=340)
 
    estado_manual = "ON" if ajuste_manual else "OFF"
    modo_txt      = "SENSOR REAL" if USAR_SENSOR_REAL else "PANEL WEB + SIMULACION"
 
    for img, es_proy in [(frame, False), (proyector, True)]:
        prefijo = "CATEGORIA" if es_proy else "Categoria"
        cv2.putText(img, f"{prefijo}: {categoria_actual}",   (20, 65),  cv2.FONT_HERSHEY_SIMPLEX, 0.8,  (255,255,255), 2)
        cv2.putText(img, f"Exploracion: {nombre_exploracion}",(20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8,  (255,255,255), 2)
        cv2.putText(img, f"Angulo ideal: Pitch {config['pitch_ideal']}deg  Roll {config['roll_ideal']}deg",
                    (20,135), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)
        if not es_proy:
            cv2.putText(img, f"Ajuste medico: {estado_manual} | Modo: {modo_txt}",
                        (20,168), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            cv2.putText(img, "c cat | n/b explo | m ajuste | flechas angulo | q salir",
                        (20,192), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150,150,150), 1)
 
    cv2.imshow("Camara - Guia Sonda", frame)
    cv2.imshow("Proyector simulado",  proyector)
 
    key = cv2.waitKey(1) & 0xFF
 
    if key == ord("q"):
        break
 
    # ── Teclas locales (teclado físico) ─────────────────────────────
    elif key == ord("c"):
        categoria_actual    = "tiroides" if categoria_actual == "abdomen" else "abdomen"
        lista_exploraciones = list(exploraciones[categoria_actual].keys())
        indice_exploracion  = 0
        offset_x = 0.0; offset_y = 0.0; scale_x = 1.0; scale_y = 1.0
    elif key == ord("n"):
        indice_exploracion = (indice_exploracion + 1) % len(lista_exploraciones)
        offset_x = 0.0; offset_y = 0.0; scale_x = 1.0; scale_y = 1.0
    elif key == ord("b"):
        indice_exploracion = (indice_exploracion - 1) % len(lista_exploraciones)
        offset_x = 0.0; offset_y = 0.0; scale_x = 1.0; scale_y = 1.0
    elif key == ord("m"):
        ajuste_manual = not ajuste_manual
 
    # Flechas para simular ángulo (sin sensor)
    elif not USAR_SENSOR_REAL:
        if   key == 82: pitch_actual = min(180, pitch_actual + 2)
        elif key == 84: pitch_actual = max(0,   pitch_actual - 2)
        elif key == 81: roll_actual  = max(-90,  roll_actual - 2)
        elif key == 83: roll_actual  = min(90,   roll_actual + 2)
 
    if ajuste_manual:
        if   key == ord("w"): offset_y -= MOVE_STEP
        elif key == ord("s"): offset_y += MOVE_STEP
        elif key == ord("a"): offset_x -= MOVE_STEP
        elif key == ord("d"): offset_x += MOVE_STEP
        elif key == ord("j"): scale_x = max(MIN_SCALE, scale_x - SCALE_STEP)
        elif key == ord("l"): scale_x = min(MAX_SCALE, scale_x + SCALE_STEP)
        elif key == ord("i"): scale_y = min(MAX_SCALE, scale_y + SCALE_STEP)
        elif key == ord("k"): scale_y = max(MIN_SCALE, scale_y - SCALE_STEP)
 
# ── Limpieza ─────────────────────────────────────────────────────────
cap.release()
if ser:
    ser.close()
cv2.destroyAllWindows()
 