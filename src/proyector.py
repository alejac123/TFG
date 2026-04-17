import cv2
import numpy as np
import time

# ---------- Config ArUco ----------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

params = cv2.aruco.DetectorParameters()
params.adaptiveThreshWinSizeMin = 3
params.adaptiveThreshWinSizeMax = 23
params.adaptiveThreshWinSizeStep = 10
params.adaptiveThreshConstant = 7
params.minMarkerPerimeterRate = 0.03
params.maxMarkerPerimeterRate = 4.0
params.polygonalApproxAccuracyRate = 0.05
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
params.cornerRefinementWinSize = 5
params.cornerRefinementMaxIterations = 30
params.cornerRefinementMinAccuracy = 0.01

detector = cv2.aruco.ArucoDetector(aruco_dict, params)

# ---------- Cámara ----------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("No se pudo abrir la cámara")
    raise SystemExit

required_ids = {0, 1, 2, 3}

# ---------- Memoria ----------
last_centers = {}
last_seen_time = {}
HOLD_SECONDS = 0.6

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ============================================================
# EXPLORACIONES ORGANIZADAS POR CATEGORÍA
# ============================================================
exploraciones = {
    "abdomen": {
        "hepatica": {
            "roi": np.array([[58, 12], [90, 12], [90, 38], [58, 38]], dtype=np.float32),
            "target": np.array([[[74, 25]]], dtype=np.float32),
            "orientacion": "vertical",
            "ancho_sonda": 70,
            "alto_sonda": 20
        },
        "epigastrica": {
            "roi": np.array([[35, 12], [65, 12], [65, 38], [35, 38]], dtype=np.float32),
            "target": np.array([[[50, 25]]], dtype=np.float32),
            "orientacion": "horizontal",
            "ancho_sonda": 90,
            "alto_sonda": 25
        },
        "esplenica": {
            "roi": np.array([[10, 12], [42, 12], [42, 38], [10, 38]], dtype=np.float32),
            "target": np.array([[[26, 25]]], dtype=np.float32),
            "orientacion": "vertical",
            "ancho_sonda": 70,
            "alto_sonda": 20
        },
        "flanco_derecho": {
            "roi": np.array([[60, 35], [88, 35], [88, 62], [60, 62]], dtype=np.float32),
            "target": np.array([[[74, 48]]], dtype=np.float32),
            "orientacion": "vertical",
            "ancho_sonda": 75,
            "alto_sonda": 22
        },
        "flanco_izquierdo": {
            "roi": np.array([[12, 35], [40, 35], [40, 62], [12, 62]], dtype=np.float32),
            "target": np.array([[[26, 48]]], dtype=np.float32),
            "orientacion": "vertical",
            "ancho_sonda": 75,
            "alto_sonda": 22
        },
        "suprapubica": {
            "roi": np.array([[35, 70], [65, 70], [65, 94], [35, 94]], dtype=np.float32),
            "target": np.array([[[50, 82]]], dtype=np.float32),
            "orientacion": "horizontal",
            "ancho_sonda": 80,
            "alto_sonda": 25
        }
    },
    "tiroides": {
        "transversal": {
            "roi": np.array([[30, 35], [70, 35], [70, 50], [30, 50]], dtype=np.float32),
            "target": np.array([[[50, 42]]], dtype=np.float32),
            "orientacion": "horizontal",
            "ancho_sonda": 70,
            "alto_sonda": 18
        },
        "longitudinal_derecha": {
            "roi": np.array([[55, 25], [70, 25], [70, 60], [55, 60]], dtype=np.float32),
            "target": np.array([[[62, 42]]], dtype=np.float32),
            "orientacion": "vertical",
            "ancho_sonda": 60,
            "alto_sonda": 18
        },
        "longitudinal_izquierda": {
            "roi": np.array([[30, 25], [45, 25], [45, 60], [30, 60]], dtype=np.float32),
            "target": np.array([[[38, 42]]], dtype=np.float32),
            "orientacion": "vertical",
            "ancho_sonda": 60,
            "alto_sonda": 18
        }
    }
}

# ---------- Estado actual ----------
categoria_actual = "abdomen"
lista_exploraciones = list(exploraciones[categoria_actual].keys())
indice_exploracion = 0

# ---------- Ajuste manual del médico ----------
ajuste_manual = False
offset_x = 0.0
offset_y = 0.0
scale_x = 1.0
scale_y = 1.0

MOVE_STEP = 2.0
SCALE_STEP = 0.05
MIN_SCALE = 0.4
MAX_SCALE = 1.8

print("Pulsa q para salir")
print("c: cambiar categoria (abdomen / tiroides)")
print("n: siguiente exploracion")
print("b: exploracion anterior")
print("m: activar/desactivar ajuste manual")
print("Mover ROI: w/a/s/d")
print("Tamano ROI: i/k/j/l")

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
                c = pts.mean(axis=0)
                last_centers[marker_id] = c
                last_seen_time[marker_id] = now

    centers = {}
    for mid in required_ids:
        if mid in last_centers and (now - last_seen_time.get(mid, 0)) <= HOLD_SECONDS:
            centers[mid] = last_centers[mid]

    nombre_exploracion = lista_exploraciones[indice_exploracion]
    config = exploraciones[categoria_actual][nombre_exploracion]

    if required_ids.issubset(centers.keys()):
        pts = np.array([centers[i] for i in [0, 1, 2, 3]], dtype=np.float32)

        idx_by_y = np.argsort(pts[:, 1])
        top = pts[idx_by_y[:2]]
        bottom = pts[idx_by_y[2:]]

        top = top[np.argsort(top[:, 0])]
        bottom = bottom[np.argsort(bottom[:, 0])]

        tl, tr = top[0], top[1]
        bl, br = bottom[0], bottom[1]

        image_pts = np.array([tl, tr, br, bl], dtype=np.float32)
        virtual_pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)

        H, _ = cv2.findHomography(virtual_pts, image_pts)

        # ---------- ROI base ----------
        roi_base = config["roi"].copy()
        target_base = config["target"].copy()

        # Centro ROI base
        centro_roi = np.mean(roi_base, axis=0)

        # Escalado respecto al centro
        roi_ajustada = (roi_base - centro_roi) * np.array([scale_x, scale_y]) + centro_roi

        # Desplazamiento manual
        roi_ajustada[:, 0] += offset_x
        roi_ajustada[:, 1] += offset_y

        target_ajustada = target_base.copy()
        target_ajustada[:, :, 0] += offset_x
        target_ajustada[:, :, 1] += offset_y

        # Limitar al espacio virtual
        roi_ajustada[:, 0] = np.clip(roi_ajustada[:, 0], 0, 100)
        roi_ajustada[:, 1] = np.clip(roi_ajustada[:, 1], 0, 100)
        target_ajustada[:, :, 0] = np.clip(target_ajustada[:, :, 0], 0, 100)
        target_ajustada[:, :, 1] = np.clip(target_ajustada[:, :, 1], 0, 100)

        roi_virtual = roi_ajustada.reshape(-1, 1, 2).astype(np.float32)
        target_virtual = target_ajustada.astype(np.float32)

        roi_real = cv2.perspectiveTransform(roi_virtual, H)
        target_real = cv2.perspectiveTransform(target_virtual, H)

        tx, ty = target_real[0][0]

        # Dibujos base en cámara
        cv2.polylines(frame, [image_pts.astype(int)], True, (0, 255, 0), 2)
        cv2.polylines(frame, [roi_real.astype(int)], True, (0, 255, 255), 2)

        # ---------- HUELLA ----------
        centro_x = int(tx)
        centro_y = int(ty)

        orientacion = config["orientacion"]
        ancho_sonda = config["ancho_sonda"]
        alto_sonda = config["alto_sonda"]

        if orientacion == "horizontal":
            x1 = centro_x - ancho_sonda // 2
            y1 = centro_y - alto_sonda // 2
            x2 = centro_x + ancho_sonda // 2
            y2 = centro_y + alto_sonda // 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.line(frame, (x1, centro_y), (x2, centro_y), (255, 0, 0), 2)

        elif orientacion == "vertical":
            x1 = centro_x - alto_sonda // 2
            y1 = centro_y - ancho_sonda // 2
            x2 = centro_x + alto_sonda // 2
            y2 = centro_y + ancho_sonda // 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.line(frame, (centro_x, y1), (centro_x, y2), (255, 0, 0), 2)

        cv2.putText(
            frame,
            "Huella objetivo",
            (centro_x - 60, centro_y - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )

        # ---------- Proyector ----------
        cv2.polylines(proyector, [roi_real.astype(int)], True, (0, 255, 255), 3)

        if orientacion == "horizontal":
            cv2.rectangle(proyector, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.line(proyector, (x1, centro_y), (x2, centro_y), (255, 0, 0), 3)

        elif orientacion == "vertical":
            cv2.rectangle(proyector, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.line(proyector, (centro_x, y1), (centro_x, y2), (255, 0, 0), 3)

        cv2.putText(
            proyector,
            "OBJETIVO",
            (int(tx) + 15, int(ty)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        missing = [mid for mid in required_ids if mid not in (ids.flatten().tolist() if ids is not None else [])]
        if missing:
            cv2.putText(
                frame,
                f"Usando HOLD para: {missing}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

    else:
        faltan = [mid for mid in required_ids if mid not in centers]
        cv2.putText(
            frame,
            f"Faltan IDs: {faltan}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    estado_manual = "ON" if ajuste_manual else "OFF"

    cv2.putText(
        frame,
        f"Categoria: {categoria_actual}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Exploracion: {nombre_exploracion}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Ajuste medico: {estado_manual}",
        (20, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "c categoria | n siguiente | b anterior | m ajuste",
        (20, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1
    )

    cv2.putText(
        proyector,
        f"CATEGORIA: {categoria_actual}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        proyector,
        f"EXPLORACION: {nombre_exploracion}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        proyector,
        f"AJUSTE MEDICO: {estado_manual}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow("Camara", frame)
    cv2.imshow("Proyector simulado", proyector)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("c"):
        if categoria_actual == "abdomen":
            categoria_actual = "tiroides"
        else:
            categoria_actual = "abdomen"

        lista_exploraciones = list(exploraciones[categoria_actual].keys())
        indice_exploracion = 0
        offset_x = 0.0
        offset_y = 0.0
        scale_x = 1.0
        scale_y = 1.0

    elif key == ord("n"):
        indice_exploracion = (indice_exploracion + 1) % len(lista_exploraciones)
        offset_x = 0.0
        offset_y = 0.0
        scale_x = 1.0
        scale_y = 1.0

    elif key == ord("b"):
        indice_exploracion = (indice_exploracion - 1) % len(lista_exploraciones)
        offset_x = 0.0
        offset_y = 0.0
        scale_x = 1.0
        scale_y = 1.0

    elif key == ord("m"):
        ajuste_manual = not ajuste_manual

    elif ajuste_manual:
        if key == ord("w"):
            offset_y -= MOVE_STEP
        elif key == ord("s"):
            offset_y += MOVE_STEP
        elif key == ord("a"):
            offset_x -= MOVE_STEP
        elif key == ord("d"):
            offset_x += MOVE_STEP
        elif key == ord("j"):
            scale_x = max(MIN_SCALE, scale_x - SCALE_STEP)
        elif key == ord("l"):
            scale_x = min(MAX_SCALE, scale_x + SCALE_STEP)
        elif key == ord("i"):
            scale_y = min(MAX_SCALE, scale_y + SCALE_STEP)
        elif key == ord("k"):
            scale_y = max(MIN_SCALE, scale_y - SCALE_STEP)

cap.release()
cv2.destroyAllWindows()