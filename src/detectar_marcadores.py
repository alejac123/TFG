import cv2
import numpy as np
import time

# ---------- Config ArUco ----------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

params = cv2.aruco.DetectorParameters()

# Parámetros más robustos (toleran iluminación peor)
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
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # si tu cámara lo soporta
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("No se pudo abrir la cámara")
    raise SystemExit

required_ids = {0, 1, 2, 3}

# Memoria de últimos centros válidos
last_centers = {}          # id -> np.array([x,y])
last_seen_time = {}        # id -> timestamp
HOLD_SECONDS = 0.6         # cuánto “aguantar” un marcador perdido

# CLAHE para mejorar contraste
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

print("Pulsa 'q' para salir")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # -------- Preprocesado --------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    corners, ids, _ = detector.detectMarkers(gray)

    now = time.time()

    # Actualiza memoria si detecta
    if ids is not None:
        ids_flat = ids.flatten()

        # Dibujar detección sobre frame original
        cv2.aruco.drawDetectedMarkers(frame, corners, ids_flat)

        for i, marker_id in enumerate(ids_flat):
            marker_id = int(marker_id)
            if marker_id in required_ids:
                pts = corners[i][0]               # 4 esquinas
                c = pts.mean(axis=0)              # centro
                last_centers[marker_id] = c
                last_seen_time[marker_id] = now

    # Construye un set de centros “vigentes” (detectados o mantenidos)
    centers = {}
    for mid in required_ids:
        if mid in last_centers and (now - last_seen_time.get(mid, 0)) <= HOLD_SECONDS:
            centers[mid] = last_centers[mid]

    # Si tenemos los 4 (aunque alguno haya sido "hold"), dibujamos ROI
    if required_ids.issubset(centers.keys()):
        pts = np.array([centers[i] for i in [0, 1, 2, 3]], dtype=np.float32)

        # ordenar por y -> top/bottom
        idx_by_y = np.argsort(pts[:, 1])
        top = pts[idx_by_y[:2]]
        bottom = pts[idx_by_y[2:]]

        # ordenar por x dentro de cada grupo
        top = top[np.argsort(top[:, 0])]
        bottom = bottom[np.argsort(bottom[:, 0])]

        tl, tr = top[0], top[1]
        bl, br = bottom[0], bottom[1]

        image_pts = np.array([tl, tr, br, bl], dtype=np.float32)

        # Plano virtual
        virtual_pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)

        H, _ = cv2.findHomography(virtual_pts, image_pts)

        # Centro virtual
        virtual_center = np.array([[[50, 50]]], dtype=np.float32)
        real_center = cv2.perspectiveTransform(virtual_center, H)[0][0]

        # Dibujo
        cv2.polylines(frame, [image_pts.astype(int)], True, (0, 255, 0), 2)
        cv2.circle(frame, (int(real_center[0]), int(real_center[1])), 10, (255, 0, 0), -1)

        # Mostrar estado de “hold”
        missing = [mid for mid in required_ids if mid not in (ids.flatten().tolist() if ids is not None else [])]
        if missing:
            cv2.putText(frame, f"Usando HOLD para: {missing}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    else:
        # feedback de qué falta
        faltan = [mid for mid in required_ids if mid not in centers]
        cv2.putText(frame, f"Faltan IDs: {faltan}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Homografia Abdomen", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
