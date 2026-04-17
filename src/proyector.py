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

# ---------- Exploraciones ----------
exploraciones = {
    "abdomen": {
        "roi": np.array([[20, 20], [80, 20], [80, 80], [20, 80]], dtype=np.float32),
        "target": np.array([[[50, 50]]], dtype=np.float32)
    },
    "suprapubica": {
        "roi": np.array([[30, 60], [70, 60], [70, 95], [30, 95]], dtype=np.float32),
        "target": np.array([[[50, 78]]], dtype=np.float32)
    },
    "hepatica": {
        "roi": np.array([[20, 10], [85, 10], [85, 50], [20, 50]], dtype=np.float32),
        "target": np.array([[[60, 30]]], dtype=np.float32)
    }
}

modo_actual = "abdomen"

print("Pulsa q para salir | 1 abdomen | 2 suprapubica | 3 hepatica")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Imagen negra para simular proyector
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

        roi_virtual = exploraciones[modo_actual]["roi"].reshape(-1, 1, 2)
        target_virtual = exploraciones[modo_actual]["target"]

        roi_real = cv2.perspectiveTransform(roi_virtual, H)
        target_real = cv2.perspectiveTransform(target_virtual, H)

        tx, ty = target_real[0][0]

        # Dibujos base en cámara
        cv2.polylines(frame, [image_pts.astype(int)], True, (0, 255, 0), 2)
        cv2.polylines(frame, [roi_real.astype(int)], True, (0, 255, 255), 2)

        # ---------- HUELLA DE SONDA ----------
        centro_x = int(tx)
        centro_y = int(ty)

        if modo_actual == "abdomen":
            ancho_sonda = 100
            alto_sonda = 25
            orientacion = "horizontal"

        elif modo_actual == "suprapubica":
            ancho_sonda = 80
            alto_sonda = 25
            orientacion = "horizontal"

        elif modo_actual == "hepatica":
            ancho_sonda = 70
            alto_sonda = 20
            orientacion = "vertical"

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

        # Dibujos en simulación de proyector
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

    cv2.putText(
        frame,
        f"Modo: {modo_actual}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        proyector,
        f"SIMULACION PROYECTOR - {modo_actual}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("Camara", frame)
    cv2.imshow("Proyector simulado", proyector)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("1"):
        modo_actual = "abdomen"
    elif key == ord("2"):
        modo_actual = "suprapubica"
    elif key == ord("3"):
        modo_actual = "hepatica"

cap.release()
cv2.destroyAllWindows()