import cv2

# Diccionario ArUco (estándar y robusto)
aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

marker_size = 200  # píxeles (para imprimir grande)
marker_ids = [0, 1, 2, 3]

for marker_id in marker_ids:
    marker_img = cv2.aruco.generateImageMarker(
        aruco_dict,
        marker_id,
        marker_size
    )
    cv2.imwrite(f"aruco_{marker_id}.png", marker_img)

print("Marcadores ArUco generados")

