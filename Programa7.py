import cv2

# Inicia la captura de la cámara
cap = cv2.VideoCapture(0)

# Verifica si la cámara está abierta correctamente
if not cap.isOpened():
    print("Error al abrir la cámara.")
    exit()

while True:
    # Lee el cuadro actual de la cámara
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el cuadro de la cámara.")
        break

    # Procesamiento del cuadro
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
    canny = cv2.Canny(gray, 10, 150)               # Aplicar filtro Canny
    canny = cv2.dilate(canny, None, iterations=1)  # Dilatar bordes
    canny = cv2.erode(canny, None, iterations=1)   # Erosionar bordes

    # Encontrar contornos
    cnts, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Detección de figuras
    for c in cnts:
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)

        if len(approx) == 3:  # Triángulo
            cv2.putText(frame, 'Triangulo', (x, y - 5), 1, 1, (0, 255, 0), 1)

        elif len(approx) == 4:  # Cuadrado o Rectángulo
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                cv2.putText(frame, 'Cuadrado', (x, y - 5), 1, 1, (0, 255, 0), 1)
            else:
                cv2.putText(frame, 'Rectangulo', (x, y - 5), 1, 1, (0, 255, 0), 1)

        elif len(approx) == 5:  # Pentágono
            cv2.putText(frame, 'Pentagono', (x, y - 5), 1, 1, (0, 255, 0), 1)

        elif len(approx) == 6:  # Hexágono
            cv2.putText(frame, 'Hexagono', (x, y - 5), 1, 1, (0, 255, 0), 1)

        elif len(approx) > 10:  # Círculo
            cv2.putText(frame, 'Circulo', (x, y - 5), 1, 1, (0, 255, 0), 1)

        cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)  # Dibuja los contornos

    # Mostrar el cuadro procesado
    cv2.imshow('Deteccion de Figuras en Tiempo Real', frame)

    # Leer tecla presionada
    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'):  # Presiona 'a' para salir
        print("Cerrando cámara...")
        break

# Libera la cámara y destruye las ventanas
cap.release()
cv2.destroyAllWindows()
