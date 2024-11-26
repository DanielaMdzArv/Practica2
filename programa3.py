import cv2

captura = cv2.VideoCapture(0)

ret, frame = captura.read()

if ret:
    cv2.imwrite("C:/Users/it-trainee/OneDrive - Nissha Co., Ltd/Escritorio/Practica2/145_22.jpg", frame)

captura.release()