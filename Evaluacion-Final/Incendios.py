import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
import tkinter as tk
from tkinter import Button, messagebox

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Función para preprocesar las imágenes
def preprocess_image(image, target_size=(64, 64)):
    if isinstance(image, str):  # Si es una ruta, carga la imagen
        if not os.path.exists(image):
            raise FileNotFoundError(f"Ruta de imagen no encontrada: {image}")
        image = cv2.imread(image)
        if image is None:
            raise FileNotFoundError(f"No se pudo leer la imagen en la ruta: {image}")
    else:
        image = image  # Si ya es una matriz (por ejemplo, desde la cámara)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    return image / 255.0  # Normalizar entre 0 y 1

# Generación de datos de ejemplo
def load_dataset(image_paths, labels, target_size=(64, 64)):
    images = [preprocess_image(img, target_size) for img in image_paths]
    return np.array(images), np.array(labels)

# Datos de ejemplo (personaliza según tus imágenes)

image_paths = [
    "C:/Users/it-trainee/Downloads/Evaluacion-Final/Imagen1.png",  
    "C:/Users/it-trainee/Downloads/Evaluacion-Final/Imagen2.jpg",
    "C:/Users/it-trainee/Downloads/Evaluacion-Final/Imagen3.jpeg",  
    "C:/Users/it-trainee/Downloads/Evaluacion-Final/Imagen4.jpeg",
    "C:/Users/it-trainee/Downloads/Evaluacion-Final/Imagen5.jpg",
    "C:/Users/it-trainee/Downloads/Evaluacion-Final/Imagen6.png",
    "C:/Users/it-trainee/Downloads/Evaluacion-Final/Imagen8.jpg",
    "C:/Users/it-trainee/Downloads/Evaluacion-Final/Imagen9.jpg",
    "C:/Users/it-trainee/Downloads/Evaluacion-Final/Imagen10.png",
    "C:/Users/it-trainee/Downloads/Evaluacion-Final/Imagen11.png",
    "C:/Users/it-trainee/Downloads/Evaluacion-Final/Imagen12.jpg",
    "C:/Users/it-trainee/Downloads/Evaluacion-Final/Imagen13.jpg",
    "C:/Users/it-trainee/Downloads/Evaluacion-Final/Imagen14.jpg",
    "C:/Users/it-trainee/Downloads/Evaluacion-Final/Imagen15.jpg",
    "C:/Users/it-trainee/Downloads/Evaluacion-Final/Imagen16.png",
    "C:/Users/it-trainee/Downloads/Evaluacion-Final/Imagen17.png",
    "C:/Users/it-trainee/Downloads/Evaluacion-Final/Imagen18.png",
    "C:/Users/it-trainee/Downloads/Evaluacion-Final/Imagen19.png",
    "C:/Users/it-trainee/Downloads/Evaluacion-Final/Imagen20.png",
    "C:/Users/it-trainee/Downloads/Evaluacion-Final/Imagen21.png",
    "C:/Users/it-trainee/Downloads/Evaluacion-Final/Imagen22.png",
    "C:/Users/it-trainee/Downloads/Evaluacion-Final/Imagen23.png",
    "C:/Users/it-trainee/Downloads/Evaluacion-Final/Imagen24.png",
    "C:/Users/it-trainee/Downloads/Evaluacion-Final/Imagen25.png",
    "C:/Users/it-trainee/Downloads/Evaluacion-Final/Imagen26.png",
    "C:/Users/it-trainee/Downloads/Evaluacion-Final/Imagen27.png",
    "C:/Users/it-trainee/Downloads/Evaluacion-Final/Imagen28.png",
    "C:/Users/it-trainee/Downloads/Evaluacion-Final/Imagen29.png",
    "C:/Users/it-trainee/Downloads/Evaluacion-Final/Imagen30.png"

]
labels = [2, 0, 0, 1,0,0,2,2, 0,1,2,1,2,1,2,2,1,2, 1,2,1,0,0, 2,2,2,1, 1, 0]
try:
    X, y = load_dataset(image_paths, labels)
    y = to_categorical(y, num_classes=3)  # Codificación one-hot
except FileNotFoundError as e:
    print(e)
    exit()

# Dividir datos en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo
model = Sequential([
    Flatten(input_shape=(64, 64, 3)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')  # Tres clases: baja, moderada, alta intensidad
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=16)

# Guardar el modelo entrenado
model.save('fire_intensity_model.h5')

# Usar el modelo para predicciones
def predict_fire_intensity(image, model, target_size=(64, 64)):
    img = preprocess_image(image, target_size)
    img = np.expand_dims(img, axis=0)  # Añadir dimensión para el batch
    predictions = model.predict(img)
    class_idx = np.argmax(predictions)
    return class_idx

# Captura de imagen desde la cámara web y predicción
def capture_and_predict():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        messagebox.showerror("Error", "No se pudo abrir la cámara.")
        return

    print("Presiona 'a' para capturar la imagen.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer el cuadro de la cámara.")
            break

        cv2.imshow('Cámara en tiempo real', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):  # Capturar imagen
            print("Imagen capturada. Procesando...")
            try:
                class_idx = predict_fire_intensity(frame, model)
                labels = ["Baja intensidad", "Moderada intensidad", "Alta intensidad"]
                label = labels[class_idx]
                print(f"Predicción: {label}")
                cv2.putText(frame, f"Predicción: {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Resultado", frame)
                cv2.waitKey(3000)  # Mostrar resultado por 3 segundos
            except Exception as e:
                print(f"Error en la predicción: {e}")
            break

    cap.release()
    cv2.destroyAllWindows()

# Crear una interfaz gráfica para activar la cámara con un botón
def create_gui():
    root = tk.Tk()
    root.title("Clasificación de incendios")

    capture_button = Button(root, text="Capturar imagen desde la cámara", command=capture_and_predict)
    capture_button.pack(pady=20)

    root.mainloop()

# Cargar el modelo entrenado
model = tf.keras.models.load_model('fire_intensity_model.h5')

# Iniciar la interfaz gráfica
create_gui()


