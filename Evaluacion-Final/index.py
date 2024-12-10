import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from flask import Flask, render_template_string, request, jsonify
import base64

# Inicializar Flask
app = Flask(__name__)

# Preprocesamiento de imágenes
def preprocess_image(image, target_size=(64, 64)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    return image / 255.0  # Normalización

# Modelo de ejemplo (ya entrenado previamente)
model = Sequential([
    Flatten(input_shape=(64, 64, 3)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')  # Tres clases
])
model.load_weights('fire_intensity_model.h5')  # Cargar el modelo entrenado

# HTML con el botón para abrir la cámara
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clasificador de Imágenes</title>
    <style>
        body { text-align: center; font-family: Arial, sans-serif; }
        video { width: 100%; max-width: 500px; }
        canvas { display: none; }
    </style>
</head>
<body>
    <h1>Clasificador de Imágenes</h1>
    <video id="camera" autoplay></video>
    <br>
    <button id="capture">Capturar y Clasificar</button>
    <p id="result"></p>
    <canvas id="canvas"></canvas>

    <script>
        const video = document.getElementById('camera');
        const canvas = document.getElementById('canvas');
        const captureButton = document.getElementById('capture');
        const resultText = document.getElementById('result');

        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => { video.srcObject = stream; })
            .catch(err => { console.error("No se pudo acceder a la cámara", err); });

        captureButton.addEventListener('click', () => {
            const ctx = canvas.getContext('2d');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const dataURL = canvas.toDataURL('image/jpeg');
            classifyImage(dataURL);
        });

        function classifyImage(dataURL) {
            fetch('/classify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: dataURL })
            })
            .then(response => response.json())
            .then(data => {
                resultText.textContent = `Resultado: ${data.label}`;
            })
            .catch(err => {
                console.error("Error al clasificar la imagen:", err);
                resultText.textContent = "Error al clasificar la imagen.";
            });
        }
    </script>
</body>
</html>
"""

# Ruta principal para mostrar el HTML
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

# Ruta para clasificar imágenes
@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    image_data = data['image'].split(',')[1]  # Eliminar el encabezado "data:image/jpeg;base64,"
    image_bytes = base64.b64decode(image_data)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    preprocessed_image = preprocess_image(image)

    # Expandir dimensiones para que coincida con el input del modelo
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    predictions = model.predict(preprocessed_image)
    label = np.argmax(predictions, axis=1)[0]
    labels = ["Baja", "Moderada", "Alta"]  # Etiquetas
    return jsonify({'label': labels[label]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
