import cv2
import numpy as np
import math
from keras.models import load_model

# Cargar el modelo previamente entrenado para QuickDraw
model = load_model('model/QuickDraw.h5')

# Función para refinar la imagen
def image_refiner(imagen):
    img_size = 28
    imagen = cv2.resize(imagen, (img_size, img_size))
    return imagen

# Función para cargar etiquetas desde un archivo
def load_labels(labels_path):
    with open(labels_path, 'r') as file:
        labels = file.read().splitlines()
    return labels

# Cargar etiquetas desde labels.txt
labels_path = 'labels.txt'
quickdraw_labels = load_labels(labels_path)

def get_predict_word(path):
    # Lee la imagen en escala de grises desde la ruta especificada
    img = cv2.imread(path, 0)

    # Invertir los colores de la imagen
    img = cv2.bitwise_not(img)

    # Redimensionar y normalizar la imagen
    refined_image = image_refiner(img)
    test_image = refined_image.reshape(-1, 28, 28, 1) / 255.0

    # Realizar la predicción y obtener el índice de la clase con mayor probabilidad
    pred = np.argmax(model.predict(test_image))

    # Convertir la predicción en la etiqueta correspondiente (de acuerdo con las etiquetas cargadas)
    predicted_label = quickdraw_labels[pred] if pred < len(quickdraw_labels) else "Desconocido"

    return predicted_label
