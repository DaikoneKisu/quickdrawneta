import tkinter as tk
from keras._tf_keras.keras.models import load_model
import numpy as np
from collections import deque
import cv2
import random
from datetime import datetime, timedelta

model = load_model('model/quickdrawSalcedinho.h5')

# Cargar nombres de clases desde el archivo labels.txt
def load_labels(filename):
    with open(filename, 'r') as f:
        labels = f.read().splitlines()
    return labels

categories = load_labels('./categories.txt')

INTERVALO_REFRESCO = 500  # En milisegundos
TIEMPO_INICIAL = 50  # En segundos
MAX_RONDAS = 6  # Número máximo de rondas

class QuickDrawApp:
    def __init__(self, root):
        self.root = root
        self.root.title("QuickDraw")
        self.canvas = tk.Canvas(root, width=640, height=480, bg='black')
        self.canvas.pack()
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.stop_paint)

        self.blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        self.pts = deque(maxlen=512)
        self.pred_class = 0

        self.category_label = tk.Label(root, text="", font=("Helvetica", 16), fg='blue')
        self.category_label.pack()

        self.label = tk.Label(root, text="", font=("Helvetica", 16))
        self.label.pack()

        self.clear_button = tk.Button(root, text="Borrar", command=self.clear)
        self.clear_button.pack()

        self.round_label = tk.Label(root, text="Ronda: 1", font=("Helvetica", 16), fg='purple')
        self.round_label.pack()
        
        # self.new_category_button = tk.Button(root, text="Nueva Categoría", command=self.set_random_category)
        # self.new_category_button.pack()

        self.points = 0
        self.points_label = tk.Label(root, text=f"Puntos: {self.points}", font=("Helvetica", 16), fg='green')
        self.points_label.pack()

        self.timer_label = tk.Label(root, text="Tiempo: 50", font=("Helvetica", 16), fg='red')
        self.timer_label.pack()

        self.used_categories = []  # Lista para almacenar categorías usadas
        self.remaining_categories = categories.copy()  # Lista de categorías restantes

        self.rondas = 0  # Contador de rondas
        self.drawing = False

        self.set_random_category()
        self.start_timer(TIEMPO_INICIAL)  # Iniciar el contador de tiempo con 50 segundos

        # Botones de reiniciar y salir, inicialmente deshabilitados
        self.restart_button = tk.Button(root, text="Reiniciar Juego", command=self.restart_game, state=tk.DISABLED)
        self.restart_button.pack(side=tk.LEFT, padx=10)

        self.exit_button = tk.Button(root, text="Salir", command=self.root.quit, state=tk.DISABLED)
        self.exit_button.pack(side=tk.RIGHT, padx=10)

    def start_timer(self, seconds):
        self.hora_final = datetime.now() + timedelta(seconds=seconds)
        self.temporizador_corriendo = True
        self.refrescar_tiempo_restante()

    def obtener_tiempo_restante(self):
        segundos_restantes = (self.hora_final - datetime.now()).total_seconds()
        if segundos_restantes < 0:
            segundos_restantes = 0
        return int(segundos_restantes)

    def refrescar_tiempo_restante(self):
        if self.temporizador_corriendo:
            tiempo_restante = self.obtener_tiempo_restante()
            self.timer_label.config(text=f"Tiempo: {tiempo_restante}")
            if tiempo_restante > 0:
                self.root.after(INTERVALO_REFRESCO, self.refrescar_tiempo_restante)
            else:
                self.timer_label.config(text="Tiempo: 0")
                self.clear()
                # self.points -= 1
                # self.points_label.config(text=f"Puntos: {self.points}")
                self.canvas.create_text(320, 240, text="TIEMPO!", fill="white", font=("Helvetica", 36))
                self.rondas += 1 
                self.root.after(3000, self.clear_and_set_new_category)

    def detener_temporizador(self):
        self.temporizador_corriendo = False

    def set_random_category(self):
        if self.rondas >= MAX_RONDAS:
            self.end_game()
            return

        if not self.remaining_categories:
            self.remaining_categories = categories.copy()  # Reiniciar la lista si se han usado todas las categorías
            self.used_categories = []  # Limpiar la lista de categorías usadas

        self.current_category = random.choice(self.remaining_categories)
        self.remaining_categories.remove(self.current_category)
        self.used_categories.append(self.current_category)
        
        self.round_label.config(text=f"Ronda: {(self.rondas)+1}")
        self.category_label.config(text=f"Dibuja: {self.current_category}")

    def paint(self, event):
        if not self.drawing:
            self.drawing = True
            self.pts = deque(maxlen=512)  # Reset points when starting a new line

        x, y = event.x, event.y
        self.pts.appendleft((x, y))
        self.canvas.create_line(x, y, x+1, y+1, fill='white', width=7)

        for i in range(1, len(self.pts)):
            if self.pts[i - 1] is None or self.pts[i] is None:
                continue
            self.canvas.create_line(self.pts[i-1], self.pts[i], fill='white', width=7)
            cv2.line(self.blackboard, self.pts[i - 1], self.pts[i], (255, 255, 255), 7)

    def stop_paint(self, event):
        self.drawing = False
        if len(self.pts) != 0:
            blackboard_gray = cv2.cvtColor(self.blackboard, cv2.COLOR_BGR2GRAY)
            blur1 = cv2.medianBlur(blackboard_gray, 15)
            blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
            thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            blackboard_cnts = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
            if len(blackboard_cnts) >= 1:
                cnt = max(blackboard_cnts, key=cv2.contourArea)
                if cv2.contourArea(cnt) > 2000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    digit = blackboard_gray[y:y + h, x:x + w]
                    pred_probab, pred_class = keras_predict(model, digit)
                    class_name = categories[pred_class]  # Obtener el nombre de la clase
                    self.label.config(text=f"Predicción: {class_name} ({pred_probab:.2f})")

                    if class_name == self.current_category:
                        self.clear()
                        self.points += 1
                        self.points_label.config(text=f"Puntos: {self.points}")
                        self.canvas.create_text(320, 240, text="FELICIDADES!", fill="white", font=("Helvetica", 36))
                        self.detener_temporizador()
                        self.rondas += 1  # Incrementar el contador de rondas
                        if self.rondas < MAX_RONDAS:
                            self.root.after(3000, self.clear_and_set_new_category)
                        else:
                            self.root.after(3000, self.end_game)

    def clear_and_set_new_category(self):
        self.clear()
        self.set_random_category()
        self.start_timer(TIEMPO_INICIAL)

    def clear(self):
        self.pts = deque(maxlen=512)
        self.blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        self.canvas.delete("all")
        self.label.config(text="")

    def end_game(self):
        self.clear()
        self.detener_temporizador()
        self.canvas.create_text(320, 240, text=f"TU PUNTUACION: {self.points}", fill="white", font=("Helvetica", 36))
        self.restart_button.config(state=tk.NORMAL)  # Habilitar el botón de reiniciar
        self.exit_button.config(state=tk.NORMAL)  # Habilitar el botón de salir

    def restart_game(self):
        self.rondas = 0
        self.points = 0
        self.points_label.config(text=f"Puntos: {self.points}")
        self.clear()
        self.set_random_category()
        self.start_timer(TIEMPO_INICIAL)
        self.round_label.config(text=f"Ronda: {(self.rondas)+1}")
        self.restart_button.config(state=tk.DISABLED)
        self.exit_button.config(state=tk.DISABLED)

def keras_predict(model, image):
    processed = keras_process_image(image)
    print("processed: " + str(processed.shape))
    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class

def keras_process_image(img):
    image_x = 28
    image_y = 28
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img

if __name__ == '__main__':
    root = tk.Tk()
    app = QuickDrawApp(root)
    root.mainloop()
