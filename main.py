import tkinter as tk
from keras.models import load_model
import numpy as np
from collections import deque
import cv2
import random

model = load_model('model/QuickDraw.h5')

# Cargar nombres de clases desde el archivo labels.txt
def load_labels(filename):
    with open(filename, 'r') as f:
        labels = f.read().splitlines()
    return labels

categories = load_labels('model/categories.txt')

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

        self.new_category_button = tk.Button(root, text="Nueva Categoría", command=self.set_random_category)
        self.new_category_button.pack()

        self.points = 0
        self.points_label = tk.Label(root, text=f"Puntos: {self.points}", font=("Helvetica", 16), fg='green')
        self.points_label.pack()

        self.timer_label = tk.Label(root, text="Tiempo: 50", font=("Helvetica", 16), fg='red')
        self.timer_label.pack()

        self.used_categories = []  # Lista para almacenar categorías usadas

        self.drawing = False

        self.set_random_category()
        self.start_timer(50)  # Iniciar el contador de tiempo con 50 segundos

    def start_timer(self, seconds):
        self.remaining_time = seconds
        self.update_timer()

    def update_timer(self):
        self.timer_label.config(text=f"Tiempo: {self.remaining_time}")
        if self.remaining_time > 0:
            self.remaining_time -= 1
            self.root.after(1000, self.update_timer)
        else:
            self.timer_label.config(text="Tiempo: 0")
            self.clear_and_set_new_category()
            self.points -= 1
            self.points_label.config(text=f"Puntos: {self.points}")
            self.canvas.create_text(320, 240, text="TIEMPO!", fill="white", font=("Helvetica", 36))
            self.root.after(3000, self.clear)
            self.stop_timer()
            self.start_timer(50)

    def set_random_category(self):
        # Obtener una categoría aleatoria que no haya sido usada antes
        while True:
            self.current_category = random.choice(categories)
            if self.current_category not in self.used_categories:
                self.used_categories.append(self.current_category)
                break
        
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
                        self.root.after(3000, self.clear_and_set_new_category)
                        self.start_timer(50)

    def stop_timer(self):
        self.root.after_cancel(self.update_timer)
    
    def clear_and_set_new_category(self):
        self.clear()
        self.set_random_category()

    def clear(self):
        self.pts = deque(maxlen=512)
        self.blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        self.canvas.delete("all")
        self.label.config(text="")

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
