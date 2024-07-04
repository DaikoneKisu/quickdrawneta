import tensorflow as tf
from keras import layers, models
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Definir la ruta del dataset
dataset_path = 'datasets'

# Inicializar listas para las imágenes y las etiquetas
images = []
labels = []

# Leer todos los archivos .npy en el directorio del dataset
for filename in os.listdir(dataset_path):
    if filename.endswith('.npy'):
        file_path = os.path.join(dataset_path, filename)
        # Cargar el archivo .npy
        data = np.load(file_path)
        
        # Asumir que cada archivo tiene imágenes con forma (n_samples, 784)
        # La etiqueta se puede deducir del nombre del archivo
        label = filename.split('_')[-1].split('.')[0]  # Extraer la etiqueta del nombre del archivo
        label = label.replace('full_numpy_bitmap_', '')  # Limpiar la etiqueta
        
        # Repetir la etiqueta para cada imagen en el archivo
        labels.extend([label] * data.shape[0])
        
        # Añadir las imágenes a la lista
        images.append(data)

# Convertir las listas a arrays de numpy
images = np.concatenate(images, axis=0)
labels = np.array(labels)

print(f"Total images: {images.shape}")
print(f"Total labels: {labels.shape}")

# Convertir las etiquetas a números
unique_labels = np.unique(labels)
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
labels = np.array([label_to_index[label] for label in labels])

# Normalizar las imágenes
images = images / 255.0

# Asegurarse de que las imágenes tengan la forma correcta
images = images.reshape(-1, 28, 28, 1)  # Ajustar según la forma de las imágenes en tu dataset

# Determinar el número de clases
number_of_classes = len(unique_labels)

# Convertir las etiquetas a categóricas
labels = tf.keras.utils.to_categorical(labels, number_of_classes)

# Dividir los datos en entrenamiento y prueba
train_x, test_x, train_y, test_y = train_test_split(images, labels, test_size=0.2, random_state=42)

# Definir la arquitectura de la red
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(number_of_classes, activation='softmax')  # Ajustar el número de clases
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Añadir early stopping para evitar el sobreajuste
early_stopping = EarlyStopping(patience=3, restore_best_weights=True)

# Entrenar el modelo
history = model.fit(train_x, train_y,
                    epochs=10,
                    batch_size=64,
                    validation_data=(test_x, test_y),
                    callbacks=[early_stopping])

# Dividir los datos nuevamente para el segundo ajuste del modelo
train_x2, test_x2, train_y2, test_y2 = train_test_split(images, labels, test_size=0.15, random_state=42)

q = len(history.history['accuracy'])

plt.figure(figsize=(10,10))
sns.lineplot(x=range(1, 1+q), y=history.history['accuracy'], label='Accuracy')
sns.lineplot(x=range(1, 1+q), y=history.history['val_accuracy'], label='Val_Accuracy')
plt.xlabel('epochs')
plt.ylabel('Accuracy')

history1 = model.fit(train_x2, train_y2, epochs=10, validation_data=(test_x2, test_y2))

q = len(history1.history['accuracy'])

plt.figure(figsize=(10,10))
sns.lineplot(x=range(1, 1+q), y=history1.history['accuracy'], label='Accuracy')
sns.lineplot(x=range(1, 1+q), y=history1.history['val_accuracy'], label='Val_Accuracy')
plt.xlabel('epochs')
plt.ylabel('Accuracy')

# Guardar el modelo entrenado
model.save('pictionary_classifier_best.h5')
