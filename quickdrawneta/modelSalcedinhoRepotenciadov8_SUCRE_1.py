import os
import glob
import numpy as np
import tensorflow as tf
from keras import layers, models
from keras._tf_keras.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from random import randint

def load_data(root, iteracionNumero = 0, vfold_ratio=0.2, max_items_per_class= 250 ):
    all_files = glob.glob(os.path.join(root, '*.npy'))

    #initialize variables 
    x = np.empty([0, 784])
    y = np.empty([0])
    class_names = []

    #load each data file 
    for idx, file in enumerate(all_files):
        data = np.load(file)
        data = data[iteracionNumero*max_items_per_class : (iteracionNumero+1)*max_items_per_class, :]
        labels = np.full(data.shape[0], idx)

        x = np.concatenate((x, data), axis=0)
        y = np.append(y, labels)

        class_name, ext = os.path.splitext(os.path.basename(file))
        class_names.append(class_name)

    data = None
    labels = None
    
    #randomize the dataset 
    permutation = np.random.permutation(y.shape[0])
    x = x[permutation, :]
    y = y[permutation]

    #separate into training and testing 
    vfold_size = int(x.shape[0]/100*(vfold_ratio*100))

    x_test = x[0:vfold_size, :]
    y_test = y[0:vfold_size]

    x_train = x[vfold_size:x.shape[0], :]
    y_train = y[vfold_size:y.shape[0]]
    return x_train, y_train, x_test, y_test, class_names

# get train datas (x_train, y_train, x_test, y_test, class_names)
def get_train_datas(iteracionNumero):
    x_train, y_train, x_test, y_test, class_names = load_data('./dataset', iteracionNumero)
    num_classes = len(class_names)
    image_size = 28

    idx = randint(0, len(x_train))
    #plt.imshow(x_train[idx].reshape(28,28)) 
    #print(class_names[int(y_train[idx].item())])

    # Reshape and normalize
    x_train = x_train.reshape(x_train.shape[0], image_size, image_size, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], image_size, image_size, 1).astype('float32')

    x_train /= 255.0
    x_test /= 255.0


    # Convert class vectors to class matrices
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test, class_names, num_classes

# Define model
def create_model(x_train, y_train, x_test, y_test, class_names, num_classes):
    model = models.Sequential()
    model.add(layers.Convolution2D(16, (3, 3),
                            padding='same',
                            input_shape=x_train.shape[1:], activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Convolution2D(32, (3, 3), padding='same', activation= 'relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Convolution2D(64, (3, 3), padding='same', activation= 'relu'))
    model.add(layers.MaxPooling2D(pool_size =(2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax')) 
    return model

# Get started for train the model
x_train, y_train, x_test, y_test, class_names, num_classes = get_train_datas(21)

adam = tf.optimizers.Adam()
#model = create_model(x_train, y_train, x_test, y_test, class_names, num_classes)
model = load_model('model/input_model.h5')
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['top_k_categorical_accuracy'])
print(model.summary())

# Train model
model.fit(x = x_train, y = y_train, validation_split=0.1, batch_size = 50, verbose=2, epochs=4)

# Delete from memory x_train, y_train, x_test, y_test, class_names and num_classes
del x_train, y_train, x_test, y_test, class_names, num_classes

# Re-train the model for 10 batches
_finalBatch = 0

for numberOfBatch in range(22, 40):
    x_train, y_train, x_test, y_test, class_names, num_classes = get_train_datas(numberOfBatch)
    print('Se esta corriendo el BATCH numero ', numberOfBatch)
    model.fit(x=x_train, y=y_train, validation_split=0.1, batch_size=50, verbose=2, epochs=4)
    del x_train, y_train, x_test, y_test, class_names, num_classes
    model.save('quickdrawSalcedinhoREPOTENCIAO_' + str(numberOfBatch) + '.h5')
    _finalBatch = numberOfBatch

# Evaluate the model
x_train, y_train, x_test, y_test, class_names, num_classes = get_train_datas(_finalBatch + 1)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test accuarcy: {:0.2f}%'.format(score[1] * 100))

idx = randint(0, len(x_test))
img = x_test[idx]
plt.imshow(img.squeeze()) 

pred = model.predict(np.expand_dims(img, axis=0))[0]
ind = (-pred).argsort()[:5]
latex = [class_names[x] for x in ind]
print(latex)

with open('class_names.txt', 'w') as file_handler:
    for item in class_names:
        file_handler.write("{}\n".format(item))

model.save('quickdrawSalcedinho2.h5')