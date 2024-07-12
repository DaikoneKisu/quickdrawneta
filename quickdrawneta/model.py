"""main file of model"""

# @title Import relevant modules
import math
import os
import numpy as np
import pandas as pd
import keras
import matplotlib as plt
from matplotlib import pyplot as plt

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# The following line improves formatting when ouputting NumPy arrays.
np.set_printoptions(linewidth=500)

AMOUNT_OF_CATEGORIES = 0  # Numbers of categories
amount_of_brightness = 255
# @title Hyperparameters
learning_rate = 0.003
batch_size = 64
validation_split = 0.3
train_dataset_path = 'quickdrawneta/train_dataset.npy'
test_dataset_path = 'quickdrawneta/test_dataset.npy'

def create_memmap_file(output_path, shape, dtype=np.uint8):
    """
    Create a memory-mapped file with the specified shape and data type.
    """
    return np.memmap(output_path, dtype=dtype, mode='w+', shape=shape)

def process_and_save_dataset():
    """
    Process all .npy files in the given dataset_path, concatenate them,
    shuffle, normalize by dividing by 255, and save into a single .npy file.
    """
    dataset_path = 'quickdrawneta/datasets'
    train_memmap_path = 'quickdrawneta/temp_train_data.dat'
    test_memmap_path = 'quickdrawneta/temp_test_data.dat'
    train_output_file = 'quickdrawneta/train_dataset.npy'
    test_output_file = 'quickdrawneta/test_dataset.npy'

    # Placeholder for collecting dataset shapes to calculate the total size
    shapes = []
    for filename in os.listdir(dataset_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(dataset_path, filename)
            data = np.load(file_path, mmap_mode='r')  # Load data in read-only mode
            shapes.append(data.shape)
    
    train_samples = sum(math.floor(shape[0] * 0.8) for shape in shapes)
    test_samples =  sum(math.floor(shape[0] * 0.2) for shape in shapes)

    feature_size = shapes[0][1]  # Assuming all files have the same feature size

    dtype = np.float16  # Assuming all data has this dtype

    # Create memory-mapped files for train and test datasets
    train_data = create_memmap_file(train_memmap_path, (train_samples, feature_size + 1), dtype=dtype)
    test_data = create_memmap_file(test_memmap_path, (test_samples, feature_size + 1), dtype=dtype)

    # Fill the memory-mapped file with data
    index_for_class = 0
    current_train_index = 0
    current_test_index = 0
    for filename in os.listdir(dataset_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(dataset_path, filename)
            print("Processing file: " + file_path)
            data = np.load(file_path, mmap_mode='c')

            labels = []

            # Assume all files have the shape (n_samples, 784)
            total_size = data.shape[0]
            train_size = math.floor(data.shape[0] * 0.8)
            test_size = math.floor(data.shape[0] * 0.2) # 20% test size, not calculated via train_size to avoid rounding errors that result in bigger test size than allocated test_samples

            train_features = data[:train_size, :]
            test_features = data[train_size:train_size + test_size, :]

            train_features = np.concatenate(train_features, axis=0)
            test_features = np.concatenate(test_features, axis=0)
            
            # Extend labels list for each image
            labels.extend([index_for_class] * total_size)
            labels = np.array(labels, dtype=data.dtype)
            labels = labels.reshape(-1, 1)
            
            train_data[current_train_index:current_train_index + train_size, :feature_size] = train_features.reshape(-1, feature_size) / 255.0
            train_data[current_train_index:current_train_index + train_size, feature_size:] = np.array(labels[:train_size], dtype=data.dtype)

            test_data[current_test_index:current_test_index + test_size, :feature_size] = test_features.reshape(-1, feature_size) / 255.0
            test_data[current_test_index:current_test_index + test_size, feature_size:] = np.array(labels[train_size:train_size + test_size], dtype=data.dtype)

            current_train_index += train_size
            current_test_index += test_size
            index_for_class += 1

    # Save the memory-mapped arrays to .npy files
    np.save(train_output_file, train_data)
    np.save(test_output_file, test_data)

    # Close the memory-mapped files and delete temporary files
    del train_data
    del test_data
    os.remove(train_memmap_path)
    os.remove(test_memmap_path)

def train():
    """train the model"""
    # @title Reading the training/test datasets from csv files

    train_features = []
    train_labels = []

    data = np.load(train_dataset_path, mmap_mode='r+')

    np.random.shuffle(data)

    # Split the train array back into train_features and train_labels
    # Assuming the last column of combined array is train_labels and the rest are train_features
    train_features = data[:, :-1].reshape(-1, 28, 28)
    train_labels = data[:, -1]

    # Uniquify the labels
    unique_labels = np.unique(train_labels)

    AMOUNT_OF_CATEGORIES = len(unique_labels)

    # Convert the labels to One-hot encoding labels
    # [1,1,2] -> [[1,0,...],[1,0,...],[0,1...]]
    train_labels = keras.utils.to_categorical(train_labels, AMOUNT_OF_CATEGORIES)

    train_x, train_y = train_features, train_labels

    # @title Define the plotting function
    def plot_curve(epochs, hist, list_of_metrics):
        """Plot a curve of one or more classification metrics vs. epoch."""
        # list_of_metrics should be one of the names shown in:
        # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics

        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Value")

        for m in list_of_metrics:
            x = hist[m]
            plt.plot(epochs[1:], x[1:], label=m)

        plt.legend()
        #plt.show()
        plt.savefig("plot-train.jpg")

        print("Loaded the plot_curve function.")

    # @title Neural Network Architecture definition
    def create_model(input_learning_rate):
        image_pixels = 28

        model = keras.models.Sequential()

        model.add(keras.layers.Conv2D(32, (3,3), activation="relu", padding="SAME", input_shape=(image_pixels, image_pixels, 1)))
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
        model.add(keras.layers.Conv2D(64, (3,3), activation="relu", padding="SAME", input_shape=(image_pixels / 2, image_pixels / 2, 32)))
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
        model.add(keras.layers.Conv2D(128, (4,4), activation="relu", padding="SAME", input_shape=(image_pixels / 4, image_pixels / 4, 64)))
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
        model.add(keras.layers.Dropout(rate=0.25))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=128, activation='relu'))
        model.add(keras.layers.Dropout(rate=0.25))
        model.add(keras.layers.Dense(units=AMOUNT_OF_CATEGORIES, activation='softmax'))

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=input_learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def train_model(
        model,
        train_features,
        train_label,
        epochs,
        batch_size_func=None,
        validation_split_func=0.1,
    ):
        history = model.fit(
            x=train_features,
            y=train_label,
            batch_size=batch_size_func,
            epochs=epochs,
            shuffle=True,
            validation_split=validation_split_func,
        )

        epochs = history.epoch
        hist = pd.DataFrame(history.history)

        return epochs, hist

    # @title Model
    # Establish the model's topography.

    my_model = create_model(learning_rate)

    epochs = 5

    # Train the model on the normalized training set.
    epochs, hist = train_model(
        my_model, train_x, train_y, epochs, batch_size, validation_split
    )

    # Plot a graph of the metric vs. epochs.
    list_of_metrics_to_plot = ["accuracy"]
    plot_curve(epochs, hist, list_of_metrics_to_plot)

    my_model.save("model.keras")

def test():
    """test the model"""

    my_model = keras.saving.load_model("model.keras", compile=True)

    if my_model is None:
        raise Exception("There is no model saved")

    data = np.load(test_dataset_path, mmap_mode='r+')

    test_features = data[:, :-1].reshape(-1, 28, 28)
    test_labels = data[:, -1]

    # See the third image and its label
    plt.imshow(test_features[0])
    plt.show()
    print(test_labels[0])

    unique_labels = np.unique(test_labels)

    AMOUNT_OF_CATEGORIES = len(unique_labels)

    test_labels = keras.utils.to_categorical(test_labels, AMOUNT_OF_CATEGORIES)

    test_x, test_y = test_features, test_labels

    # Evaluate against the test set.
    print("\n Evaluate the new model against the test set:")
    my_model.evaluate(x=test_x, y=test_y, batch_size=batch_size)

def summary():
    """summarize the model"""

    my_model = keras.saving.load_model("model.keras", compile=True)

    if my_model is None:
        raise Exception("There is no model saved")

    print(my_model.summary())