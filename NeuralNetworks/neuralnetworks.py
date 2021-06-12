import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# Importing the Dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# # Visualizing the Dataset.
# print(train_images.shape) #look at the shape
# print(train_images[0,23,23]) #look at one pixel
# print(train_labels[:10]) #looking at the first 10 labels.
# plt.figure()
# plt.imshow(train_images[1])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# Names of the labels.
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Data Preprocessing
train_images = train_images / 255.0
test_images = test_images / 255.0

# Creating the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1)
    keras.layers.Dense(128, activation='relu'),  # hidden layer (2)
    keras.layers.Dense(10, activation='softmax') # output layer (3)
])

# Compiling the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs = 10)

# Testing the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
print('Test accuracy:', test_acc)

# Making predictions
predictions = model.predict([test_images[0]])