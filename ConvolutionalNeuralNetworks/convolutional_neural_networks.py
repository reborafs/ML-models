import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


# Load and Split dataset from keras.
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values between 0-1.
train_images, test_images = train_images/255.0, test_images/255.0
class_names = ['airplanes', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# # Looking at one image:
# IMG_INDEX = 1
# plt.imshow(train_images[IMG_INDEX], cmap=plt.cm.binary)
# plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
# plt.show()

# Creating the Convolutional Base.
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))

# Adding the dense layers. (Classifier)
model.add(layers.Flatten()) #shapes into 1D.
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10)) #Output layer, 1 node for each class.

model.summary() #Checking the model

# Training the model.
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

# Evaluating the model.
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)
