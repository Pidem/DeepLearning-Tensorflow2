import tensorflow as tf

print("Tensorflow version: {}".format(tf.__version__))
print("Found GPU: {}".format(tf.test.gpu_device_name()))
print("Executing eagerly: {}".format(tf.executing_eagerly()))

assert tf.__version__ >= '2.0.0'

import numpy as np
import matplotlib.pyplot as plt

LEARNING_RATE = 0.01
NUM_EPOCHS = 25
BATCH_SIZE = 100

(train_X, train_Y), (test_X, test_Y) = tf.keras.datasets.mnist.load_data()
train_X, test_X = train_X / 255.0, test_X / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_X, train_Y, epochs=NUM_EPOCHS)
model.evaluate(test_X, test_Y)
