import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Import data

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(100)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(100)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.SimpleRNN(64, input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
