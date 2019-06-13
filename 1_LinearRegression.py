import tensorflow as tf

print("Tensorflow version: {}".format(tf.__version__))
print("Found GPU: {}".format(tf.test.gpu_device_name()))
print("Executing eagerly: {}".format(tf.executing_eagerly()))

assert tf.__version__ >= '2.0.0'

import numpy as np
import matplotlib.pyplot as plt
NUM_EXAMPLES = 150
NUM_EPOCHS = 100

X = tf.random.uniform([NUM_EXAMPLES], 0, 10)
Y = X * 3 + 2 + tf.random.normal([NUM_EXAMPLES], 0, 3)


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.W = tf.Variable(5., name='weight')
        self.B = tf.Variable(10., name='bias')

    def call(self, inputs):
        return inputs * self.W + self.B


def loss(model, inputs, targets):
    error = model(inputs) - targets
    return tf.reduce_mean(tf.square(error))


def train(model, inputs, targets, learning_rate):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)

    dW, dB = tape.gradient(loss_value, [model.W, model.B])
    model.W.assign_sub(learning_rate * dW)
    model.B.assign_sub(learning_rate * dB)


model = Model()

for epoch in range(NUM_EPOCHS):
    train(model, X, Y, learning_rate=0.1)
    current_loss = loss(model, X, Y)

    if epoch % 20 == 0:
        print("Loss at step {:03d}: {:.3f}".format(epoch, current_loss))

print("Final loss: {:.3f}".format(loss(model, X, Y)))
print("W= {}, b={}".format(model.W.numpy(), model.B.numpy()))

x = np.linspace(0, 10, 1000)
plt.scatter(X, Y, marker='+')
plt.plot(x, model.W.numpy() * x + model.B.numpy(), marker='.')
plt.title("Linear regression")
plt.show()
