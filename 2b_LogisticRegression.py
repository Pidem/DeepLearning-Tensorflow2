import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


print("Tensorflow version: {}".format(tf.__version__))
print("Found GPU: {}".format(tf.test.gpu_device_name()))
print("Executing eagerly: {}".format(tf.executing_eagerly()))

assert tf.__version__ >= '2.0.0'



LEARNING_RATE = 0.01
NUM_EPOCHS = 25
BATCH_SIZE = 100

(train_X, train_Y), (test_X, test_Y) = tf.keras.datasets.mnist.load_data()
train_X, test_X = train_X / 255.0, test_X / 255.0
train_X = train_X.reshape(60000, -1)


class Model(tf.keras.Model):

    def __init__(self):
        super(Model, self).__init__()
        self.W = tf.Variable(initial_value=tf.random_normal_initializer, shape=(784, 10), dtype='float32')
        self.b = tf.Variable(initial_value=tf.random_normal_initializer, shape=(10,), dtype='float32')

    def call(self, inputs):
        inputs = tf.keras.Input(shape=(784,), name="digits")
        return tf.nn.softmax(tf.matmul(inputs, self.W) + self.b)


def loss(model, inputs, targets):
    prediction = model(inputs)
    return tf.losses.softmax_cross_entropy(targets, logits=prediction)


optimizer = tf.optimizers.Adam()


def train(model, inputs, targets, learning_rate):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    dW, db = tape.gradient(loss_value, [model.W, model.b])
    optimizer.apply_gradients(zip([dW, db], [model.W, model.b]))


def accuracy(model, inputs, targets):
    logits = model(inputs)
    prediction = tf.argmax(logits, 1).numpy()
    equality = tf.equal(prediction, tf.argmax(targets, 1))
    return tf.reduce_mean(tf.cast(equality, tf.float32))


model = Model()
print(train_X.shape)
tf.nn.softmax(tf.matmul(tf.cast(train_X, tf.float32), model.W) + model.b)

for epoch in range(NUM_EPOCHS):
    train(model, train_X, train_Y, learning_rate=0.1)
    current_loss = loss(model, train_X, train_Y)

    if epoch % 20 == 0:
        print("Loss at step {:03d}: {:.3f}".format(epoch, current_loss))

print("Final Loss: {}".format(accuracy(model, test_X, test_Y)))

