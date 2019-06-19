import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Import data

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[..., tf.newaxis].astype('float32')
x_test = x_test[..., tf.newaxis].astype('float32')

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(100)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(100)

# Define Multilayer Perceptron model defined with the tf.nn module

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        n_input = 784
        n_hidden_1 = 512  # 1st layer num features
        n_hidden_2 = 512  # 2nd layer num features
        n_hidden_3 = 256  # 3rd layer num features
        n_classes = 10

        # NETWORK PARAMETERS
        stddev = 0.1
        self.w1 = tf.Variable(tf.random.normal([n_input, n_hidden_1], stddev=stddev), name='w1')
        self.w2 = tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], stddev=stddev), name='w2')
        self.w3 = tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3], stddev=stddev))
        self.wout = tf.Variable(tf.random.normal([n_hidden_3, n_classes], stddev=stddev), name='wout')
        self.b1 = tf.Variable(tf.random.normal([n_hidden_1]), name='b1')
        self.b2 = tf.Variable(tf.random.normal([n_hidden_2]), name='b2')
        self.b3 = tf.Variable(tf.random.normal([n_hidden_3]))
        self.bout = tf.Variable(tf.random.normal([n_classes]), name='bout')

    def call(self, x):
        inputs = tf.reshape(x, (-1, 28 * 28))
        layer_1 = tf.nn.relu(tf.add(tf.matmul(inputs, self.w1), self.b1))
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, self.w2), self.b2))
        layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, self.w3), self.b3))
        output = tf.nn.dropout(layer_3, keep_prob=0.6)
        return tf.nn.softmax(tf.matmul(output, self.wout) + self.bout)

model = Model()


# Define Multilayer Perceptron model defined with the tf.keras module

class Model_Keras(tf.keras.Model):
    def __init__(self):
        super(Model_Keras, self).__init__()
        self.reshape = tf.keras.layers.Reshape((-1, 28*28))
        self.dense1 = tf.keras.layers.Dense(units=512, activation='relu', use_bias=True,
                                            kernel_initializer='glorot_normal', bias_initializer='glorot_normal')
        self.dense2 = tf.keras.layers.Dense(units=512, activation='relu', use_bias=True,
                                            kernel_initializer='glorot_normal', bias_initializer='glorot_normal')
        self.dense3 = tf.keras.layers.Dense(units=256, activation='relu', use_bias=True,
                                            kernel_initializer='glorot_normal', bias_initializer='glorot_normal')
        self.dropout = tf.keras.layers.Dropout(rate=0.4)
        self.dense_out = tf.keras.layers.Dense(units=10, activation='softmax', use_bias=True,
                                               kernel_initializer='glorot_normal', bias_initializer='glorot_normal')
    def call(self, x):
        x = self.reshape(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dropout(x)
        return self.dense_out(x)

# model = Model_Keras()


# Verify the that the model has the right amount of trainable variables
total_parameters = 0
for var in model.trainable_variables:
    total_parameters += np.prod(var.get_shape())

print("Model implemented. Total trainable variables: %d, Total parameters: %d" % (
len(model.trainable_variables), total_parameters))


# Build training parameters

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)


# Run model
EPOCHS = 20

for epoch in range(EPOCHS):
    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))