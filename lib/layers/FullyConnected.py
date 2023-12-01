import tensorflow as tf
from lib.layers import Activation


class FullyConnected(tf.keras.layers.Layer):
    def __init__(self, units : int, activation: str = 'relu', **kwargs):
        super(FullyConnected, self).__init__(**kwargs)
        self.units = units
        self.activation_name = activation

    def build(self, shape):
        if self.activation_name == 'none':
            activation_class = None
        else:
            activation_class = Activation(self.activation_name)
        self.dense = tf.keras.layers.Dense(units=self.units, activation=activation_class)

    def call(self, inputs, training: bool = True):
        return self.dense(inputs, training=training)
