import tensorflow as tf


class MeanSquaredError(tf.keras.losses.MeanSquaredError):
    def __init__(self, multiplayer: float = 1, name: str = 'mean_squared_error', **kwargs):
        super(MeanSquaredError, self).__init__(name=name, **kwargs)
        self._multiplayer = multiplayer
