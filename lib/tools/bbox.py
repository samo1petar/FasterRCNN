import tensorflow as tf


def convert_from_exact_to_percentage(proposals, input_image_shape):
    """
    proposals -> [Batch, N, 4]
    input_image_shape -> [3] -> [Batch, Height, Width, Channels]
    """
    return proposals / tf.constant([input_image_shape[1], input_image_shape[2],
                                    input_image_shape[1], input_image_shape[2]], dtype=tf.float32)


def convert_from_percentage_to_exact(proposals, input_image_shape):
    """
    proposals -> [Batch, N, 4]
    input_image_shape -> [3] -> [Batch, Height, Width, Channels]
    """
    return proposals * tf.constant([input_image_shape[1], input_image_shape[2],
                                    input_image_shape[1], input_image_shape[2]], dtype=tf.float32)


def convert_yxyx_to_yxhw(proposals):
    """
    proposals ->
    """
    y1 = proposals[:, 0]
    x1 = proposals[:, 1]
    y2 = proposals[:, 2]
    x2 = proposals[:, 3]
    y = (y2 + y1) / 2
    x = (x2 + x1) / 2
    h = y2 - y1
    w = x2 - x1

    return tf.stack((y, x, h, w), axis=-1)


def convert_yxhw_to_yxyx(proposals):
    y = proposals[:, 0]
    x = proposals[:, 1]
    h = proposals[:, 2]
    w = proposals[:, 3]
    y1 = y - h / 2
    x1 = x - w / 2
    y2 = y + h / 2
    x2 = x + w / 2

    return tf.stack((y1, x1, y2, x2), axis=-1)


def convert_positional_ywhw_to_yxyx(proposals):
    y = proposals[:, :, :, 0]
    x = proposals[:, :, :, 1]
    h = proposals[:, :, :, 2]
    w = proposals[:, :, :, 3]
    y1 = y - h / 2
    x1 = x - w / 2
    y2 = y + h / 2
    x2 = x + w / 2

    return tf.stack((y1, x1, y2, x2), axis=3)


def convert_positional_yxyx_to_ywhw(proposals):
    y = (proposals[:, :, :, 0] + proposals[:, :, :, 2]) / 2
    x = (proposals[:, :, :, 1] + proposals[:, :, :, 3]) / 2
    h = proposals[:, :, :, 2] - proposals[:, :, :, 0]
    w = proposals[:, :, :, 3] - proposals[:, :, :, 1]

    return tf.stack((y, x, h, w), axis=-1)
