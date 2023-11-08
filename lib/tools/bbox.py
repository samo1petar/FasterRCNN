import tensorflow as tf


def convert_ywhw_to_yxyx(proposals):
    y = proposals[:, :, :, 0]
    x = proposals[:, :, :, 1]
    h = proposals[:, :, :, 2]
    w = proposals[:, :, :, 3]
    y1 = y - h / 2
    x1 = x - w / 2
    y2 = y + h / 2
    x2 = x + w / 2

    return tf.stack((y1, x1, y2, x2), axis=3)


def convert_yxyx_to_ywhw(proposals):
    y = (proposals[:, :, :, 0] + proposals[:, :, :, 2]) / 2
    x = (proposals[:, :, :, 1] + proposals[:, :, :, 3]) / 2
    h = proposals[:, :, :, 2] - proposals[:, :, :, 0]
    w = proposals[:, :, :, 3] - proposals[:, :, :, 1]

    return tf.stack((y, x, h, w), axis=-1)
