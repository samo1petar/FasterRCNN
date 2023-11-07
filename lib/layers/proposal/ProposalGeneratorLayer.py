import tensorflow as tf


class ProposalGeneratorLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            clip : bool = True,
            correct_proposals: bool = True,
            format : str = 'yxhw',
            name : str  = 'proposal_layer',
    ):
        super(ProposalGeneratorLayer, self).__init__(name=name)
        self.clip = clip
        self.format = format # yxhw or yxyx
        self.correct_proposals = correct_proposals

        assert self.format in ['yxhw', 'yxyx'], 'Proposal Generator Layer got wrong format parametr.'

    def convert_ywhw_to_yxyx(self, proposals): # ToDo extract out of this class
        y = proposals[:, :, :, 0]
        x = proposals[:, :, :, 1]
        h = proposals[:, :, :, 2]
        w = proposals[:, :, :, 3]
        y1 = y - h / 2
        x1 = x - w / 2
        y2 = y + h / 2
        x2 = x + w / 2

        return tf.stack((y1, x1, y2, x2), axis=3)

    def convert_yxyx_to_ywhw(self, proposals): # ToDo extract out of this class
        y = (proposals[:, :, :, 0] + proposals[:, :, :, 2]) / 2
        x = (proposals[:, :, :, 1] + proposals[:, :, :, 3]) / 2
        h = proposals[:, :, :, 2] - proposals[:, :, :, 0]
        w = proposals[:, :, :, 3] - proposals[:, :, :, 1]

        return tf.stack((y, x, h, w), axis=-1)

    def call(
            self,
            predictions : tf.Tensor,
            input_shape : tf.Tensor,
            anchors     : tf.Tensor,
    ):
        anchors = tf.constant(anchors, dtype=tf.float32)

        _, h, w, _= predictions.shape

        h_ratio = input_shape[1] / predictions.shape[1]
        w_ratio = input_shape[2] / predictions.shape[2]

        h_centers = tf.linspace(h_ratio / 2, input_shape[1] - h_ratio / 2, h, name='h_centers_linspace')
        w_centers = tf.linspace(w_ratio / 2, input_shape[2] - w_ratio / 2, w, name='w_centers_linspace')

        h_centers_tiled = tf.transpose(tf.reshape(tf.tile(h_centers, w_centers.shape), [w_centers.shape[0], h_centers.shape[0]]))
        w_centers_tiled = tf.reshape(tf.tile(w_centers, h_centers.shape), [h_centers.shape[0], w_centers.shape[0]])

        centers = tf.stack((h_centers_tiled, w_centers_tiled), axis=-1)
        centers = tf.concat((centers, centers), axis=-1)
        centers = tf.tile(centers, [1, 1, anchors.shape[0]])
        centers = tf.reshape(centers, [centers.shape[0], centers.shape[1], anchors.shape[0], 4])
        anchor_offsets = tf.concat((-anchors/2, anchors/2), axis=1)

        proposals_yxyx = centers + anchor_offsets

        proposals_yxhw = self.convert_yxyx_to_ywhw(proposals_yxyx)

        if self.correct_proposals:
            predictions = tf.reshape(predictions, proposals_yxhw.shape)

            y1 = proposals_yxhw[:, :, :, 0]
            x1 = proposals_yxhw[:, :, :, 1]
            h1 = proposals_yxhw[:, :, :, 2]
            w1 = proposals_yxhw[:, :, :, 3]

            delta_y = predictions[:, :, :, 0]
            delta_x = predictions[:, :, :, 1]
            delta_h = predictions[:, :, :, 2]
            delta_w = predictions[:, :, :, 3]

            y2 = delta_y * h1 + y1
            x2 = delta_x * w1 + x1
            h2 = (delta_h + 1) * h1
            w2 = (delta_w + 1) * w1

            proposals_yxhw = tf.stack((y2, x2, h2, w2), axis=-1)

        if self.clip:
            proposals_yxyx = self.convert_ywhw_to_yxyx(proposals_yxhw)
            proposals_yxyx = tf.clip_by_value(proposals_yxyx, clip_value_min=0, clip_value_max=tf.tile(input_shape[1:3], [2]))
            proposals_yxhw = self.convert_yxyx_to_ywhw(proposals_yxyx)

        if self.format == 'yxhw':
            return proposals_yxhw[tf.newaxis, ...]
        elif self.format == 'yxyx':
            proposals_yxyx = self.convert_ywhw_to_yxyx(proposals_yxhw)
            return proposals_yxyx[tf.newaxis, ...]


if __name__ == '__main__':

    import numpy as np

    predictions_np = np.zeros((1, 3, 3, 8))
    predictions_np[0, 0, 0, 1] = 1

    predictions = tf.constant(predictions_np, dtype=tf.float32)
    input_shape = tf.constant([1, 60, 100, 3], dtype=tf.float32)
    anchors = [[100, 100], [20, 20]] #, [7, 7], [10, 10]]

    generate_proposal_layer = ProposalGeneratorLayer()
    proposals = generate_proposal_layer(predictions, input_shape, anchors)

    # from IPython import embed
    # embed()
