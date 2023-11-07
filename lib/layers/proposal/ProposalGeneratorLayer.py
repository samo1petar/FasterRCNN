import tensorflow as tf


class ProposalGeneratorLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            clip : bool = True,
            name : str  = 'proposal_layer',
    ):
        super(ProposalGeneratorLayer, self).__init__(name=name)
        self.clip = clip


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

        proposals = centers + anchor_offsets

        if self.clip:
            proposals = tf.clip_by_value(proposals, clip_value_min=0, clip_value_max=tf.tile(input_shape[1:3], [2]))

        # Question: Should predictions be cut :-4?
        # Question: Should proposals be corrected here?

        return proposals[tf.newaxis, ...], predictions[..., :-4]


if __name__ == '__main__':

    import numpy as np

    predictions = tf.constant(np.ones((1, 3, 3, 4)), dtype=tf.float32)
    input_shape = tf.constant([1, 60, 100, 3], dtype=tf.float32)
    anchors = [[100, 100], [20, 20]] #, [7, 7], [10, 10]]

    generate_proposal_layer = ProposalGeneratorLayer()
    proposals, cls_prob = generate_proposal_layer(predictions, input_shape, anchors)

    # from IPython import embed
    # embed()
