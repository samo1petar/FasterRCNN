import tensorflow as tf


class ProposalSelectorLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            name          : str   = 'select_proposal_layer',
            cls_threshold : float = 0.7,
    ):
        super(ProposalSelectorLayer, self).__init__(name=name)
        self.cls_threshold = cls_threshold


    def call(self, proposals: tf.Tensor, cls_prob: tf.Tensor) -> tf.Tensor:

        mask = tf.argmax(tf.reshape(cls_prob, (cls_prob.shape[0], cls_prob.shape[1], cls_prob.shape[2], -1, 2)), axis=-1)

        proposals = tf.boolean_mask(proposals, mask)

        return proposals[tf.newaxis, ...]


if __name__ == '__main__':

    import numpy as np
    from ProposalGeneratorLayer import ProposalGeneratorLayer

    bbox_deltas_np = np.zeros((1, 3, 3, 8))
    bbox_deltas_np[0, 0, 0, 1] = 1

    bbox_deltas = tf.constant(bbox_deltas_np, dtype=tf.float32)

    input_shape = tf.constant([1, 60, 100, 3], dtype=tf.float32)
    anchors = [[100, 100], [20, 20]] #, [7, 7], [10, 10]]

    generate_proposal_layer = ProposalGeneratorLayer()
    proposals = generate_proposal_layer(bbox_deltas, input_shape, anchors)

    cls_deltas_np = np.random.random((1, 3, 3, 4))
    cls_deltas = tf.constant(cls_deltas_np, dtype=tf.float32)

    proposal_selector_layer = ProposalSelectorLayer()
    selected_proposals = proposal_selector_layer(proposals, cls_deltas)

    from IPython import embed
    embed()
