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

        cls_max_prob = tf.reduce_max(cls_prob, axis=-1)
        cls_mask = cls_max_prob > self.cls_threshold
        cls_index = tf.argmax(cls_prob, axis=-1)

        proposals = tf.boolean_mask(proposals, cls_mask)
        cls_index = tf.boolean_mask(cls_index, cls_mask)
        cls_max_prob = tf.boolean_mask(cls_max_prob, cls_mask)

        return proposals, cls_index, cls_max_prob
