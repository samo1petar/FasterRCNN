import tensorflow as tf
from lib.tools.bbox import iou_tf


class ProposalTargetLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            name          : str = 'target_proposal_layer',
            cls_threshold : float = 0.7,
    ):
        super(ProposalTargetLayer, self).__init__(name=name)
        self.cls_threshold = cls_threshold


    def call(
            self,
            proposals    : tf.Tensor,
            cls_prob     : tf.Tensor,
            gt_proposals : tf.Tensor,
            gt_cls       : tf.Tensor,
    ) -> tf.Tensor:

        iou = iou_tf(proposals, gt_proposals)

        a = tf.math.top_k(tf.reshape(iou, [-1]), k=10) # TODO parametrize k

        i = tf.tensor_scatter_nd_update(
            tensor=tf.reshape(tf.zeros_like(iou), [-1]),
            indices=tf.reshape(a[1], [-1, 1]),
            updates=tf.ones_like(a[1], dtype=tf.float32),
        )
        i = tf.cast(tf.reshape(i, iou.shape), dtype=tf.bool)

        print('Proposal Target Layer')
        from IPython import embed
        embed()
        exit()

        cls_max_prob = tf.reduce_max(cls_prob, axis=-1)
        cls_mask = cls_max_prob > self.cls_threshold
        cls_index = tf.argmax(cls_prob, axis=-1)

        proposals = tf.boolean_mask(proposals, cls_mask)
        cls_index = tf.boolean_mask(cls_index, cls_mask)
        cls_max_prob = tf.boolean_mask(cls_max_prob, cls_mask)

        return proposals, cls_index, cls_max_prob
