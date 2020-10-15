import tensorflow as tf
from lib.tools.bbox import iou_tf


class ProposalTargetLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            name            : str = 'target_proposal_layer',
            iou_threshold   : float = 0.7,
            top_k           : int = 100,
            take_positive_N : int = 10,
    ):
        super(ProposalTargetLayer, self).__init__(name=name)

        self.iou_threshold = cls_threshold
        self.top_k = top_k
        self.take_positive_N = take_positive_N


    def call(
            self,
            proposals    : tf.Tensor,
            cls_prob     : tf.Tensor,
            anchor_prob  : tf.Tensor,
            gt_proposals : tf.Tensor,
            gt_cls       : tf.Tensor,
    ) -> tf.Tensor:

        iou = iou_tf(proposals, gt_proposals)

        ########################## POSITIVE EXAMPLES ##########################

        a = tf.math.top_k(tf.reshape(iou, [-1]), k=self.top_k)

        a_over_threshold_indices = tf.gather_nd(a[1], tf.where(a[0] > self.iou_threshold))

        i = tf.tensor_scatter_nd_update(
            tensor=tf.reshape(tf.zeros_like(iou), [-1]),
            indices=tf.reshape(a_over_threshold_indices, [-1, 1]),
            updates=tf.ones_like(a_over_threshold_indices, dtype=tf.float32),
        )
        i = tf.cast(tf.reshape(i, iou.shape), dtype=tf.bool)

        proposals_index = tf.where(i)

        proposals_index = tf.random.shuffle(proposals_index)[:self.take_positive_N]

        positive_proposals = tf.gather_nd(
            proposals,
            tf.concat((proposals_index[:, :3], tf.reshape(proposals_index[:, -1], (-1, 1))), axis=-1)
        )

        positive_cls_prob = tf.gather_nd(cls_prob, proposals_index[:, :3])

        positive_gt_proposals = tf.gather(gt_proposals, proposals_index[:, 3])

        positive_gt_cls = tf.gather(gt_cls, proposals_index[:, 3])

        positive_anchor = proposals_index[:, -1]

        positive_anchor_cls = tf.gather_nd(anchor_prob, proposals_index[:, :3])

        ########################## NEGATIVE EXAMPLES ##########################

        iou_per_position = tf.reduce_max(iou, axis=[3, 4])

        negative_index = tf.where(iou_per_position < self.iou_threshold)

        negative_index = tf.random.shuffle(negative_index)[:positive_data.shape[0]]

        negative_cls_prob = tf.gather_nd(cls_prob, negative_index)

        return proposals_index, positive_proposals, positive_cls_prob, positive_gt_proposals, positive_gt_cls,\
               positive_anchor, positive_anchor_cls, negative_index, negative_cls_prob
