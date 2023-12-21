import tensorflow as tf
from lib.tools.iou import iou_tf_multiple
from lib.tools.bbox import convert_yxyx_to_yxhw, convert_yxhw_to_yxyx
from lib.loader_coco.classes import classes


class RCNNTargetLayer(tf.keras.layers.Layer): # ToDo Adjust names
    def __init__(
            self,
            name                   : str = 'final_proposal_target_layer',
            iou_positive_threshold : float = 0.4,
            iou_negative_threshold : float = 0.3,
            top_k                  : int = 100,
            take_positive_N        : int = 1000,
    ):
        super(RCNNTargetLayer, self).__init__(name=name)

        self.iou_positive_threshold = iou_positive_threshold
        self.iou_negative_threshold = iou_negative_threshold
        self.top_k = top_k
        self.take_positive_N = take_positive_N

    def call(
            self,
            proposals    : tf.Tensor,
            anchors      : tf.Tensor,
            gt_classes   : tf.Tensor,
            gt_proposals : tf.Tensor,
            cls_prob     : tf.Tensor,
    ) -> tf.Tensor:

        iou = iou_tf_multiple(proposals[0], gt_proposals)

        ########################## POSITIVE EXAMPLES ##########################

        a = tf.math.top_k(tf.reshape(iou, [-1]), k=tf.reduce_min((self.top_k, tf.reduce_sum(iou.shape))))

        a_over_threshold_indices = tf.gather_nd(a[1], tf.where(a[0] > self.iou_positive_threshold))

        i = tf.tensor_scatter_nd_update(
            tensor=tf.reshape(tf.zeros_like(iou), [-1]),
            indices=tf.reshape(a_over_threshold_indices, [-1, 1]),
            updates=tf.ones_like(a_over_threshold_indices, dtype=tf.float32),
        )
        i = tf.cast(tf.reshape(i, iou.shape), dtype=tf.bool)

        proposals_index = tf.where(i)

        proposals_index = tf.random.shuffle(proposals_index)[:self.take_positive_N]

        positive_proposals = tf.gather_nd(proposals[0], tf.reshape(proposals_index[:, 0], [-1, 1]))

        positive_cls_prob = tf.gather_nd(cls_prob[0], tf.reshape(proposals_index[:, 0], [-1, 1]))

        positive_gt_proposals = tf.gather(gt_proposals, proposals_index[:, 1])

        positive_gt_cls_prob = tf.one_hot(tf.gather_nd(gt_classes, tf.reshape(proposals_index[:, 1], [-1, 1])), len(classes))

        ########################## NEGATIVE EXAMPLES ##########################

        negative_index = tf.where(iou < self.iou_negative_threshold)

        negative_index = tf.random.shuffle(negative_index)[:tf.shape(proposals_index)[0]]

        negative_cls_prob = tf.gather_nd(cls_prob[0], tf.reshape(negative_index[:, 0], [-1, 1]))

        negative_gt_cls_prob = tf.one_hot(tf.tile([len(classes)-1], [negative_cls_prob.shape[0]]), len(classes))

        ########################## CALCULATE DELTAS ##########################

        positive_proposals_yxhw = convert_yxyx_to_yxhw(positive_proposals)
        positive_gt_proposals_yxhw = convert_yxyx_to_yxhw(positive_gt_proposals)

        proposal_deltas = self.calculate_proposals_deltas(positive_proposals_yxhw, positive_gt_proposals_yxhw)

        return proposals_index, positive_proposals, positive_cls_prob, positive_gt_proposals, positive_gt_cls_prob,\
               negative_index, negative_cls_prob, negative_gt_cls_prob, proposal_deltas

    def calculate_proposals_deltas(self, proposals_yxhw, gt_proposals_yxhw):
        pr_y = proposals_yxhw[:, 0]
        pr_x = proposals_yxhw[:, 1]
        pr_h = proposals_yxhw[:, 2]
        pr_w = proposals_yxhw[:, 3]

        gt_y = gt_proposals_yxhw[:, 0]
        gt_x = gt_proposals_yxhw[:, 1]
        gt_h = gt_proposals_yxhw[:, 2]
        gt_w = gt_proposals_yxhw[:, 3]

        delta_y = (gt_y - pr_y) / pr_h
        delta_x = (gt_x - pr_x) / pr_w
        delta_h = (gt_h - pr_h) / pr_h
        delta_w = (gt_w - pr_w) / pr_w

        return tf.stack((delta_y, delta_x, delta_h, delta_w), axis=-1)
