import tensorflow as tf
from typing import Tuple
from lib.tools.bbox import convert_from_exact_to_percentage, convert_yxyx_to_yxhw


class RPNProposalSelectorLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            name          : str   = 'select_proposal_layer',
            cls_threshold : float = 0.7,
    ):
        super(RPNProposalSelectorLayer, self).__init__(name=name)
        self.cls_threshold = cls_threshold
        self.softmax = tf.keras.layers.Softmax()
        self.nms_threshold = 0.7 # ToDo extract
        self.nms_max_outputs = 100 # ToDo extract
        self.min_size_threshold = 3


    def call(
            self,
            proposals         : tf.Tensor,
            cls_score         : tf.Tensor,
            input_image_shape : Tuple[int],
            format            : str = 'percentage',
            remove_small_proposals : bool = True,
    ) -> tf.Tensor:

        proposals_shape = tf.shape(proposals)

        cls_score = tf.reshape(cls_score, (tf.shape(cls_score)[0], tf.shape(cls_score)[1], tf.shape(cls_score)[2], -1, 2))

        cls_prob = self.softmax(cls_score)

        mask = cls_prob > self.cls_threshold

        mask = mask[:, :, :, :, 0]

        cls_mask = tf.stack((mask, mask), axis=-1)

        cls_prob_positive = tf.reshape(tf.boolean_mask(cls_prob, cls_mask), [proposals_shape[0], -1, 2])[:, :, 0]

        proposal_mask = tf.stack((mask, mask, mask, mask), axis=-1)

        proposals_positive = tf.reshape(tf.boolean_mask(proposals, proposal_mask), [proposals_shape[0], -1, 4])

        # NMS
        selected_indices = tf.image.non_max_suppression(
            tf.reshape(proposals_positive, [-1, 4]), tf.reshape(cls_prob_positive, [-1]),
            self.nms_max_outputs,
            self.nms_threshold)
        selected_boxes = tf.gather(tf.reshape(proposals_positive, [-1, 4]), selected_indices)
        selected_cls = tf.gather(tf.reshape(cls_prob_positive, [-1]), selected_indices)

        # ToDo NMS step implemented in this way supports only batch = 1. Create a loop over batches.
        # ToDo https://stackoverflow.com/questions/35330117/how-can-i-run-a-loop-with-a-tensor-as-its-range-in-tensorflow
        # ToDo take a look at example three at https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/while_loop

        if remove_small_proposals:
            selected_boxes_yxhw = convert_yxyx_to_yxhw(selected_boxes)

            mask_h = selected_boxes_yxhw[:, 2] > self.min_size_threshold
            mask_w = selected_boxes_yxhw[:, 3] > self.min_size_threshold

            mask = tf.logical_and(mask_h, mask_w)

            selected_boxes = tf.boolean_mask(selected_boxes, mask)
            selected_cls = tf.boolean_mask(selected_cls, mask)

        selected_boxes = tf.reshape(selected_boxes, [1, -1, 4])
        selected_cls = tf.reshape(selected_cls, [1, -1])

        if format == 'percentage':
            selected_boxes = convert_from_exact_to_percentage(selected_boxes, input_image_shape)
        else:
            raise ValueError # ToDo add other options

        return selected_boxes, selected_cls


if __name__ == '__main__':

    # ToDo remove main part

    import numpy as np
    from RPNProposalGeneratorLayer import RPNProposalGeneratorLayer

    bbox_deltas_np = np.zeros((1, 3, 3, 8))
    bbox_deltas_np[0, 0, 0, 1] = 1

    bbox_deltas = tf.constant(bbox_deltas_np, dtype=tf.float32)

    input_shape = tf.constant([1, 60, 100, 3], dtype=tf.float32)
    anchors = [[100, 100], [20, 20]] #, [7, 7], [10, 10]]

    generate_proposal_layer = RPNProposalGeneratorLayer()
    proposals = generate_proposal_layer(bbox_deltas, input_shape, anchors)

    cls_deltas_np = np.random.random((1, 3, 3, 4))
    cls_deltas = tf.constant(cls_deltas_np, dtype=tf.float32)

    proposal_selector_layer = RPNProposalSelectorLayer()
    selected_proposals = proposal_selector_layer(proposals, cls_deltas)

    from IPython import embed
    embed()
