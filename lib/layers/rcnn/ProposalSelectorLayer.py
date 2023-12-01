import tensorflow as tf
from lib.tools.bbox import convert_from_percentage_to_exact, convert_yxhw_to_yxyx
from typing import Tuple


class ProposalSelectorLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            name          : str   = 'select_proposal_layer',
            cls_threshold : float = 0.7,
    ):
        super(ProposalSelectorLayer, self).__init__(name=name)
        self.cls_threshold = cls_threshold
        self.softmax = tf.keras.layers.Softmax()
        self.nms_threshold = 0.5
        self.nms_max_outputs = 100


    def call(
            self,
            proposals: tf.Tensor,
            bbox_deltas: tf.Tensor,
            cls_score: tf.Tensor,
            input_image_shape : Tuple[int],
            correct_proposals: bool = True,
    ) -> tf.Tensor:

        cls_score_max_index = tf.reshape(tf.argmax(cls_score, axis=2), (-1, 1))

        r = tf.reshape(tf.range(cls_score_max_index.shape[0], dtype=tf.int64), (-1, 1))

        indices = tf.concat((r, cls_score_max_index), axis=1)

        max_scores = tf.gather_nd(indices=indices, params=cls_score[0])

        mask = max_scores > self.cls_threshold

        proposals = tf.boolean_mask(proposals[0], mask)

        bbox_deltas = tf.boolean_mask(bbox_deltas[0], mask)

        max_scores = tf.boolean_mask(max_scores, mask)

        cls_indexes = tf.boolean_mask(cls_score_max_index, mask)

        proposals = convert_from_percentage_to_exact(proposals, input_image_shape)

        if correct_proposals:
            y1 = proposals[:, 0]
            x1 = proposals[:, 1]
            h1 = proposals[:, 2]
            w1 = proposals[:, 3]

            bbox_deltas = tf.math.tanh(bbox_deltas)

            delta_y = bbox_deltas[:, 0]
            delta_x = bbox_deltas[:, 1]
            delta_h = bbox_deltas[:, 2]
            delta_w = bbox_deltas[:, 3]

            y2 = delta_y * h1 + y1
            x2 = delta_x * w1 + x1
            h2 = (delta_h + 1) * h1
            w2 = (delta_w + 1) * w1

            proposals = tf.stack((y2, x2, h2, w2), axis=-1)
            proposals = convert_positional_ywhw_to_yxyx(proposals)

        if self.clip:
            proposals = tf.clip_by_value(proposals,
                                         clip_value_min=0,
                                         clip_value_max=tf.cast(tf.tile(input_shape[1:3], [2]), dtype=tf.float32))

        print('ProposalSelectorLayer')
        from IPython import embed
        embed()

        exit()

        # # NMS - Not used yet
        selected_indices = tf.image.non_max_suppression(
            proposals, max_scores,
            self.nms_max_outputs,
            self.nms_threshold)
        selected_boxes = tf.gather(proposals, selected_indices)
        selected_cls = tf.gather(tf.reshape(cls_prob_positive, [-1]), selected_indices) # ToDo
        #
        # # ToDo NMS step implemented in this way supports only batch = 1. Create a loop over batches.
        # # ToDo https://stackoverflow.com/questions/35330117/how-can-i-run-a-loop-with-a-tensor-as-its-range-in-tensorflow
        # # ToDo take a look at example three at https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/while_loop
        #
        # selected_boxes = tf.reshape(selected_boxes, [1, -1, 4])
        # selected_cls = tf.reshape(selected_cls, [1, -1])
        #
        # return selected_boxes, selected_cls
