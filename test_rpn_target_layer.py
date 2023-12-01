import numpy as np
import tensorflow as tf
from lib.layers import RPNProposalGeneratorLayer
from lib.layers import RPNTargetLayer

if __name__ == '__main__':

    import numpy as np

    bbox_deltas_np = np.random.random((1, 3, 3, 8))

    bbox_deltas = tf.constant(bbox_deltas_np, dtype=tf.float32)

    input_shape = tf.constant([1, 60, 100, 3], dtype=tf.float32)
    anchors = [[100, 100], [20, 20]] #, [7, 7], [10, 10]]

    generate_proposal_layer = RPNProposalGeneratorLayer(clip=False, format='yxyx')
    proposals = generate_proposal_layer(bbox_deltas, input_shape, anchors, correct_proposals=False)

    gt_bboxes = tf.constant(
        [
            [0, 70, 20, 90],
            # [10, 10, 80, 80],
        ], dtype=tf.float32)

    cls_deltas_np = np.random.random((1, 3, 3, 4))
    cls_deltas = tf.constant(cls_deltas_np, dtype=tf.float32)

    rpn_target_layer = RPNTargetLayer()
    proposals_index, \
    positive_proposals, \
    positive_cls_prob, \
    positive_gt_proposals, \
    positive_gt_cls_prob, \
    negative_index, \
    negative_cls_prob, \
    negative_gt_cls_prob, \
    proposal_deltas \
        = rpn_target_layer(proposals, anchors, gt_bboxes, cls_deltas)

    print('Calculate Deltas')
    from IPython import embed

    embed()