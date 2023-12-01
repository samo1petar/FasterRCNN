import numpy as np
import tensorflow as tf
from lib.layers import RPNProposalSelectorLayer, RPNProposalGeneratorLayer

bbox_deltas_np = np.zeros((1, 3, 3, 8))
bbox_deltas_np[0, 0, 0, 1] = 1

bbox_deltas = tf.constant(bbox_deltas_np, dtype=tf.float32)

input_shape = tf.constant([1, 60, 100, 3], dtype=tf.float32)
anchors = [[100, 100], [20, 20]]  # , [7, 7], [10, 10]]

generate_proposal_layer = RPNProposalGeneratorLayer()
proposals = generate_proposal_layer(bbox_deltas, input_shape, anchors)

cls_deltas_np = np.random.random((1, 3, 3, 4))
cls_deltas = tf.constant(cls_deltas_np, dtype=tf.float32)

proposal_selector_layer = RPNProposalSelectorLayer()
selected_proposals = proposal_selector_layer(proposals, cls_deltas)

from IPython import embed
embed()