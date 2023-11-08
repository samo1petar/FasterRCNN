import tensorflow as tf
from lib.layers import Conv, ProposalGeneratorLayer, ProposalSelectorLayer, ProposalTargetLayer
from lib.loader_coco.classes import classes
from typing import List


class FasterRCNNModel(tf.keras.Model): # ToDo Max Pool still not implemented
    def __init__(
            self,
            name              : str,
            feature_extractor : tf.keras.layers.Layer,
            anchors           : List[List[int]],
            inference         : bool = False,
    ):
        super(FasterRCNNModel, self).__init__(name=name)

        self.feature_extractor = feature_extractor
        self.anchors = anchors
        self.cls_conv = Conv(len(anchors) * 2, 1, name='rpn_cls_conv')
        self.bbox_conv = Conv(len(anchors) * 4, 1, name='rpn_bbox_conv')
        self.proposal_generator_layer = ProposalGeneratorLayer(correct_proposals=False, clip=True, format='yxyx')
        self.proposal_selector_layer = ProposalSelectorLayer()
        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs: tf.Tensor, training: bool = False):

        x = self.feature_extractor(inputs, training=training)

        bbox_conv = self.bbox_conv(x)

        cls_conv = self.cls_conv(x)

        # cls_conv = self.softmax(cls_conv)

        proposals = self.proposal_generator_layer(bbox_conv, inputs.shape, self.anchors)

        # proposals = self.proposal_selector_layer(proposals, cls_conv)

        # if training:
            # ToDo - also return bbox_conv and cls_conv because those need to be trained

        # ToDO continue

        # ToDo - ROI Pooling
        # ToDo - Cls Prediction
        # ToDo - Bbox Prediction

        # ToDo - fix return - this is just for now
        return proposals, bbox_conv, cls_conv
