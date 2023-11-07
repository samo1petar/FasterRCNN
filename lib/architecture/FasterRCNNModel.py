import tensorflow as tf
from lib.layers import Conv, ProposalGeneratorLayer, ProposalSelectorLayer
from lib.loader_coco.classes import classes
from typing import List


class FasterRCNNModel(tf.keras.Model): # ToDo Max Pool still not implemented
    def __init__(
            self,
            name              : str,
            feature_extractor : tf.keras.layers.Layer,
            anchors           : List[List[int]],
    ):
        super(SSDModel, self).__init__(name=name)

        self.feature_extractor = feature_extractor
        self.anchors = anchors
        self.cls_conv = Conv(len(anchors) * 2, 1, name='rpn_cls_conv')
        self.bbox_conv = Conv(len(anchors) * 4, 1, name='rpn_bbox_conv')
        self.proposal_generator_layer = ProposalGeneratorLayer()
        self.proposal_selector_layer = ProposalSelectorLayer()

    def call(self, inputs: tf.Tensor, training: bool = False):

        x = self.feature_extractor(inputs, training=training)

        bbox_conv = self.bbox_conv(x)

        cls_conv = self.cls_conv(x)

        proposals = self.proposal_generator_layer(bbox_conv, inputs.shape, self.anchors)

        proposals = self.proposal_selector_layer(proposals, cls_conv)

        # ToDO continue

        from IPython import embed
        print('Model')
        embed()
