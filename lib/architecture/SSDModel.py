import tensorflow as tf
from lib.layers import Conv, ProposalGeneratorLayer, ProposalSelectorLayer
from lib.loader_coco.classes import classes
from typing import List


class SSDModel(tf.keras.Model):
    def __init__(
            self,
            name              : str,
            feature_extractor : tf.keras.layers.Layer,
            anchors           : List[List[int]],
    ):
        super(SSDModel, self).__init__(name=name)

        self.feature_extractor = feature_extractor
        self.anchors = anchors
        self.anchor_cls = Conv(len(anchors), 3, name='anchor_prob_conv')
        self.bbox_conv = Conv(len(classes) + 4, 3, name='bbox_conv') # TODO separate Conv for cls and bbox
        self.proposal_generator_layer = ProposalGeneratorLayer()
        self.proposal_selector_layer = ProposalSelectorLayer()

    def call(self, inputs: tf.Tensor, training: bool = False):

        x = self.feature_extractor(inputs, training=training)

        features = self.bbox_conv(x)

        anchor_cls = self.anchor_cls(x)

        proposals, cls_prob = self.proposal_generator_layer(features, inputs.shape, self.anchors)

        if training:
            return proposals, cls_prob, anchor_cls
        else:
            proposals, cls_index, cls_prob = self.proposal_selector_layer(proposals, cls_prob) # TODO check Proposal Selector Layer
            return proposals, cls_index, cls_prob
