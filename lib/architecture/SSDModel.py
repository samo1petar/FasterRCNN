import tensorflow as tf
from lib.layers import Conv, ProposalLayer
from lib.loader_coco.classes import classes
from typing import List


class SSDModel(tf.keras.Model):
    def __init__(
            self,
            name              : str,
            feature_extractor : tf.keras.layers.Layer,
            anchors           : tf.Tensor,
    ):
        super(SSDModel, self).__init__(name=name)

        self.feature_extractor = feature_extractor
        self.anchors = anchors
        self.bbox_conv = Conv(len(classes) + 4, 3, name='bbox_conv')
        self.proposal_payer = ProposalLayer()

    def call(self, inputs: tf.Tensor, training: bool = False):

        x = self.feature_extractor(inputs, training=training)

        features = self.bbox_conv(x)

        proposals = self.proposal_payer(features, inputs.shape, self.anchors)

        return features, proposals
