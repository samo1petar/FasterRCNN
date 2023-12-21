import tensorflow as tf
from lib.layers import Conv, FullyConnected, ProposalSelectorLayer, RCNNTargetLayer, RPNProposalGeneratorLayer, \
    RPNProposalSelectorLayer, ROIMaxPool
from lib.loader_coco.classes import classes
from lib.tools.bbox import convert_from_percentage_to_exact
from typing import List


class FasterRCNNModel(tf.keras.Model):
    def __init__(
            self,
            name              : str,
            feature_extractor : tf.keras.layers.Layer,
            anchors           : List[List[int]],
            inference         : bool = False,
    ):
        super(FasterRCNNModel, self).__init__(name=name)

        # ToDo CHANGE NAMES OF LAYERS!!!

        self.feature_extractor = feature_extractor
        self.anchors = anchors
        self.rpn_cls_conv = Conv(len(anchors) * 2, 1, name='rpn_cls_conv')
        self.rpn_bbox_conv = Conv(len(anchors) * 4, 1, name='rpn_bbox_conv')
        self.proposal_generator_layer = RPNProposalGeneratorLayer(clip=True, format='yxyx')
        self.proposal_selector_layer = RPNProposalSelectorLayer()
        self.roi_max_pool = ROIMaxPool(3, 3)
        self.fc_1 = FullyConnected(units=512, activation='relu', name='fc_1')
        self.fc_2 = FullyConnected(units=256, activation='relu', name='fc_2')
        self.fp_cls_fc = FullyConnected(units=len(classes), activation='sigmoid', name='fp_cls_fc')
        self.fp_bbox_fc = FullyConnected(units=4, activation='none', name='fp_bbox_fc')

        self.rcnn_proposal_selector_layer = ProposalSelectorLayer()

    def call(self, inputs: tf.Tensor, training: bool = False):

        x = self.feature_extractor(inputs, training=training)

        rpn_bbox_conv = self.rpn_bbox_conv(x)

        rpn_cls_conv = self.rpn_cls_conv(x)

        if training:
            uncorrected_proposals = self.proposal_generator_layer(rpn_bbox_conv, inputs.shape, self.anchors,
                                                                  correct_proposals=False)

            proposals = self.proposal_generator_layer(rpn_bbox_conv, inputs.shape, self.anchors, correct_proposals=True) # ToDo correct_proposals=True
            selected_proposals, selected_prob = self.proposal_selector_layer(proposals, rpn_cls_conv,
                                                                             input_image_shape=inputs.shape,
                                                                             format='percentage')

            pooled_features = self.roi_max_pool([x, selected_proposals])

            pooled_features_flattened = tf.reshape(pooled_features, (
                pooled_features.shape[0],
                pooled_features.shape[1],
                pooled_features.shape[2] * pooled_features.shape[3] * pooled_features.shape[4]))

            fc_1_out = self.fc_1(pooled_features_flattened) # ToDo rename
            fc_2_out = self.fc_2(fc_1_out) # ToDo rename

            fp_cls_delt = self.fp_cls_fc(fc_2_out) # ToDo rename
            fp_bbox_delt = self.fp_bbox_fc(fc_2_out) # ToDo rename

            selected_proposals = convert_from_percentage_to_exact(selected_proposals, inputs.shape)

            rcnn_proposals, rcnn_scores, rcnn_classes = self.rcnn_proposal_selector_layer(selected_proposals, fp_bbox_delt, fp_cls_delt, inputs.shape)

            return uncorrected_proposals, rpn_bbox_conv, rpn_cls_conv, selected_proposals, fp_bbox_delt, fp_cls_delt # ToDo add rcnn_proposals, rcnn_scores, rcnn_classes

        if not training: # ToDo most parts are the same as the training = True block.
            proposals = self.proposal_generator_layer(rpn_bbox_conv, inputs.shape, self.anchors, correct_proposals=True)
            selected_proposals, selected_prob = self.proposal_selector_layer(proposals, rpn_cls_conv)

            pooled_features = self.roi_max_pool([x, selected_proposals])

            pooled_features_flattened = tf.reshape(pooled_features, (
                pooled_features.shape[0],
                pooled_features.shape[1],
                pooled_features.shape[2] * pooled_features.shape[3] * pooled_features.shape[4]))

            fc_1_out = self.fc_1(pooled_features_flattened) # ToDo rename
            fc_2_out = self.fc_2(fc_1_out) # ToDo rename

            fp_cls_delt = self.fp_cls_fc(fc_2_out) # ToDo rename
            fp_bbox_delt = self.fp_bbox_fc(fc_2_out) # ToDo rename

            return selected_proposals, selected_prob

        # ToDO continue

        # ToDo - Cls Prediction
        # ToDo - Bbox Prediction

        # ToDo - fix return - this is just for now
        return selected_proposals, rpn_bbox_conv, rpn_cls_conv
