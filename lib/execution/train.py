import os
import tensorflow as tf
import time
from lib.loader_coco.RecordReader import RecordReader
from lib.tools.TrainSupport import TrainSupport


def train(
        model                       : tf.keras.Model,
        loader                      : RecordReader,
        loss_cls_object             : tf.keras.losses.Loss,
        loss_reg_object             : tf.keras.losses.Loss,
        optimizer                   : tf.keras.optimizers.Optimizer,
        proposal_target_layer       : tf.keras.layers.Layer,
        final_proposal_target_layer : tf.keras.layers.Layer,
        print_every_iter            : int,
        eval_every_iter             : int,
        max_iter                    : int,
        results_dir                 : str,
        name                        : str,
        clip_gradients              : float,
) -> None:

    train_support = TrainSupport(save_dir=results_dir, name=name)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_print_loss = tf.keras.metrics.Mean(name='train_print_loss')
    # test_loss = tf.keras.metrics.Mean(name='test_loss')
    time_measurement = tf.keras.metrics.Mean(name='time_measurement')
    rpn_positive_bbox_loss = tf.keras.metrics.Mean(name='rpn_positive_bbox_loss')
    rpn_positive_cls_loss = tf.keras.metrics.Mean(name='rpn_positive_cls_loss')
    rpn_negative_cls_loss = tf.keras.metrics.Mean(name='rpn_negative_cls_loss')

    fp_positive_bbox_loss = tf.keras.metrics.Mean(name='fp_positive_bbox_loss')
    fp_positive_cls_loss = tf.keras.metrics.Mean(name='fp_positive_cls_loss')
    fp_negative_cls_loss = tf.keras.metrics.Mean(name='fp_negative_cls_loss')

    iterator_train = loader.read_record('train')

    logdir = train_support._save_dir
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    # @tf.function
    def train_step(image, class_ids, bboxes):
        with tf.GradientTape() as tape:

            uncorrected_proposals, rpn_bbox_conv, rpn_cls_conv, selected_proposals, fp_bbox_delt, fp_cls_delt = model(image, training=True)

            proposals_index, positive_proposals, positive_cls_prob, positive_gt_proposals, positive_gt_cls_prob, _, \
            negative_cls_prob, negative_gt_cls_prob, proposal_deltas \
                = proposal_target_layer(uncorrected_proposals, model.anchors, bboxes, rpn_cls_conv)

            rpn_bbox_conv = tf.reshape(rpn_bbox_conv, (tf.shape(rpn_bbox_conv)[0], tf.shape(rpn_bbox_conv)[1], tf.shape(rpn_bbox_conv)[2], -1, 4))
            rpn_bbox_conv_selected = tf.gather_nd(
                rpn_bbox_conv,
                tf.concat((proposals_index[:, :3], tf.reshape(proposals_index[:, -1], (-1, 1))), axis=-1)
            )

            rpn_positive_proposal_bbox_loss = loss_reg_object(proposal_deltas, rpn_bbox_conv_selected)

            rpn_positive_proposal_cls_loss = tf.cast(loss_cls_object(positive_gt_cls_prob, positive_cls_prob), dtype=tf.float32)

            rpn_negative_proposal_cls_loss = tf.cast(loss_cls_object(negative_gt_cls_prob, negative_cls_prob), dtype=tf.float32)

            # Add FP layer losses
            proposals_index, positive_proposals, positive_cls_prob, positive_gt_proposals, positive_gt_cls_prob, \
            negative_index, negative_cls_prob, negative_gt_cls_prob, proposal_deltas\
                = final_proposal_target_layer(selected_proposals, model.anchors, class_ids, bboxes, fp_cls_delt)

            fp_bbox_delt_selected = tf.gather_nd(fp_bbox_delt[0], tf.reshape(proposals_index[:, 0], [-1, 1]))

            fp_positive_proposal_bbox_loss = loss_reg_object(proposal_deltas, fp_bbox_delt_selected)

            fp_positive_proposal_cls_loss = tf.cast(loss_cls_object(positive_gt_cls_prob, positive_cls_prob), dtype=tf.float32)

            fp_negative_proposal_cls_loss = tf.cast(loss_cls_object(negative_gt_cls_prob, negative_cls_prob), dtype=tf.float32)

            loss = tf.add(
                tf.add(tf.add(rpn_negative_proposal_cls_loss, rpn_positive_proposal_cls_loss), rpn_positive_proposal_bbox_loss),
                tf.add(tf.add(fp_positive_proposal_cls_loss, fp_negative_proposal_cls_loss), fp_positive_proposal_bbox_loss),
            )

        gradients = tape.gradient(loss, model.trainable_variables)

        # print('Training')
        # from IPython import embed
        # embed()

        # clipped_gradients = []
        # for grad in gradients:
        #     clipped_gradients.append(tf.clip_by_value(grad, -clip_gradients, clip_gradients))

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_print_loss(loss)
        rpn_positive_bbox_loss(rpn_positive_proposal_bbox_loss)
        rpn_positive_cls_loss(rpn_positive_proposal_cls_loss)
        rpn_negative_cls_loss(rpn_negative_proposal_cls_loss)
        fp_positive_bbox_loss(fp_positive_proposal_bbox_loss)
        fp_positive_cls_loss(fp_positive_proposal_cls_loss)
        fp_negative_cls_loss(fp_negative_proposal_cls_loss)
        train_loss(loss)

    # train_loss.reset_states() # Question - Is this needed before any entry?
    # train_print_loss.reset_states() # Question - Is this needed before any entry?
    # test_loss.reset_states() # Question - Is this needed before any entry?

    iter = tf.constant(0, dtype=tf.int64)
    while iter < max_iter:
        iter += 1

        # #######################  EVAL  #######################
        if iter % eval_every_iter == 0:
            tf.summary.scalar('train_loss', train_loss.result(), iter)
            train_loss.reset_states()

            model.save_weights(os.path.join(train_support.model_saving_dir, 'model_' + str(iter.numpy())), save_format='tf')

            train_support.sample_from(model, iterator_train, train_support.sample_train_dir, save_count=1, N=iter)
            # train_support.sample_from(model, loader.read_record('test'), train_support.sample_test_dir)
            #
            # for name, cls, cls_name, image in loader.read_record('test'):
            #     prediction = model(image, training=False)
            #     loss = loss_object(cls, prediction)
            #     test_loss(loss)
            # tf.summary.scalar('test_loss', test_loss.result(), iter)
            # test_loss.reset_states()

        ########################  ITER  ########################
        if iter % print_every_iter == 0:
            print('Iter: {} \tLoss: {:.8f} \tRPN Bbox: {:.8f} \t+cls: {:.8f} \t-cls: {:.8f} \t FP Bbox: {:.8f} \t+cls: {:.8f} \t-cls: {:.8f} \tTime: {}'.format(
                iter, train_print_loss.result(), rpn_positive_bbox_loss.result(), rpn_positive_cls_loss.result(),
                rpn_negative_cls_loss.result(), fp_positive_bbox_loss.result(), fp_positive_cls_loss.result(), fp_negative_cls_loss.result(), time_measurement.result()))
            train_print_loss.reset_states()
            rpn_positive_bbox_loss.reset_states()
            rpn_positive_cls_loss.reset_states()
            rpn_negative_cls_loss.reset_states()
            fp_positive_bbox_loss.reset_states()
            fp_positive_cls_loss.reset_states()
            fp_negative_cls_loss.reset_states()
            time_measurement.reset_states()

        ######################  TRAIN STEP ######################
        index, class_ids, bboxes, image = iterator_train.__next__()
        start = time.time()
        train_step(image, class_ids, bboxes)
        end = time.time()
        time_measurement(end - start)
