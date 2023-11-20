import os
import tensorflow as tf
import time
from lib.loader_coco.RecordReader import RecordReader
from lib.tools.TrainSupport import TrainSupport


def train(
        model                 : tf.keras.Model,
        loader                : RecordReader,
        loss_cls_object       : tf.keras.losses.Loss,
        loss_reg_object       : tf.keras.losses.Loss,
        optimizer             : tf.keras.optimizers.Optimizer,
        proposal_target_layer : tf.keras.layers.Layer,
        print_every_iter      : int,
        eval_every_iter       : int,
        max_iter              : int,
        results_dir           : str,
        name                  : str,
        clip_gradients        : float,
) -> None:

    train_support = TrainSupport(save_dir=results_dir, name=name)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_print_loss = tf.keras.metrics.Mean(name='train_print_loss')
    # test_loss = tf.keras.metrics.Mean(name='test_loss')
    time_measurement = tf.keras.metrics.Mean(name='time_measurement')
    positive_bbox_loss = tf.keras.metrics.Mean(name='positive_bbox_loss')
    positive_cls_loss = tf.keras.metrics.Mean(name='positive_cls_loss')
    negative_cls_loss = tf.keras.metrics.Mean(name='negative_cls_loss')

    iterator_train = loader.read_record('train')

    logdir = train_support._save_dir
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    @tf.function
    def train_step(image, class_ids, bboxes):
        with tf.GradientTape() as tape:
            proposals, bbox_pred, cls_prob = model(image, training=True)
            proposals_index, positive_proposals, positive_cls_prob, positive_gt_proposals, positive_gt_cls_prob, _, \
            negative_cls_prob, negative_gt_cls_prob, proposal_deltas \
                = proposal_target_layer(proposals, model.anchors, bboxes, cls_prob)

            bbox_pred = tf.reshape(bbox_pred, (tf.shape(bbox_pred)[0], tf.shape(bbox_pred)[1], tf.shape(bbox_pred)[2], -1, 4))
            bbox_pred_selected = tf.gather_nd(
                bbox_pred,
                tf.concat((proposals_index[:, :3], tf.reshape(proposals_index[:, -1], (-1, 1))), axis=-1)
            )

            positive_proposal_bbox_loss = tf.cast(loss_reg_object(proposal_deltas, bbox_pred_selected), dtype=tf.float16)

            positive_proposal_cls_loss = loss_cls_object(positive_gt_cls_prob, positive_cls_prob)

            negative_proposal_cls_loss = loss_cls_object(negative_gt_cls_prob, negative_cls_prob)

            loss = tf.add(tf.add(positive_proposal_bbox_loss, positive_proposal_cls_loss), negative_proposal_cls_loss)

        gradients = tape.gradient(loss, model.trainable_variables)

        # clipped_gradients = []
        # for grad in gradients:
        #     clipped_gradients.append(tf.clip_by_value(grad, -clip_gradients, clip_gradients))

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_print_loss(loss)
        positive_bbox_loss(positive_proposal_bbox_loss)
        positive_cls_loss(positive_proposal_cls_loss)
        negative_cls_loss(negative_proposal_cls_loss)
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
            print('Iter: {} \tLoss: {:.8f} \tBbox: {:.8f} \t+cls: {:.8f} \t-cls: {:.8f} \tTime: {}'.format(
                iter, train_print_loss.result(), positive_bbox_loss.result(), positive_cls_loss.result(),
                negative_cls_loss.result(), time_measurement.result()))
            train_print_loss.reset_states()
            positive_bbox_loss.reset_states()
            positive_cls_loss.reset_states()
            negative_cls_loss.reset_states()
            time_measurement.reset_states()

        ######################  TRAIN STEP ######################
        index, class_ids, bboxes, image = iterator_train.__next__()
        start = time.time()
        train_step(image, class_ids, bboxes)
        end = time.time()
        time_measurement(end - start)
