from lib.architecture.FasterRCNNModel import FasterRCNNModel
from lib.feature_extractor.basic_feature_extractor import BasicFE
from lib.layers import RPNTargetLayer, RCNNTargetLayer
from lib.loader_coco.RecordReader import RecordReader
from lib.loader_coco.RecordWriter import RecordWriter
from lib.loss.softmax_cross_entropy import SoftmaxCrossEntropy
from lib.loss.mean_squared_error import MeanSquaredError
from lib.optimizer.adam import optimizer_and_learning_rate


params = {
    'batch_size'       : 1, # Must be one. NMS, ROI Pooling don't work well otherwise because part of tensor would be empty
    'learning_rate'    : 0.0001,
    'decay_steps'      : 10000,
    'decay_rate'       : 0.9,
    'print_every_iter' : 10,
    'eval_every_iter'  : 500,
    'max_iter'         : 1000000,
    'clip_gradients'   : 2.0,
    'results_dir'      : 'results',
    'name'             : 'RCNN_test',
}


class Definition:

    writer = RecordWriter(
        data_path='/media/david/A/Datasets/COCO',
        record_dir='records',
        record_name='one_image',
        save_n_test_images=1,
        save_n_train_images=1,
    )

    reader = RecordReader(
        record_dir='records',
        record_name='one_image',
        batch_size=1,
        shuffle_buffer=1,
        num_parallel_calls=1,
        num_parallel_reads=1,
        prefatch_buffer_size=1,
        count=-1,
    )

    feature_extractor = BasicFE(name='basic_feature_extractor')
    anchors = [[250, 315], [150, 120], [100, 80]]

    model = FasterRCNNModel(
        name='Model',
        feature_extractor=feature_extractor,
        anchors=anchors,
    )

    rpn_target_layer = RPNTargetLayer()
    fp_target_layer = RCNNTargetLayer()

    loss_cls = SoftmaxCrossEntropy()
    loss_reg = MeanSquaredError()

    optimizer = optimizer_and_learning_rate(
        learning_rate = params['learning_rate'],
        batch_size    = params['batch_size'],
        decay_steps   = params['decay_steps'],
        decay_rate    = params['decay_rate'],
    )
