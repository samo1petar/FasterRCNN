import cv2
import numpy as np
from os import listdir, mkdir, remove
from os.path import join
from shutil import copyfile
import tensorflow as tf
from typing import Iterator
from lib.tools.file import mkdir, choose_one_from_dir
from lib.tools.time import get_time
from lib.tools.plot import save_detections


class TrainSupport:
    def __init__(
            self,
            save_dir : str = 'results',
            name     : str = '',
    ):
        self._save_dir = join(save_dir, get_time() + '_' + name)
        self.model_saving_dir = join(self._save_dir, 'model')
        self.sample_train_dir = join(self._save_dir, 'sample_train')
        self.sample_test_dir  = join(self._save_dir, 'sample_test')

        for path in [self._save_dir, self.model_saving_dir, self.sample_train_dir, self.sample_test_dir]:
            mkdir(path)
        copyfile('params.py', join(self._save_dir, 'params.py'))

    def restore(self, save_dir):
        experiment = choose_one_from_dir(save_dir)

        checkpoint_files = [x for x in listdir(join(experiment, 'model'))]

        for x in checkpoint_files:
            copyfile(join(experiment, 'model', x), join(self.model_saving_dir, x))
        for x in listdir(join(experiment, 'metrics')):
            copyfile(join(experiment, 'metrics', x), join(self._save_dir, 'metrics', x))

        return join(experiment, 'model')

    @staticmethod
    def sample_from(model: tf.keras.Model, iterator : Iterator, save_dir: str, save_count: int = 20, N : int = 0):
        # for x in listdir(save_dir): # Question - isn't this done automatically when writing new images over existing ones?
        #     remove(join(save_dir, x))

        count = 0
        for index, class_ids, bboxes, image in iterator:

            proposals, cls_probs = model(image, training=False)

            bboxes = bboxes.numpy()
            proposals = proposals.numpy()
            cls_probs = cls_probs.numpy()
            image = image.numpy()

            image = (image * 255).astype(np.uint8)

            for x in range(image.shape[0]):
                count += 1
                if count > save_count:
                    return

                if N > 0:
                    save_path = join(save_dir, f'rpn_{N}.png')
                else:
                    save_path = join(save_dir, 'rpn.png')
                save_detections(image[x], bboxes, proposals[x], 'rpn', save_path)
