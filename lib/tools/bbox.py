import numpy as np
import tensorflow as tf


def iou_np(bbox: np.ndarray, bboxes: np.ndarray) -> np.ndarray:

    bbox = np.tile(bbox, bboxes.shape[0]).reshape(bboxes.shape[0], 4)

    first_points_mask  = bbox[:, :2] >= bboxes[:, :2]
    second_points_mask = bbox[:, 2:] <= bboxes[:, 2:]

    first_points = bboxes[:, :2].copy()
    first_points[first_points_mask] = bbox[:, :2][first_points_mask]

    second_points = bboxes[:, 2:].copy()
    second_points[second_points_mask] = bbox[:, 2:][second_points_mask]

    mid_bboxes = np.concatenate((first_points, second_points), axis=1)

    no_intersection_mask = np.logical_or(mid_bboxes[:, 0] > mid_bboxes[:, 2], mid_bboxes[:, 1] > mid_bboxes[:, 3])

    mid_bboxes_area = (mid_bboxes[:, 2] - mid_bboxes[:, 0]) * (mid_bboxes[:, 3] - mid_bboxes[:, 1])
    bbox_area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
    bboxes_area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

    iou = mid_bboxes_area / (bbox_area + bboxes_area - mid_bboxes_area)
    iou[no_intersection_mask] = 0

    return iou


def iou_tf(bbox: tf.Tensor, bboxes: tf.Tensor) -> tf.Tensor:


    bbox = tf.reshape(tf.tile(bbox, [bboxes.shape[0]]), (bboxes.shape[0], 4))

    first_points_mask = bbox[:, :2] >= bboxes[:, :2]
    second_points_mask = bbox[:, 2:] <= bboxes[:, 2:]

    mask = tf.concat((first_points_mask, second_points_mask), axis=1)

    mid_bboxes = bbox * tf.cast(mask, bbox.dtype) + bboxes * tf.cast(tf.logical_not(mask), bboxes.dtype)

    no_intersection_mask = tf.logical_or(mid_bboxes[:, 0] > mid_bboxes[:, 2], mid_bboxes[:, 1] > mid_bboxes[:, 3])

    mid_bboxes_area = (mid_bboxes[:, 2] - mid_bboxes[:, 0]) * (mid_bboxes[:, 3] - mid_bboxes[:, 1])
    bbox_area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
    bboxes_area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

    iou = mid_bboxes_area / (bbox_area + bboxes_area - mid_bboxes_area)
    iou = iou * tf.cast(tf.logical_not(no_intersection_mask), iou.dtype)

    return iou


if __name__ == '__main__':
    bbox = np.array([10, 10, 20, 20])

    bboxes = np.array([
        [5, 15, 15, 25],
        [12, 5, 18, 25],
        [5, 5, 12, 12],
        [8, 8, 15, 15],
        [10, 10, 20, 20],
        [16, 16, 30, 30],
        [100, 100, 200, 200],
        [0, 0, 2, 2],
    ])

    print (iou_np(bbox, bboxes))
    exit()

    error_n = 1
    n = 0
    while 1:
        print (n, end='\r')
        n += 1
        bbox = np.concatenate((np.random.uniform(0, 50, 2), np.random.uniform(50, 100, 2)))
        bboxes = np.concatenate((np.random.uniform(0, 70, 2 * 1000).reshape(-1, 2), np.random.uniform(50, 100, 2 * 1000).reshape(-1, 2)), axis=1)

        np_output = iou_np(bbox, bboxes)
        tf_output = iou_tf(tf.constant(bbox), tf.constant(bboxes)).numpy()

        if np.sum(np.abs(np_output - tf_output)) > 0.00001:
            print ('error {}\nbbox:\n{}\nbboxes: \n{}'.format(error_n, bbox, bboxes))
            error_n += 1
            print ()

    # import time
    # import matplotlib
    # matplotlib.use('tkagg')
    # from matplotlib import pyplot as plt
    # times = []
    # lista = [10, 100, 1000, 10000, 100000, 1000000]
    # for x in lista:
    #     bboxes_test = np.tile(bboxes, x).reshape(-1, 4)
    #     start = time.time()
    #     iou_np(bbox, bboxes_test)
    #     end = time.time()
    #     print ('{} time for x = {} shape'.format(end - start, bboxes_test.shape))
    #     times.append(end-start)
    # plt.plot(lista, times)
    # plt.show()