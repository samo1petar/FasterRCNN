from gluoncv import data, utils
import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt

val_dataset = data.COCODetection(root='/media/david/A/Dataset/COCO', splits=['instances_val2017'])
print('Num of validation images:', len(val_dataset))

from IPython import embed
embed()
exit()

test_image, test_label = val_dataset[0]
bounding_boxes = test_label[:, :4]
class_ids = test_label[:, 4:5]

print('Image size (height, width, RGB):', test_image.shape)
print('Num of objects:', bounding_boxes.shape[0])
print('Bounding boxes (num_boxes, x_min, y_min, x_max, y_max):\n', bounding_boxes)
print('Class IDs (num_boxes, ):\n', class_ids)

print (test_label)

utils.viz.plot_bbox(test_image.asnumpy(), bounding_boxes, scores=None,
                    labels=class_ids, class_names=val_dataset.classes)
plt.show()
