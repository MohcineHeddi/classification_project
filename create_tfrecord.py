import tensorflow as tf
import utils
import os
import numpy as np

path = '/Users/mohcine/Documents/Code/Scor/Database'
name = 'englishlmg'
nbr_classes = 62

labels = [el[1] for el in utils.get_path_label()]
cnt_labels = [(labels.count(el + 1), int(0.30 * labels.count(el + 1))) for el in range(nbr_classes)]
path_lbl = utils.get_path_label()
filename_train = os.path.join(path, name + '_train.tfrecords')
filename_test = os.path.join(path, name + '_test.tfrecords')
writer_train = tf.python_io.TFRecordWriter(filename_train)
writer_test = tf.python_io.TFRecordWriter(filename_test)

cnt = 0
for el in path_lbl:
    cnt += 1
    print('###writing image count:', str(cnt), '###')
    label = el[1]
    path = el[0]
    im = utils.normalize_im(utils.reshape_im(utils.get_image(path)))
    im_rot90 = np.asarray(list(zip(*im[::-1])))
    im_rot180 = im[::-1]
    data_raw = im.tostring()
    data_rot90 = im_rot90.tostring()
    data_rot180 = im_rot180.tostring()
    rows = im.shape[0]
    cols = im.shape[1]
    example = tf.train.Example(features=tf.train.Features(feature={
        'width': utils._int64_feature(cols),
        'height': utils._int64_feature(rows),
        'label': utils._int64_feature(int(label)),
        'image_raw': utils._bytes_feature(data_raw)}))
    example_rot90 = tf.train.Example(features=tf.train.Features(feature={
            'width': utils._int64_feature(cols),
            'height': utils._int64_feature(rows),
            'label': utils._int64_feature(int(label)),
            'image_raw': utils._bytes_feature(data_rot90)}))
    example_rot180 = tf.train.Example(features=tf.train.Features(feature={
            'width': utils._int64_feature(cols),
            'height': utils._int64_feature(rows),
            'label': utils._int64_feature(int(label)),
            'image_raw': utils._bytes_feature(data_rot180)}))
    if cnt_labels[int(label) - 1][0] >= cnt_labels[int(label) - 1][1]:
        cnt_labels[int(label) - 1] = (cnt_labels[int(label) - 1][0] - 1, cnt_labels[int(label) - 1][1])
        writer_train.write(example.SerializeToString())
        writer_train.write(example_rot90.SerializeToString())
        writer_train.write(example_rot180.SerializeToString())
    else:
        writer_test.write(example.SerializeToString())
        writer_test.write(example_rot90.SerializeToString())
        writer_test.write(example_rot180.SerializeToString())

writer_train.close()
writer_test.close()
