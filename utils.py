from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import tensorflow as tf


# The operation used to print out the configuration
def print_configuration_op(FLAGS):
    print('[Configurations]:')
    # pdb.set_trace()
    for name, value in FLAGS.flag_values_dict().items():
        if type(value) == float:
            print('\t%s: %f' % (name, value))
        elif type(value) == int:
            print('\t%s: %d' % (name, value))
        elif type(value) == str:
            print('\t%s: %s' % (name, value))
        elif type(value) == bool:
            print('\t%s: %s' % (name, value))
        else:
            print('\t%s: %s' % (name, value))

    print('End of configuration')


def update(it, image, image_d, image_white, axis):
    image_d = tf.cond(tf.less(tf.random.uniform([], minval=0, maxval=1), 0.5),
                      lambda: tf.concat([image_d, image_white], axis=axis),
                      lambda: tf.concat([image_d, image], axis=axis))
    it = it + 1

    return it, image, image_d, image_white, axis


def duplicate(image, times, axis_mode="height", mode="train"):
    times = tf.cast(times, dtype=tf.int32)
    if axis_mode == "height":
        axis = tf.constant(0)
        tile_shape = (times, 1, 1)
    elif axis_mode == "width":
        axis = tf.constant(1)
        tile_shape = (1, times, 1)
    else:
        raise ValueError("[ERROR]: Unknown mode for duplicate: " + axis_mode)

    if mode == "train":
        image_d = tf.identity(image)
        # image_white = tf.ones_like(image, dtype=tf.float32) * 0.999
        image_white = tf.random_uniform(tf.shape(image), minval=0.94, maxval=0.999, dtype=tf.float32)
        it = tf.constant(0)
        condition = lambda it, image, image_d, image_white, axis: tf.less(it, times - 1)
        _, _, image_d, _, _ = tf.while_loop(condition, update, (it, image, image_d, image_white, axis),
                                            shape_invariants=(it.get_shape(), tf.TensorShape([None, None, 3]),
                                                              tf.TensorShape([None, None, None]),
                                                              tf.TensorShape([None, None, 3]), axis.get_shape()))
    elif mode == "val":
        image_d = tf.tile(image, tile_shape)
    else:
        raise ValueError("[ERROR]: Unknown mode for duplicate: " + mode)

    return image_d


def process_singe_image(image_path, FLAGS, mode):
    image = tf.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    assertion = tf.assert_equal(tf.shape(image)[2], 3, message="image does not have 3 channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    # scale image, new min(height,  width) = FLAGS.image_size
    with tf.name_scope("scaling"):
        h, w, _ = tf.shape(image)
        scale = tf.cast(FLAGS.image_size, dtype=tf.float32) / tf.cond(tf.less(h, w), lambda: w, lambda: h)
        image = tf.cond(tf.less_equal(scale, 1.0), lambda: tf.identity(image),
                        lambda: tf.image.resize_bilinear(image, [tf.cast(tf.floor(scale * h), dtype=tf.int32),
                                                                 tf.cast(tf.floor(scale * w), dtype=tf.int32)]))

    with tf.name_scope("extrapolate"):
        with tf.name_scope("height"):
            h, _, _ = tf.shape(image)
            scale_h = tf.cast(FLAGS.image_size, dtype=tf.float32) / h
            image = tf.cond(tf.less(scale_h, 2.0), lambda: tf.identity(image),
                            duplicate(image, tf.floor(scale_h), "height", mode))

        with tf.name_scope("width"):
            _, w, _ = tf.shape(image)
            scale_w = tf.cast(FLAGS.image_size, dtype=tf.float32) / w
            image = tf.cond(tf.less(scale_w, 2.0), lambda: tf.identity(image),
                            duplicate(image, tf.floor(scale_w), "height", mode))

    with tf.name_scope("pad"):
        h, w, _ = tf.shape(image)
        h_diff, w_diff = FLAGS.image_size - h, FLAGS.image_size - w
        assert_positive_hdiff = tf.assert_greater_equal(h_diff, 0)
        assert_positive_wdiff = tf.assert_greater_equal(w_diff, 0)
        with tf.control_dependencies([assert_positive_hdiff, assert_positive_wdiff]):
            image = tf.pad(image, ([0, h_diff], [0, w_diff], [0, 0]), constant_values=0.999)

    image = tf.expand_dims(image, 0)
    with tf.name_scope("brightness_contrast_hue_saturation"):
        image = tf.image.random_brightness(image, FLAGS.max_delta)
        image = tf.image.random_contrast(image, 0, FLAGS.max_delta)
        image = tf.image.random_hue(image, FLAGS.max_delta)
        image = tf.image.random_saturation(image, 0, FLAGS.max_saturation_delta)

    with tf.name_scope("random_crop"):
        random_size = tf.random_uniform([], minval=0.6, maxval=1.0, dtype=tf.float32)
        image = tf.image.crop_and_resize(image, boxes=[[0, 0, random_size, random_size]], box_ind=[0],
                                         crop_size=[FLAGS.image_size, FLAGS.image_size])

    return image


def pre_process(image_paths_tensor, FLAGS, mode='train'):
    with tf.variable_scope('pre-process'):
        # image_paths_list = data.images_path
        # image_paths_tensor = tf.convert_to_tensor(image_paths_list, dtype=tf.string)
        image_batch = tf.map_fn(lambda image_path: process_singe_image(image_path, FLAGS, mode), image_paths_tensor)
        image_batch = tf.stack(image_batch, axis=0)
        return image_batch
        # return image_batch, tf.convert_to_tensor(data.labels, dtype=tf.int32)


# def val_pre_process(data_dict: dict, FLAGS):
#     data = collections.namedtuple('data', 'images_path, labels')
#     enrollment_dict = {}
#     for label, images_path in data_dict.items():
#         dummy_data = data(images_path=images_path, labels=[0])
#         enrollment_dict[label], _ = pre_process(dummy_data, FLAGS, mode='val')
#     return enrollment_dict


def enroll(net, image_path_tensor, FLAGS):
    with tf.variable_scope('enroll'):
        images = pre_process(image_path_tensor, FLAGS, mode='val')
        embeddings = net.forward_pass(images)
        return tf.reduce_mean(embeddings, axis=0)

def infer(net, image_path_tensor, FLAGS):
    with tf.variable_scope('infer'):
        images = pre_process(image_path_tensor, FLAGS, mode='val')
        return net.forward_pass(images)


def get_closest_emb_label(enrolled_emb_dic: dict, embedding_list, np_ord=2):
    labels = []
    for emb in embedding_list:
        min_dist = sys.maxsize
        closest_lab = None
        for l, l_emb in enrolled_emb_dic.items():
            dist = np.linalg.norm((emb - l_emb), ord=np_ord)
            if dist < min_dist:
                min_dist = dist
                closest_lab = l
        labels.append(closest_lab)
    return labels


def validate(sess: tf.Session, net, val_enroll_dict: dict, val_batch_dict: dict, FLAGS):
    enrolled_emb_dict = {}
    images_path_tensor = tf.placeholder(tf.string, shape=[None, ], name='images_path_tensor')
    _enroll_embeddings = enroll(net, images_path_tensor, FLAGS)
    _embedding_list = infer(net, images_path_tensor, FLAGS)
    for l, images_paths in val_enroll_dict.items():
        enrolled_emb_dict[l] = sess.run(_enroll_embeddings, feed_dict={images_path_tensor: images_paths})

    labels = []
    predicted = []
    for l, images_paths in val_batch_dict.items():
        embedding_list = sess.run(_embedding_list, feed_dict={images_path_tensor: images_paths})
        labels.extend([l] * len(embedding_list))
        predicted.extend(get_closest_emb_label(enrolled_emb_dict, embedding_list))

    return (np.array(labels) == np.array(predicted)).mean()
