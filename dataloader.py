from __future__ import division, absolute_import, print_function

import collections
import os
import re

import numpy as np
import tensorflow as tf


def prepare_image_paths(image_dir):
    images_list = os.listdir(image_dir)
    images_list = [image_path for image_path in images_list if image_path.endswith(".png")]
    labels = [image_path[0:3] for image_path in images_list]
    images_list = [os.path.join(image_dir, image_path) for image_path in images_list]
    images_dict = {}
    for ind, image_path in enumerate(images_list):
        label = labels[ind]
        if label in images_dict:
            images_dict[label].append(image_path)
        else:
            images_dict[label] = [image_path]

    labels = list(set(labels))
    for label in labels:
        assert re.match('^[a-z]0[0-9]$', label)

    return images_dict, len(images_list), labels


class DataLoader:

    def __init__(self, FLAGS):
        self.FLAGS = FLAGS

        self.val_enroll_images_path, self.val_enroll_dict = [], {}
        self.enrollment_size = 0
        self.train_dict, self.train_len, self.labels = prepare_image_paths(self.FLAGS.train_dir)
        self.val_dict, self.val_len, self.val_labels = prepare_image_paths(self.FLAGS.val_dir)

        if len(self.labels) > len(self.val_labels):
            print("[WARNING]: some label are missing from validation set")

        if not all(l in self.labels for l in self.val_labels):
            raise ValueError("[ERROR]: validation labels not subset of train labels")

    def get_batch(self, name='train'):
        assert name == 'train' or name == 'val'
        data = collections.namedtuple('data', 'images_path, labels')

        if name == 'train':
            all_labels = self.labels
            data_dict = self.train_dict
        else:
            all_labels = self.val_labels
            data_dict = self.val_dict

        labels_list = np.random.choice(all_labels, size=self.FLAGS.batch_labels_size, replace=False)
        images_path = []
        labels = []
        for l in labels_list:
            labels.extend([self.labels.index(l)] * self.FLAGS.batch_image_per_label)
            if name == 'train':
                images_path.extend(np.random.choice(data_dict[l], size=self.FLAGS.batch_image_per_label, replace=False))
            else:
                assert len(data_dict[l]) > self.enrollment_size + self.FLAGS.batch_image_per_label
                inserted = 0
                while inserted < self.FLAGS.batch_image_per_label:
                    ran_image_path = np.random.choice(data_dict[l])
                    if ran_image_path not in self.val_enroll_images_path:
                        images_path.append(ran_image_path)
                        inserted += 1

        return data(
            images_path=images_path,
            labels=labels
        )  # TODO: check whether list will work

    def get_data_size(self):
        data_size = collections.namedtuple('data_size', 'train, val')
        return data_size(
            train=self.train_len,
            val=self.val_len
        )

    def get_val_enrollment_batch(self, enrollment_size=2):
        self.enrollment_size = enrollment_size
        self.val_enroll_images_path, self.val_enroll_dict = [], {}
        data = collections.namedtuple('data', 'images_path, label_dict')
        all_labels = self.val_labels
        data_dict = self.val_dict

        for l in all_labels:
            assert len(data_dict[l]) > enrollment_size
            _batch = np.random.choice(data_dict[l], size=enrollment_size, replace=False)
            self.val_enroll_dict[l] = _batch
            self.val_enroll_images_path.extend(_batch)

        return data(
            images_path=self.val_enroll_images_path,
            label_dict=self.val_enroll_dict
        )


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


def pre_process(data, FLAGS, mode="train"):
    image_paths_list = data.images_path
    image_paths_tensor = tf.convert_to_tensor(image_paths_list, dtype=tf.string)
    image_batch = tf.map_fn(lambda image_path: process_singe_image(image_path, FLAGS, mode), image_paths_tensor)
    image_batch = tf.stack(image_batch, axis=0)
    return image_batch, tf.convert_to_tensor(data.labels, dtype=tf.int32)


def pre_process_enroll(data_dict: dict, FLAGS):
    data = collections.namedtuple('data', 'images_path, labels')
    enrollement_dict = {}
    for label, images_path in data_dict.items():
        dummy_data = data(images_path=images_path, labels=[0])
        enrollement_dict[label], _ = pre_process(dummy_data, FLAGS, mode='val')
    return enrollement_dict
