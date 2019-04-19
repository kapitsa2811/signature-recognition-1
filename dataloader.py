from __future__ import division, absolute_import, print_function

import numpy as np
import tensorflow as tf
import os
import re
import collections


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
        assert re.match("^[a-z]0[0-9]$", label)

    return images_dict, len(images_list), labels


class Dataloader:

    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.train_dict = {}

        self.val_dict = {}
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
            images_path.extend(np.random.choice(data_dict[l], size=self.FLAGS.batch_image_per_label, replace=False))

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


def duplicate(image, times, mode="height"):
    return image


def process_singe_image(image_path, FLAGS):
    image = tf.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    assertion = tf.assert_equal(tf.shape(image)[2], 3, message="image does not have 3 channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    # scale image, new min(height,  width) = FLAGS.image_size
    with tf.name_scope("scaling"):
        h, w, _ = tf.shape(image)
        scale = tf.cond(tf.less(h, w), lambda: w, lambda: h) / tf.cast(FLAGS.image_size, dtype=tf.float32)
        image = tf.cond(tf.less_equal(scale, 1.0), lambda: tf.identity(image),
                        lambda: tf.image.resize_bilinear(image, [tf.cast(tf.ceil(scale * h), dtype=tf.int32),
                                                                 tf.cast(tf.ceil(scale * w), dtype=tf.int32)]))

    with tf.name_scope("extrapolate"):
        with tf.name_scope("height"):
            h, _, _ = tf.shape(image)
            scale_h = tf.cast(FLAGS.image_size, dtype=tf.float32) / h
            image = tf.cond(tf.less(scale_h, 2.0), lambda: tf.identity(image),
                            duplicate(image, tf.floor(scale_h), "height"))

        with tf.name_scope("width"):
            _, w, _ = tf.shape(image)
            scale_w = tf.cast(FLAGS.image_size, dtype=tf.float32) / w
            image = tf.cond(tf.less(scale_w, 2.0), lambda: tf.identity(image),
                            duplicate(image, tf.floor(scale_w), "height"))

    with tf.name_scope("pad"):
        image = tf.identity(image)
    # with tf.name_scope("random_crop") TODO: check whether required

    return image


def pre_process(data, FLAGS):
    image_paths_list = data.images_path
    image_paths_tensor = tf.convert_to_tensor(image_paths_list, dtype=tf.string)
    image_batch = tf.map_fn(lambda image_path: process_singe_image(image_path, FLAGS), image_paths_tensor)
    return image_batch, tf.convert_to_tensor(data.labels, dtype=tf.int32)
