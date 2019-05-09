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

        # if len(self.labels) > len(self.val_labels):
        #     print("[WARNING]: some label are missing from validation set")
        #
        # if not all(l in self.labels for l in self.val_labels):
        #     raise ValueError("[ERROR]: validation labels not subset of train labels")

    def get_train_batch(self):
        data = collections.namedtuple('data', 'images_path, labels')
        all_labels = self.labels
        data_dict = self.train_dict

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

    def get_val_batch(self):
        val_batch_dict = {}
        all_labels = self.val_labels
        data_dict = self.val_dict

        for l in all_labels:
            assert len(data_dict[l]) > self.enrollment_size + self.FLAGS.val_batch_image_per_label
            inserted = 0
            images_path = []
            while inserted < self.FLAGS.val_batch_image_per_label:
                ran_image_path = np.random.choice(data_dict[l])
                if ran_image_path not in self.val_enroll_images_path:
                    images_path.append(ran_image_path)
                    inserted += 1
            val_batch_dict[l] = images_path
        return val_batch_dict

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


