from __future__ import division, absolute_import, print_function

import collections
import os

import numpy as np


def prepare_image_paths(image_dir, dataset_name):
    images_list = os.listdir(image_dir)
    images_list = [image_path for image_path in images_list if image_path.lower().endswith(".png")]
    if dataset_name == 'kaggle_signature':
        labels = [image_path.split("_")[1] for image_path in images_list]
    elif dataset_name == 'SigComp2009-training':
        labels = [image_path[7:10] for image_path in images_list]
    else:
        raise ValueError("Unknown dataset: " + dataset_name)
    images_list = [os.path.join(image_dir, image_path) for image_path in images_list]
    images_dict = {}
    for ind, image_path in enumerate(images_list):
        label = labels[ind]
        if label in images_dict:
            images_dict[label].append(image_path)
        else:
            images_dict[label] = [image_path]

    labels = list(set(labels))
    # for label in labels:
    #     assert re.match('^[a-z]0[0-9]$', label)

    return images_dict, len(images_list), labels


class DataLoader:

    def __init__(self, FLAGS):
        self.FLAGS = FLAGS

        self.val_enroll_images_path, self.val_enroll_dict = [], {}
        self.enrollment_size = self.FLAGS.val_enrollment_size
        self.train_dict, self.train_len, self.labels = prepare_image_paths(self.FLAGS.train_dir,
                                                                           FLAGS.train_dataset_name)
        self.val_dict, self.val_len, self.val_labels = prepare_image_paths(self.FLAGS.val_dir, FLAGS.val_dataset_name)
        print("[INFO] train labels:", self.labels)
        print("[INFO] val labels:", self.val_labels)

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
        data_size = collections.namedtuple('data_size', 'train, val,train_labels, val_labels')
        return data_size(
            train=self.train_len,
            train_labels=len(self.labels),
            val=self.val_len,
            val_labels=len(self.val_labels)
        )

    def get_val_batch(self):
        val_batch_dict = {}
        all_labels = self.val_labels
        data_dict = self.val_dict

        for l in all_labels:
            # print('[LABEL]: ', l, len(data_dict[l]))
            if len(data_dict[l]) < self.enrollment_size:
                continue
            if len(data_dict[l]) < self.enrollment_size + self.FLAGS.val_batch_image_per_label:
                # print('[WARNING] Skipped validation label:', l, "from batch. size:", len(data_dict[l]))
                continue
            inserted = 0
            images_path = []
            while inserted < self.FLAGS.val_batch_image_per_label:
                ran_image_path = np.random.choice(data_dict[l])
                if ran_image_path not in self.val_enroll_images_path:
                    images_path.append(ran_image_path)
                    inserted += 1
            val_batch_dict[l] = images_path
        return val_batch_dict

    def get_val_enrollment_batch(self):
        self.val_enroll_images_path, self.val_enroll_dict = [], {}
        data = collections.namedtuple('data', 'images_path, val_enroll_dict')
        all_labels = self.val_labels
        data_dict = self.val_dict

        for l in all_labels:
            if len(data_dict[l]) < self.enrollment_size:
                print('[WARNING] Skipped validation label:', l, ". size:", len(data_dict[l]))
                continue
            _batch = np.random.choice(data_dict[l], size=self.enrollment_size, replace=False)
            self.val_enroll_dict[l] = _batch
            self.val_enroll_images_path.extend(_batch)

        return data(
            images_path=self.val_enroll_images_path,
            val_enroll_dict=self.val_enroll_dict
        )
