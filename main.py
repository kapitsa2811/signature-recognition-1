from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import math
import tensorflow as tf
from utils import print_configuration_op
from dataloader import Dataloader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

Flags = tf.app.flags

# The system parameter
Flags.DEFINE_string('output_dir', None, 'The output directory of the checkpoint')
Flags.DEFINE_string('summary_dir', None, 'The dirctory to output the summary')
Flags.DEFINE_string('mode', 'train', 'The mode of the model train, test.')
Flags.DEFINE_string('checkpoint', None, 'If provided, the weight will be restored from the provided checkpoint.'
                                        'Checkpoint folder (Latest checkpoint will be taken)')
Flags.DEFINE_boolean('pre_trained_model', False,
                     'If set True, the weight will be loaded but the global_step will still '
                     'be 0. If set False, you are going to continue the training. That is, '
                     'the global_step will be initiallized from the checkpoint, too')
Flags.DEFINE_string('vgg_ckpt', './vgg19/vgg_19.ckpt', 'path to checkpoint file for the vgg19')
# Flags.DEFINE_string('task', None, 'The task: Slomo, Slogan')
# Flags.DEFINE_string('pre_trained_model_type', 'Slomo', 'The type of pretrained model (Slomo or Slogan)')

# DataLoader Parameters
Flags.DEFINE_string('train_dir',
                    '/mnt/069A453E9A452B8D/Ram/slomo_data/DeepVideoDeblurring_Dataset_Original_High_FPS_Videos/'
                    'original_high_fps_videos',
                    'Video data folder')
Flags.DEFINE_string('val_dir', None, 'The directory to extract videos temporarily')
Flags.DEFINE_integer('batch_labels_size', 3, 'Number of labels in each batch. min 2, P')
Flags.DEFINE_integer('batch_image_per_label', 2, 'Number of images per label. min 2, K, batch size = P*K')
Flags.DEFINE_integer('batch_thread', 4, 'The number of threads to process image queue for generating batches')
Flags.DEFINE_integer('slim_num_readers', 4, 'The number reader for slim TFreader')
Flags.DEFINE_integer('tfrecord_threads', 5, 'The number of threads for tfrecord extraction.')
Flags.DEFINE_integer('resize_width', 256, 'The width of the training image')
Flags.DEFINE_integer('resize_height', 256, 'The width of the training image')
Flags.DEFINE_integer('train_data_count', None, 'The number of samples in training tfrecords')
Flags.DEFINE_integer('val_data_count', None, 'The number of samples in training tfrecords')

# model configurations
Flags.DEFINE_integer('first_kernel', 7, 'First conv kernel size in flow computation network')
Flags.DEFINE_integer('second_kernel', 5, 'First conv kernel size in flow computation network')
Flags.DEFINE_float('epsilon', 1e-12, 'The eps added to prevent nan')
Flags.DEFINE_string('perceptual_mode', 'VGG54', 'The type of feature used in perceptual loss')
Flags.DEFINE_float('reconstruction_scaling', 0.1, 'The scaling factor for the reconstruction loss')
Flags.DEFINE_float('perceptual_scaling', 1.0, 'The scaling factor for the perceptual loss')
Flags.DEFINE_float('wrapping_scaling', 1.0, 'The scaling factor for the wrapping loss')
Flags.DEFINE_float('smoothness_scaling', 50.0, 'The scaling factor for the smoothness loss')

# Trainer Parameters
Flags.DEFINE_float('learning_rate', 0.0001, 'The learning rate for the network')
Flags.DEFINE_integer('decay_step', 500000, 'The steps needed to decay the learning rate')
Flags.DEFINE_float('decay_rate', 0.1, 'The decay rate of each decay step')
Flags.DEFINE_boolean('stair', False, 'Whether perform staircase decay. True => decay in discrete interval.')
Flags.DEFINE_float('beta', 0.9, 'The beta1 parameter for the Adam optimizer')
Flags.DEFINE_integer('max_epoch', None, 'The max epoch for the training')
Flags.DEFINE_integer('max_iter', 1000000, 'The max iteration of the training')
Flags.DEFINE_integer('display_freq', 20, 'The diplay frequency of the training process')
Flags.DEFINE_integer('summary_freq', 100, 'The frequency of writing summary')
Flags.DEFINE_integer('save_freq', 10000, 'The frequency of saving checkpoint')

FLAGS = Flags.FLAGS

# Print the configuration of the model
print_configuration_op(FLAGS)
