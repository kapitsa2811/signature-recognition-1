from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf

from dataloader import DataLoader
from model import Network
from utils import pre_process
from utils import print_configuration_op

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

# DataLoader Parameters
Flags.DEFINE_string('train_dir', None, 'The train data directory')
Flags.DEFINE_string('val_dir', None, 'The validation data directory')
Flags.DEFINE_integer('batch_labels_size', 8, 'Number of labels in each batch. min 2, P')
Flags.DEFINE_integer('batch_image_per_label', 4, 'Number of images per label. min 2, K, batch size = P*K')
Flags.DEFINE_integer('val_batch_image_per_label', 10, 'Number of images per label for validation.')
Flags.DEFINE_integer('val_enrollment_size', 5, 'Number of images per label for enrollment size.')
Flags.DEFINE_integer('batch_thread', 4, 'The number of threads to process image queue for generating batches')
Flags.DEFINE_integer('image_size', 224, 'Image crop size (image_size x image_size)')
Flags.DEFINE_float('max_delta', 0.4, 'max delta for brightness, contrast and hue [0,0.5]')
Flags.DEFINE_float('max_saturation_delta', 2, 'max delta for saturation [0,3]')

# model configurations
Flags.DEFINE_integer('embedding_size', 128, 'output embedding size')
Flags.DEFINE_string('loss', 'semi-hard', 'primary loss function. (semi-hard: triplet loss with semi-hard negative '
                                         'mining | hard: triplet loss with hard negative mining)')
Flags.DEFINE_float('loss_margin', 1.0, 'The learning rate for the network')

# Trainer Parameters
Flags.DEFINE_float('learning_rate', 0.001, 'The learning rate for the network')
Flags.DEFINE_integer('decay_step', 500000, 'The steps needed to decay the learning rate')
Flags.DEFINE_float('decay_rate', 0.1, 'The decay rate of each decay step')
Flags.DEFINE_boolean('stair', False, 'Whether perform staircase decay. True => decay in discrete interval.')
Flags.DEFINE_float('beta', 0.9, 'The beta1 parameter for the Adam optimizer')
Flags.DEFINE_integer('max_iter', 1000000, 'The max iteration of the training')
Flags.DEFINE_integer('display_freq', 20, 'The diplay frequency of the training process')
Flags.DEFINE_integer('summary_freq', 100, 'The frequency of writing summary')
Flags.DEFINE_integer('save_freq', 1000, 'The frequency of saving checkpoint')

FLAGS = Flags.FLAGS

# Print the configuration of the model
print_configuration_op(FLAGS)

# Check Directories
if FLAGS.output_dir is None or FLAGS.summary_dir is None:
    raise ValueError('The output directory and summary directory are needed')

if FLAGS.train_dir is None or FLAGS.val_dir is None:
    raise ValueError('The train directory and val directory are needed')

# Check the output directory to save the checkpoint
if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)

# Check the summary directory to save the event
if not os.path.exists(FLAGS.summary_dir):
    os.mkdir(FLAGS.summary_dir)

# Initialize DataLoader
data_loader = DataLoader(FLAGS)
data_size = data_loader.get_data_size()
print('[DATA LOADED] train size: %d with %d writers, val size: %d with %d writers' % (
    data_size.train, data_size.train_labels, data_size.val, data_size.val_labels))

# Defining Placeholder
images_path_tensor = tf.placeholder(tf.string, shape=[None, ], name='image_path_tensors')
images_label_tensor = tf.placeholder(tf.int32, shape=[None, ], name='image_lables_tensor')
images_path_tensor_val = tf.placeholder(tf.string, shape=[None, ], name='images_path_tensor_val')
# # A hack to add validation accuracy in tensorboard
val_accuracy = tf.placeholder(tf.double, shape=[], name='val_accuracy')

# Training
print('[INFO]: getting training model')
net = Network(FLAGS)
images_tensor = pre_process(images_path_tensor, FLAGS)
_print_shape = tf.Print(images_tensor, [tf.shape(images_tensor)], message="[INFO] current train batch shape: ",
                        first_n=1)
with tf.control_dependencies([_print_shape]):
    train = net(images_tensor, images_label_tensor)

# Validation
# val_image_tensor = pre_process(images_path_tensor_val, FLAGS, mode='val')
# val_forward_pass = net.forward_pass(images_path_tensor_val)

# Add summaries
print('[INFO]: Adding summaries')
tf.summary.histogram("embeddings_histogram", train.embeddings)
tf.summary.image("train_images", images_tensor, max_outputs=10)
tf.summary.scalar("train_loss", train.loss)
tf.summary.scalar("learning_rate", net.learning_rate)
tf.summary.scalar("val_accuracy", val_accuracy)

# Define the saver and weight initiallizer
saver = tf.train.Saver(max_to_keep=10)

# Get trainable variable
train_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="network")
weight_initializer = tf.train.Saver(train_var_list)

# Start the session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Use supervisor to coordinate all queue and summary writer
# TODO: Deprecated, Update with tf.train.MonitoredTrainingSession
sv = tf.train.Supervisor(logdir=FLAGS.summary_dir, save_summaries_secs=0, saver=None)

with sv.managed_session(config=config) as sess:
    # TODO: check the saving checkpoint part for below both
    if (FLAGS.checkpoint is not None) and (FLAGS.pre_trained_model is False):
        print('[INFO]: Loading model from the checkpoint...')
        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint)
        saver.restore(sess, checkpoint)

    elif (FLAGS.checkpoint is not None) and (FLAGS.pre_trained_model is True):
        print('[INFO]: Loading weights from the pre-trained model')
        weight_initializer.restore(sess, FLAGS.checkpoint)

    print('[INFO] Optimization starts!!!')
    start = time.time()
    val_acc = 0

    for step in range(FLAGS.max_iter):

        batch = data_loader.get_train_batch()
        images_path, images_label = batch.images_path, batch.labels

        # Validation
        # TODO: add validation images to tensorboard
        # if ((step + 1) % FLAGS.display_freq) == 0 or ((step + 1) % FLAGS.summary_freq) == 0:
        #     # print("[INFO]: Validation Step.")
        #     val_enroll_dict = data_loader.get_val_enrollment_batch().val_enroll_dict
        #     validation_batch_dict = data_loader.get_val_batch()
        #     val_acc = validate(sess, val_forward_pass, images_path_tensor_val, val_enroll_dict, validation_batch_dict,
        #                        FLAGS)

        fetches = {
            "train": train.train,
            "global_step": net.global_step,
        }

        if ((step + 1) % FLAGS.display_freq) == 0:
            fetches["training_loss"] = train.loss
            fetches["learning_rate"] = net.learning_rate

        if ((step + 1) % FLAGS.summary_freq) == 0:
            fetches["summary"] = sv.summary_op

        results = sess.run(fetches, feed_dict={images_path_tensor: images_path, images_label_tensor: images_label,
                                               val_accuracy: val_acc})

        if ((step + 1) % FLAGS.summary_freq) == 0:
            print('[INFO]: Recording summary !!!!')
            sv.summary_writer.add_summary(results['summary'], results['global_step'])

        if ((step + 1) % FLAGS.display_freq) == 0:
            print("[PROGRESS]: global step: %d | learning rate: %f | training_loss: %f | val_accuracy %0.1f" % (
                results['global_step'], results['learning_rate'], results['training_loss'], val_acc))

        if ((step + 1) % FLAGS.save_freq) == 0:
            print('[INFO]: Save the checkpoint !!!!')
            # TODO: Check wehter result['global_step'] needs to be passed instead
            saver.save(sess, os.path.join(FLAGS.output_dir, 'model'), global_step=net.global_step)

    print('[INFO]: Optimization done!!!!!!!!!!!!')
