from __future__ import division, absolute_import, print_function

import collections
import functools

import tensorflow as tf

from triplet_loss import batch_hard_triplet_loss

# layers = tf.layers
layers = tf.keras.layers


# Loss Helper Functions

def semihard_mining_triplet_loss(labels, embeddings, margin=1.0):
    return tf.contrib.losses.metric_learning.triplet_semihard_loss(labels, embeddings, margin=margin)


def hard_mining_triplet_loss(labels, embeddings, margin=1.0):
    return None  # TODO: Implement


# Model Helper Classes and Functions

class ConvBlock(tf.keras.Model):
    def __init__(self, filters, stage, block, regularizer, drop_rate=0.1, kernel=3, strides=(2, 2)):
        super(ConvBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        # bn_name_base = 'bn' + str(stage) + block + '_branch'
        do_name_base = 'do' + str(stage) + block + '_branch'

        self.conv2a = layers.Conv2D(filters1, kernel_size=(1, 1), strides=strides, name=conv_name_base + '2a',
                                    kernel_regularizer=regularizer, bias_regularizer=regularizer)
        # self.bn2a = layers.BatchNormalization(name=bn_name_base + '2a')
        self.do2a = layers.Dropout(drop_rate, name=do_name_base + '2a')

        self.conv2b = layers.Conv2D(filters2, kernel_size=kernel, padding='same', name=conv_name_base + '2b',
                                    kernel_regularizer=regularizer, bias_regularizer=regularizer)
        # self.bn2b = layers.BatchNormalization(name=bn_name_base + '2b')
        self.do2b = layers.Dropout(drop_rate, name=do_name_base + '2b')

        self.conv2c = layers.Conv2D(filters3, kernel_size=(1, 1), name=conv_name_base + '2c',
                                    kernel_regularizer=regularizer, bias_regularizer=regularizer)
        # self.bn2c = layers.BatchNormalization(name=bn_name_base + '2c')

        self.conv_shortcut = layers.Conv2D(filters3, kernel_size=(1, 1), strides=strides, name=conv_name_base + '1',
                                           kernel_regularizer=regularizer, bias_regularizer=regularizer)
        # self.bn_shortcut = layers.BatchNormalization(name=bn_name_base + '1')

    def call(self, input_tensor, training=False, mask=None):
        x = self.conv2a(input_tensor)
        # x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)
        x = self.do2a(x, training=training)

        x = self.conv2b(x)
        # x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)
        x = self.do2b(x, training=training)

        x = self.conv2c(x)
        # x = self.bn2c(x, training=training)

        shortcut = self.conv_shortcut(input_tensor)
        # shortcut = self.bn_shortcut(shortcut, training=training)

        x += shortcut
        return tf.nn.relu(x)


class IdentityBlock(tf.keras.Model):

    def __init__(self, filters, stage, block, regularizer, drop_rate=0.1, kernel_size=3):
        super(IdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        # bn_name_base = 'bn' + str(stage) + block + '_branch'
        do_name_base = 'do' + str(stage) + block + '_branch'

        self.conv2a = layers.Conv2D(filters1, (1, 1), name=conv_name_base + '2a', kernel_regularizer=regularizer,
                                    bias_regularizer=regularizer)
        # self.bn2a = layers.BatchNormalization(name=bn_name_base + '2a')
        self.do2a = layers.Dropout(drop_rate, name=do_name_base + '2a')

        self.conv2b = layers.Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b',
                                    kernel_regularizer=regularizer, bias_regularizer=regularizer)
        # self.bn2b = layers.BatchNormalization(name=bn_name_base + '2b')
        self.do2b = layers.Dropout(drop_rate, name=do_name_base + '2b')

        self.conv2c = layers.Conv2D(filters3, (1, 1), name=conv_name_base + '2c', kernel_regularizer=regularizer,
                                    bias_regularizer=regularizer)
        # self.bn2c = layers.BatchNormalization(name=bn_name_base + '2c')

    def call(self, input_tensor, training=False, mask=None):
        x = self.conv2a(input_tensor)
        # x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)
        x = self.do2a(x, training=training)

        x = self.conv2b(x)
        # x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)
        x = self.do2b(x, training=training)

        x = self.conv2c(x)
        # x = self.bn2c(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)


# Model
# TODO: Add variable scope if it doesnt work
# TODO: Add dropout and regularization
class Resnet50(tf.keras.Model):

    def __init__(self, emb_size, drop_rate, regularizer):
        super(Resnet50, self).__init__(name='')

        self.conv1 = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1'
                                   , kernel_regularizer=regularizer, bias_regularizer=regularizer)
        # self.bn_conv1 = layers.BatchNormalization(name='bn_conv1')
        self.max_pool = layers.MaxPooling2D((3, 3), strides=(2, 2), name='mx_pool1')

        self.l2a = ConvBlock([64, 64, 256], stage=2, block='a', drop_rate=drop_rate, regularizer=regularizer,
                             strides=(1, 1))
        self.l2b = IdentityBlock([64, 64, 256], stage=2, drop_rate=drop_rate, regularizer=regularizer, block='b')
        self.l2c = IdentityBlock([64, 64, 256], stage=2, drop_rate=drop_rate, regularizer=regularizer, block='c')

        self.l3a = ConvBlock([128, 128, 512], stage=3, drop_rate=drop_rate, regularizer=regularizer, block='a')
        self.l3b = IdentityBlock([128, 128, 512], stage=3, drop_rate=drop_rate, regularizer=regularizer, block='b')
        self.l3c = IdentityBlock([128, 128, 512], stage=3, drop_rate=drop_rate, regularizer=regularizer, block='c')
        self.l3d = IdentityBlock([128, 128, 512], stage=3, drop_rate=drop_rate, regularizer=regularizer, block='d')

        self.l4a = ConvBlock([256, 256, 1024], stage=4, drop_rate=drop_rate, regularizer=regularizer, block='a')
        self.l4b = IdentityBlock([256, 256, 1024], stage=4, drop_rate=drop_rate, regularizer=regularizer, block='b')
        self.l4c = IdentityBlock([256, 256, 1024], stage=4, drop_rate=drop_rate, regularizer=regularizer, block='c')
        self.l4d = IdentityBlock([256, 256, 1024], stage=4, drop_rate=drop_rate, regularizer=regularizer, block='d')
        self.l4e = IdentityBlock([256, 256, 1024], stage=4, drop_rate=drop_rate, regularizer=regularizer, block='e')
        self.l4f = IdentityBlock([256, 256, 1024], stage=4, drop_rate=drop_rate, regularizer=regularizer, block='f')

        self.l5a = ConvBlock([512, 512, 2048], stage=5, drop_rate=drop_rate, regularizer=regularizer, block='a')
        self.l5b = IdentityBlock([512, 512, 2048], stage=5, drop_rate=drop_rate, regularizer=regularizer, block='b')
        self.l5c = IdentityBlock([512, 512, 2048], stage=5, drop_rate=drop_rate, regularizer=regularizer, block='c')

        self.avg_pool = layers.AveragePooling2D((7, 7), strides=(7, 7), name='avg_pool1')

        self.flatten = layers.Flatten()
        self.fc = layers.Dense(emb_size, name='fc1', kernel_regularizer=regularizer, bias_regularizer=regularizer)

    def call(self, input_tensor, training=True, mask=None):
        x = self.conv1(input_tensor)
        # x = self.bn_conv1(x, training=training)
        x = tf.nn.relu(x)
        x = self.max_pool(x)

        x = self.l2a(x, training=training)
        x = self.l2b(x, training=training)
        x = self.l2c(x, training=training)

        x = self.l3a(x, training=training)
        x = self.l3b(x, training=training)
        x = self.l3c(x, training=training)
        x = self.l3d(x, training=training)

        x = self.l4a(x, training=training)
        x = self.l4b(x, training=training)
        x = self.l4c(x, training=training)
        x = self.l4d(x, training=training)
        x = self.l4e(x, training=training)
        x = self.l4f(x, training=training)

        x = self.l5a(x, training=training)
        x = self.l5b(x, training=training)
        x = self.l5c(x, training=training)

        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


# Network and Training op
# TODO: Check wether placeholder needed, Take care of private structure, make properties
class Network:

    def __init__(self, FLAGS, reuse=False, var_scope='network'):
        self.FLAGS = FLAGS
        self.embedding_size = FLAGS.embedding_size
        self.drop_rate = FLAGS.dropout_rate
        self.var_scope = var_scope
        self.global_step = tf.train.get_or_create_global_step()
        self.learning_rate = FLAGS.learning_rate
        self.var_scope = var_scope
        self.reuse = reuse
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

        self.net = Resnet50(self.embedding_size, self.drop_rate,
                            self.regularizer)

        if FLAGS.loss == 'semi-hard':
            self.loss_fn = functools.partial(semihard_mining_triplet_loss, margin=FLAGS.loss_margin)
        elif FLAGS.loss == 'hard':
            self.loss_fn = functools.partial(batch_hard_triplet_loss, margin=FLAGS.loss_margin)
            # raise ValueError("loss fn not implemented: " + FLAGS.loss)
        else:
            raise ValueError("unknown loss fn: " + FLAGS.loss)

    def __call__(self, inputs, labels, training=True):
        net_output = collections.namedtuple('net_output', 'embeddings, loss, l2_loss, triplet_loss, train')
        with tf.variable_scope(self.var_scope, reuse=self.reuse):
            embeddings = self.net(inputs, training=training)
        # NOTE: tf.losses.get_regularization_loss() doesnt work with tf.keras.layers,
        # So here l2 loss will always be zero.
        # If need to use regularization loss use tf.layers.
        l2_loss = tf.losses.get_regularization_loss()
        triplet_loss = self.loss_fn(labels=labels, embeddings=embeddings)
        loss = triplet_loss + l2_loss

        with tf.variable_scope("optimizer"):
            self.learning_rate = tf.train.exponential_decay(self.FLAGS.learning_rate, self.global_step,
                                                            self.FLAGS.decay_step,
                                                            self.FLAGS.decay_rate,
                                                            staircase=self.FLAGS.stair)
            incr_global_step = tf.assign(self.global_step, self.global_step + 1)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.var_scope)
                optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.FLAGS.beta)
                grads_and_vars = optimizer.compute_gradients(loss, tvars)
                train_op = optimizer.apply_gradients(grads_and_vars)

        # TODO: Add regularization loss
        return net_output(
            embeddings=embeddings,
            loss=loss,
            l2_loss=l2_loss,
            triplet_loss=triplet_loss,
            train=tf.group(loss, incr_global_step, train_op)
        )

    def forward_pass(self, inputs):
        with tf.variable_scope(self.var_scope, reuse=tf.AUTO_REUSE):
            output = self.net(inputs, training=False)
        return output
