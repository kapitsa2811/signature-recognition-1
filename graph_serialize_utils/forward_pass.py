import argparse
import collections

import tensorflow as tf

from model import Network
from utils import pre_process

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", default='./graph_serialize_utils/testR.png', help="input image 224 224")
parser.add_argument("--model_dir", default='./graph_serialize_utils/model-sign-pre', help="output folder for pb")

args = parser.parse_args()

# FLAGS for model, Parameters should be same as training
_FLAGS = collections.namedtuple('_FLAGS', 'embedding_size, loss, learning_rate, image_size, loss_margin, dropout_rate')
FLAGS = _FLAGS(
    loss='semi-hard',
    embedding_size=128,
    learning_rate=0.0001,
    image_size=224,
    loss_margin=0.5,
    dropout_rate=0.1
)

# Model
print('[INFO]: getting validation model')
net = Network(FLAGS)

path_tensor = tf.placeholder(tf.string, shape=[None,], name='image_path_tensors')
images_tensor = pre_process(path_tensor, FLAGS)

# input_image = tf.placeholder(tf.float32, shape=[None, FLAGS.image_size, FLAGS.image_size, 3], name='input_images')
input_images = tf.identity(images_tensor, name='input_images')
output = net.forward_pass(input_images)
embeddings = tf.identity(output, name='embeddings')
output_node_names = ['embeddings']

with tf.Session() as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], args.model_dir)
    print(sess.run(output, feed_dict={path_tensor: [args.image_path]}))
