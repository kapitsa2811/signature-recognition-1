import argparse
import collections

import tensorflow as tf

from model import Network

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", default='./graph_serialize_utils/test.png', help="input image 224 224")
parser.add_argument("--model_dir", default='./graph_serialize_utils/model-sign', help="output folder for pb")

args = parser.parse_args()

# FLAGS for model, Parameters should be same as training
_FLAGS = collections.namedtuple('_FLAGS', 'embedding_size, loss, learning_rate, image_size, loss_margin')

FLAGS = _FLAGS(
    loss='semi-hard',
    embedding_size=128,
    learning_rate=0.0001,
    image_size=224,
    loss_margin=0.5
)

print('[INFO]: getting validation model')
net = Network(FLAGS)

path_tensor = tf.placeholder(tf.string, shape=[], name='image_path_tensors')
image = tf.read_file(path_tensor, name='image_bytes')
image = tf.image.decode_png(image, channels=3)
image = tf.image.convert_image_dtype(image, dtype=tf.float32)
image = tf.expand_dims(image, axis=0)
image = tf.image.resize_bilinear(image, size=[224, 224])
image.set_shape([1, 224, 224, 3])

# input_image = tf.placeholder(tf.float32, shape=[None, FLAGS.image_size, FLAGS.image_size, 3], name='input_images')
input_image = tf.identity(image, name='input_images')
output = net.forward_pass(input_image)
embeddings = tf.identity(output, name='embeddings')
output_node_names = ['embeddings']

with tf.Session() as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], args.model_dir)
    print(sess.run(output, feed_dict={path_tensor: args.image_path}))
