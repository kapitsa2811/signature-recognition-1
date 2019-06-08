import argparse
import collections

import tensorflow as tf

from model import Network

parser = argparse.ArgumentParser()
parser.add_argument("--model_path",
                    default='/handwritten-data/experiment_sign_semi/model-5000',
                    help="path for model")
parser.add_argument("--output_dir", default='./graph_serialize_utils/model-sign', help="output folder for pb")
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

# Model
print('[INFO]: getting validation model')
net = Network(FLAGS)

path_tensor = tf.placeholder(tf.string, shape=[], name='image_path_tensors')
image = tf.read_file(path_tensor)
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

# Weight Initializer
train_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="network")
weight_initializer = tf.train.Saver(train_var_list)

# Builder
builder = tf.saved_model.builder.SavedModelBuilder(args.output_dir)

# Start the session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sv = tf.train.Supervisor(save_summaries_secs=0, saver=None)
# with sv.managed_session(config=config) as sess:
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    weight_initializer.restore(sess, args.model_path)

    # Put name of all nodes in txt file
    output_nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
    with open("nodes.txt", 'w') as file:
        for _node in output_nodes:
            file.write(_node + "\n")

    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.TRAINING], strip_default_attrs=True)
    builder.add_meta_graph([tf.saved_model.tag_constants.SERVING], strip_default_attrs=True)

builder.save()
