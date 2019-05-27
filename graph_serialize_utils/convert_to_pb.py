import argparse
import collections
import os

import tensorflow as tf

from model import Network

parser = argparse.ArgumentParser()
parser.add_argument("--model_path",
                    default='/mnt/069A453E9A452B8D/Ram/handwritten-data/experiment_2/model-83000',
                    help="path for model")
parser.add_argument("--output_dir", default='./graph_serialize_utils/pb', help="output folder for pb")
# parser.add_argument("--output_node", default='network/resnet50/fc1/BiasAdd', help="output operation node")

args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

meta_path = args.model_path + '.meta'  # Your .meta file

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
input_image = tf.placeholder(tf.float32, shape=[None, FLAGS.image_size, FLAGS.image_size, 3], name='input_images')
net = Network(FLAGS)
output = net.forward_pass(input_image)
embeddings = tf.identity(output, name='embeddings')
output_node_names = ['embeddings']

# Weight Initializer
train_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="network")
weight_initializer = tf.train.Saver(train_var_list)

# Start the session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sv = tf.train.Supervisor(save_summaries_secs=0, saver=None)
with sv.managed_session(config=config) as sess:
    weight_initializer.restore(sess, args.model_path)

    # Put name of all nodes in txt file
    output_nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
    with open("nodes.txt", 'w') as file:
        for _node in output_nodes:
            file.write(_node + "\n")

    # Remove Locking (Useful in batch normalization case)
    gd = sess.graph.as_graph_def()

    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        gd,
        output_node_names)

    output_path = os.path.join(args.output_dir, 'output_graph.pb')
    # Save the frozen graph
    with open(output_path, 'wb') as file:
        file.write(frozen_graph_def.SerializeToString())
