import argparse
import collections

import tensorflow as tf

from model import Network

parser = argparse.ArgumentParser()
parser.add_argument("--model_path",
                    default='/mnt/069A453E9A452B8D/Ram/handwritten-data/experiment_sign_semi/model-5000',
                    help="path for model")
parser.add_argument("--image_path", default='./graph_serialize_utils/test.png', help="input image 224 224")
parser.add_argument("--output_dir", default='./graph_serialize_utils/model-sign', help="output folder for pb")
# parser.add_argument("--output_node", default='network/resnet50/fc1/BiasAdd', help="output operation node")

args = parser.parse_args()

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
path_tesnor = tf.convert_to_tensor(args.image_path, dtype=tf.string)
image = tf.read_file(path_tesnor)
image = tf.image.decode_png(image, channels=3)
image = tf.image.convert_image_dtype(image, dtype=tf.float32)
image = tf.expand_dims(image, axis=0)
image.set_shape([1, 224, 224, 3])

input_image = tf.image.resize_bilinear(image, size=[224, 224])
input_image.set_shape([1, 224, 224, 3])

net = Network(FLAGS)
_print = tf.Print(input_image[0, 0:5, 0:5, 0], [input_image[0, 0:5, 0:5, 0]], message="[IMAGE] : ",
                  first_n=1)
with tf.control_dependencies([_print]):
    output = net.forward_pass(image)

# Weight Initializer
# train_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="network")
# weight_initializer = tf.train.Saver(train_var_list)

# Start the session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sv = tf.train.Supervisor(save_summaries_secs=0, saver=None)
# with sv.managed_session(config=config) as sess:
with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    # weight_initializer.restore(sess, args.model_path)
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], args.output_dir)
    print(sess.run(output))
