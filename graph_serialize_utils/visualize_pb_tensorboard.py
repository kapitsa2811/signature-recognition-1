import tensorflow as tf
from tensorflow.python.platform import gfile

with tf.Session() as sess:
    model_filename = './graph_serialize_utils/pb/output_graph.pb'
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)

LOG_DIR = './graph_serialize_utils/log'
train_writer = tf.summary.FileWriter(LOG_DIR)
train_writer.add_graph(sess.graph)
train_writer.flush()
train_writer.close()
