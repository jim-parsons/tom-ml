import tensorflow as tf
import numpy as np

CHECK_POINT_DIR = './model/model/'
INCEPTION_MODEL_FILE = './model/inception_v3.ckpt'

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'  # inception-v3模型中代表瓶颈层结果的张量名称
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'  # 图像输入张量对应的名称

file_path = ''
y_test = [4]

image_data = tf.gfile.FastGFile(file_path, 'rb').read()

checkpoint_file = tf.train.latest_checkpoint(CHECK_POINT_DIR)

with tf.Graph().as_default() as graph:
    with tf.Session() as sess:
        # 读取训练好的inception-v3模型
        with tf.gfile.FastGFile(INCEPTION_MODEL_FILE, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def, return_elements=[BOTTLENECK_TENSOR_NAME,
                                                                                              JPEG_DATA_TENSOR_NAME])

        # 将四维数组压缩成一维数组，由于全连接层输入时有batch的维度，所以用列表作为输入
        bottleneck_values = sess.run(bottleneck_tensor, {jpeg_data_tensor: image_data})
        bottleneck_values = np.squeeze(bottleneck_values)

        saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # 通过名字从图中获取输入占位符
        input_x = graph.get_operation_by_name('BottleneckInputPlaceholder').outputs[0]

        predictions = graph.get_operation_by_name('evaluation/ArgMax').outputs[0]

        all_predictions = []

        all_predictions = sess.run(predictions, {input_x: bottleneck_values})

if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print('\nTotal number of test examples: {}'.format(len(y_test)))
    print('Accuracy: {:g}'.format(correct_predictions / float(len(y_test))))