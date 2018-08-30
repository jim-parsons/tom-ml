import tensorflow as tf
import numpy as np

CHECK_POINT_DIR = './runs/1535611282/checkpoints'
LABELS_DIR = './runs/1535611282/labels.txt'
INCEPTION_MODEL_FILE = './model/classify_image_graph_def.pb'

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'  # inception-v3模型中代表瓶颈层结果的张量名称
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'  # 图像输入张量对应的名称

file_path = 'E:/flower_photos/daisy/5547758_eea9edfd54_n.jpg'
BOTTLENECK_TENSOR_SIZE = 2048

image_data = tf.gfile.FastGFile(file_path, 'rb').read()
label_lines = [line.rstrip() for line in tf.gfile.GFile(LABELS_DIR)]

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

        # softmax_tensor = graph.get_operation_by_name('evaluation/ArgMax').outputs[0]
        softmax_tensor = graph.get_operation_by_name('final_training_ops/Softmax').outputs[0]

        bottleneck_values = bottleneck_values.reshape(-1, BOTTLENECK_TENSOR_SIZE)
        predictions = sess.run(softmax_tensor, {input_x: bottleneck_values})
        predictions = predictions[0]
        # [i: j: s]
        # [2 4 1 3 0]
        # [0 3 1 4 2]
        # top_k = predictions.argsort()[-len(predictions):][::-1]
        top_k = predictions.argsort()[::-1]
        print(top_k)

        predict_result = {}

        for node_id in top_k:
            flower = label_lines[node_id]
            score = str(predictions[node_id])
            predict_result[flower] = score
        print(predict_result)

# if y_test is not None:
#     correct_predictions = float(sum(predictions == y_test))
#     print('\nTotal number of test examples: {}'.format(len(y_test)))
#     print('Accuracy: {:g}'.format(correct_predictions / float(len(y_test))))
