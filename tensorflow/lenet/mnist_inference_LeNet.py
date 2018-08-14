# -*- coding: utf-8 -*-
import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# 第一层卷积的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5

# 第二层卷积的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 5

# 全连接层的节点个数
FC_SIZE = 512


def inference(input_tensor, train, regularizer):
    # 第一层 卷积层
    with tf.variable_scope('layer1-conv1'):
        # CONV1_SIZE, CONV1_SIZE 代表过滤器尺寸
        # NUM_CHANNELS 当前层的深度
        # CONV1_DEEP 过滤器的深度
        conv1_weights = tf.get_variable('weight', [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))

        conv1_biases = tf.get_variable('bias', [CONV1_DEEP], initializer=tf.constant_initializer(0.0))

        # tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
        # input:指需要做卷积的输入图像，它要求是一个Tensor，具有[batch, in_height, in_width, in_channels]这样的shape，
        # 具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，注意这是一个4维的Tensor，要求类型为float32和float64其中之一

        # filter: 相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]这样的shape，
        # [卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input的第四维

        # strides: 卷积时在图像每一维的步长，这是一个一维的向量，长度4

        # padding：string类型的量，只能是"SAME","VALID"其中之一
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')

        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # 第二层 pool层 边长为2 步长为2 上一层输出为28*28*32 这一层输出为14*14*32
    with tf.name_scope('layer2-pool1'):
        # max_pool(value, ksize, strides, padding, data_format='NHWC', name=None)
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 第三层 卷积层  输入14*14*32 输出14*14*64
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable('weight', [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))

        conv2_biases = tf.get_variable('bias', [CONV2_DEEP], initializer=tf.constant_initializer(0.0))

        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')

        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # 第四层  pool层 输入14*14*32 输出7*7*64
    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # 将第四层输出转化为第五层输入, 第四层输出7*7*64,把这个拉成一个向量
        pool_shape = pool2.get_shape().as_list()
        # 将矩阵拉成向量,长度就为=7*7*64  pool_shape[0]为batch中数据个数
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    # 第五层 输入为7*7*64=3136 一维向量,输出512
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable('weight', [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))

        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc1_weights))

        fc1_biases = tf.get_variable('bias', [FC_SIZE], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)

        if train:
            # 训练时,随机将部分节点输出改为0,避免过拟合
            fc1 = tf.nn.dropout(fc1, 0.5)

    # 第六层 输入为512向量,输出为10向量,通过softmax得到分类结果
    with tf.variable_scope('layer5-fc2'):
        fc2_weights = tf.get_variable('weight', [FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))

        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc2_weights))

        fc2_biases = tf.get_variable('bias', [NUM_LABELS], initializer=tf.constant_initializer(0.1))

        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit
