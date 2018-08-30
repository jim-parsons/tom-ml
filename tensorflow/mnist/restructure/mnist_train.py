# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8  # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 学习衰减率

REGULARIZATION_RATE = 0.0001  # 正则化在损失函数中的系数
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率
MODEL_SAVE_PATH = './model'
MODEL_NAME = 'model.ckpt'


def train(mnist):
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    # 在所有代表神经网络参数的辩论上使用滑动平均
    # tf.trainable_variables()返回所有没有指定 trainable=False的参数
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 交叉熵
    # 当分类只有一个正确结果的时候,使用tf.nn.sparse_softmax_cross_entropy_with_logits()
    # 第一个参数为不包含softmax的前向传播结果
    # 第二个参数是训练数据的正确答案
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 总损失 交叉熵损失 + 正则化损失和
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # tf.train.exponential_decay 学习率指数衰减法
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,  # 基础学习率
                                               global_step,  # 当前迭代轮数
                                               mnist.train.num_examples / BATCH_SIZE,  # 总共需要的次数
                                               LEARNING_RATE_DECAY)  # 学习率衰减速度

    # 梯度下降
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 每过一遍数据,需要反向传播跟新参数,又要跟新每一个参数的滑动平均,
    # train_step 反向传播更新参数
    # variable_averages_op 跟新参数的滑动平均
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print('After %d training step(s), loss on training batch is %g' % (step, loss_value))
                # saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
                saver.save(sess, './model/model.ckpt', global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets('../../../data/mnist', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
