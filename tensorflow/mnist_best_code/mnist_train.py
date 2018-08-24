import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference

# 配置神经网络参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVEING_AVERAGE_DEACY = 0.99

# 模型保存路径和文件名
MODEL_SAVE_PATH = './save/'
MODEL_NAME = 'model.ckpt'


def train(mnist):
    # 定义输入输出占位符placeholder
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')

    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    # l2正则化惩罚项
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

    # 使用 mnist_inference的前向传播
    y = mnist_inference.inference(x, regularizer)

    global_step = tf.Variable(0, trainable=False)

    # 滑动平均模型
    # 初始衰减率 MOVEING_AVERAGE_DEACY
    # 控制衰减率变量 global_step
    variable_averages = tf.train.ExponentialMovingAverage(MOVEING_AVERAGE_DEACY, global_step)

    # 类似于ema.applay([v1])
    # 置变量v使用滑动平均模型
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # 使用指数衰减学习率
    # 通过tf.train.exponential_decay生成学习率
    # decayed_learning_rate = LEARNING_RATE_BASE * LEARNING_RATE_DECAY ^ (mnist.train.num_examples / BATCH_SIZE)
    # 每隔global_step次训练后,都要乘以 LEARNING_RATE_DECAY
    # decayed_learning_rate = 0.8 * 0.99  ^ 550 (每隔100轮乘以0.99,一共550轮)
    decayed_learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,  # 55000 / 100
        LEARNING_RATE_DECAY)

    # global_step会自动跟新,从而学习率也会更新
    train_step = tf.train.GradientDescentOptimizer(decayed_learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):  # 30000
            # 每次拿100个
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})

            if i % 1000 == 0:
                # 输出模型在当前train batch上的损失数大小
                print('Atfer %d training step(s), loss is %g' % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets('../../data/mnist/', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
