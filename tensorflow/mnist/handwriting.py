import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST相关参数
INPUT_NODE = 784  # 输入层节点数
OUTPUT_NODE = 10  # 输出层节点数  类别个数

LAYER_NODE = 500  # 影层节点数
BATCH_SIZE = 100  # 一个训练batch中训练数据个数,越小过程越接近随机梯度下降

LEARNING_RATE_BASE = 0.8  # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 学习衰减率

REGULARIZATION_RATE = 0.0001  # 正则化在损失函数中的系数
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率


def inference(input_tensor, avg_class, weight1, biases1, weight2, biases2):
    # Relu激活函数
    # 计算损失函数是会计算softmax函数,所以前向传播不计算softmax
    if avg_class is None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weight1) + biases1)
        return tf.matmul(layer1, weight2) + biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weight1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weight2)) + avg_class.average(biases2)


def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    # tf.truncate_noraml(shape, mean, stddev) 产生正太分布
    # shape 张量维度
    # mean 均值
    # stddev 标准差

    weight1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER_NODE], stddev=0.1))  # 隐层参数 --权重
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER_NODE]))

    weight2 = tf.Variable(tf.truncated_normal([LAYER_NODE, OUTPUT_NODE], stddev=0.1))  # 输出层参数
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    y = inference(x, None, weight1, biases1, weight2, biases2)

    # 定义存贮训练轮数的变量,这个变量无需计算滑动平均, 所以trainable=False
    global_step = tf.Variable(0, trainable=False)

    # 初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    # 在所有代表神经网络参数的辩论上使用滑动平均
    # tf.trainable_variables()返回所有没有指定 trainable=False的参数
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算使用滑动平均类的前向传播结果
    average_y = inference(x, variable_averages, weight1, biases1, weight2, biases2)

    # 交叉熵
    # 当分类只有一个正确结果的时候,使用tf.nn.sparse_softmax_cross_entropy_with_logits()
    # 第一个参数为不包含softmax的前向传播结果
    # 第二个参数是训练数据的正确答案
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # L2正则化损失函数,一般只计算权重,不计算偏置项
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    regularization = regularizer(weight1) + regularizer(weight2)
    # 总损失 交叉熵损失 + 正则化损失和
    loss = cross_entropy_mean + regularization

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

    # average_y是一个batch_size*10的二维数组,每一行表示一个阳历的前向结果
    # tf.argmax的第二个参数1,表示选取最大值的操作仅在第一个维度中进行,即只在每一行选取最大值对应的下标
    # 结果就是一个长度为btach的一维数组,每个值就表示了每个样子对应的数字识别结果
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print('After %d training step(s), validation accuracy using average model is %g' % (i, validate_acc))

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print('After %d training step(s), test accuracy using average model is %g' % (TRAINING_STEPS, test_acc))


def main(argv=None):
    mnist = input_data.read_data_sets('../../data/mnist', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
