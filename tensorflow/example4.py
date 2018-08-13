import tensorflow as tf
from numpy.random import RandomState

# 定义训练数据batch大小
batch_size = 8

# 定义神经网络参数
w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1))

# 在shape的一个维度上使用None,可以方便使用不同的batch大小,
# 在训练是需要把数据分词比较小的batch,但是在测试时,可以一次性使用全部数据
# 当数据集比较小的时候方便测试, 数据集比较大是,大量数据放入batch可能oom
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# 定义前向传播过程
# tf.matmul为矩阵乘法
# x * w1 为两个元素对应相乘
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数和方向传播算法
y = tf.sigmoid(y)

cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
    + (1 - y_) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)))
# y_ 正确结果
# y 预测结果
# tf.log(tf.clip_by_value(x, a, b)将x限制在[a, b]范围内
train_step = tf.train.AdadeltaOptimizer(0.001).minimize(cross_entropy)

# 通过随机数模拟生成数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)

# 定义规则来给出样本标签, 这里x1+x2<1样例被认为正样本,其他为负样本
# 0表示负样本, 1表示正样本

Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

with tf.Session() as sess:
    # 全局变量初始化
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    """
    print(sess.run(w1))
    print(sess.run(w2))
    [[-0.8113182 1.4845988 0.06532937] [-2.4427042 0.0992484 0.5912243 ]]
    [[-0.8113182 ]
    [ 1.4845988 ]
    [ 0.06532937]]
    """

    STEPS = 5000
    for i in range(STEPS):
        # 每次选取batch_size个样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        # 通过选取的样本训练神经网络并跟新参数
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})

        if i % 1000 == 0:
            total_cross_entropy = sess.run(
                cross_entropy, feed_dict={x: X, y_: Y})
            # print('After %d training step(s), cross entropy on all data is %g' % (i, total_cross_entropy))

    print(sess.run(w1))
    print(sess.run(w2))
    """
    [[-0.8090058   1.4823178   0.06302097]
     [-2.4402      0.09674797  0.5887192]]
    [[-0.8086925]
     [1.4822694]
     [0.06277986]]
    
    """
