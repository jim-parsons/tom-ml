import tensorflow as tf

w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1))

# 定义placeholder作为存放数据的地方, 维度不一定需要定义,但是维度确定后则可降低出错概率
x = tf.placeholder(tf.float32, shape=(3, 2), name='input')
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

with tf.Session() as sess:
    # 单个变量初始化
    # sess.run(w1.initializer)
    # sess.run(w2.initializer)

    # 全局变量初始化
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 下面这一行会报错,
    # print(sess.run(y))
    print(sess.run(y, feed_dict={x: [[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]]}))  # feed_dict给出每个用到的placeholder
