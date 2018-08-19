import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets('../data/mnist/', one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)  # 预估的值

y_ = tf.placeholder(tf.float32, [None, 10])  # 正确的值

cross_entropy = -tf.reduce_mean(y_ * tf.log(y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
corr = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accr = tf.reduce_mean(tf.cast(corr, tf.float32))

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(60000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    # 0.9043
    print(sess.run(accr, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
