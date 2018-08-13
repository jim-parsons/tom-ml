import tensorflow as tf

x = tf.constant([[[1, 1, 1],
                  [2, 2, 2]],
                 [[3, 3, 3],
                  [4, 4, 4]]])
y = tf.reduce_mean(x, [0, 1])
with tf.Session() as sess:
    print(sess.run(x))
    print(sess.run(y))
