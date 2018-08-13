import tensorflow as tf

v1 = tf.Variable(0, dtype=tf.float32)

step = tf.Variable(0, trainable=False)

ema = tf.train.ExponentialMovingAverage(0.99, step)

maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # ema.average(v1) 获取滑动平均
    print(sess.run([v1, ema.average(v1)]))

    sess.run(tf.assign(v1, 5))
    # 跟新v1,衰减率为min{0.99, (1+step)/(10+step)=0.1} = 0.1
    # 所以v1的滑动平均跟新为 0.1x0 + (1-0.1)x5 = 4.5
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))

    sess.run(tf.assign(step, 10000))

    sess.run(tf.assign(v1, 10))

    # 跟新v1,衰减率为min{0.99, (1+step)/(10+step)=0.1} = min{0.99, (1+10000)/(10+10000)=0.999} = 0.99
    # 所以v1的滑动平均跟新为 0.99x4.5 + (1-0.99)x10 = 4.555

    sess.run(maintain_averages_op)

    print(sess.run([v1, ema.average(v1)]))



