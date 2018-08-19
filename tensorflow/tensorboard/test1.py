import tensorflow as tf

input1 = tf.constant([1.0, 2.0, 3.0], name='input-1')
input2 = tf.Variable(tf.random_uniform([3]), name='input-2')
output = tf.add_n([input1, input2], name='add')

a = tf.constant([1, 2, 3], shape=[3], name='a')
b = tf.constant([1, 2, 3], shape=[3], name='b')

c = a + b

config = tf.ConfigProto()
config.log_device_placement = True  # 输出设备信息
config.allow_soft_placement = True  # 动态使用设备
config.gpu_options.allow_growth = True  # GPU按需分配显存

# # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True))
sess = tf.Session(config=config)
# print(sess.run(c))
#
# writer = tf.summary.FileWriter('../../data/log', tf.get_default_graph())
# writer.close()
#

d = tf.constant([[3, 2, 1], [1, 3, 2], [2, 3, 1]], shape=[3, 3])
e = tf.constant([[4, 3, 2, 1], [1, 2, 3, 4], [2, 1, 4, 3]], shape=[3, 4])
f = tf.matmul(d, e)
print(sess.run(f))
