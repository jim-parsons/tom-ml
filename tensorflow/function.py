# %matplotlib inline
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

num_points = 1000

vectors_set = []

for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

plt.scatter(x_data, y_data, c='r')

plt.show()

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='W')
b = tf.Variable(tf.zeros([1]), name='b')

y = W * x_data + b

# 预估值y和实际值y_data之间的均方误差作为损失
loss = tf.reduce_mean(tf.square(y - y_data), name='loss')
# 梯度下降
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 训练过程就是最小化误差
train = optimizer.minimize(loss, name='train')

sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)

print("W =", sess.run(W), "b = ", sess.run(b), "loss = ", sess.run(loss))
for step in range(20):
    sess.run(train)
    print("W =", sess.run(W), "b = ", sess.run(b), "loss = ", sess.run(loss))

plt.scatter(x_data, y_data, c='r')
plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
plt.show()
writer = tf.summary.FileWriter("./tmp", sess.graph)

