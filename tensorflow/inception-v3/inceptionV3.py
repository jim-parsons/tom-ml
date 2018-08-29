import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow.contrib.slim as slim

# 加载通过TensorFlow-Slim定义好的inception_v3模型。
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3

INPUT_DATA = './model/flower_processed_data.npy'

# 新的训练数据训练模型保存路径
TRAIN_FILE = './model/model/'

# 训练好的模型
CKPT_FILE = './model/inception_v3.ckpt'

# 定义训练中使用的参数
LEARNING_RATE = 0.0001
STEPS = 300
BATCH = 32

# 种类个数
N_CLASSES = 5

# 不需要从谷歌训练好的模型中加载的参数。
# 这里是最后的FC, 这里是参数的前缀
CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'
# 需要训练的网络层参数明层，在inceptionV3.py的过程中就是最后的全联接层。
TRAINABLE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogit'


# 获取所有需要从谷歌训练好的模型中加载的参数
def get_tuned_variables():
    # strip 移除字符串头尾指定的字符（默认为空格或换行符）
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(',')]  # 获取所有子目录
    print(slim.get_model_variables())
    variables_to_restore = []
    # 枚举inception-v3模型中所有的参数，然后判断是否需要从加载列表中移除。
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore


# 获取所有训练的变量列表
def get_trainable_variables():
    scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(',')]
    variables_to_train = []

    # 枚举所有需要训练的参数前缀，并通过这些前缀找到所有需要训练的参数。
    for scope in scopes:
        # 神经网络中的参数
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)

    return variables_to_train


def main():
    # 在data_process.py中最后存储的是,这边依次取出
    # np.asarray([training_images, training_labels,
    #             validation_images, validation_labels,
    #             testing_images, testing_labels])
    processed_data = np.load(INPUT_DATA)
    training_images = processed_data[0]
    n_training_example = len(training_images)
    print(n_training_example)
    training_labels = processed_data[1]

    validation_images = processed_data[2]
    validation_labels = processed_data[3]

    testing_images = processed_data[4]
    testing_labels = processed_data[5]
    print("%d training examples, %d validation examples and %d testing examples." % (
        n_training_example, len(validation_labels), len(testing_labels)))

    # 定义inception-v3的输入，images为输入图片，labels为每一张图片对应的标签。
    images = tf.placeholder(tf.float32, [None, 299, 299, 3], name='input_images')
    labels = tf.placeholder(tf.int64, [None], name='labels')

    # 定义inception-v3模型。因为谷歌给出的只有模型参数取值，所以这里
    # 需要在这个代码中定义inception-v3的模型结构。虽然理论上需要区分训练和
    # 测试中使用到的模型，也就是说在测试时应该使用is_training=False，但是
    # 因为预先训练好的inception-v3模型中使用的batch normalization参数与
    # 新的数据会有出入，所以这里直接使用同一个模型来做测试。

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(images, num_classes=N_CLASSES)

    # 获取需要训练的变量
    trainable_variables = get_trainable_variables()

    # 定义损失函数和训练过程。
    # tf.losses.softmax_cross_entropy(
    #     onehot_labels,  #  [batch_size, num_classes] one_hot类型的label.
    #     logits,   [batch_size, num_classes] 神经网络的logits输出.
    #     weights=1.0, loss的系数
    #     label_smoothing=0 如果大于0，则对label进行平滑,公式如下
    #                              new_onehot_labels = onehot_labels*(1-label_smoothing) + label_smoothing/num_classes
    #     scope=None,  命名空间
    #     loss_collection=tf.GraphKeys.LOSSES, 指定loss集合
    #     reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
    # )
    tf.losses.softmax_cross_entropy(tf.one_hot(labels, N_CLASSES), logits=logits, weights=1.0)
    total_loss = tf.losses.get_total_loss()
    train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(total_loss)

    # 计算正确率。
    with tf.name_scope('evaluation'):
        correction_prediction = tf.equal(tf.argmax(logits, 1), labels)
        evaluation_step = tf.reduce_mean(tf.cast(correction_prediction, tf.float32))

    # 定义加载Google训练好的Inception-v3模型的Saver。
    load_fn = slim.assign_from_checkpoint_fn(CKPT_FILE, get_tuned_variables(), ignore_missing_vars=True)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # 初始化没有加载进来的变量。
        tf.global_variables_initializer().run()

        # 加载谷歌已经训练好的模型。
        print('Loading tuned variables from %s' % CKPT_FILE)
        load_fn(sess)

        start = 0
        end = BATCH
        for i in range(STEPS):
            _, loss = sess.run([train_step, total_loss], feed_dict={
                images: training_images[start: end],
                labels: training_labels[start: end]
            })

            if i % 30 == 0 or i + 1 == STEPS:
                saver.save(sess, TRAIN_FILE, global_step=i)
                validation_accuracy = sess.run(evaluation_step, feed_dict={
                    images: validation_images, labels: validation_labels})
                print('Step %d: Training loss is %.1f Validation accuracy = %.1f%%' % (
                    i, loss, validation_accuracy * 100.0))

            start = end
            if start == n_training_example:
                start = 0
            end = start + BATCH
            if end > n_training_example:
                end = n_training_example
        # 在最后的测试数据上测试正确率。
        test_accuracy = sess.run(evaluation_step, feed_dict={
            images: testing_images, labels: testing_labels})
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100))


def test():
    hello = tf.constant('hello')
    with tf.Session() as sess:
        print(sess.run(hello))

if __name__ == '__main__':
    test()
