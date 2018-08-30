import glob
import os.path
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

# 原始数据一个文件夹一个类别
INPUT_DATA = '../../data/flowers/flower_photos/'

OUTPUT_FILE = './flower_processed_data.npy'

# 测试数据和验证数据比例。
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10


def create_image_lists(sess, testing_percentage, validation_percentage):
    # ['../../data/flowers']
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]  # 获取所有子目录
    is_root_dir = True  # 第一个目录为当前目录，需要忽略

    # 初始化各个数据集
    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []
    validation_images = []
    validation_labels = []
    current_label = 0

    # 读取所有子目录
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        # 获取一个子目录中所有的图片文件
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        # 去掉目录路径，返回文件名
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            # 列表作为参数，并将该参数的每个元素都添加到原有的列表中
            # 返回所有匹配的文件路径列表
            file_list.extend(glob.glob(file_glob))

        if not file_list:
            continue
        print('processing :' + dir_name + ' : ' + str(len(file_list)) + ' 个图片')

        i = 0
        # 处理图片数据
        for file_name in file_list:
            i += 1
            # 读取并解析图片，将图片转化为299*299以方便inception-v3模型来处理。
            # tf.gfile.FastGFile(path, decodestyle)
            # path：图片宽度所在路径(2)
            # decodestyle: 图片的解码方式。(‘r’:UTF-8编码; ‘rb’:非UTF-8编码)
            image_raw_data = gfile.FastGFile(file_name, 'rb').read()
            image = tf.image.decode_jpeg(image_raw_data)
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.image.resize_images(image, [299, 299])
            image_value = sess.run(image)

            # 随机划分数据集
            chance = np.random.randint(100)

            if chance < validation_percentage:
                validation_images.append(image_value)
                validation_labels.append(current_label)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(image_value)
                testing_labels.append(current_label)
            else:
                training_images.append(image_value)
                training_labels.append(current_label)

            if i % 200 == 0:
                print(str(i) + ' images processed')
        current_label += 1

    # 将训练数据随机打乱以获得更好的训练效果。
    # 获取随机生成器 np.random的状态
    state = np.random.get_state()
    np.random.shuffle(training_images)
    # 使随机生成器random保持相同的状态（state）
    np.random.set_state(state)
    np.random.shuffle(training_labels)

    return np.asarray([training_images, training_labels,
                       validation_images, validation_labels,
                       testing_images, testing_labels])


with tf.Session() as sess:
    start = time.time()
    processed_data = create_image_lists(sess, TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
    np.save(OUTPUT_FILE, processed_data)
    print('共耗时 %f: ' % (time.time() - start))
