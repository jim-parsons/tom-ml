import tensorflow as tf
import matplotlib.pyplot as plt

image_raw_data = tf.gfile.FastGFile('../../data/img/Am314.jpeg', 'rb').read()

with tf.Session() as sess:
    img_data = tf.image.decode_image(image_raw_data)
    print(sess.run(img_data))
    plt.imshow(img_data.eval())
    plt.show()
