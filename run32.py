import tensorflow as tf
from model32 import generator, discriminator
from read import label_size as laten_size
import cv2
import numpy as np
from ops import *
import glob

x = tf.placeholder(tf.float32, shape=[None, 208, 208, 3])
latent = tf.placeholder(tf.float32, shape=[None, laten_size])

with tf.name_scope('latent_input'):
    latent_std = tf.random_normal(
        tf.shape(latent), mean=0.0, stddev=0.00, dtype=tf.float32)
    latent_ = latent + latent_std
    latent_random = tf.random_normal([tf.shape(latent)[0], 128], dtype=tf.float32)
    # latent_random = tf.zeros([tf.shape(latent)[0], 128], dtype=tf.float32)
    latent_ = tf.concat([latent_, latent_random], 1)

with tf.variable_scope('generator'):
    y = generator(x, latent_, name="generator")

with tf.variable_scope('discriminator'):
    d_real, latent_real_ = discriminator(x, laten_size)

y = tf.minimum(y, 1)

saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


files = glob.glob('input/*')
print(files)
images = []

for file in files:
    image = cv2.imread(file)
    image = cv2.resize(image, (208, 208))
    image = image / 255.
    image = image[..., ::-1]
    images.append(image)

with tf.Session(config=config) as sess:
    ckpt = glob.glob('checkpoints/*.meta')[-1]
    print(ckpt)
    saver = tf.train.import_meta_graph(ckpt)
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))

    # Our operations on the frame come here
    label = np.array([[0., 0., 0., 0., 0.],
                      [1., 1., 1., 0., 0.],
                      [1., 1., 1., 1., 0.],
                      [0., 1., .5, .5, 0.],
                      [0., 1., 0., 0., 0.5],
                      [0., 0., 1., 0., 0.]])
    label2 = np.array([[1., 1., 0., 0., 0., 1., 0.],
                       [1., 0., 0., 0., 1., 0., -1.],
                       [-1., -1., 1., 0., 0., 1., 1.],
                       [0., 0., 0., 0., 0., 1., 1.],
                       [0., 0., 0., 0., 0., 1., 1.],
                       [-1.5, 1.5, 0., 0., 1., 1., 1.]])
    images_in, images_out = sess.run([x, y], feed_dict={x: images, latent: label2})

    images_out = images_out[..., ::-1] * 255
    images_in = images_in[..., ::-1] * 255
    for idy, (image_, image) in enumerate(zip(images_in, images_out)):
        cv2.imwrite(f'output/{idy:02d}.jpg', image)
        cv2.imwrite(f'output/{idy:02d}_.jpg', image_)

    # idx = 0
    # for idy in range(51):
    #     inter = idy / 50.0
    #     label_in = (1.0 - inter) * label + inter * label2
    #     images_out = sess.run(y, feed_dict={x: images, latent: label_in})
    #     images_out = images_out[..., ::-1] * 255
    #     cv2.imwrite(f'output/{idy:02d}.jpg', images_out[idx])
