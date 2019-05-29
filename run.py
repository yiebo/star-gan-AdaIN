import tensorflow as tf
from model import generator
from read import label_size as laten_size
from read import iterator
import cv2
import numpy as np
from ops import *
import os
import glob

x = tf.placeholder(tf.float32, shape=[None, 208, 208, 3])
latent = tf.placeholder(tf.float32, shape=[None, laten_size])
training = tf.constant(False, dtype=tf.bool)

with tf.name_scope('latent_input'):
    latent_std = tf.random_normal(tf.shape(latent), mean=0.0, stddev=0.00, dtype=tf.float32)
    latent_ = latent + latent_std
    # latent_random = tf.random_normal([tf.shape(latent)[0], 128], dtype=tf.float32)
    latent_random = tf.zeros([tf.shape(latent)[0], 128], dtype=tf.float32)
    latent_ = tf.concat([latent_, latent_random], 1)

with tf.variable_scope('generator', dtype=tf.float16,
                       custom_getter=float32_variable_storage_getter):
    y = generator(x, latent_, name="generator", dtype=tf.float16, training=training)

y = tf.minimum(y, 1)

saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

if not os.path.exists('input'):
    os.makedirs('input')
if not os.path.exists('output'):
    os.makedirs('output')

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
    saver = tf.train.import_meta_graph('checkpoints/model.ckpt-20000.meta')
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints0'))

    # Our operations on the frame come here
    label = np.array([[0., 1., 0., 1., 0.],
                      [0., 1., 1., 0., 0.],
                      [0., 1., 1., 0., 0.],
                      [1., 1., 0., 1., 1.],
                      [1., 1., .5, 1., 0.],
                      [1., 1., 0., 1., 0.]])

    label2 = np.array([[1., 1., 0., 1., 0.],
                       [1., 1., 0., 0., 1.],
                       [1., 0., 0., 0., 1.],
                       [1., 1., 0., 1., 1.],
                       [0., 0., .5, 1., 0.],
                       [0., 1., 0., 1., 0.]])
    # images_in, images_out = sess.run([x, y], feed_dict={x: images, latent: label})
    # # images_in, images_out, lat = sess.run([x, y, latent])
    # # print(lat)
    # images_out = images_out[..., ::-1] * 255
    # images_in = images_in[..., ::-1] * 255
    # for idy, (image_, image) in enumerate(zip(images_in, images_out)):
    #     cv2.imwrite(f'output/{idy:02d}.jpg', image)
    #     cv2.imwrite(f'output/{idy:02d}_.jpg', image_)
    idx = 1
    for idy in range(51):
        inter = idy / 50.0
        label_in = (1.0 - inter) * label + inter * label2
        print(label_in[idx])
        images_in, images_out = sess.run([x, y], feed_dict={x: images, latent: label_in})
        images_out = images_out[..., ::-1] * 255
        cv2.imwrite(f'output/{idy:02d}.jpg', images_out[idx])
