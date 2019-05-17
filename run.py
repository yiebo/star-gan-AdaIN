import tensorflow as tf
from model import generator
from read import label_size as laten_size
import cv2
from ops import *
import os
import glob

x = tf.placeholder(tf.float32, shape=[None, None, None, 3])
latent = tf.placeholder(tf.float32, shape=[None, laten_size])

with tf.name_scope('latent_input'):
    latent_std = tf.random_normal(tf.shape(latent), mean=0.0, stddev=0.01, dtype=tf.float32)
    latent_ = latent + latent_std
    latent_random = tf.random_normal([tf.shape(latent)[0], 128], dtype=tf.float32)
    latent_ = tf.concat([latent_, latent_random], 1)

with tf.variable_scope('generator', dtype=tf.float16,
                       custom_getter=float32_variable_storage_getter):
    y = generator(x, latent_, name="generator", dtype=tf.float16)

y = tf.minimum(y, 1)

saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

if not os.path.exists('input'):
    os.makedirs('input')
if not os.path.exists('output'):
    os.makedirs('output')

files = glob.glob('input/*')
images = []

for file in files:
    image = cv2.imread(file)
    image = cv2.resize(image, (208, 208))
    image = image / 255.0
    image = image[..., ::-1]
    images.append(image)

with tf.Session(config=config) as sess:
    saver = tf.train.import_meta_graph('checkpoints/model.ckpt-34000.meta')
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))

    # Our operations on the frame come here
    for idx in range(0, len(images), 5):
        images_out = sess.run(y, feed_dict={x: images[idx:idx + 5]})
        images_out = images_out[..., ::-1] * 255
        for idy, image in enumerate(images_out):
            cv2.imwrite(f'output/{idx+idy:02d}.jpg', image)
