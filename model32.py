import tensorflow as tf
from ops import *


def u_block_latent(x, latent, filters):
    with tf.variable_scope('layer_1'):
        x = tf.layers.conv2d(inputs=x, filters=filters,
                             kernel_size=3, strides=1, padding='SAME')
        x = tf.contrib.layers.instance_norm(x, center=False, scale=False)
        x = ada_in(x, latent)
        x = tf.nn.relu(x)

    with tf.variable_scope('layer_2'):
        x = tf.layers.conv2d(inputs=x, filters=filters,
                             kernel_size=3, strides=1, padding='SAME')
        x = tf.contrib.layers.instance_norm(x, center=False, scale=False)
        x = ada_in(x, latent)
        x = tf.nn.relu(x)
    return x


def u_block(x, filters):

    with tf.variable_scope('layer_1'):
        x = tf.layers.conv2d(inputs=x, filters=filters,
                             kernel_size=3, strides=1, padding='SAME')
        x = tf.contrib.layers.instance_norm(x)
        x = tf.nn.relu(x)

    with tf.variable_scope('layer_2'):
        x = tf.layers.conv2d(inputs=x, filters=filters,
                             kernel_size=3, strides=1, padding='SAME')
        x = tf.contrib.layers.instance_norm(x)
        x = tf.nn.relu(x)

    return x


def u_net(x, latent, name='u-net'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        depth = [64, 128, 256, 512]
        skip_connections = []
        for idx, filters in enumerate(depth):
            with tf.variable_scope(f'down_block{idx}'):
                x = u_block(x, filters)
                skip_connections.append(x)
                x = tf.layers.max_pooling2d(inputs=x, pool_size=2, strides=2, padding='SAME')

        with tf.variable_scope('conv_block'):
            x = u_block_latent(x, latent, 1028)

        depth = [512, 256, 128, 64]
        skip_connections.reverse()

        for idx, (filters, skip_connection) in enumerate(zip(depth, skip_connections)):
            with tf.variable_scope(f'up_block{idx}'):
                with tf.variable_scope('sub_pixel'):
                    x = sub_pixel_conv(x, filters=filters, kernel_size=3, uprate=2)
                    x = tf.contrib.layers.instance_norm(x)
                    x = tf.nn.relu(x)
                    x = tf.concat([x, skip_connection], 3)
                x = u_block_latent(x, latent, filters)

        with tf.variable_scope('final_block'):
            x = tf.layers.conv2d(inputs=x, filters=3,
                                 kernel_size=1, strides=1, padding='SAME')
            x = tf.nn.relu(x)
    return x


def generator_mapping(x, name='G_mapping'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        depth = [256] * 5
        for idx, filters in enumerate(depth):
            with tf.variable_scope(f'dense_{idx}'):
                x = tf.layers.dense(inputs=x, units=filters)
                x = tf.nn.leaky_relu(x)
        return x


def generator(x, latent, name='generator', dtype=tf.float32):
    x = tf.cast(x, dtype=dtype)
    latent = tf.cast(latent, dtype=dtype)

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE, dtype=dtype):
        latent = generator_mapping(latent)
        x = u_net(x, latent)

    x = tf.cast(x, dtype=tf.float32)
    return x


def discriminator(x, laten_size, name='discriminator', dtype=tf.float32):
    x = tf.cast(x, dtype=dtype)

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        depth = [64, 128, 256, 512, 1028]
        for idx, filters in enumerate(depth):
            with tf.variable_scope(f'conv{idx}'):
                x = tf.layers.conv2d(inputs=x, filters=filters,
                                     kernel_size=3, strides=1, padding='SAME')
                x = tf.contrib.layers.instance_norm(x)
                x = tf.nn.leaky_relu(x)
                x = tf.layers.max_pooling2d(inputs=x, pool_size=2, strides=2, padding='SAME')

        with tf.variable_scope('conv_adv'):
            x_adv = tf.layers.conv2d(inputs=x, filters=512,
                                     kernel_size=3, strides=1, padding='SAME')
            x_adv = tf.contrib.layers.instance_norm(x_adv)
            x_adv = tf.nn.leaky_relu(x_adv)
            x_adv = tf.layers.conv2d(inputs=x_adv, filters=1,
                                     kernel_size=1, strides=1, padding='SAME')

        with tf.variable_scope('conv_cls'):
            x_cls = tf.layers.conv2d(inputs=x, filters=laten_size,
                                     kernel_size=1, strides=1, padding='SAME')

    x_adv = tf.cast(x_adv, dtype=tf.float32)
    x_cls = tf.cast(x_cls, dtype=tf.float32)
    return x_adv, x_cls
# 218x178
