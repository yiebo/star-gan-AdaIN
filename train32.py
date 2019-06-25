import tensorflow as tf
import glob
import tqdm
import os

from ops import *
from model32 import generator, discriminator
from read import iterator
from read import label_size as laten_size

LAMBDA = 10
BATCH_SIZE = 4


file_list = glob.glob("TFRECORD/celebA*.tfrecord")
it = iterator(file_list, BATCH_SIZE)

x, latent_real = it.get_next()
# training = tf.constant(True, dtype=tf.bool)

with tf.name_scope('latent_input'):
    latent_std = tf.random_normal(tf.shape(latent_real), mean=0.0, stddev=0.01, dtype=tf.float32)
    latent_real_in = latent_real + latent_std

    """shuffle doesnt randomize well with small batch sizes"""
    # latent_fake = tf.random_shuffle(latent_real)
    latent_fake = tf.reverse(latent_real, [0])
    latent_std = tf.random_normal(tf.shape(latent_real), mean=0.0, stddev=0.01, dtype=tf.float32)
    latent_fake_in = latent_fake + latent_std

    latent_fake_in_ = latent_fake_in - latent_real_in
    latent_real_in_ = latent_real_in - latent_fake_in

    latent_random = tf.random_normal([tf.shape(latent_real)[0], 128], dtype=tf.float32)
    latent_real_in_ = tf.concat([latent_real_in_, latent_random], 1)

    latent_random = -latent_random
    latent_fake_in_ = tf.concat([latent_fake_in_, latent_random], 1)

with tf.variable_scope('generator'):
    y = generator(x, latent_fake_in_)
    x_ = generator(y, latent_real_in_)


with tf.variable_scope('discriminator'):
    d_real, latent_real_ = discriminator(x, laten_size)
    d_fake, latent_fake_ = discriminator(y, laten_size)


tf.train.create_global_step()
global_step = tf.train.get_global_step()
learning_rate_ = tf.train.exponential_decay(0.001, global_step,
                                            decay_steps=1000, decay_rate=0.99)

G_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
D_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

with tf.name_scope('D_optimizer'):
    D_optimizer = tf.train.AdamOptimizer(learning_rate_, beta1=0.5, name='D_solver')

with tf.name_scope('G_optimizer'):
    G_optimizer = tf.train.AdamOptimizer(learning_rate_, beta1=0.5, name='G_solver')

# -------------GAN SIMPLE GP R1
with tf.name_scope('losses'):

    d_real = tf.reduce_mean(d_real)
    d_fake = tf.reduce_mean(d_fake)
    latent_real_ = tf.reduce_mean(latent_real_, axis=[1, 2])
    latent_fake_ = tf.reduce_mean(latent_fake_, axis=[1, 2])

    # latent_real_shift = -2 * latent_real + 1
    # latent_fake_shift = -2 * latent_fake + 1
    # loss_d_cls = latent_real_shift * latent_real_
    # loss_g_cls = latent_fake_shift * latent_fake_
    # loss_d_cls = tf.reduce_mean(loss_d_cls, axis=0)
    # loss_g_cls = tf.reduce_mean(loss_g_cls, axis=0)
    # loss_d_cls = tf.reduce_mean(tf.nn.softplus(loss_d_cls))
    # loss_g_cls = tf.reduce_mean(tf.nn.softplus(loss_g_cls))

    loss_d_cls = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=latent_real_, labels=latent_real_in))
    loss_g_cls = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=latent_fake_, labels=latent_fake_in))

    loss_cycle = LAMBDA * tf.reduce_mean(tf.abs(x - x_))

    cycle_decay = tf.train.exponential_decay(.5, global_step,
                                             decay_steps=10000, decay_rate=0.95)
    cycle_decay += .5

    loss_g_x = tf.nn.softplus(-d_fake)
    loss_g = loss_g_x / cycle_decay + loss_g_cls + cycle_decay * loss_cycle

    gp = gradient_penalty_simple(d_real, x)

    loss_d_x = tf.nn.softplus(d_fake) + tf.nn.softplus(-d_real)
    loss_d = loss_d_x + loss_d_cls + gp

    D_solver = D_optimizer.minimize(loss_d, var_list=D_var, global_step=global_step)
    G_solver = G_optimizer.minimize(loss_g, var_list=G_var)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    training_op = tf.group([D_solver, G_solver, update_ops])

# losses
tf.summary.scalar('cls/d', loss_d_cls)
tf.summary.scalar('cls/g', loss_g_cls)
tf.summary.scalar('loss/d', loss_d_x)
tf.summary.scalar('loss/g', loss_g_x)
tf.summary.scalar('loss/total/d', loss_d)
tf.summary.scalar('loss/total/g', loss_g)
tf.summary.scalar('loss/cycle_loss', loss_cycle)
tf.summary.scalar('loss/gp', gp)

# images
tf.summary.image('img/a', x, 2)
tf.summary.image('img/b', tf.minimum(y, 1), 2)
tf.summary.image('img/c', tf.minimum(x_, 1), 2)

latent_fake_ = tf.maximum(tf.minimum(latent_fake_, 1), 0)
latent_real_ = tf.maximum(tf.minimum(latent_real_, 1), 0)

tf.summary.image('label/a', tf.reshape(latent_real, [-1, 1, laten_size, 1]), 2)
tf.summary.image('label/b', tf.reshape(latent_fake, [-1, 1, laten_size, 1]), 2)

tf.summary.image('label_pred/a', tf.reshape(latent_real_, [-1, 1, laten_size, 1]), 2)
tf.summary.image('label_pred/b', tf.reshape(latent_fake_, [-1, 1, laten_size, 1]), 2)

# params
tf.summary.scalar('other/learning_rate', learning_rate_)
tf.summary.scalar('other/cycle_decay', cycle_decay)

log_dir = 'logs'
checkpoint_dir = 'checkpoints'

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=10))
with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir, summary_dir=log_dir,
                                       save_checkpoint_steps=2000,
                                       save_summaries_steps=200, config=config) as sess:
    with tqdm.tqdm(total=400000, dynamic_ncols=True) as pbar:
        while True:
            _, step = sess.run([training_op, global_step])
            if step - pbar.n > 0:
                pbar.update(step - pbar.n)
                if step >= pbar.total:
                    print('finished training')
                    break
