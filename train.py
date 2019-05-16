import tensorflow as tf
import glob
import tqdm
import os

from ops import *
from model import generator, discriminator
from read import iterator
from read import label_size as laten_size

LAMBDA = 10
BATCH_SIZE = 6

# [black, brown, blond]

file_list = glob.glob("TFRECORD/celebA*.tfrecord")
it = iterator(file_list, BATCH_SIZE)

x, latent_real = it.get_next()

with tf.name_scope('latent_input'):
    latent_std = tf.random_normal(tf.shape(latent_real), mean=0.0, stddev=0.01, dtype=tf.float32)
    latent_real_in = latent_real + latent_std
    latent_random = tf.random_normal([tf.shape(latent_real)[0], 128], dtype=tf.float32)
    latent_real_in_ = tf.concat([latent_real_in, latent_random], 1)

    """shuffle doesnt randomize well with small batch sizes"""
    # latent_fake = tf.random_shuffle(latent_real)
    latent_fake = tf.reverse(latent_real, [0])

    latent_std = tf.random_normal(tf.shape(latent_real), mean=0.0, stddev=0.01, dtype=tf.float32)
    latent_fake_in = latent_fake + latent_std
    latent_random = tf.random_normal([tf.shape(latent_real)[0], 128], dtype=tf.float32)
    latent_fake_in_ = tf.concat([latent_fake_in, latent_random], 1)

with tf.variable_scope('generator', dtype=tf.float16,
                       custom_getter=float32_variable_storage_getter):
    y = generator(x, latent_fake_in_, name="generator", dtype=tf.float16, training=True)
    x_ = generator(y, latent_real_in_, name="generator", dtype=tf.float16, training=True)


with tf.variable_scope('discriminator', dtype=tf.float16,
                       custom_getter=float32_variable_storage_getter):
    d_real, latent_real_ = discriminator(x, laten_size, name='discriminator',
                                         dtype=tf.float16, training=True)
    d_fake, latent_fake_ = discriminator(y, laten_size, name='discriminator',
                                         dtype=tf.float16, training=True)


tf.train.create_global_step()
global_step = tf.train.get_global_step()
learning_rate_ = tf.train.exponential_decay(0.0005, global_step,
                                            decay_steps=2000, decay_rate=0.95)

G_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
D_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

with tf.name_scope('D_optimizer'):
    D_optimizer = tf.train.AdamOptimizer(learning_rate_, beta1=0.5, name='D_solver')
    loss_scale_manager_D = tf.contrib.mixed_precision.ExponentialUpdateLossScaleManager(5000, 100)
    loss_scale_optimizer_D = tf.contrib.mixed_precision.LossScaleOptimizer(D_optimizer,
                                                                           loss_scale_manager_D)

with tf.name_scope('G_optimizer'):
    G_optimizer = tf.train.AdamOptimizer(learning_rate_, beta1=0.5, name='G_solver')
    loss_scale_manager_G = tf.contrib.mixed_precision.ExponentialUpdateLossScaleManager(5000, 100)
    loss_scale_optimizer_G = tf.contrib.mixed_precision.LossScaleOptimizer(G_optimizer,
                                                                           loss_scale_manager_G)

# -------------GAN SIMPLE GP R1
with tf.name_scope('losses'):

    d_real = tf.reduce_mean(d_real)
    d_fake = tf.reduce_mean(d_fake)
    # latent_real_ = tf.reduce_mean(latent_real_, axis=[1, 2])
    # latent_fake_ = tf.reduce_mean(latent_fake_, axis=[1, 2])
    loss_d_cls = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=latent_real_, labels=latent_real_in))
    loss_g_cls = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=latent_fake_, labels=latent_fake_in))

    loss_cycle = LAMBDA * tf.reduce_mean(tf.abs(x - x_))

    loss_g_x = tf.nn.softplus(-d_fake)
    loss_g = loss_g_x + loss_cycle + loss_g_cls

    gp = gradient_penalty_simple(d_real, x, loss_scale_manager_D.get_loss_scale())

    loss_d_x = tf.nn.softplus(d_fake) + tf.nn.softplus(-d_real)
    loss_d = loss_d_x + gp + loss_d_cls

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    D_solver = loss_scale_optimizer_D.minimize(loss_d, var_list=D_var, global_step=global_step)
    G_solver = loss_scale_optimizer_G.minimize(loss_g, var_list=G_var)

    training_op = tf.group(D_solver, G_solver)

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
tf.summary.image('img/x', x, 1)
tf.summary.image('img/x->y', tf.minimum(y, 1), 1)
tf.summary.image('img/y->x_', tf.minimum(x_, 1), 1)
tf.summary.image('label/x', tf.reshape(latent_real, [-1, 1, laten_size, 1]), 1)
tf.summary.image('label/y', tf.reshape(latent_fake, [-1, 1, laten_size, 1]), 1)

# params
tf.summary.scalar('loss_scale/g', loss_scale_manager_G.get_loss_scale())
tf.summary.scalar('loss_scale/d', loss_scale_manager_D.get_loss_scale())
tf.summary.scalar('learning_rate', learning_rate_)

if not os.path.exists('logs'):
    os.makedirs('logs')
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


with tf.train.MonitoredTrainingSession(checkpoint_dir='checkpoints', summary_dir='logs',
                                       save_checkpoint_steps=2000,
                                       save_summaries_steps=200, config=config) as sess:
    with tqdm.tqdm(total=200000, dynamic_ncols=True) as pbar:
        while True:
            _, step = sess.run([training_op, global_step])
            if step - pbar.n > 0:
                pbar.update(step - pbar.n)
                if step >= pbar.total:
                    print('finished training')
                    break
