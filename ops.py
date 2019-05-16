import tensorflow as tf
from tensorflow.python.framework import ops


def weight_variable(name="weight", shape=None, init=tf.glorot_uniform_initializer):
    weight = tf.get_variable(name=name, shape=shape, initializer=init)
    return weight


def bias_variable(name="bias", shape=None, init=tf.glorot_uniform_initializer):
    bias = tf.get_variable(name=name, shape=shape, initializer=init)
    return bias


def conv2d_dilated(x, k, co, rate=1, padding='SAME'):
    if isinstance(k, list):
        W = weight_variable([k[0], k[1], shape(x)[-1], co])
    else:
        W = weight_variable([k, k, shape(x)[-1], co])
    B = bias_variable([co])
    L = tf.nn.atrous_conv2d(x, W, rate=rate, padding=padding) + B
    return L


class Reverse_Gradient_Builder(object):
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "reverse_gradient{}".format(self.num_calls)

        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self.num_calls += 1
        return y


reverse_gradient = Reverse_Gradient_Builder()


def float32_variable_storage_getter(getter, name, shape=None, dtype=None,
                                    initializer=None, regularizer=None,
                                    trainable=True,
                                    *args, **kwargs):
    """Custom variable getter that forces trainable variables to be stored in
    float32 precision and then casts them to the training precision.
    """
    storage_dtype = tf.float32 if trainable else dtype
    variable = getter(name, shape, dtype=storage_dtype,
                      initializer=initializer, regularizer=regularizer,
                      trainable=trainable,
                      *args, **kwargs)
    if trainable and dtype != tf.float32:
        variable = tf.cast(variable, dtype)
    return variable


def shape(tensor):
    return tensor.get_shape().as_list()


def pixel_norm(x, epsilon=1e-8):
    with tf.name_scope('PixelNorm'):
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float32)
        x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + epsilon)
        x = tf.cast(x, orig_dtype)
    return x


def instance_norm(x, epsilon=1e-8):
    with tf.variable_scope('InstanceNorm'):
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float32)
        x -= tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
        x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis=[1, 2], keepdims=True) + epsilon)
        x = tf.cast(x, orig_dtype)
    return x


def ada_in(x, latent):
    with tf.variable_scope('ada_in'):
        with tf.variable_scope('style_scale'):
            style_scale = tf.layers.dense(inputs=latent, units=shape(x)[-1])
        with tf.variable_scope('style_bias'):
            style_bias = tf.layers.dense(inputs=latent, units=shape(x)[-1])
        style_scale = tf.reshape(style_scale,
                                 [tf.shape(style_scale)[0], 1, 1, tf.shape(style_scale)[1]])
        style_bias = tf.reshape(style_bias,
                                [tf.shape(style_bias)[0], 1, 1, tf.shape(style_bias)[1]])
        x = x * (style_scale + 1) + style_bias
    return x


def sub_pixel_conv(x, filters, kernel_size=2, stride=1, padding='SAME', uprate=2):
    x = tf.layers.conv2d(inputs=x, filters=filters * uprate**2,
                         kernel_size=kernel_size, strides=stride,
                         padding=padding)
    x = tf.depth_to_space(input=x, block_size=uprate)
    return x


def gradient_penalty_wgan(real, fake, discriminator, name_d, gamma=10.0):
    dtype = fake.dtype
    with tf.name_scope("gradient_penalty_name_d"):
        epsilon = tf.random_uniform(shape=[tf.shape(real)[0], 1, 1, 1],
                                    minval=0.0, maxval=1.0)
        real = tf.cast(real, tf.float32)
        fake = tf.cast(fake, tf.float32)
        x_hat = real + epsilon * (fake - real)
        x_hat_type = tf.cast(x_hat, dtype)
        real = tf.reduce_sum(real)
        D_false_w = discriminator(x_hat_type, name=name_d)
        D_false_w = tf.cast(D_false_w, tf.float32)

        gradients = tf.gradients(real, x_hat)[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
        gp = gamma * tf.reduce_mean(tf.square(slopes - 1.0))

    return gp


def gradient_penalty_simple(d_output, x, scale=None, gamma=10.0):
    with tf.name_scope("gradient_penalty_R1"):
        d_output = tf.reduce_sum(d_output)
        if scale is not None:
            gradients = tf.gradients(d_output * scale, x)[0] / scale
        else:
            gradients = tf.gradients(d_output, x)[0]
        gp = gamma * tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3])
        gp = tf.reduce_mean(gp)

    return gp
