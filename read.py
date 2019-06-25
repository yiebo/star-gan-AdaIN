import tensorflow as tf

size = 208
label_size = 7


def _parse_function(example_proto, dtype=tf.float32):
    features = {'path': tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),
                'label': tf.io.FixedLenFeature([label_size], dtype=tf.int64,
                                               default_value=tf.zeros([label_size], dtype=tf.int64))
                }
    parsed = tf.io.parse_single_example(example_proto, features)
    path = parsed['path']
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image)
    image = tf.reshape(image, [218, 178, 3])
    image = tf.contrib.image.translate(image, tf.random_uniform(shape=[2], minval=-10, maxval=10))
    image = tf.image.crop_and_resize([image], [[0.0917, 0, 0.9083, 1]], [0], [size, size])[0]
    image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_brightness(image, max_delta=32)
    # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.cast(tf.reshape(image, [size, size, 3]), dtype=dtype) / 255.0
    label = tf.cast(parsed['label'], dtype=dtype)
    return image, label


def iterator(file_list, batch_size, shuffle_size=1000, dtype=tf.float32):
    dataset = tf.data.TFRecordDataset(file_list)
    dataset = dataset.map(map_func=lambda example_proto: _parse_function(example_proto, dtype),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(shuffle_size).batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    return iterator
