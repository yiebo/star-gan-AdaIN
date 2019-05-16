#!/usr/bin/env python
import tensorflow as tf
from tqdm import tqdm
import csv
import os


def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def tfrecord_example(image_path, label):
    features = {'path': bytes_feature(image_path.encode()),
                'label': int64_feature(label),
                }

    example = tf.train.Example(features=tf.train.Features(feature=features))

    return example


if __name__ == '__main__':
    TFRECORD_SIZE = 1000
    tfrecord_idx = 1
    idx = 0
    males = []
    females = []
    path = '../DATASETS/celebA'
    output_path = 'TFRECORD'
    image_path = f'{path}/img_align_celeba'
    labels = [21, 40] + [9, 12, 10]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(f'{path}/list_attr_celeba.txt') as csvfile:
        reader = list(csv.reader(csvfile, delimiter=" "))

    tfrecord_writer = tf.python_io.TFRecordWriter(
        f'{output_path}/celebA_{tfrecord_idx:03d}.tfrecord')

    for instance in tqdm(reader[1:]):
        if idx >= TFRECORD_SIZE:
            idx = 0
            tfrecord_idx += 1
            tfrecord_writer.close()
            tfrecord_writer = tf.python_io.TFRecordWriter(
                f'{output_path}/celebA_{tfrecord_idx:03d}.tfrecord')

        label = [int(instance[x]) for x in labels]
        file_path = f'{image_path}/{instance[0]}'
        example = tfrecord_example(file_path, label)
        tfrecord_writer.write(example.SerializeToString())
        idx += 1
    tfrecord_writer.close()
