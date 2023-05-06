from typing import Dict

import numpy as np
import tensorflow as tf


def read_image_tfds(image, bbox):
    """
    Resize `image` and `bbox`.
    - Resize the image to size 224x224
    - Standardize the image
    - Normalize bbox
    """
    image = tf.cast(image, tf.float32)
    shape = tf.shape(image)

    factor_x = tf.cast(shape[1], tf.float32)
    factor_y = tf.cast(shape[0], tf.float32)

    image = tf.image.resize(
        image,
        (
            224,
            224,
        ),
    )

    image = image / 127.5
    image -= 1

    bbox_list = [
        bbox[0] / factor_x,
        bbox[1] / factor_y,
        bbox[2] / factor_x,
        bbox[3] / factor_y,
    ]

    return image, bbox_list


def read_image_with_shape(image, bbox):
    """Makes a copy of the given images and returns it along with preprocessed one."""
    original_image = image
    image, bbox_list = read_image_tfds(image, bbox)
    return original_image, image, bbox_list


def read_image_tfds_with_original_bbox(data: Dict):
    """Unpacks image and bbox from `data` and denormalizes bbox."""
    image = data["image"]
    bbox = data["bbox"]

    shape = tf.shape(image)
    factor_x = tf.cast(shape[1], tf.float32)
    factor_y = tf.cast(shape[0], tf.float32)

    bbox_list = [
        bbox[1] * factor_x,
        bbox[0] * factor_y,
        bbox[3] * factor_x,
        bbox[2] * factor_y,
    ]
    return image, bbox_list


def dataset_to_numpy_util(dataset, batch_size=0, N=0):
    # eager execution: loop through datasets normally
    take_dataset = dataset.shuffle(1024)

    if batch_size > 0:
        take_dataset = take_dataset.batch(batch_size)

    if N > 0:
        take_dataset = take_dataset.take(N)

    if tf.executing_eagerly():
        ds_images, ds_bboxes = [], []
        for images, bboxes in take_dataset:
            ds_images.append(images.numpy())
            ds_bboxes.append(bboxes.numpy())

    return (np.array(ds_images), np.array(ds_bboxes))


def dataset_to_numpy_with_original_bboxes_util(dataset, batch_size=0, N=0):
    normalized_dataset = dataset.map(read_image_with_shape)
    if batch_size > 0:
        normalized_dataset = normalized_dataset.batch(batch_size)

    if N > 0:
        normalized_dataset = normalized_dataset.take(N)

    if tf.executing_eagerly():
        ds_original_images, ds_images, ds_bboxes = [], [], []

    for original_images, images, bboxes in normalized_dataset:
        ds_images.append(images.numpy())
        ds_bboxes.append(bboxes.numpy())
        ds_original_images.append(original_images.numpy())

    return np.array(ds_original_images), np.array(ds_images), np.array(ds_bboxes)
