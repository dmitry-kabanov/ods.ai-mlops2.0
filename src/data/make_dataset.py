# -*- coding: utf-8 -*-
import os
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import tensorflow as tf
import tensorflow_datasets as tfds
from src.data.preprocessing import read_image_tfds_with_original_bbox
from src.data.preprocessing import read_image_tfds


@click.command()
@click.argument("input_ds", type=click.Path())
@click.argument("interim_train_ds_marker", type=click.Path())
@click.argument("interim_val_ds_marker", type=click.Path())
@click.argument("processed_train_ds_marker", type=click.Path())
@click.argument("processed_val_ds_marker", type=click.Path())
@click.option("--batch_size", default=64, type=click.INT)
def main(
    input_ds: str,
    interim_train_ds_marker: str,
    interim_val_ds_marker: str,
    processed_train_ds_marker: str,
    processed_val_ds_marker: str,
    batch_size: int,
):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    data_path = "/".join(input_ds.split("/")[:-1])
    ds_name = input_ds.split("/")[-1]

    visualization_train_dataset = get_visualization_training_dataset(ds_name, data_path)
    logger.info("Save interim training dataset")
    interim_train_ds_path = "/".join(interim_train_ds_marker.split("/")[:-1])
    os.makedirs(interim_train_ds_path, exist_ok=True)
    visualization_train_dataset.save(interim_train_ds_path)
    open(interim_train_ds_marker, "w").close()

    visualization_val_dataset = get_visualization_validation_dataset(ds_name, data_path)
    logger.info("Save interim validation dataset")
    interim_val_ds_path = "/".join(interim_val_ds_marker.split("/")[:-1])
    os.makedirs(interim_val_ds_path, exist_ok=True)
    visualization_val_dataset.save(interim_val_ds_path)
    open(interim_val_ds_marker, "w").close()

    logger.info("Run get_training_dataset with batch_size %d", batch_size)
    training_dataset = visualization_train_dataset.map(
        read_image_tfds, num_parallel_calls=16
    )
    logger.info("Run get_validation_dataset with batch_size %d", batch_size)
    validation_dataset = visualization_val_dataset.map(
        read_image_tfds, num_parallel_calls=16
    )

    logger.info("Save processed training dataset")
    processed_train_ds_path = "/".join(processed_train_ds_marker.split("/")[:-1])
    os.makedirs(processed_train_ds_path, exist_ok=True)
    training_dataset.save(processed_train_ds_path)
    open(processed_train_ds_marker, "w").close()

    logger.info("Save processed validation dataset")
    processed_val_ds_path = "/".join(processed_val_ds_marker.split("/")[:-1])
    os.makedirs(processed_val_ds_path, exist_ok=True)
    validation_dataset.save(processed_val_ds_path)
    open(processed_val_ds_marker, "w").close()


def get_visualization_training_dataset(ds_name, data_dir) -> tf.data.Dataset:
    dataset, info = tfds.load(
        ds_name,
        split="train",
        with_info=True,
        data_dir=data_dir,
        download=True,
    )
    print(info)
    visualization_training_dataset = dataset.map(
        read_image_tfds_with_original_bbox, num_parallel_calls=16
    )
    return visualization_training_dataset


def get_visualization_validation_dataset(ds_name, data_dir) -> tf.data.Dataset:
    dataset = tfds.load(ds_name, split="test", data_dir=data_dir, download=False)
    visualization_validation_dataset = dataset.map(
        read_image_tfds_with_original_bbox, num_parallel_calls=16
    )
    return visualization_validation_dataset


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
