import logging

import click
import numpy as np
import tensorflow as tf

from src.data.preprocessing import dataset_to_numpy_with_original_bboxes_util
from src.models.build_model import build_and_compile_model_mobilenetv2
from src.models.evaluate import intersection_over_union
from src.visualization import display_digits_with_boxes, plot_metrics

EPOCHS = 50


@click.command()
@click.argument("visualization_train_ds_path", type=click.Path(exists=True))
@click.argument("visualization_val_ds_path", type=click.Path(exists=True))
@click.argument("train_ds_path", type=click.Path(exists=True))
@click.argument("val_ds_path", type=click.Path(exists=True))
@click.option("--batch_size", default=64, type=click.INT)
def main(
    visualization_train_ds_path: str,
    visualization_val_ds_path: str,
    train_ds_path: str,
    val_ds_path: str,
    batch_size,
):
    logger = logging.getLogger(__name__)
    marker_str = "/marker"

    visualization_train_ds_path = visualization_train_ds_path[: -len(marker_str)]
    logger.info("Load visualization_train_ds from path %s", visualization_train_ds_path)
    visualization_train_ds = tf.data.Dataset.load(visualization_train_ds_path)

    visualization_val_ds_path = visualization_val_ds_path[: -len(marker_str)]
    logger.info("Load visualization_val_ds from path %s", visualization_val_ds_path)
    visualization_val_ds = tf.data.Dataset.load(visualization_val_ds_path)

    logger.info("Load training_dataset")
    train_ds_path = train_ds_path[: -len(marker_str)]
    training_dataset = tf.data.Dataset.load(train_ds_path)
    training_dataset = training_dataset.shuffle(512, reshuffle_each_iteration=True)
    training_dataset = training_dataset.repeat()
    training_dataset = training_dataset.batch(batch_size)
    training_dataset = training_dataset.prefetch(-1)

    logger.info("Load validation_dataset")
    val_ds_path = val_ds_path[: -len(marker_str)]
    validation_dataset = tf.data.Dataset.load(val_ds_path)
    validation_dataset = validation_dataset.batch(batch_size)
    validation_dataset = validation_dataset.repeat()

    # Get the length of the training set
    length_of_training_dataset = len(visualization_train_ds)

    # Get the length of the validation set
    length_of_validation_dataset = len(visualization_val_ds)

    BATCH_SIZE = batch_size

    steps_per_epoch = length_of_training_dataset // BATCH_SIZE
    if length_of_training_dataset % BATCH_SIZE > 0:
        steps_per_epoch += 1

    validation_steps = length_of_validation_dataset // BATCH_SIZE
    if length_of_validation_dataset % BATCH_SIZE > 0:
        validation_steps += 1

    model = build_and_compile_model_mobilenetv2()
    model.summary()

    history = model.fit(
        training_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_dataset,
        validation_steps=validation_steps,
        epochs=EPOCHS,
    )

    loss = model.evaluate(validation_dataset, steps=validation_steps)
    print("Loss: ", loss)

    model.save("models/birds_mobilenetv2.h5")

    plot_metrics(history, "loss", "Bounding Box Loss", ylim=0.2)

    # Makes predictions
    (
        original_images,
        normalized_images,
        normalized_bboxes,
    ) = dataset_to_numpy_with_original_bboxes_util(visualization_val_ds, N=500)
    predicted_bboxes = model.predict(normalized_images, batch_size=32)

    # Calculates IOU and reports true positives and false positives
    # based on IOU threshold.
    iou = intersection_over_union(predicted_bboxes, normalized_bboxes)
    iou_threshold = 0.5

    print(
        "Number of predictions where iou > threshold(%s): %s"
        % (iou_threshold, (iou >= iou_threshold).sum())
    )
    print(
        "Number of predictions where iou < threshold(%s): %s"
        % (iou_threshold, (iou < iou_threshold).sum())
    )

    n = 10
    indexes = np.random.choice(len(predicted_bboxes), size=n)

    # iou_to_draw = iou[indexes]
    # norm_to_draw = original_images[indexes]
    display_digits_with_boxes(
        original_images[indexes],
        predicted_bboxes[indexes],
        normalized_bboxes[indexes],
        iou[indexes],
        "True and Predicted values",
        iou_threshold,
        bboxes_normalized=True,
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
