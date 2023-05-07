import tensorflow as tf


def build_and_compile_model_mobilenetv2() -> tf.keras.Model:
    # define the input layer
    inputs = tf.keras.layers.Input(shape=(224, 224, 3))

    # create the model
    model = final_model_mobilenetv2(inputs)

    # compile your model
    model.compile(
        optimizer=tf.keras.optimizers.SGD(momentum=0.9),
        loss="mse",
    )

    model.summary()

    return model


def final_model_mobilenetv2(inputs):
    # features
    feature_cnn = feature_extractor_mobilenetv2(inputs)
    # dense layers
    last_dense_layer = dense_layers(feature_cnn)
    # bounding box
    bounding_box_output = bounding_box_regression(last_dense_layer)

    # define the TensorFlow Keras model using the inputs and outputs to your model
    model = tf.keras.Model(inputs=inputs, outputs=bounding_box_output)

    return model


def feature_extractor_mobilenetv2(inputs):
    # Create a mobilenet version 2 model object
    mobilenet_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )

    # pass the inputs into this modle object to get a feature extractor for these inputs
    feature_extractor = mobilenet_model(inputs)

    # return the feature_extractor
    return feature_extractor


def dense_layers(features):
    # global average pooling 2d layer
    x = tf.keras.layers.GlobalAveragePooling2D()(features)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)

    return x


def bounding_box_regression(x):
    # Dense layer named `bounding_box`
    bounding_box_regression_output = tf.keras.layers.Dense(4, name="bounding_box")(x)

    return bounding_box_regression_output
