import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
import tensorflow as tf
from spatial_transformer_3D import SpatialTransformer3D

def test_spatial_transformer_3D_train():
    """Test a simple training loop."""

    seed = 0
    params_spatial_transformer = dict(interp_method="bilinear", padding_mode="border")
    volume_shape = tf.TensorShape((1, 10, 10, 10, 1))
    # source and target volumes are 3D-tensors of shape [A0, W, H, D, C]
    source = tf.random.normal(shape=volume_shape, dtype=tf.float32, seed=seed)
    target = source + tf.random.normal(shape=volume_shape, stddev=1e-3, dtype=tf.float32, seed=seed)
    # train with 5 iterations
    num_iters = 5
    # initialize weights with identity transformation slightly pertubed
    init_quaternion = tf.constant([0., 0., 0., 1., 0., 0., 0., 1., 1., 1.])
    init_quaternion = init_quaternion + tf.random.normal(shape=(10,), stddev=1e-6, dtype=tf.float32, seed=seed)
    init_weights_quaternion = [tf.zeros((5, 10), dtype=tf.float32), init_quaternion]

    if tf.executing_eagerly():
        # input and model definition
        inp_source = tf.keras.Input(shape=volume_shape[1:], dtype="float32")
        reshaped = tf.keras.layers.Flatten()(inp_source)
        reshaped = tf.expand_dims(reshaped, axis=-1)
        max_pooled = tf.keras.layers.MaxPool1D(pool_size=2, strides=200, padding="SAME")(reshaped)
        flattened = tf.keras.layers.Flatten()(max_pooled)
        transformation = tf.keras.layers.Dense(units=10, activation=None, weights=init_weights_quaternion)(flattened)
        output = SpatialTransformer3D(**params_spatial_transformer)([inp_source, transformation])
        model = tf.keras.models.Model(inputs=[inp_source], outputs=[output])
        # optimizer function
        optimizer = tf.keras.optimizers.SGD(learning_rate=1e-6)
        # training loop
        for _ in range(num_iters):
            with tf.GradientTape() as tape:
                loss = tf.nn.l2_loss(model([source]) - target)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            # Check that gradients has correct shape
            tf.debugging.assert_equal(len(grads), 2)
            tf.debugging.assert_equal(grads[0].shape, tf.TensorShape((5, 10)))
            tf.debugging.assert_equal(grads[1].shape, tf.TensorShape((10)))

def test_spatial_transformer_3D_forward_methods():
    """Test a simple forward pass with different interpolation and padding methods."""

    seed = 0
    params_spatial_transformer = [
        dict(interp_method="bilinear", padding_mode="border")
        , dict(interp_method="nn", padding_mode="border")
        , dict(interp_method="nn", padding_mode="zeros")
        , dict(interp_method="nn", padding_mode="min")
    ]
    volume_shape = (1, 10, 10, 10, 1)
    # source volume is 3D-tensors of shape [A0, W, H, D, C]
    source = tf.random.normal(shape=volume_shape, dtype=tf.float32, seed=seed)
    # resample with identity transformation slightly pertubed
    quaternion = tf.constant([[0., 0., 0., 1., 0., 0., 0., 1., 1., 1.]], dtype=tf.float32)
    quaternion = quaternion + tf.random.normal(shape=(1, 10), stddev=1e-9, dtype=tf.float32, seed=seed)

    if tf.executing_eagerly():
        for param_spatial_transformer in params_spatial_transformer:
            output = SpatialTransformer3D(**param_spatial_transformer, trainable=False)([source, quaternion])
            tf.debugging.assert_near(source, output, atol=1e-6, message="{}".format(param_spatial_transformer))

def test_spatial_transformer_3D_forward_transformations():
    """Test a simple forward pass with different transformations."""

    seed = 0
    params_spatial_transformer = dict(interp_method="bilinear", padding_mode="border")
    volume_shape = (1, 10, 10, 10, 1)
    # source volume is 3D-tensors of shape [A0, W, H, D, C]
    source = tf.random.normal(shape=volume_shape, dtype=tf.float32, seed=seed)
    # resample with identity transformation slightly pertubed
    quaternions = [
        tf.constant([[0., 0., 0., 1., 0., 0., 0., 1., 1., 1.]], dtype=tf.float32) \
            + tf.random.normal(shape=(1, 10), stddev=1e-6, dtype=tf.float32, seed=seed)
        , tf.constant([[0., 0., 0., 1., 0., 0., 0.]], dtype=tf.float32) \
            + tf.random.normal(shape=(1, 7), stddev=1e-6, dtype=tf.float32, seed=seed)
        , tf.constant([[0., 0., 0., 1.]], dtype=tf.float32) \
            + tf.random.normal(shape=(1, 4), stddev=1e-6, dtype=tf.float32, seed=seed)
    ]

    if tf.executing_eagerly():
        for quaternion in quaternions:
            output = SpatialTransformer3D(**params_spatial_transformer, trainable=False)([source, quaternion])
            tf.debugging.assert_near(source, output, atol=1e-3, message="{}".format(quaternion))

if __name__ == "__main__":
    test_spatial_transformer_3D_train()
    test_spatial_transformer_3D_forward_methods()
    test_spatial_transformer_3D_forward_transformations()