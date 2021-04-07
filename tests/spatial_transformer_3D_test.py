import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
import tensorflow as tf
from spatial_transformer_3D import SpatialTransformer3D

def test_spatial_transformer_3D_train():
    """Test a simple training loop."""

    seed = 0
    params_spatial_transformer = dict(
        min_ref_grid=[-1., -1., -1.],
        max_ref_grid=[1., 1., 1.],
        interp_method="bilinear",
        padding_mode="min")
    volume_shape = tf.TensorShape((1, 10, 10, 10, 1))
    # source and target volumes are 3D-tensors of shape [A0, W, H, D, C]
    source = tf.random.normal(shape=volume_shape, dtype=tf.float32, seed=seed)
    target = source + 1e-6
    # train with 5 iterations
    num_iters = 5

    if tf.executing_eagerly():
        # inputs
        inp_source = tf.keras.Input(shape=volume_shape[1:], dtype="float32")
        inp_target = tf.keras.Input(shape=volume_shape[1:], dtype="float32")
        # model definition
        summed = tf.keras.layers.Add()([inp_source, inp_target])
        reshaped = tf.keras.layers.Flatten()(summed)
        reshaped = tf.expand_dims(reshaped, axis=-1)
        max_pooled = tf.keras.layers.MaxPool1D(pool_size=2, strides=200, padding="SAME")(reshaped)
        flattened = tf.keras.layers.Flatten()(max_pooled)
        transformation = tf.keras.layers.Dense(units=7, activation=None)(flattened)
        output = SpatialTransformer3D(**params_spatial_transformer)([inp_source, transformation])
        model = tf.keras.models.Model(inputs=[inp_target, inp_source], outputs=[output])
        # optimizer function
        optimizer = tf.keras.optimizers.SGD(learning_rate=1e-6)
        # training loop
        for _ in range(num_iters):
            with tf.GradientTape() as tape:
                loss = tf.nn.l2_loss(model([source, target]) - target)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            # Check that gradients has correct shape
            tf.debugging.assert_equal(len(grads), 2)
            tf.debugging.assert_equal(grads[0].shape, tf.TensorShape((5, 7)))
            tf.debugging.assert_equal(grads[1].shape, tf.TensorShape((7)))

def test_spatial_transformer_3D_forward_bilinear():
    """Test a simple forward pass with bilinear resampling."""

    params_spatial_transformer = dict(
            min_ref_grid=[-1., -1., -1.],
            max_ref_grid=[1., 1., 1.],
            interp_method="bilinear",
            padding_mode="min")
    volume_shape = (1, 10, 10, 10, 1)
    # source volume is 3D-tensors of shape [A0, W, H, D, C]
    source = tf.random.normal(shape=volume_shape, dtype=tf.float32, seed=0)
    # resample with identity quaternion and no translation
    quaternion = tf.constant([[1., 0., 0., 0., 0., 0., 0.]], dtype=tf.float32)

    if tf.executing_eagerly():
        output = SpatialTransformer3D(**params_spatial_transformer, trainable=False)([source, quaternion])
        tf.debugging.assert_near(source, output)

def test_spatial_transformer_3D_forward_nn():
    """Test a simple forward pass with nn resampling."""

    params_spatial_transformer = dict(
            min_ref_grid=[-1., -1., -1.],
            max_ref_grid=[1., 1., 1.],
            interp_method="nn",
            padding_mode="min")
    volume_shape = (1, 10, 10, 10, 1)
    # source volume is 3D-tensors of shape [A0, W, H, D, C]
    source = tf.random.normal(shape=volume_shape, dtype=tf.float32, seed=0)
    # resample with identity quaternion and no translation
    quaternion = tf.constant([[1., 0., 0., 0., 0., 0., 0.]], dtype=tf.float32)

    if tf.executing_eagerly():
        output = SpatialTransformer3D(**params_spatial_transformer, trainable=False)([source, quaternion])
        tf.debugging.assert_near(source, output)

if __name__ == "__main__":
    test_spatial_transformer_3D_train()
    test_spatial_transformer_3D_forward_bilinear()
    test_spatial_transformer_3D_forward_nn()