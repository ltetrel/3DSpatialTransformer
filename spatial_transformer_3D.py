"""Implementation of the Spatial Transformer in 3D.

@misc{jaderberg2016spatial,
      title={Spatial Transformer Networks}, 
      author={Max Jaderberg and Karen Simonyan and Andrew Zisserman and Koray Kavukcuoglu},
      year={2016},
      eprint={1506.02025},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

NOTE: Nearest neighbor interpolation is not available during training, because of non-derivable operations.
"""

import tensorflow as tf

def repeat(x, num_reps):
    num_reps = tf.cast(num_reps, dtype=tf.int32)
    x = tf.expand_dims(x, axis=1)
    return tf.tile(x, multiples=(1,num_reps))

def quat_to_3D_matrix(q):
    size_q = tf.shape(q[...,0])
    R = tf.stack([1 - 2.*(q[...,2]**2 + q[...,3]**2), 2*(q[...,1]*q[...,2] - q[...,0]*q[...,3]), 2*(q[...,0]*q[...,2] + q[...,1]*q[...,3]), tf.zeros(size_q),
                    2.*(q[...,1]*q[...,2] + q[...,0]*q[...,3]), 1 - 2.*(q[...,1]**2 + q[...,3]**2), 2.*(q[...,2]*q[...,3] - q[...,0]*q[...,1]), tf.zeros(size_q),
                    2.*(q[...,1]*q[...,3] - q[...,0]*q[...,2]), 2.*(q[...,0]*q[...,1] + q[...,2]*q[...,3]), 1 - 2.*(q[...,1]**2 + q[...,2]**2), tf.zeros(size_q),
                    tf.zeros(size_q), tf.zeros(size_q), tf.zeros(size_q), tf.ones(size_q)],axis=-1)

    return tf.reshape(R, (-1, 4, 4))

def get_matrix_from_params(transfos, num_elems):
    num_batch = tf.shape(transfos)[0]
    scaling = tf.shape(transfos)[-1] == 10
    trans = tf.shape(transfos)[-1] == 7

    #if the transformations is [q0, q1, q2, q3, tx, ty, tz, sx, sy, sz], then we apply scaling
    def apply_scaling(): return tf.linalg.diag(1./transfos[:, -3:])
    def default_scaling(): return tf.eye(num_rows=3, batch_shape=[num_batch])
    thetas = tf.cond(scaling, apply_scaling, default_scaling)

    #if the transformations is [q0, q1, q2, q3, tx, ty, tz], then we apply translation
    def apply_trans(): return tf.concat(axis=2, values=[thetas, transfos[:, 4:7, tf.newaxis]])
    def default_trans(): return tf.concat(axis=2, values=[thetas, tf.zeros((num_batch, 3, 1))])
    thetas = tf.cond(trans, apply_trans, default_trans)

    #transformations should always be at least [q0, q1, q2, q3]
    R = quat_to_3D_matrix(transfos[:, :4])
    thetas = tf.linalg.matmul(thetas, R)

    return thetas

class SpatialTransformer3D(tf.keras.layers.Layer):
    """ The 3D Spatial Transformer derivable layer."""
    def __init__(self, min_ref_grid=[-1., -1., -1.], max_ref_grid=[1., 1., 1.], interp_method="bilinear", padding_mode="min", **kwargs):
        """Constructs a 3D Spatial Transformer layer.
        
        Args:
            min_ref_grid: A `list` of `float`.
                The starting points to define the resampling grid for each spatial dimension (default: [-1., -1., -1.]).
            max_ref_grid: A `list` of `float`.
                The end points to define the resampling grid for each spatial dimension (default: [1., 1., 1.]).
            interp_method: A `string` between `"bilinear"` or `"nn"` (case-insensitive).
                `"bilinear"` takes the weighted sum of each neighboring pixel,
                `"nn"` takes instead the nearest neighboring pixel (default: `"bilinear"`).
            padding_mode: A `string` between `"border"`, `"zeros"` or `"min"` (case-insensitive).
                It defines which default value should be used for pixels that are outside the grid after the transformation.
                `"border"` to use the same value as the border,
                `"zeros"` to nullify them,
                `"min"` to use the minimum value from the input tensor (default: `"min"`).
            **kwargs: Additional keyword arguments passed to the base layer.
        """
        super(self.__class__, self).__init__(**kwargs)
        self.min_ref_grid = tf.constant(min_ref_grid, dtype=tf.float32)
        self.max_ref_grid = tf.constant(max_ref_grid, dtype=tf.float32)
        self.interp_method = tf.constant(interp_method, dtype=tf.string)
        self.padding_mode = tf.constant(padding_mode, dtype=tf.string)

        if tf.math.logical_and(self.trainable == tf.constant(True), self.interp_method == tf.constant("nn")):
            raise Exception("Cannot train with nearest-neighbor interpolator because it is not derivable!") 

    def build(self, input_shape):
        super(self.__class__, self).build(input_shape)
        num_dims = input_shape[0].ndims - 2
        shape_grid = tf.shape(self.min_ref_grid)[0]

        def ref_grid():
            self.min_ref_grid = (-1) * tf.ones(num_dims, dtype=tf.float32)
            self.max_ref_grid = tf.ones(num_dims, dtype=tf.float32)
        tf.cond(num_dims != shape_grid, ref_grid, lambda *args: None)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        return {
            'min_ref_grid': self.min_ref_grid,
            'max_ref_grid': self.max_ref_grid,
            'interp_method': self.interp_method,
            'padding_mode': self.padding_mode
        }

    def call(self, inputs):
        img, transfos = inputs
        output = self._resample(img, transfos)
        return output

    def _resample(self, img, transfos):
        """Constructs a 3D Spatial Transformer layer.
        
        Args:
            inputs: a dense tensor of size `[B, N, 1, D]`.
            training: flag to control batch normalization update statistics.

        Returns:
            Tensor with shape `[B, N, 1, C]`.
        """
        input_shape = tf.shape(img)
        ref_size = input_shape[1:-1]
        ref_size_xyz = tf.concat([ref_size[1::-1], ref_size[2:]], axis=0)

        input_transformed = self._transform_grid(ref_size_xyz, transfos=transfos, min_ref_grid=self.min_ref_grid, max_ref_grid=self.max_ref_grid)
        input_transformed = self._interpolate(im=img
                                            , points=input_transformed
                                            , min_ref_grid=self.min_ref_grid
                                            , max_ref_grid=self.max_ref_grid
                                            , method=self.interp_method
                                            , padding_mode=self.padding_mode)
        output = tf.reshape(input_transformed, shape=input_shape)

        return output
    
    def _transform_grid(self, ref_size_xyz, transfos, min_ref_grid, max_ref_grid):
        num_batch = tf.shape(transfos)[0]
        num_elems = tf.reduce_prod(ref_size_xyz)
        thetas = get_matrix_from_params(transfos, num_elems)

        # grid creation from volume affine
        mz, my, mx = tf.meshgrid(tf.linspace(min_ref_grid[2], max_ref_grid[2], ref_size_xyz[2])
                                , tf.linspace(min_ref_grid[1], max_ref_grid[1], ref_size_xyz[1])
                                , tf.linspace(min_ref_grid[0], max_ref_grid[0], ref_size_xyz[0])
                                , indexing='ij')

        # preparing grid for quaternion rotation
        grid = tf.concat([tf.reshape(mx, (1, -1)), tf.reshape(my, (1, -1)), tf.reshape(mz, (1, -1))], axis=0)
        grid = tf.expand_dims(grid, axis=0)
        grid = tf.tile(grid, (num_batch, 1, 1))

        # preparing grid for augmented transformation
        grid = tf.concat([grid, tf.ones((num_batch, 1, num_elems))], axis=1)
        return tf.linalg.matmul(thetas, grid)
    
    def _interpolate(self, im, points, min_ref_grid, max_ref_grid, method="bilinear", padding_mode="zeros"):
        num_batch = tf.shape(im)[0]
        vol_shape_xyz = tf.cast(tf.concat([tf.shape(im)[1:-1][1::-1], tf.shape(im)[1:-1][2:]], axis=0), dtype=tf.float32)
        width = vol_shape_xyz[0]
        height = vol_shape_xyz[1]
        depth = vol_shape_xyz[2]
        width_i = tf.cast(width, dtype=tf.int32)
        height_i = tf.cast(height, dtype=tf.int32)
        depth_i = tf.cast(depth, dtype=tf.int32)
        channels = tf.shape(im)[-1]
        num_row_major = tf.cast(tf.math.cumprod(vol_shape_xyz), dtype=tf.int32)
        shape_output = tf.stack([num_batch, num_row_major[-1] , 1])
        zero = tf.zeros([], dtype=tf.float32)
        zero_i = tf.zeros([], dtype=tf.int32)
        ibatch = repeat(num_row_major[-1] * tf.range(num_batch, dtype=tf.int32), num_row_major[-1])

        # scale positions to [0, width/height - 1]
        coeff_x = (width - 1.)/(max_ref_grid[0] - min_ref_grid[0])
        coeff_y = (height - 1.)/(max_ref_grid[1] - min_ref_grid[1])
        coeff_z = (depth - 1.)/(max_ref_grid[2] - min_ref_grid[2])
        ix = (coeff_x * points[:, 0, :]) - (coeff_x *  min_ref_grid[0])
        iy = (coeff_y * points[:, 1, :]) - (coeff_y *  min_ref_grid[1])
        iz = (coeff_z * points[:, 2, :]) - (coeff_z *  min_ref_grid[2])

        # zeros padding mode, for positions outside of refrence grid
        cond = tf.math.logical_or(tf.math.equal(padding_mode, tf.constant("zeros", dtype=tf.string))
                                  , tf.math.equal(padding_mode, tf.constant("min", dtype=tf.string)))
        def evaluate_valid(): return tf.expand_dims(tf.cast(tf.less_equal(ix, width - 1.) & tf.greater_equal(ix, zero)
                                             & tf.less_equal(iy, height - 1.) & tf.greater_equal(iy, zero)
                                             & tf.less_equal(iz, depth - 1.) & tf.greater_equal(iz, zero)
                                             , dtype=tf.float32), -1)
        def default(): return tf.ones([], dtype=tf.float32)
        valid = tf.cond(cond, evaluate_valid, default)

        # if using bilinear interpolation, calculate each area between corners and positions to get the weights for each input pixel
        def bilinear():
            output = tf.zeros(shape_output, dtype=tf.float32)
            
            # get north-west-top corner indexes based on the scaled positions
            ix_nwt = tf.clip_by_value(tf.floor(ix), zero, width - 1.)
            iy_nwt = tf.clip_by_value(tf.floor(iy), zero, height - 1.)
            iz_nwt = tf.clip_by_value(tf.floor(iz), zero, depth - 1.)
            ix_nwt_i = tf.cast(ix_nwt, dtype=tf.int32)
            iy_nwt_i = tf.cast(iy_nwt, dtype=tf.int32)
            iz_nwt_i = tf.cast(iz_nwt, dtype=tf.int32)       

            #gettings all offsets to create corners
            offset_corner = tf.constant([ [0., 0., 0.]
                                        , [0., 0., 1.]
                                        , [0., 1., 0.]
                                        , [0., 1., 1.]
                                        , [1., 0., 0.]
                                        , [1., 0., 1.]
                                        , [1., 1., 0.]
                                        , [1., 1., 1.]], dtype=tf.float32)
            offset_corner_i =  tf.cast(offset_corner, dtype=tf.int32)

            for c in range(8):
                # getting all corner indexes from north-west-top corner
                ix_c = ix_nwt + offset_corner[-c - 1, 0]
                iy_c = iy_nwt + offset_corner[-c - 1, 1]
                iz_c = iz_nwt + offset_corner[-c - 1, 2]

                # area is computed using the opposite corner
                nc = tf.expand_dims(tf.abs((ix - ix_c) * (iy - iy_c) * (iz - iz_c)), -1)

                # current corner position
                ix_c = ix_nwt_i + offset_corner_i[c, 0]
                iy_c = iy_nwt_i + offset_corner_i[c, 1]
                iz_c = iz_nwt_i + offset_corner_i[c, 2]

                # gather input image values from corners idx, and calculate weighted pixel value
                idx_c = ibatch + tf.clip_by_value(ix_c, zero_i, width_i - 1) \
                        + num_row_major[0] * tf.clip_by_value(iy_c, zero_i, height_i - 1) \
                        + num_row_major[1] * tf.clip_by_value(iz_c, zero_i, depth_i - 1)
                Ic = tf.gather(tf.reshape(im, [-1, channels]), idx_c)

                output += nc * Ic
            return output
        # else if using nearest neighbor, get the nearest corner
        def nearest_neighbor():
            # get rounded indice corner based on the scaled positions
            ix_nn = tf.cast(tf.clip_by_value(tf.round(ix), zero, width - 1.), dtype=tf.int32)
            iy_nn = tf.cast(tf.clip_by_value(tf.round(iy), zero, height - 1.), dtype=tf.int32)
            iz_nn = tf.cast(tf.clip_by_value(tf.round(iz), zero, depth - 1.), dtype=tf.int32)

            # gather input pixel values from nn corner indexes
            idx_nn = ibatch + ix_nn + num_row_major[0] * iy_nn + num_row_major[1] * iz_nn
            output = tf.gather(tf.reshape(im, [-1, channels]), idx_nn)
            return output

        cond_bilinear = tf.math.equal(method, tf.constant("bilinear", dtype=tf.string))
        cond_nn = tf.math.equal(method, tf.constant("nn", dtype=tf.string))
        output = tf.case([(cond_bilinear, bilinear), (cond_nn, nearest_neighbor)], exclusive=True)
        
        # padding mode
        cond_border = tf.math.equal(padding_mode, tf.constant("border", dtype=tf.string))
        cond_zero = tf.math.equal(padding_mode, tf.constant("zeros", dtype=tf.string))
        cond_value = tf.math.equal(padding_mode, tf.constant("min", dtype=tf.string))
        def border_padding_mode(): return output
        def zero_padding_mode(): return output * valid
        def min_padding_mode(): return output * valid + tf.reduce_min(im) * (1. - valid)
        output = tf.case([(cond_border, border_padding_mode), (cond_zero, zero_padding_mode), (cond_value, min_padding_mode)], exclusive=True)

        return output