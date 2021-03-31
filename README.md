# 3DSpatialTransformer
A tf.keras derivable layer to transform and resample a 3D volume using the quaternion and translation parameters.

It basically implements https://arxiv.org/pdf/1506.02025.pdf in 3D.
This derivable layer can be used either to improve the performance on a model with spatial un-normalized data (like for MNIST classification) or to perform (non-)rigid registration within a deep learning framework. 
