# import random
# import numpy as np
# import tensorflow as tf
# import cv2

# SEED = 202
# random.seed(SEED)
# np.random.seed(SEED)
# tf.set_random_seed(SEED)

#### kitti dataset orijection from lidar to top, front and rgb ####

TOP_Y_MIN=-20  #40
TOP_Y_MAX=+20
TOP_X_MIN=0
TOP_X_MAX=40   #70.4
TOP_Z_MIN=-2.0    ###<todo> determine the correct values!
TOP_Z_MAX= 0.4

TOP_X_DIVISION=0.1  #0.1
TOP_Y_DIVISION=0.1
TOP_Z_DIVISION=0.4


#rgb camera
MATRIX_Mt = ([[  2.34773698e-04,   1.04494074e-02,   9.99945389e-01,  0.00000000e+00],
              [ -9.99944155e-01,   1.05653536e-02,   1.24365378e-04,  0.00000000e+00],
              [ -1.05634778e-02,  -9.99889574e-01,   1.04513030e-02,  0.00000000e+00],
              [  5.93721868e-02,  -7.51087914e-02,  -2.72132796e-01,  1.00000000e+00]])

MATRIX_Kt = ([[ 721.5377,    0.    ,    0.    ],
              [   0.    ,  721.5377,    0.    ],
              [ 609.5593,  172.854 ,    1.    ]])