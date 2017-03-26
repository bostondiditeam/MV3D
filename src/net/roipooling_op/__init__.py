# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
#import roipooling_op
import tensorflow as tf
import os
from tensorflow.python.framework import ops

print ('running init code of roi pooling')
filename = os.path .join(os.path .dirname(__file__), 'roi_pooling.so')
_roi_pooling_module = tf.load_op_library(filename)
roi_pool      = _roi_pooling_module.roi_pool
roi_pool_grad = _roi_pooling_module.roi_pool_grad



#import roi_pooling_op_grad
@ops.RegisterGradient("RoiPool")
def _roi_pool_grad(op, grad, _):
  """The gradients for `roi_pool`.
  Args:
    op: The `roi_pool` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `roi_pool` op.
  Returns:
    Gradients with respect to the input of `zero_out`.
  """
  data   = op.inputs[0]
  rois   = op.inputs[1]
  argmax = op.outputs[1]
  height = op.get_attr('pooled_height')#('pooled_height')
  width  = op.get_attr('pooled_width')#('pooled_width')
  scale  = op.get_attr('spatial_scale')#('spatial_scale')

  # compute gradient
  data_grad = roi_pool_grad(data, rois, argmax, grad, height, width, scale)

  return [data_grad, None]  # List of one Tensor, since we have one input
