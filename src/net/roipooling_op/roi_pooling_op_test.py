import tensorflow as tf
import numpy as np
from net.roipooling_op import *


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# main --------------------------------------------------------------------------
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    data_array = np.random.rand(32, 100, 100, 3)
    data = tf.convert_to_tensor(data_array, dtype=tf.float32)
    rois = tf.convert_to_tensor([[0, 10, 10, 20, 20], [31, 30, 30, 40, 40]], dtype=tf.float32)

    #dummy net
    W = weight_variable([3, 3, 3, 1])
    h = conv2d(data, W)
    y, argmax = roi_pool(h, rois, 6, 6, 1.0/3)


    #loss
    y_hat = tf.convert_to_tensor(np.ones((2, 6, 6, 1)), dtype=tf.float32)
    loss = tf.reduce_mean(tf.square(y - y_hat))   # Minimize the mean squared errors.
    optimizer      = tf.train.GradientDescentOptimizer(learning_rate=0.008)
    optimizer_step = optimizer.minimize(loss)


    print ('-----------------------------')
    print ('y_hat, y, argmax')
    print (y_hat, y, argmax)


    #start training here! ---------------------------------
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess.run(tf.global_variables_initializer())

    for iter in range(500):
        _,ls = sess.run([optimizer_step,loss])


        print ('-----------------------------')
        print ('iter=%d,  loss=%f'%(iter,ls))
        #weights   = sess.run(W)
        #estimates = sess.run(y)
        #print ('weights:\n',weights)
        #print ('estimates:\n',estimates)
        print ('')

    weights   = sess.run(W)
    estimates = sess.run(y)
    features = sess.run(h)
    print ('weights:\n',weights)
    print ('estimates:\n',estimates)
    print ('features:\n',features)

    #with tf.device('/gpu:0'):
    #  result = module.roi_pool(data, rois, 1, 1, 1.0/1)
    #  print result.eval()
    #with tf.device('/cpu:0'):
    #  run(init)
