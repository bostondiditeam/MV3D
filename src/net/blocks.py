'''
building blocks of network
#http://programtalk.com/vs2/python/3069/image_captioning/utils/nn.py/
'''
import tensorflow as tf
import numpy as np




##  global varaiables ##
IS_TRAIN_PHASE = tf.placeholder(dtype=tf.bool, name='is_train_phase')

## net summaries ## -------------------------------
## bug for deconvolution ##
def print_macs_to_file(log=None):

    #nodes = tf.get_default_graph().as_graph_def().node
    #variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    if log is not None:
        log.write( 'MAC for conv layers : \n')
        log.write( 'MAC  param_size  :   name           (op)    params   out    in \n')
        log.write( '----------------------------------------------------------------\n')


    all =0
    all_param_size=0
    all_mac=0

    ops = tf.Graph.get_operations(tf.get_default_graph())
    for op in ops:
        if hasattr(op.op_def, 'name'):
            op_name = op.op_def.name
            if op_name =='Conv2D':

                #print(op.name)
                #print(op.inputs)
                #print(op.outputs)
                #print(op.op_def)
                # assert(op.inputs[1].name == op.name + '_weight/read:0')
                # input_shape  = op.inputs[0].get_shape().as_list()
                # output_shape = op.outputs[0].get_shape().as_list()
                # kernel_shape = op.inputs[1].get_shape().as_list()
                # print(input_shape)
                # print(output_shape)
                # print(kernel_shape)

                g=1 # how do we handle group (e.g used in caffe, mxnet) here ???
                assert(op.inputs[1].name == op.name + '_weight/read:0')
                inum, ih, iw, ic  = op.inputs[0].get_shape().as_list()
                onum, oh, ow, oc  = op.outputs[0].get_shape().as_list()
                h, w, ki, ko    = op.inputs[1].get_shape().as_list()
                assert(ic==ki)
                assert(oc==ko)


                name=op.name
                input_name =op.inputs [0].name
                output_name=op.outputs[0].name
                try:
                    mac = w*h*ic *oc* oh*ow /1000000./g   #10^6 "multiply-accumulate count"
                    param_size = oc*h*w*ic/1000000.


                    all_param_size += param_size
                    all_mac += mac
                    all += 1

                    if log is not None:
                        log.write('%10.1f  %5.2f  :  %-26s (%s)   %4d  %dx%dx%4d   %-30s %3d, %3d, %4d,   %-30s %3d, %3d, %5d\n'%
                               (mac, param_size , name,'Conv2D', oc,h,w,ic,  output_name, oh, ow, oc, input_name, ih, iw, ic ))


                except:
                    print ('error in shape?')


            if op_name == 'MatMul':
                #raise Exception('xxx')
                # print(op.name)
                # print(op.inputs)
                # print(op.outputs)
                # print(op.op_def)

                #assert (op.inputs[1].name == op.name + ':0')
                inum, ic  = op.inputs[0].get_shape().as_list()
                onum, oc  = op.outputs[0].get_shape().as_list()

                name = op.name
                input_name = op.inputs[0].name
                output_name = op.outputs[0].name

                mac =  ic * oc  / 1000000. / g  # 10^6 "multiply-accumulate count"
                param_size = oc * ic / 1000000.

                all_param_size += param_size
                all_mac += mac
                all += 1

                if log is not None:
                    log.write('%10.1f  %5.2f  :  %-26s (%s)   %4d  %dx%dx%3d   %-30s %3d, %3d, %4d,   %-30s %3d, %3d, %5d\n' %
                        (mac, param_size, name, 'Conv2D', oc, 1, 1, ic, output_name, 1, 1, oc, input_name, 1, 1, ic))
    if log is not None:
        log.write( '\n')
        log.write('summary : \n')
        log.write( 'num of conv     = %d\n'%all)
        log.write( 'all mac         = %.1f (M)\n'%all_mac)
        log.write( 'all param_size  = %.1f (M)\n'%all_param_size)

    return all, all_mac, all_param_size



## loss and metric ## -------------------------------
def l2_regulariser(decay):

    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    for v in variables:
        name = v.name
        if 'weight' in name:  #this is weight
            l2 = decay * tf.nn.l2_loss(v)
            tf.add_to_collection('losses', l2)
        elif 'bias' in name:  #this is bias
            pass
        elif 'beta' in name:
            pass
        elif 'gamma' in name:
            pass
        elif 'moving_mean' in name:
            pass
        elif 'moving_variance' in name:
            pass
        elif 'moments' in name:
            pass

        else:
            #pass
            raise Exception('unknown variable type: %s ?'%name)
            pass

    l2_loss = tf.add_n(tf.get_collection('losses'))
    return l2_loss



## op layers ## -------------------------------

# http://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
def conv2d(input, num_kernels=1, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', has_bias=True, name='conv'):

    input_shape = input.get_shape().as_list()
    assert len(input_shape)==4
    C = input_shape[3]
    H = kernel_size[0]
    W = kernel_size[1]
    K = num_kernels

    ##[filter_height, filter_width, in_channels, out_channels]
    w    = tf.get_variable(name=name+'_weight', shape=[H, W, C, K], initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv = tf.nn.conv2d(input, w, strides=stride, padding=padding, name=name)
    if has_bias:
        b = tf.get_variable(name=name + '_bias', shape=[K], initializer=tf.constant_initializer(0.0))
        conv = conv+b

    return conv


def relu(input, name='relu'):
    act = tf.nn.relu(input, name=name)
    return act


def dropout(input, keep=1.0, name='drop'):
    #drop = tf.cond(IS_TRAIN_PHASE, lambda: tf.nn.dropout(input, keep), lambda: input)
    drop = tf.cond(IS_TRAIN_PHASE,
                   lambda: tf.nn.dropout(input, keep),
                   lambda: tf.nn.dropout(input, 1))
    return drop


#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/api_docs/python/functions_and_classes/shard4/tf.contrib.layers.batch_norm.md
#http://www.bubufx.com/detail-1792794.html
def bn (input, decay=0.9, eps=1e-5, name='bn'):
    with tf.variable_scope(name) as scope:
        bn = tf.cond(IS_TRAIN_PHASE,
            lambda: tf.contrib.layers.batch_norm(input,  decay=decay, epsilon=eps, center=True, scale=True,
                              is_training=1,reuse=None,
                              updates_collections=None, scope=scope),
            lambda: tf.contrib.layers.batch_norm(input, decay=decay, epsilon=eps, center=True, scale=True,
                              is_training=0, reuse=True,
                              updates_collections=None, scope=scope))

    return bn


def maxpool(input, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='max' ):
    H = kernel_size[0]
    W = kernel_size[1]
    pool = tf.nn.max_pool(input, ksize=[1, H, W, 1], strides=stride, padding=padding, name=name)
    return pool

def avgpool(input, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', has_bias=True, is_global_pool=False, name='avg'):

    if is_global_pool==True:
        input_shape = input.get_shape().as_list()
        assert len(input_shape) == 4
        H = input_shape[1]
        W = input_shape[2]

        pool = tf.nn.avg_pool(input, ksize=[1, H, W, 1], strides=[1,H,W,1], padding='VALID', name=name)
        pool = flatten(pool)

    else:
        H = kernel_size[0]
        W = kernel_size[1]
        pool = tf.nn.avg_pool(input, ksize=[1, H, W, 1], strides=stride, padding=padding, name=name)

    return pool


def concat(input, axis=3, name='cat'):
    cat = tf.concat(axis=axis, values=input, name=name)
    return cat

def flatten(input, name='flat'):
    input_shape = input.get_shape().as_list()        # list: [None, 9, 2]
    dim   = np.prod(input_shape[1:])                 # dim = prod(9,2) = 18
    flat  = tf.reshape(input, [-1, dim], name=name)  # -1 means "all"
    return flat

def linear(input, num_hiddens=1,  has_bias=True, name='linear'):
    input_shape = input.get_shape().as_list()
    assert len(input_shape)==2

    C = input_shape[1]
    K = num_hiddens

    w = tf.get_variable(name=name + '_weight', shape=[C,K], initializer=tf.truncated_normal_initializer(stddev=0.1))
    dense = tf.matmul(input, w, name=name)
    if has_bias:
        b = tf.get_variable(name=name + '_bias', shape=[K], initializer=tf.constant_initializer(0.0))
        dense = dense + b

    return dense





#http://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/
#https://github.com/MarvinTeichmann/tensorflow-fcn/blob/master/fcn16_vgg.py
#http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/18/image-segmentation-with-tensorflow-using-cnns-and-conditional-random-fields/
def upsample2d(input, factor = 2, has_bias=True, trainable=True, name='upsample2d'):

    def make_upsample_filter(size):
        '''
        Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
        '''
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        return (1 - abs(og[0] - center) / factor) * \
               (1 - abs(og[1] - center) / factor)

    input_shape = input.get_shape().as_list()
    assert len(input_shape)==4
    N = input_shape[0]
    H = input_shape[1]
    W = input_shape[2]
    C = input_shape[3]
    K = C

    size = 2 * factor - factor % 2
    filter = make_upsample_filter(size)
    weights = np.zeros(shape=(size,size,C,K), dtype=np.float32)
    for c in range(C):
        weights[:, :, c, c] = filter
    init= tf.constant_initializer(value=weights, dtype=tf.float32)

    #https://github.com/tensorflow/tensorflow/issues/833
    output_shape=tf.stack([tf.shape(input)[0], tf.shape(input)[1]*factor,tf.shape(input)[2]*factor, tf.shape(input)[3]])#[N, H*factor, W*factor, C],
    w = tf.get_variable(name=name+'_weight', shape=[size, size, C, K], initializer=init, trainable=trainable)
    deconv = tf.nn.conv2d_transpose(name=name, value=input, filter=w, output_shape=output_shape, strides=[1, factor, factor, 1], padding='SAME')

    if has_bias:
        b = tf.get_variable(name=name+'_bias', shape=[K], initializer=tf.constant_initializer(0.0))
        deconv = deconv+b

    return deconv






## basic blocks ## -------------------------------
def conv2d_bn_relu(input, num_kernels=1, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='conv'):
    with tf.variable_scope(name) as scope:
        block = conv2d(input, num_kernels=num_kernels, kernel_size=kernel_size, stride=stride, padding=padding, has_bias=False)
        block = bn(block)
        block = relu(block)
    return block

def conv2d_relu(input, num_kernels=1, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='conv'):
    with tf.variable_scope(name) as scope:
        block = conv2d(input, num_kernels=num_kernels, kernel_size=kernel_size, stride=stride, padding=padding, has_bias=True)
        block = relu(block)
    return block

def linear_bn_relu(input,  num_hiddens=1, name='conv'):
    with tf.variable_scope(name) as scope:
        block = linear(input, num_hiddens=num_hiddens, has_bias=False)
        block = bn(block)
        block = relu(block)
    return block

def linear_dropout_relu(input, keep=1.0, num_hiddens=1, name='linear_dropout_relu'):
    with tf.variable_scope(name) as scope:
        block = linear(input, num_hiddens=num_hiddens, has_bias=False)
        block = dropout(block,keep=keep)
        block = relu(block)
    return block



