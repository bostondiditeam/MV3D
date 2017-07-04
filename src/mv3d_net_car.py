from net.utility.file import *
from net.blocks import *
from net.rpn_nms_op import tf_rpn_nms
from net.lib.roi_pooling_layer.roi_pooling_op import roi_pool as tf_roipooling
# from net.roipooling_op import roi_pool as tf_roipooling
from config import cfg
from net.resnet import ResnetBuilder
from keras.models import Model
import keras.applications.xception as xcep
from keras.preprocessing import image
from keras.models import Model
from keras import layers
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    SeparableConv2D,
    Conv2D,
    BatchNormalization,
    MaxPooling2D
    )
import config

top_view_rpn_name = 'top_view_rpn'
imfeature_net_name = 'image_feature'
fusion_net_name = 'fusion'


def top_feature_net(input, anchors, inds_inside, num_bases):
    """temporary net for debugging only. may not follow the paper exactly .... 
    :param input: 
    :param anchors: 
    :param inds_inside: 
    :param num_bases: 
    :return: 
            top_features, top_scores, top_probs, top_deltas, proposals, proposal_scores
    """
    stride=1.
    #with tf.variable_scope('top-preprocess') as scope:
    #    input = input

    with tf.variable_scope('top-block-1') as scope:
        block = conv2d_bn_relu(input, num_kernels=32, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=32, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        block = maxpool(block, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
        stride *=2

    with tf.variable_scope('top-block-2') as scope:
        block = conv2d_bn_relu(block, num_kernels=64, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=64, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        block = maxpool(block, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
        stride *=2

    with tf.variable_scope('top-block-3') as scope:
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='3')
        block = maxpool(block, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
        stride *=2

    with tf.variable_scope('top-block-4') as scope:
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='3')


    with tf.variable_scope('top') as scope:
        #up     = upsample2d(block, factor = 2, has_bias=True, trainable=True, name='1')
        #up     = block
        up      = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        scores  = conv2d(up, num_kernels=2*num_bases, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='score')
        probs   = tf.nn.softmax( tf.reshape(scores,[-1,2]), name='prob')
        deltas  = conv2d(up, num_kernels=4*num_bases, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='delta')

    #<todo> flip to train and test mode nms (e.g. different nms_pre_topn values): use tf.cond
    with tf.variable_scope('top-nms') as scope:    #non-max
        batch_size, img_height, img_width, img_channel = input.get_shape().as_list()
        img_scale = 1
        rois, roi_scores = tf_rpn_nms( probs, deltas, anchors, inds_inside,
                                       stride, img_width, img_height, img_scale,
                                       nms_thresh=0.7, min_size=stride, nms_pre_topn=500, nms_post_topn=100,
                                       name ='nms')

    #<todo> feature = upsample2d(block, factor = 4,  ...)
    feature = block

    print ('top: scale=%f, stride=%d'%(1./stride, stride))
    return feature, scores, probs, deltas, rois, roi_scores, stride


def top_feature_net_r(input, anchors, inds_inside, num_bases):
    """
    :param input: 
    :param anchors: 
    :param inds_inside: 
    :param num_bases: 
    :return: 
            top_features, top_scores, top_probs, top_deltas, proposals, proposal_scores
    """
    stride=1.
    #with tf.variable_scope('top-preprocess') as scope:
    #    input = input
    batch_size, img_height, img_width, img_channel = input.get_shape().as_list()

    with tf.variable_scope('feature-extract-resnet') as scope:
        print('build_resnet')
        block = ResnetBuilder.resnet_tiny_smaller_kernel(input)
        feature = block

        top_feature_stride = 4
        # resnet_input = resnet.get_layer('input_1').input
        # resnet_output = resnet.get_layer('add_7').output
        # resnet_f = Model(inputs=resnet_input, outputs=resnet_output)  # add_7
        # # print(resnet_f.summary())
        # block = resnet_f(input)
        block = upsample2d(block, factor=2, has_bias=True, trainable=True, name='upsampling')



    with tf.variable_scope('predict') as scope:
        # block = upsample2d(block, factor=4, has_bias=True, trainable=True, name='1')
        # up     = block
        # kernel_size = config.cfg.TOP_CONV_KERNEL_SIZE
        top_anchors_stride = 2
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(1, 1), stride=[1, 1, 1, 1], padding='SAME',
                               name='2')
        scores = conv2d(block, num_kernels=2 * num_bases, kernel_size=(1, 1), stride=[1, 1, 1, 1], padding='SAME',name='score')
        probs = tf.nn.softmax(tf.reshape(scores, [-1, 2]), name='prob')
        deltas = conv2d(block, num_kernels=4 * num_bases, kernel_size=(1, 1), stride=[1, 1, 1, 1], padding='SAME',name='delta')

    #<todo> flip to train and test mode nms (e.g. different nms_pre_topn values): use tf.cond
    with tf.variable_scope('NMS') as scope:    #non-max

        img_scale = 1
        rois, roi_scores = tf_rpn_nms( probs, deltas, anchors, inds_inside,
                                       stride, img_width, img_height, img_scale,
                                       nms_thresh=0.3, min_size=stride, nms_pre_topn=6000, nms_post_topn=100,
                                       name ='nms')



    print ('top: scale=%f, stride=%d'%(1./stride, stride))
    return feature, scores, probs, deltas, rois, roi_scores, top_anchors_stride,top_feature_stride



#------------------------------------------------------------------------------
def rgb_feature_net(input):

    stride=1.
    #with tf.variable_scope('rgb-preprocess') as scope:
    #   input = input-128

    with tf.variable_scope('rgb-block-1') as scope:
        block = conv2d_bn_relu(input, num_kernels=32, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=32, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        block = maxpool(block, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
        stride *=2

    with tf.variable_scope('rgb-block-2') as scope:
        block = conv2d_bn_relu(block, num_kernels=64, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=64, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        block = maxpool(block, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
        stride *=2

    with tf.variable_scope('rgb-block-3') as scope:
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='3')
        block = maxpool(block, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
        stride *=2

    with tf.variable_scope('rgb-block-4') as scope:
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='3')


    #<todo> feature = upsample2d(block, factor = 4,  ...)
    feature = block


    print ('rgb : scale=%f, stride=%d'%(1./stride, stride))
    return feature, stride

def rgb_feature_net_r(input):

    #with tf.variable_scope('rgb-preprocess') as scope:
    #   input = input-128

    batch_size, img_height, img_width, img_channel = input.get_shape().as_list()

    with tf.variable_scope('resnet-block-1') as scope:
        print('build_resnet')
        block = ResnetBuilder.resnet_tiny(input)
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(1, 1), stride=[1, 1, 1, 1], padding='SAME', name='2')
        stride = 8

    #<todo> feature = upsample2d(block, factor = 4,  ...)
    feature = block


    print ('rgb : scale=%f, stride=%d'%(1./stride, stride))
    return feature, stride


def rgb_feature_net_x(input):

    # Xception feature extractor

    batch_size, img_height, img_width, img_channel = input.get_shape().as_list()

    # with tf.variable_scope('xception_model'):
    #     base_model= xcep.Xception(include_top=False, weights=None,
    #                               input_shape=(img_height, img_width, img_channel ))
    # # print(base_model.summary())
    #
    #     base_model_input = base_model.get_layer('input_2').input
    #     base_model_output = base_model.get_layer('block12_sepconv3_bn').output
    # # print(model.summary())

    with tf.variable_scope('preprocess'):
        block = maxpool(input, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
        block = xcep.preprocess_input(block)

    with tf.variable_scope('feature_extract'):
        # keras/applications/xception.py
        print('build Xception')
        x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1')(block)
        x = BatchNormalization(name='block1_conv1_bn')(x)
        x = Activation('relu', name='block1_conv1_act')(x)
        x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
        x = BatchNormalization(name='block1_conv2_bn')(x)
        x = Activation('relu', name='block1_conv2_act')(x)

        residual = Conv2D(128, (1, 1), strides=(2, 2),
                          padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
        x = BatchNormalization(name='block2_sepconv1_bn')(x)
        x = Activation('relu', name='block2_sepconv2_act')(x)
        x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
        x = BatchNormalization(name='block2_sepconv2_bn')(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
        x = layers.add([x, residual])

        residual = Conv2D(256, (1, 1), strides=(2, 2),
                          padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = Activation('relu', name='block3_sepconv1_act')(x)
        x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
        x = BatchNormalization(name='block3_sepconv1_bn')(x)
        x = Activation('relu', name='block3_sepconv2_act')(x)
        x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
        x = BatchNormalization(name='block3_sepconv2_bn')(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
        x = layers.add([x, residual])

        residual = Conv2D(728, (1, 1), strides=(2, 2),
                          padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = Activation('relu', name='block4_sepconv1_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
        x = BatchNormalization(name='block4_sepconv1_bn')(x)
        x = Activation('relu', name='block4_sepconv2_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
        x = BatchNormalization(name='block4_sepconv2_bn')(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
        x = layers.add([x, residual])

        i = None
        for i in range(7):
            residual = x
            prefix = 'block' + str(i + 5)

            x = Activation('relu', name=prefix + '_sepconv1_act')(x)
            x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
            x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
            x = Activation('relu', name=prefix + '_sepconv2_act')(x)
            x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
            x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
            x = Activation('relu', name=prefix + '_sepconv3_act')(x)
            x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
            x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)
            x = layers.add([x, residual])

        i += 1
        prefix = 'block' + str(i + 5)
        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
        x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
        x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
        x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

        block = x
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(1, 1), stride=[1, 1, 1, 1],
                               padding='SAME', name='conv')
        stride = 32

        feature = block

    print ('rgb : scale=%f, stride=%d'%(1./stride, stride))
    return feature, stride

#------------------------------------------------------------------------------
def front_feature_net(input):

    feature = None
    return feature

# feature_list:
# ( [top_features,     top_rois,     6,6,1./stride],
#   [front_features,   front_rois,   0,0,1./stride],  #disable by 0,0
#   [rgb_features,     rgb_rois,     6,6,1./stride],)
#
def fusion_net(feature_list, num_class, out_shape=(8,3)):

    with tf.variable_scope('fuse-net') as scope:
        num = len(feature_list)
        feature_names = ['top', 'front', 'rgb']
        roi_features_list = []
        for n in range(num):
            feature = feature_list[n][0]
            roi = feature_list[n][1]
            pool_height = feature_list[n][2]
            pool_width = feature_list[n][3]
            pool_scale = feature_list[n][4]
            if (pool_height == 0 or pool_width == 0): continue

            with tf.variable_scope(feature_names[n] + '-roi-pooling'):
                roi_features, roi_idxs = tf_roipooling(feature, roi, pool_height, pool_width,
                                                       pool_scale, name='%s-roi_pooling' % feature_names[n])
            with tf.variable_scope(feature_names[n]+ '-feature-conv'):

                with tf.variable_scope('block1') as scope:
                    block = conv2d_bn_relu(roi_features, num_kernels=128, kernel_size=(3, 3),
                                           stride=[1, 1, 1, 1], padding='SAME',name=feature_names[n]+'_conv_1')
                    residual=block

                    block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3, 3), stride=[1, 1, 1, 1],
                                           padding='SAME',name=feature_names[n]+'_conv_2')+residual

                    block = avgpool(block, kernel_size=(2, 2), stride=[1, 2, 2, 1],
                                    padding='SAME', name=feature_names[n]+'_max_pool')
                with tf.variable_scope('block2') as scope:

                    block = conv2d_bn_relu(block, num_kernels=256, kernel_size=(3, 3), stride=[1, 1, 1, 1], padding='SAME',
                                           name=feature_names[n]+'_conv_1')
                    residual = block
                    block = conv2d_bn_relu(block, num_kernels=256, kernel_size=(3, 3), stride=[1, 1, 1, 1], padding='SAME',
                                           name=feature_names[n]+'_conv_2')+residual

                    block = avgpool(block, kernel_size=(2, 2), stride=[1, 2, 2, 1],
                                    padding='SAME', name=feature_names[n]+'_max_pool')
                with tf.variable_scope('block3') as scope:

                    block = conv2d_bn_relu(block, num_kernels=512, kernel_size=(3, 3), stride=[1, 1, 1, 1], padding='SAME',
                                           name=feature_names[n]+'_conv_1')
                    residual = block
                    block = conv2d_bn_relu(block, num_kernels=512, kernel_size=(3, 3), stride=[1, 1, 1, 1], padding='SAME',
                                           name=feature_names[n]+'_conv_2')+residual

                    block = avgpool(block, kernel_size=(2, 2), stride=[1, 2, 2, 1],
                                    padding='SAME', name=feature_names[n]+'_max_pool')


                roi_features = flatten(block)
                tf.summary.histogram(feature_names[n], roi_features)
                roi_features_list.append(roi_features)

        with tf.variable_scope('rois-feature-concat'):
            block = concat(roi_features_list, axis=1, name='concat')

        with tf.variable_scope('fusion-feature-fc'):
            print('\nUse fusion-feature-2fc')
            block = linear_bn_relu(block, num_hiddens=512, name='1')
            block = linear_bn_relu(block, num_hiddens=512, name='2')

    return  block


def fuse_loss(scores, deltas, rcnn_labels, rcnn_targets):

    def modified_smooth_l1( deltas, targets, sigma=3.0):
        '''
            ResultLoss = outside_weights * SmoothL1(inside_weights * (box_pred - box_targets))
            SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                          |x| - 0.5 / sigma^2,    otherwise
        '''
        sigma2 = sigma * sigma
        diffs  =  tf.subtract(deltas, targets)
        smooth_l1_signs = tf.cast(tf.less(tf.abs(diffs), 1.0 / sigma2), tf.float32)

        smooth_l1_option1 = tf.multiply(diffs, diffs) * 0.5 * sigma2
        smooth_l1_option2 = tf.abs(diffs) - 0.5 / sigma2
        smooth_l1_add = tf.multiply(smooth_l1_option1, smooth_l1_signs) + tf.multiply(smooth_l1_option2, 1-smooth_l1_signs)
        smooth_l1 = smooth_l1_add

        return smooth_l1


    _, num_class = scores.get_shape().as_list()
    dim = np.prod(deltas.get_shape().as_list()[1:])//num_class

    with tf.variable_scope('get_scores'):
        rcnn_scores   = tf.reshape(scores,[-1, num_class], name='rcnn_scores')
        rcnn_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=rcnn_scores, labels=rcnn_labels))

    with tf.variable_scope('get_detals'):
        num = tf.identity( tf.shape(deltas)[0], 'num')
        idx = tf.identity(tf.range(num)*num_class + rcnn_labels,name='idx')
        deltas1      = tf.reshape(deltas,[-1, dim],name='deltas1')
        rcnn_deltas_with_fp  = tf.gather(deltas1,  idx, name='rcnn_deltas_with_fp')  # remove ignore label
        rcnn_targets_with_fp =  tf.reshape(rcnn_targets,[-1, dim], name='rcnn_targets_with_fp')

        #remove false positive
        fp_idxs = tf.where(tf.not_equal(rcnn_labels, 0), name='fp_idxs')
        rcnn_deltas_no_fp  = tf.gather(rcnn_deltas_with_fp,  fp_idxs, name='rcnn_deltas_no_fp')
        rcnn_targets_no_fp =  tf.gather(rcnn_targets_with_fp,  fp_idxs, name='rcnn_targets_no_fp')

    with tf.variable_scope('modified_smooth_l1'):
        rcnn_smooth_l1 = modified_smooth_l1(rcnn_deltas_no_fp, rcnn_targets_no_fp, sigma=3.0)

    rcnn_reg_loss  = tf.reduce_mean(tf.reduce_sum(rcnn_smooth_l1, axis=1))

    return rcnn_cls_loss, rcnn_reg_loss


def rpn_loss(scores, deltas, inds, pos_inds, rpn_labels, rpn_targets):

    def modified_smooth_l1( box_preds, box_targets, sigma=3.0):
        '''
            ResultLoss = outside_weights * SmoothL1(inside_weights * (box_pred - box_targets))
            SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                          |x| - 0.5 / sigma^2,    otherwise
        '''
        sigma2 = sigma * sigma
        diffs  =  tf.subtract(box_preds, box_targets)
        smooth_l1_signs = tf.cast(tf.less(tf.abs(diffs), 1.0 / sigma2), tf.float32)

        smooth_l1_option1 = tf.multiply(diffs, diffs) * 0.5 * sigma2
        smooth_l1_option2 = tf.abs(diffs) - 0.  / sigma2
        smooth_l1_add = tf.multiply(smooth_l1_option1, smooth_l1_signs) + tf.multiply(smooth_l1_option2, 1-smooth_l1_signs)
        smooth_l1 = smooth_l1_add   #tf.multiply(box_weights, smooth_l1_add)  #

        return smooth_l1

    scores1      = tf.reshape(scores,[-1,2])
    rpn_scores   = tf.gather(scores1,inds)  # remove ignore label
    rpn_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_scores, labels=rpn_labels))

    deltas1       = tf.reshape(deltas,[-1,4])
    rpn_deltas    = tf.gather(deltas1, pos_inds)  # remove ignore label

    with tf.variable_scope('modified_smooth_l1'):
        rpn_smooth_l1 = modified_smooth_l1(rpn_deltas, rpn_targets, sigma=3.0)

    rpn_reg_loss  = tf.reduce_mean(tf.reduce_sum(rpn_smooth_l1, axis=1))
    return rpn_cls_loss, rpn_reg_loss

def load(top_shape, front_shape, rgb_shape, num_class, len_bases):


    out_shape = (8, 3)
    stride = 8

    top_anchors = tf.placeholder(shape=[None, 4], dtype=tf.int32, name='anchors')
    top_inside_inds = tf.placeholder(shape=[None], dtype=tf.int32, name='inside_inds')

    top_view = tf.placeholder(shape=[None, *top_shape], dtype=tf.float32, name='top')
    front_view = tf.placeholder(shape=[None, *front_shape], dtype=tf.float32, name='front')
    rgb_images = tf.placeholder(shape=[None, *rgb_shape], dtype=tf.float32, name='rgb')
    top_rois = tf.placeholder(shape=[None, 5], dtype=tf.float32, name='top_rois')  # todo: change to int32???
    front_rois = tf.placeholder(shape=[None, 5], dtype=tf.float32, name='front_rois')
    rgb_rois = tf.placeholder(shape=[None, 5], dtype=tf.float32, name='rgb_rois')

    with tf.variable_scope(top_view_rpn_name):
        # top feature
        if cfg.USE_RESNET_AS_TOP_BASENET==True:
            top_features, top_scores, top_probs, top_deltas, proposals, proposal_scores, \
            top_anchors_stride, top_feature_stride = \
                top_feature_net_r(top_view, top_anchors, top_inside_inds, len_bases)
        else:
            top_features, top_scores, top_probs, top_deltas, proposals, proposal_scores, top_feature_stride = \
                top_feature_net(top_view, top_anchors, top_inside_inds, len_bases)

        with tf.variable_scope('loss'):
            # RRN
            top_inds = tf.placeholder(shape=[None], dtype=tf.int32, name='top_ind')
            top_pos_inds = tf.placeholder(shape=[None], dtype=tf.int32, name='top_pos_ind')
            top_labels = tf.placeholder(shape=[None], dtype=tf.int32, name='top_label')
            top_targets = tf.placeholder(shape=[None, 4], dtype=tf.float32, name='top_target')
            top_cls_loss, top_reg_loss = rpn_loss(top_scores, top_deltas, top_inds, top_pos_inds,
                                                  top_labels, top_targets)


    with tf.variable_scope(imfeature_net_name) as scope:
        if cfg.RGB_BASENET =='resnet':
            rgb_features, rgb_stride= rgb_feature_net_r(rgb_images)
        elif cfg.RGB_BASENET =='xception':
            rgb_features, rgb_stride = rgb_feature_net_x(rgb_images)
        elif cfg.RGB_BASENET =='VGG':
            rgb_features, rgb_stride = rgb_feature_net(rgb_images)

    with tf.variable_scope('front_feature_net') as scope:
        front_features = front_feature_net(front_view)

    #debug roi pooling
    # with tf.variable_scope('after') as scope:
    #     roi_rgb, roi_idxs = tf_roipooling(rgb_images, rgb_rois, 100, 200, 1)
    #     tf.summary.image('roi_rgb',roi_rgb)

    with tf.variable_scope(fusion_net_name) as scope:
        if cfg.IMAGE_FUSION_DIABLE==True:
            fuse_output = fusion_net(
                    ([top_features, top_rois, 6, 6, 1. / top_feature_stride],
                     [front_features, front_rois, 0, 0, 1. / stride],  # disable by 0,0
                     [rgb_features, rgb_rois*0, 6, 6, 1. / rgb_stride],),
                    num_class, out_shape)
            print('\n\n!!!! disable image fusion\n\n')

        elif 0:
            # for test
            fuse_output = fusion_net(
                    ([top_features, top_rois*0, 6, 6, 1. / top_feature_stride],
                     [front_features, front_rois, 0, 0, 1. / stride],  # disable by 0,0
                     [rgb_features, rgb_rois, 6, 6, 1. / rgb_stride],),
                    num_class, out_shape)
            print('\n\n!!!! disable top view fusion\n\n')

        else:
            fuse_output = fusion_net(
                    ([top_features, top_rois, 6, 6, 1. / top_feature_stride],
                     [front_features, front_rois, 0, 0, 1. / stride],  # disable by 0,0
                     [rgb_features, rgb_rois, 6, 6, 1. / rgb_stride],),
                    num_class, out_shape)


        # include background class
        with tf.variable_scope('predict') as scope:
            dim = np.product([*out_shape])
            fuse_scores = linear(fuse_output, num_hiddens=num_class, name='score')
            fuse_probs = tf.nn.softmax(fuse_scores, name='prob')
            fuse_deltas = linear(fuse_output, num_hiddens=dim * num_class, name='box')
            fuse_deltas = tf.reshape(fuse_deltas, (-1, num_class, *out_shape))

        with tf.variable_scope('loss') as scope:
            fuse_labels = tf.placeholder(shape=[None], dtype=tf.int32, name='fuse_label')
            fuse_targets = tf.placeholder(shape=[None, *out_shape], dtype=tf.float32, name='fuse_target')
            fuse_cls_loss, fuse_reg_loss = fuse_loss(fuse_scores, fuse_deltas, fuse_labels, fuse_targets)


    return {
        'top_anchors':top_anchors,
        'top_inside_inds':top_inside_inds,
        'top_view':top_view,
        'front_view':front_view,
        'rgb_images':rgb_images,
        'top_rois':top_rois,
        'front_rois':front_rois,
        'rgb_rois': rgb_rois,

        'top_cls_loss': top_cls_loss,
        'top_reg_loss': top_reg_loss,
        'fuse_cls_loss': fuse_cls_loss,
        'fuse_reg_loss': fuse_reg_loss,

        'top_features': top_features,
        'top_scores': top_scores,
        'top_probs': top_probs,
        'top_deltas': top_deltas,
        'proposals': proposals,
        'proposal_scores': proposal_scores,

        'top_inds': top_inds,
        'top_pos_inds':top_pos_inds,

        'top_labels':top_labels,
        'top_targets' :top_targets,

        'fuse_labels':fuse_labels,
        'fuse_targets':fuse_targets,

        'fuse_probs':fuse_probs,
        'fuse_scores':fuse_scores,
        'fuse_deltas':fuse_deltas,

        'top_feature_stride':top_anchors_stride

    }


if __name__ == '__main__':
    import  numpy as np
    x =tf.placeholder(tf.float32,(None),name='x')
    y = tf.placeholder(tf.float32,(None),name='y')
    idxs = tf.where(tf.not_equal(x,0))
    # weights = tf.cast(tf.not_equal(x,0),tf.float32)
    y_w = tf.gather(y,idxs)
    sess = tf.Session()
    with sess.as_default():
        ret= sess.run(y_w, feed_dict={
            x:np.array([1.0,1.0,0.,2.]),
            y: np.array([1., 2., 2., 3.]),
        })
        print(ret)
