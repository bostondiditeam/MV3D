from net.utility.file import *
from net.blocks import *
from net.rpn_nms_op import tf_rpn_nms
from net.roipooling_op import roi_pool as tf_roipooling
import net.rpn_loss_op as rpnloss
import net.rcnn_loss_op as rcnnloss


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
        block = conv2d_bn_relu(block, num_kernels=32, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='3')
        block = maxpool(block, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
        stride *=2

    with tf.variable_scope('top-block-2') as scope:
        block = conv2d_bn_relu(block, num_kernels=64, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=64, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        block = conv2d_bn_relu(block, num_kernels=64, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='3')
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
    return feature, scores, probs, deltas, rois, roi_scores



#------------------------------------------------------------------------------
def rgb_feature_net(input):

    stride=1.
    #with tf.variable_scope('rgb-preprocess') as scope:
    #   input = input-128

    with tf.variable_scope('rgb-block-1') as scope:
        block = conv2d_bn_relu(input, num_kernels=32, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=32, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        block = conv2d_bn_relu(block, num_kernels=32, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='3')
        block = maxpool(block, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
        stride *=2

    with tf.variable_scope('rgb-block-2') as scope:
        block = conv2d_bn_relu(block, num_kernels=64, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=64, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        block = conv2d_bn_relu(block, num_kernels=64, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='3')
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
    return feature

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

    num=len(feature_list)

    input = None
    with tf.variable_scope('fuse-input') as scope:
        for n in range(num):
            feature     = feature_list[n][0]
            roi         = feature_list[n][1]
            pool_height = feature_list[n][2]
            pool_width  = feature_list[n][3]
            pool_scale  = feature_list[n][4]
            if (pool_height==0 or pool_width==0): continue

            roi_features,  roi_idxs = tf_roipooling(feature,roi, pool_height, pool_width, pool_scale, name='%d/pool'%n)
            roi_features = flatten(roi_features)
            if input is None:
                input = roi_features
            else:
                input = concat([input,roi_features], axis=1, name='%d/cat'%n)

    with tf.variable_scope('fuse-block-1') as scope:
        block = linear_bn_relu(input, num_hiddens=512, name='1')
        block = linear_bn_relu(block, num_hiddens=512, name='2')
        block = linear_bn_relu(block, num_hiddens=512, name='3')
        block = linear_bn_relu(block, num_hiddens=512, name='4')

    #include background class
    with tf.variable_scope('fuse') as scope:
        dim = np.product([*out_shape])
        scores  = linear(block, num_hiddens=num_class,     name='score')
        probs   = tf.nn.softmax (scores, name='prob')
        deltas  = linear(block, num_hiddens=dim*num_class, name='box')
        deltas  = tf.reshape(deltas,(-1,num_class,*out_shape))

    return  scores, probs, deltas

import tensorflow as tf
import numpy as np

def rcnn_predict(scores, deltas):

    _, num_class = scores.get_shape().as_list()
    dim = np.prod(deltas.get_shape().as_list()[1:])//num_class

    rcnn_scores   = tf.reshape(scores,[-1, num_class])
    class_probability= tf.nn.softmax(rcnn_scores)

    num = tf.shape(deltas)[0]
    idx = tf.range(num)*num_class + class_probability
    deltas1      = tf.reshape(deltas,[-1, dim])
    rcnn_deltas  = tf.gather(deltas1,  idx)  # remove ignore label

    return class_probability, rcnn_deltas


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

    # top feature

    top_features, top_scores, top_probs, top_deltas, proposals, proposal_scores = \
        top_feature_net(top_view, top_anchors, top_inside_inds, len_bases)

    # RRN
    top_inds = tf.placeholder(shape=[None], dtype=tf.int32, name='top_ind')
    top_pos_inds = tf.placeholder(shape=[None], dtype=tf.int32, name='top_pos_ind')
    top_labels = tf.placeholder(shape=[None], dtype=tf.int32, name='top_label')
    top_targets = tf.placeholder(shape=[None, 4], dtype=tf.float32, name='top_target')
    top_cls_loss, top_reg_loss = rpnloss.rpn_loss(top_scores, top_deltas, top_inds, top_pos_inds, top_labels, top_targets)



    front_features = front_feature_net(front_view)
    rgb_features = rgb_feature_net(rgb_images)

    # fusion todo: Add NMS
    fuse_scores, fuse_probs, fuse_deltas = \
        fusion_net(
            ([top_features, top_rois, 6, 6, 1. / stride],
             [front_features, front_rois, 0, 0, 1. / stride],  # disable by 0,0
             [rgb_features, rgb_rois, 6, 6, 1. / stride],),
            num_class, out_shape)

    fuse_labels = tf.placeholder(shape=[None], dtype=tf.int32, name='fuse_label')
    fuse_targets = tf.placeholder(shape=[None, *out_shape], dtype=tf.float32, name='fuse_target')
    fuse_cls_loss, fuse_reg_loss = rcnnloss.rcnn_loss(fuse_scores, fuse_deltas, fuse_labels, fuse_targets)

    # predict
    # predict_scores, predict_deltas=rcnn_predict(fuse_scores,fuse_deltas)

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
        'fuse_deltas':fuse_deltas

        # 'predict_scores':predict_scores,
        # 'predict_deltas':predict_deltas
    }


# # main ###########################################################################
# # to start in tensorboard:
# #    /opt/anaconda3/bin
# #    ./python tensorboard --logdir /root/share/out/didi/tf
# #     http://http://localhost:6006/
#
# if __name__ == '__main__':
#     print( '%s: calling main function ... ' % os.path.basename(__file__))
#     out_dir='/root/share/out/didi/tf'
#     log = Logger('/root/share/out/udacity/00/xxx_log.txt', mode='a')  # log file
#
#
#     # draw graph to check connections
#     with tf.Session()  as sess:
#         tf.global_variables_initializer().run(feed_dict={IS_TRAIN_PHASE:True})
#         summary_writer = tf.summary.FileWriter(out_dir, sess.graph)
#
#         print_macs_to_file(log)
#     print ('sucess!')