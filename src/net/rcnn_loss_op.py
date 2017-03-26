from net.common import *

def rcnn_loss(scores, deltas, rcnn_labels, rcnn_targets):

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

    rcnn_scores   = tf.reshape(scores,[-1, num_class])
    rcnn_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rcnn_scores, labels=rcnn_labels))

    num = tf.shape(deltas)[0]
    idx = tf.range(num)*num_class + rcnn_labels
    deltas1      = tf.reshape(deltas,[-1, dim])
    rcnn_deltas  = tf.gather(deltas1,  idx)  # remove ignore label
    rcnn_targets =  tf.reshape(rcnn_targets,[-1, dim])

    rcnn_smooth_l1 = modified_smooth_l1(rcnn_deltas, rcnn_targets, sigma=3.0)
    rcnn_reg_loss  = tf.reduce_mean(tf.reduce_sum(rcnn_smooth_l1, axis=1))

    return rcnn_cls_loss, rcnn_reg_loss