from net.common import *



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
    rpn_smooth_l1 = modified_smooth_l1(rpn_deltas, rpn_targets, sigma=3.0)
    rpn_reg_loss  = tf.reduce_mean(tf.reduce_sum(rpn_smooth_l1, axis=1))

    return rpn_cls_loss, rpn_reg_loss