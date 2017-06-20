from net.utility.draw import *

from net.processing.boxes import *
from net.rpn_target_op import make_bases, make_anchors
import tensorflow as tf



#customised tf ops:
#  http://stackoverflow.com/questions/39921607/tensorflow-how-to-make-a-custom-activation-function-with-only-python
#  http://stackoverflow.com/questions/38347485/how-to-create-an-op-like-conv-ops-in-tensorflow
#
#


#nms for rpn (region proposal net) ----------------------------
def draw_rpn_deltal_apply(image, probs, deltas, anchors, inside_inds, threshold=0.75, darker=0.7):

    ## yellow (thick): box regression results
    ## red: box classification results

    img_rpn = image.copy()*darker
    probs = probs.reshape(-1,2)
    probs = probs[:,1]

    deltas = deltas.reshape(-1,4)
    inds = np.argsort(probs)[::-1]       #sort ascend #[::-1]

    num_anchors = len(anchors)
    insides = np.zeros((num_anchors),dtype=np.int32)
    insides[inside_inds]=1
    for j in range(100):
        i = inds[j]
        if insides[i]==0:
            continue

        a = anchors[i]
        t = deltas[i]
        b = box_transform_inv(a.reshape(1,4), t.reshape(1,4))
        #b = clip_boxes(b,img_width,img_height)
        b = b.reshape(-1)
        s = probs[i]
        if s<threshold:
            continue

        v = s*255
        cv2.rectangle(img_rpn,(a[0], a[1]), (a[2], a[3]), (0,0,v), 1)
        cv2.rectangle(img_rpn,(b[0], b[1]), (b[2], b[3]), (0,v,v), 1)

    return img_rpn


def draw_rpn_proposal(image, rois, roi_scores, draw_num=100):
    img_rpn_nms = image.copy()

    scores = roi_scores
    inds = np.argsort(scores)       #sort ascend #[::-1]
    # num = draw_num if draw_num<len(inds) else len(inds)
    num=len(inds)
    for n in range(0, num):
        i   = inds[n]
        box = rois[i,1:5].astype(np.int)
        v=254*(roi_scores[i])+1
        color = (0,v,v)
        cv2.rectangle(img_rpn_nms,(box[0], box[1]), (box[2], box[3]), color, 1)

    return img_rpn_nms


def filter_boxes(boxes, min_size):
    '''Remove all boxes with any side smaller than min_size.'''
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep



def rpn_nms_generator(
    stride, img_width, img_height, img_scale=1,
    nms_thresh   =CFG.TRAIN.RPN_NMS_THRESHOLD,
    min_size     =CFG.TRAIN.RPN_NMS_MIN_SIZE,
    nms_pre_topn =CFG.TRAIN.RPN_NMS_PRE_TOPN,
    nms_post_topn=CFG.TRAIN.RPN_NMS_POST_TOPN):


    def rpn_nms(scores, deltas, anchors, inside_inds):
        # 1. Generate proposals from box deltas and shifted anchors
        #batch_size, H, W, C = scores.shape
        #assert(C==2)
        scores = scores.reshape((-1, 2,1))
        scores = scores[:,1,:]
        deltas = deltas.reshape((-1, 4))

        scores = scores[inside_inds]
        deltas = deltas[inside_inds]
        anchors = anchors[inside_inds]

        # Convert anchors into proposals via box transformations
        proposals = box_transform_inv(anchors, deltas)

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, img_width, img_height)

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep      = filter_boxes(proposals, min_size*img_scale)
        proposals = proposals[keep, :]
        scores    = scores[keep]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if nms_pre_topn > 0:
            order = order[:nms_pre_topn]
            proposals = proposals[order, :]
            scores = scores[order]

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals
        keep = nms(np.hstack((proposals, scores)), nms_thresh)
        if nms_post_topn > 0:
            keep = keep[:nms_post_topn]
            proposals = proposals[keep, :]
            scores = scores[keep]

        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        roi_scores=scores.squeeze()

        num_proposals = len(proposals)
        batch_inds = np.zeros((num_proposals, 1), dtype=np.float32)
        rois = np.hstack((batch_inds, proposals))

        return rois, roi_scores
    return rpn_nms



def tf_rpn_nms(
    scores, deltas, anchors, inside_inds,
    stride, img_width,img_height,img_scale,
    nms_thresh, min_size, nms_pre_topn, nms_post_topn,
    name='rpn_mns'):

    #<todo>
    #assert batch_size == 1, 'Only single image batches are supported'

    rpn_nms = rpn_nms_generator(stride, img_width, img_height, img_scale, nms_thresh, min_size, nms_pre_topn, nms_post_topn)
    return  \
        tf.py_func(
            rpn_nms,
            [scores, deltas, anchors, inside_inds],
            [tf.float32, tf.float32],
        name = name)

## main ##----------------------------------------------------------------
if __name__ == '__main__':
    print  ('\"%s\" running main function ...' % os.path.basename(__file__))

    bases = make_bases(
            base_size = 16,
            #ratios=[0.5, 1, 2],
            #scales=2**np.arange(3, 6))
            ratios=[0.5, 1, 2],
            scales=2**np.arange(3, 4 ))
    num_bases = len(bases)
    stride = 16
    image_shape   = (480,640,3)
    feature_shape = (480//stride,640//stride,64)
    anchors, inside_inds =  make_anchors(bases, stride, image_shape[0:2], feature_shape[0:2])

    img_height,img_width,_ = image_shape
    H,W,_ = feature_shape
    scores = np.random.uniform(0,255,size=(1, H,W,num_bases*2)).astype(np.float32)
    deltas = np.random.uniform(0,1,  size=(1, H,W,num_bases*4)).astype(np.float32)

    rpn_nms = rpn_nms_generator(stride, img_width, img_height)
    rois, roi_scores = rpn_nms(scores, deltas, anchors, inside_inds)

    print  ('sucess!')