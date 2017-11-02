from net.configuration import CFG
from net.lib.utils.bbox import bbox_overlaps ,box_vote
from net.lib.nms.cpu_nms import cpu_nms as nms
import numpy as np

#     roi  : i, x1,y1,x2,y2  i=image_index  
#     box : x1,y1,x2,y2,


def clip_boxes(boxes, width, height):
    ''' Clip process to image boundaries. '''

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], width - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], height - 1), 0)
    # x2 < width
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], width - 1), 0)
    # y2 < height
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], height - 1), 0)
    return boxes


# et_boxes = estimated 
# gt_boxes = ground truth 

def box_transform(et_boxes, gt_boxes):
    et_ws  = et_boxes[:, 2] - et_boxes[:, 0] + 1.0
    et_hs  = et_boxes[:, 3] - et_boxes[:, 1] + 1.0
    et_cxs = et_boxes[:, 0] + 0.5 * et_ws
    et_cys = et_boxes[:, 1] + 0.5 * et_hs
     
    gt_ws  = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
    gt_hs  = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0
    gt_cxs = gt_boxes[:, 0] + 0.5 * gt_ws
    gt_cys = gt_boxes[:, 1] + 0.5 * gt_hs
     
    dxs = (gt_cxs - et_cxs) / et_ws
    dys = (gt_cys - et_cys) / et_hs
    dws = np.log(gt_ws / et_ws)
    dhs = np.log(gt_hs / et_hs)

    deltas = np.vstack((dxs, dys, dws, dhs)).transpose()
    return deltas



def box_transform_inv(et_boxes, deltas):

    num = len(et_boxes)
    boxes = np.zeros((num,4), dtype=np.float32)
    if num == 0: return boxes

    et_ws  = et_boxes[:, 2] - et_boxes[:, 0] + 1.0
    et_hs  = et_boxes[:, 3] - et_boxes[:, 1] + 1.0
    et_cxs = et_boxes[:, 0] + 0.5 * et_ws
    et_cys = et_boxes[:, 1] + 0.5 * et_hs

    et_ws  = et_ws [:, np.newaxis]
    et_hs  = et_hs [:, np.newaxis]
    et_cxs = et_cxs[:, np.newaxis]
    et_cys = et_cys[:, np.newaxis]

    dxs = deltas[:, 0::4]
    dys = deltas[:, 1::4]
    dws = deltas[:, 2::4]
    dhs = deltas[:, 3::4]

    cxs = dxs * et_ws + et_cxs
    cys = dys * et_hs + et_cys
    # print('value for et_ws: ', et_ws)
    # print('value for dws: ', dws)
    ws  = np.exp(dws) * et_ws
    hs  = np.exp(dhs) * et_hs
    # print('ws is here: ', ws)
    # print('hs is here: ', hs)

    boxes[:, 0::4] = cxs - 0.5 * ws  # x1, y1,x2,y2
    boxes[:, 1::4] = cys - 0.5 * hs
    boxes[:, 2::4] = cxs + 0.5 * ws
    boxes[:, 3::4] = cys + 0.5 * hs

    return boxes

# nms  ###################################################################
def non_max_suppress(boxes, scores, num_classes,
                     nms_after_thesh=CFG.TEST.RCNN_NMS_AFTER, 
                     nms_before_score_thesh=0.05, 
                     is_box_vote=False,
                     max_per_image=100 ):

   
    # nms_before_thesh = 0.05 ##0.05   # set low number to make roc curve.
                                       # else set high number for faster speed at inference
 
    #non-max suppression 
    nms_boxes = [[]for _ in range(num_classes)]
    for j in range(1, num_classes): #skip background
        inds = np.where(scores[:, j] > nms_before_score_thesh)[0]
         
        cls_scores = scores[inds, j]
        cls_boxes  = boxes [inds, j*4:(j+1)*4]
        cls_dets   = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False) 

        # is_box_vote=0
        if len(inds)>0:
            keep = nms(cls_dets, nms_after_thesh) 
            dets_NMSed = cls_dets[keep, :] 
            if is_box_vote:
                cls_dets = box_vote(dets_NMSed, cls_dets)
            else:
                cls_dets = dets_NMSed 

        nms_boxes[j] = cls_dets
      

    ##Limit to MAX_PER_IMAGE detections over all classes
    if max_per_image > 0:
        image_scores = np.hstack([nms_boxes[j][:, -1] for j in range(1, num_classes)])
        if len(image_scores) > max_per_image:
            image_thresh = np.sort(image_scores)[-max_per_image]
            for j in range(1, num_classes):
                keep = np.where(nms_boxes[j][:, -1] >= image_thresh)[0]
                nms_boxes[j] = nms_boxes[j][keep, :]

    return nms_boxes  
