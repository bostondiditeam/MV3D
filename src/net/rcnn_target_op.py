from net.configuration import *
from net.processing.boxes import *
from net.processing.boxes3d import *
from net.utility.draw import *



# gt_boxes    : (x1,y1,  x2,y2  label)  #projected 2d
# gt_boxes_3d : (x1,y1,z1,  x2,y2,z2,  ....    x8,y8,z8,  label)


def rcnn_target(rois, gt_labels, gt_boxes, gt_boxes3d):

    # Include "ground-truth" in the set of candidate rois
    rois = rois.reshape(-1,5)  # Proposal (i, x1, y1, x2, y2) coming from RPN
    num           = len(gt_boxes)
    zeros         = np.zeros((num, 1), dtype=np.float32)
    extended_rois = np.vstack((rois, np.hstack((zeros, gt_boxes))))
    assert np.all(extended_rois[:, 0] == 0), 'Only single image batches are supported'


    rois_per_image    = CFG.TRAIN.RCNN_BATCH_SIZE
    fg_rois_per_image = np.round(CFG.TRAIN.RCNN_FG_FRACTION * rois_per_image)

    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(extended_rois[:,1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float)
    )
    max_overlaps  = overlaps.max(axis=1)
    gt_assignment = overlaps.argmax(axis=1)
    labels        = gt_labels[gt_assignment]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= CFG.TRAIN.RCNN_FG_THRESH_LO)[0]
    fg_rois_per_this_image = int(min(fg_rois_per_image, fg_inds.size))
    if fg_inds.size > 0:
        fg_inds = np.random.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < CFG.TRAIN.RCNN_BG_THRESH_HI) &
                       (max_overlaps >= CFG.TRAIN.RCNN_BG_THRESH_LO))[0]
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = int(min(bg_rois_per_this_image, bg_inds.size))
    if bg_inds.size > 0:
        bg_inds = np.random.choice(bg_inds, size=bg_rois_per_this_image, replace=False)


    # The indices that we're selecting (both fg and bg)
    keep   = np.append(fg_inds, bg_inds)
    rois   = extended_rois[keep]
    labels = labels[keep]                # Select sampled values from various arrays:
    labels[fg_rois_per_this_image:] = 0  # Clamp la bels for the background RoIs to 0


    gt_boxes3d = gt_boxes3d[gt_assignment[keep]]
    et_boxes=rois[:,1:5]
    if gt_boxes3d.shape[1:]==gt_boxes.shape[1:]:
        #normal image faster-rcnn .... for debug
        targets = box_transform(et_boxes, gt_boxes3d)
        #targets = targets / np.array(CFG.TRAIN.RCNN_box_NORMALIZE_STDS)  # this is for each box
    else:
        et_boxes3d = top_box_to_box3d(et_boxes)
        targets = box3d_transform(et_boxes3d, gt_boxes3d)
        #exit(0)

    return rois, labels, targets


# gt_boxes    : (x1,y1,  x2,y2  label)  #projected 2d
# gt_boxes_3d : (x1,y1,z1,  x2,y2,z2,  ....    x8,y8,z8,  label)


def fusion_target(rois, gt_labels, gt_boxes, gt_boxes3d):

    # Include "ground-truth" in the set of candidate rois
    rois = rois.reshape(-1,5)  # Proposal (i, x1, y1, x2, y2) coming from RPN
    num           = len(gt_boxes)
    zeros         = np.zeros((num, 1), dtype=np.float32)
    extended_rois = np.vstack((rois, np.hstack((zeros, gt_boxes))))
    assert np.all(extended_rois[:, 0] == 0), 'Only single image batches are supported'


    rois_per_image    = CFG.TRAIN.RCNN_BATCH_SIZE
    fg_rois_per_image = np.round(CFG.TRAIN.RCNN_FG_FRACTION * rois_per_image)

    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(extended_rois[:,1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float)
    )
    max_overlaps  = overlaps.max(axis=1)
    gt_assignment = overlaps.argmax(axis=1)
    labels        = gt_labels[gt_assignment]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= CFG.TRAIN.RCNN_FG_THRESH_LO)[0]
    # fg_rois_per_this_image = int(min(10, fg_inds.size))
    # if fg_inds.size > 0:
    #     fg_inds = np.random.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select false positive
    fp_inds = np.where((max_overlaps < 0.01))[0]
    # fp_rois_per_this_image = int(min(10, fp_inds.size))
    # if fp_inds.size > 0:
    #     fp_inds = np.random.choice(fp_inds, size=fp_rois_per_this_image, replace=False)


    # The indices that we're selecting (both fg and bg)
    keep   = np.append(fg_inds, fp_inds)
    rois   = extended_rois[keep]
    labels = labels[keep]                # Select sampled values from various arrays:
    labels[fg_inds.size:] = 0  # Clamp la bels for the background RoIs to 0


    gt_boxes3d = gt_boxes3d[gt_assignment[keep]]
    et_boxes=rois[:,1:5]

    et_boxes3d = top_box_to_box3d(et_boxes)
    targets = box3d_transform(et_boxes3d, gt_boxes3d)
    targets[np.where(labels == 0), :, :] = 0

    return rois, labels, targets

def proprosal_to_top_rois(rois):

    # Include "ground-truth" in the set of candidate rois
    rois = rois.reshape(-1,5)  # Proposal (i, x1, y1, x2, y2) coming from RPN
    num           = len(rois)
    zeros         = np.zeros((num, 1), dtype=np.float32)
    extended_rois = np.vstack((rois, np.hstack((zeros, gt_boxes))))
    assert np.all(extended_rois[:, 0] == 0), 'Only single image batches are supported'


    rois_per_image    = CFG.TRAIN.RCNN_BATCH_SIZE
    fg_rois_per_image = np.round(CFG.TRAIN.RCNN_FG_FRACTION * rois_per_image)

    # # overlaps: (rois x gt_boxes)
    # overlaps = box_overlaps(
    #     np.ascontiguousarray(extended_rois[:,1:5], dtype=np.float),
    #     np.ascontiguousarray(gt_boxes, dtype=np.float)
    # )
    # max_overlaps  = overlaps.max(axis=1)
    # gt_assignment = overlaps.argmax(axis=1)
    # labels        = gt_labels[gt_assignment]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= CFG.TRAIN.RCNN_FG_THRESH_LO)[0]
    fg_rois_per_this_image = int(min(fg_rois_per_image, fg_inds.size))
    if fg_inds.size > 0:
        fg_inds = np.random.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < CFG.TRAIN.RCNN_BG_THRESH_HI) &
                       (max_overlaps >= CFG.TRAIN.RCNN_BG_THRESH_LO))[0]
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = int(min(bg_rois_per_this_image, bg_inds.size))
    if bg_inds.size > 0:
        bg_inds = np.random.choice(bg_inds, size=bg_rois_per_this_image, replace=False)


    # The indices that we're selecting (both fg and bg)
    keep   = np.append(fg_inds, bg_inds)
    rois   = extended_rois[keep]
    # labels = labels[keep]                # Select sampled values from various arrays:
    # labels[fg_rois_per_this_image:] = 0  # Clamp la bels for the background RoIs to 0
    #
    #
    # gt_boxes3d = gt_boxes3d[gt_assignment[keep]]
    # et_boxes=rois[:,1:5]
    # if gt_boxes3d.shape[1:]==gt_boxes.shape[1:]:
    #     #normal image faster-rcnn .... for debug
    #     targets = box_transform(et_boxes, gt_boxes3d)
    #     #targets = targets / np.array(CFG.TRAIN.RCNN_box_NORMALIZE_STDS)  # this is for each box
    # else:
    #     et_boxes3d = top_box_to_box3d(et_boxes)
    #     targets = box3d_transform(et_boxes3d, gt_boxes3d)
    #     #exit(0)

    return rois


def draw_rcnn_labels(image, rois,  labels, darker=0.7):
    is_print=0

    ## draw +ve/-ve labels ......
    boxes = rois[:,1:5]
    labels = labels.reshape(-1)

    fg_label_inds = np.where(labels != 0)[0]
    bg_label_inds = np.where(labels == 0)[0]
    num_pos_label = len(fg_label_inds)
    num_neg_label = len(bg_label_inds)
    if is_print: print ('rcnn label : num_pos=%d num_neg=%d,  all = %d'  %(num_pos_label, num_neg_label,num_pos_label+num_neg_label))

    img_label = image.copy()*darker
    if 1:
        for i in bg_label_inds:
            a = boxes[i]
            cv2.rectangle(img_label,(a[0], a[1]), (a[2], a[3]), (32,32,32), 1)
            cv2.circle(img_label,(a[0], a[1]),2, (32,32,32), -1)

    for i in fg_label_inds:
        a = boxes[i]
        cv2.rectangle(img_label,(a[0], a[1]), (a[2], a[3]), (0,0,255), 1)
        cv2.circle(img_label,(a[0], a[1]),2, (0,0,255), -1)

    return img_label

def draw_rcnn_targets(image, rois, labels,  targets, darker=0.7):
    is_print=0

    #draw +ve targets ......
    boxes = rois[:,1:5]

    fg_target_inds = np.where(labels != 0)[0]
    num_pos_target = len(fg_target_inds)
    if is_print: print ('rcnn target : num_pos=%d'  %(num_pos_target))

    img_target = image.copy()*darker
    for n,i in enumerate(fg_target_inds):
        a = boxes[i]
        cv2.rectangle(img_target,(a[0], a[1]), (a[2], a[3]), (255,0,255), 1)

        if targets.shape[1:]==(4,):
            t = targets[n]
            b = box_transform_inv(a.reshape(1,4), t.reshape(1,4))
            b = b.reshape(4)
            cv2.rectangle(img_target,(b[0], b[1]), (b[2], b[3]), (255,255,255), 1)

        if targets.shape[1:]==(8,3):
            t = targets[n]
            a3d = top_box_to_box3d(a.reshape(1,4))
            b3d = box3d_transform_inv(a3d, t.reshape(1,8,3))
            #b3d = b3d.reshape(1,8,3)
            img_target = draw_box3d_on_top(img_target, b3d)

    return img_target





