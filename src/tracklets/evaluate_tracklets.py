#! /usr/bin/python
""" Udacity Self-Driving Car Challenge Tracklet evaluation script
"""

from __future__ import print_function, division
from shapely.geometry import Polygon
from collections import Counter
import numpy as np
import argparse
import os
import sys
import yaml

from parse_tracklet import *


def lwh_to_box(l, w, h):
    box = np.array([
        [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
        # FIXME constrain height to range from ground or relative to centroid like l & w?
        [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2],
        #[0.0, 0.0, 0.0, 0.0, h, h, h, h]
    ])
    return box


def iou_3d_yaw(pred_vol, pred_box, gt_vol, gt_box):
    """
    A simplified calculation of 3d bounding box intersection
    over union. It is assumed that the bounding box is only rotated
    around Z axis (yaw) from an axis-aligned box.

    :param pred_vol: predicted obstacle volume
    :param pred_box: predicted obstacle bounding box
    :param gt_vol: ground truth obstacle volume
    :param gt_box: ground truth obstacle bounding box
    :return: iou float, intersection volume float
    """
    # height (Z) overlap
    pred_min_h = np.min(pred_box[2])
    pred_max_h = np.max(pred_box[2])
    gt_min_h = np.min(gt_box[2])
    gt_max_h = np.max(gt_box[2])
    max_of_min = np.max([pred_min_h, gt_min_h])
    min_of_max = np.min([pred_max_h, gt_max_h])
    z_intersection = np.max([0, min_of_max - max_of_min])
    if z_intersection == 0:
        return 0, 0

    # oriented XY overlap
    pred_xy_poly = Polygon(zip(*pred_box[0:2, 0:4]))
    gt_xy_poly = Polygon(zip(*gt_box[0:2, 0:4]))
    xy_intersection = gt_xy_poly.intersection(pred_xy_poly).area
    if xy_intersection == 0:
        return 0, 0

    intersection = z_intersection * xy_intersection
    union = pred_vol + gt_vol - intersection
    iou = intersection / union
    return iou, intersection


class Obs(object):

    def __init__(self, tracklet_idx, object_type, box_vol, oriented_box):
        self.tracklet_idx = tracklet_idx
        self.object_type = object_type
        self.box_vol = box_vol
        self.oriented_box = oriented_box

    def __repr__(self):
        return str(self.tracklet_idx) + ' ' + str(self.object_type)


class EvalFrame(object):

    def __init__(self):
        self.gt_obs = []
        self.pred_obs = []

    def score(self, intersection_count, union_count, pr_at_ious):
        # Perform IOU calculations between all gt and predicted obstacle pairings and greedily match those
        # with the largest IOU. Possibly other matching algorithms will work better/be more efficient.
        # NOTE: This is not a tracking oriented matching like MOTA, predicted -> gt affinity context
        # would need to be passed between frame evaluations for that.

        intersections = []
        fn = set(range(len(self.gt_obs)))  # gt idx for gt that don't have any intersecting pred
        fp = set(range(len(self.pred_obs)))  # pred idx for pred not intersecting any gt

        # Compute IOU between all obstacle gt <-> prediction pairing possibilities (of same obj type)
        for p_idx, p in enumerate(self.pred_obs):
            for g_idx, g in enumerate(self.gt_obs):
                if p.object_type == g.object_type:
                    iou_val, intersection_vol = iou_3d_yaw(
                        p.box_vol, p.oriented_box,
                        g.box_vol, g.oriented_box)
                    if iou_val > 0:
                        intersections.append((iou_val, intersection_vol, p_idx, g_idx))

        # Traverse calculated intersections, greedily consume intersections with largest overlap first,
        # summing volumes and marking TP/FP/FN at specific IOU thresholds as we go.
        intersections.sort(key=lambda x: x[0], reverse=True)
        for iou_val, intersection_vol, p_idx, g_idx in intersections:
            if g_idx in fn and p_idx in fp:
                fn.remove(g_idx)  # consume the groundtruth
                fp.remove(p_idx)  # consume the prediction
                obs = self.gt_obs[g_idx]
                #print('IOU: ', iou_val, intersection_vol)
                intersection_count[obs.object_type] += intersection_vol
                union_count[obs.object_type] += obs.box_vol + self.pred_obs[p_idx].box_vol - intersection_vol
                for iou_threshold in pr_at_ious.keys():
                    if iou_val > iou_threshold:
                        pr_at_ious[iou_threshold]['TP'] += 1
                    else:
                        # It's already determined at this point that this is the highest IOU score
                        # there is for this match and another match for these two bbox won't be considered.
                        # Thus both the gt and prediction should be considered unmatched.
                        pr_at_ious[iou_threshold]['FP'] += 1
                        pr_at_ious[iou_threshold]['FN'] += 1

        # sum remaining false negative volume (unmatched ground truth box volume)
        for g_idx in fn:
            obs = self.gt_obs[g_idx]
            union_count[obs.object_type] += obs.box_vol
            for iou_threshold in pr_at_ious.keys():
                pr_at_ious[iou_threshold]['FN'] += 1

        # sum remaining false positive volume (unmatched prediction volume)
        for p_idx in fp:
            obs = self.pred_obs[p_idx]
            union_count[obs.object_type] += obs.box_vol
            for iou_threshold in pr_at_ious.keys():
                pr_at_ious[iou_threshold]['FP'] += 1


def generate_boxes(tracklets):
    for tracklet_idx, tracklet in enumerate(tracklets):
        h, w, l = tracklet.size
        box_vol = h * w * l
        tracklet_box = lwh_to_box(l, w, h)
        # print(tracklet_box)
        frame_idx = tracklet.first_frame
        for trans, rot in zip(tracklet.trans, tracklet.rots):
            # calc 3D bound box in capture vehicle oriented coordinates
            yaw = rot[2]  # rotations besides yaw should be 0
            rot_mat = np.array([
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0]])
            oriented_box = np.dot(rot_mat, tracklet_box) + np.tile(trans, (8, 1)).T
            yield frame_idx, tracklet_idx, tracklet.object_type, box_vol, oriented_box
            frame_idx += 1


def main():
    parser = argparse.ArgumentParser(description='Evaluate two tracklet files.')
    parser.add_argument('prediction', type=str, nargs='?', default='tracklet_labels.xml',
        help='Predicted tracklet label filename')
    parser.add_argument('groundtruth', type=str, nargs='?', default='tracklet_labels_gt.xml',
        help='Groundtruth tracklet label filename')
    parser.add_argument('-o', '--outdir', type=str, nargs='?', default=None,
        help='Output folder')
    parser.add_argument('-d', dest='debug', action='store_true', help='Debug print enable')
    parser.set_defaults(debug=False)
    args = parser.parse_args()
    output_dir = args.outdir

    pred_file = args.prediction
    if not os.path.exists(pred_file):
        sys.stderr.write('Error: Prediction file %s not found.\n' % pred_file)
        exit(-1)

    gt_file = args.groundtruth
    if not os.path.exists(gt_file):
        sys.stderr.write('Error: Ground-truth file %s not found.\n' % gt_file)
        exit(-1)

    pred_tracklets = parse_xml(pred_file)
    if not pred_tracklets:
        sys.stderr.write('Error: No Tracklets parsed for predictions.\n')
        exit(-1)

    gt_tracklets = parse_xml(gt_file)
    if not gt_tracklets:
        sys.stderr.write('Error: No Tracklets parsed for ground truth.\n')
        exit(-1)

    num_gt_frames = 0
    for gt_tracklet in gt_tracklets:
        num_gt_frames = max(num_gt_frames, gt_tracklet.first_frame + gt_tracklet.num_frames)

    num_pred_frames = 0
    for p_idx, pred_tracklet in enumerate(pred_tracklets):
        num_pred_frames = max(num_pred_frames, pred_tracklet.first_frame + pred_tracklet.num_frames)
        # FIXME START TEST HACK
        if False:
            blah = np.random.normal(0, 0.3, pred_tracklet.trans.shape)
            pred_tracklets[p_idx].trans = pred_tracklets[p_idx].trans + blah
        # FIXME END HACK

    num_frames = max(num_gt_frames, num_pred_frames)
    if not num_frames:
        print('Error: No frames to evaluate')
        exit(-1)
    eval_frames = [EvalFrame() for _ in range(num_frames)]

    for frame_idx, tracklet_idx, object_type, box_vol, oriented_box in generate_boxes(gt_tracklets):
        eval_frames[frame_idx].gt_obs.append(
            Obs(tracklet_idx, object_type, box_vol, oriented_box))

    for frame_idx, tracklet_idx, object_type, box_vol, oriented_box in generate_boxes(pred_tracklets):
        eval_frames[frame_idx].pred_obs.append(
            Obs(tracklet_idx, object_type, box_vol, oriented_box))

    iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    pr_at_ious = {k: Counter() for k in iou_thresholds}

    intersection_count = Counter()
    union_count = Counter()
    for frame_idx in range(num_frames):
        #  calc scores
        eval_frames[frame_idx].score(
            intersection_count,
            union_count,
            pr_at_ious)

    results_table = {'iou_per_obj': {}, 'pr_per_iou': {}}

    # FIXME determine how we want to combined IOU scores between object classes:
    # - combine volumes across all classes before ratio (vehicles dominate pedestrians)
    # - simple mean of per class ratio (current)
    # - weighted mean of per class ratio
    iou_sum = 0.0
    for k in intersection_count.keys():
        iou = intersection_count[k] / union_count[k]
        results_table['iou_per_obj'][k] = float(iou)
        iou_sum += iou
    results_table['iou_per_obj']['All'] = float(iou_sum / len(intersection_count))

    # FIXME add support for per class P/R scores?
    # NOTE P/R scores need further analysis given their use with the greedy pred - gt matching
    for k, v in pr_at_ious.items():
        p = v['TP'] / (v['TP'] + v['FP']) if v['TP'] else 0.0
        r = v['TP'] / (v['TP'] + v['FN']) if v['TP'] else 0.0
        # f1 = 2 * (p * r) / (p + r) if (p + r != 0) else 0.0
        results_table['pr_per_iou'][k] = {'precision': p, 'recall': r}

    print('\nResults')
    print(yaml.safe_dump(results_table, default_flow_style=False, explicit_start=True))

    if output_dir is not None:
        with open(os.path.join(output_dir, 'iou_per_obj.csv'), 'w') as f:
            f.write('object_type,iou\n')
            [f.write('{0},{1}\n'.format(k, v))
             for k, v in sorted(results_table['iou_per_obj'].items(), key=lambda x: x[0])]
        with open(os.path.join(output_dir, 'pr_per_iou.csv'), 'w') as f:
            f.write('iou_threshold,p,r\n')
            [f.write('{0},{1},{2}\n'.format(k, v['precision'], v['recall']))
             for k, v in sorted(results_table['pr_per_iou'].items(), key=lambda x: x[0])]


if __name__ == '__main__':
    main()
