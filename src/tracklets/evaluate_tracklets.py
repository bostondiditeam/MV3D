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
from tracklets.parse_tracklet import *

VOLUME_METHODS = ['box', 'sphere']


def lwh_to_box(l, w, h):
    box = np.array([
        [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
        [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2],
    ])
    return box


def iou_bbox_with_yaw(vol_a, box_a, vol_b, box_b):
    """
    A simplified calculation of 3d bounding box intersection
    over union. It is assumed that the bounding box is only rotated
    around Z axis (yaw) from an axis-aligned box.

    :param vol_a, vol_b: precalculated obstacle volumes for comparison
    :param box_a, box_b: obstacle bounding boxes for comparison
    :return: iou float, intersection volume float
    """
    # height (Z) overlap
    min_h_a = np.min(box_a[2])
    max_h_a = np.max(box_a[2])
    min_h_b = np.min(box_b[2])
    max_h_b = np.max(box_b[2])
    max_of_min = np.max([min_h_a, min_h_b])
    min_of_max = np.min([max_h_a, max_h_b])
    z_intersection = np.max([0, min_of_max - max_of_min])
    if z_intersection == 0:
        return 0., 0.

    # oriented XY overlap
    xy_poly_a = Polygon(zip(*box_a[0:2, 0:4]))
    xy_poly_b = Polygon(zip(*box_b[0:2, 0:4]))
    xy_intersection = xy_poly_a.intersection(xy_poly_b).area
    if xy_intersection == 0:
        return 0., 0.

    intersection = z_intersection * xy_intersection
    union = vol_a + vol_b - intersection
    iou = intersection / union
    return iou, intersection


def iou_sphere(vol_a, sphere_a, vol_b, sphere_b):
    dist = np.linalg.norm(sphere_a[0:3] - sphere_b[0:3])
    r_a, r_b = sphere_a[3], sphere_b[3]
    if dist >= r_a + r_b:
        # spheres do not overlap in any way
        return 0., 0.
    elif dist <= abs(r_a - r_b):
        # one sphere fully inside the other (includes coincident)
        # take volume of smaller sphere as intersection
        intersection = 4/3. * np.pi * min(r_a, r_b)**3
    else:
        # spheres partially overlap, calculate intersection as per
        # http://mathworld.wolfram.com/Sphere-SphereIntersection.html
        intersection = (r_a + r_b - dist)**2
        intersection *= (dist**2 + 2*dist*(r_a + r_b) - 3*(r_a - r_b)**2)
        intersection *= np.pi / (12*dist)
    union = vol_a + vol_b - intersection
    iou = intersection / union
    return iou, intersection


class Obs(object):

    def __init__(self, tracklet_idx, object_type, size, position, rotation):
        self.tracklet_idx = tracklet_idx
        self.object_type = object_type
        self.h, self.w, self.l = size
        self.position = position
        self.yaw = rotation[2]
        self._oriented_bbox = None  # for caching

    def get_bbox(self):
        if self._oriented_bbox is None:
            bbox = lwh_to_box(self.l, self.w, self.h)
            # calc 3D bound box in capture vehicle oriented coordinates
            rot_mat = np.array([
                [np.cos(self.yaw), -np.sin(self.yaw), 0.0],
                [np.sin(self.yaw), np.cos(self.yaw), 0.0],
                [0.0, 0.0, 1.0]])
            self._oriented_bbox = np.dot(rot_mat, bbox) + np.tile(self.position, (8, 1)).T
        return self._oriented_bbox

    def get_sphere(self):
        # For a quick and dirty bounding sphere we will take 1/2 the largest
        # obstacle dim as the radius and call it close enough, for our purpose won't
        # make a noteworthy difference from a circumscribed sphere of the bbox
        r = max(self.h, self.w, self.l)/2
        # sphere passed as 4 element vector with radius as last element
        return np.append(self.position, r)

    def get_vol_sphere(self):
        r = max(self.h, self.w, self.l)/2
        return 4/3. * np.pi * r**3

    def get_vol_box(self):
        return self.h * self.w * self.l

    def get_vol(self, method='box'):
        return self.get_vol_sphere() if method == 'sphere' else self.get_vol_box()

    def intersection(self, other, method='box'):
        if method == 'sphere':
            iou_val, intersection_vol = iou_sphere(
                self.get_vol_sphere(), self.get_sphere(),
                other.get_vol_sphere(), other.get_sphere())
        else:
            iou_val, intersection_vol = iou_bbox_with_yaw(
                self.get_vol_box(), self.get_bbox(),
                other.get_vol_box(), other.get_bbox())
        return iou_val, intersection_vol

    def __repr__(self):
        return str(self.tracklet_idx) + ' ' + str(self.object_type)


def generate_obstacles(tracklets, override_size=None):
    for tracklet_idx, tracklet in enumerate(tracklets):
        frame_idx = tracklet.first_frame
        for trans, rot in zip(tracklet.trans, tracklet.rots):
            obstacle = Obs(
                tracklet_idx,
                tracklet.object_type,
                override_size if override_size is not None else tracklet.size,
                trans,
                rot)
            yield frame_idx, obstacle
            frame_idx += 1


class EvalFrame(object):

    def __init__(self):
        self.gt_obs = []
        self.pred_obs = []

    def score(self, intersection_count, union_count, pr_at_ious, method='box'):
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
                    iou_val, intersection_vol = g.intersection(p, method=method)
                    if iou_val > 0:
                        intersections.append((iou_val, intersection_vol, p_idx, g_idx))

        # Traverse calculated intersections, greedily consume intersections with largest overlap first,
        # summing volumes and marking TP/FP/FN at specific IOU thresholds as we go.
        intersections.sort(key=lambda x: x[0], reverse=True)
        for iou_val, intersection_vol, p_idx, g_idx in intersections:
            if g_idx in fn and p_idx in fp:
                fn.remove(g_idx)  # consume the ground truth
                fp.remove(p_idx)  # consume the prediction
                obs = self.gt_obs[g_idx]
                #print('IOU: ', iou_val, intersection_vol)
                intersection_count[obs.object_type] += intersection_vol
                union_count[obs.object_type] += \
                    obs.get_vol(method) + self.pred_obs[p_idx].get_vol(method) - intersection_vol
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
            union_count[obs.object_type] += obs.get_vol(method)
            for iou_threshold in pr_at_ious.keys():
                pr_at_ious[iou_threshold]['FN'] += 1

        # sum remaining false positive volume (unmatched prediction volume)
        for p_idx in fp:
            obs = self.pred_obs[p_idx]
            union_count[obs.object_type] += obs.get_vol(method)
            for iou_threshold in pr_at_ious.keys():
                pr_at_ious[iou_threshold]['FP'] += 1


def load_indices(indices_file):
    with open(indices_file, 'r') as f:
        def safe_int(x):
            try:
                return int(x.split(',')[0])
            except ValueError:
                return 0
        indices = [safe_int(line) for line in f][1:]  # skip header line
    return indices


def tracklet_score(pred_file, gt_file, filter_indices_file=None, exclude_indices_file=None, output_dir = None,
                   volume_method='sphere'):
    parser = argparse.ArgumentParser(description='Evaluate two tracklet files.')
    parser.add_argument('-g', dest='override_lwh_with_gt', action='store_true',
        help='Override predicted lwh values with value from first gt tracklet.')
    parser.add_argument('-d', dest='debug', action='store_true', help='Debug print enable')
    parser.set_defaults(debug=False)
    parser.set_defaults(override_lwh_with_gt=False)
    args = parser.parse_args()
    override_lwh_with_gt = args.override_lwh_with_gt

    if volume_method not in VOLUME_METHODS:
        sys.stderr.write('Error: Invalid volume method argument "%s". Must be one of %s\n'
                         % (volume_method, VOLUME_METHODS))
        exit(-1)

    if not os.path.exists(pred_file):
        sys.stderr.write('Error: Prediction file %s not found.\n' % pred_file)
        exit(-1)

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
            trans_noise = np.random.normal(0, 0.3, pred_tracklet.trans.shape)
            #rots_noise = np.random.normal(0, 0.1, pred_tracklet.rots.shape)
            pred_tracklets[p_idx].trans = pred_tracklet.trans + trans_noise
            #pred_tracklets[p_idx].rots = pred_tracklet.rots + rots_noise
            pred_tracklets[p_idx].size = pred_tracklet.size + np.random.normal(0, 0.2, pred_tracklet.size.shape)
        # FIXME END HACK

    num_frames = max(num_gt_frames, num_pred_frames)
    if not num_frames:
        print('Error: No frames to evaluate')
        exit(-1)

    if filter_indices_file:
        if not os.path.exists(filter_indices_file):
            sys.stderr.write('Error: Filter indices files specified but does not exist.\n')
            exit(-1)
        eval_indices = load_indices(filter_indices_file)
        print('Filter file %s loaded with %d frame indices to include for evaluation.' %
              (filter_indices_file, len(eval_indices)))
    else:
        eval_indices = list(range(num_frames))

    if exclude_indices_file:
        if not os.path.exists(exclude_indices_file):
            sys.stderr.write('Error: Exclude indices files specified but does not exist.\n')
            exit(-1)
        exclude_indices = set(load_indices(exclude_indices_file))
        eval_indices = [x for x in eval_indices if x not in exclude_indices]
        print('Exclude file %s loaded with %d frame indices to exclude from evaluation.' %
              (exclude_indices_file, len(exclude_indices)))

    eval_frames = {i: EvalFrame() for i in eval_indices}

    included_gt = 0
    excluded_gt = 0
    for frame_idx, obstacle in generate_obstacles(gt_tracklets):
        if frame_idx in eval_frames:
            eval_frames[frame_idx].gt_obs.append(obstacle)
            included_gt += 1
        else:
            excluded_gt += 1

    included_pred = 0
    excluded_pred = 0
    gt_size = gt_tracklets[0].size if override_lwh_with_gt else None
    for frame_idx, obstacle in generate_obstacles(pred_tracklets, override_size=gt_size):
        if frame_idx in eval_frames:
            eval_frames[frame_idx].pred_obs.append(obstacle)
            included_pred += 1
        else:
            excluded_pred += 1

    print('%d ground truth object instances included in evaluation, % s excluded' % (included_gt, excluded_gt))
    print('%d predicted object instances included in evaluation, % s excluded' % (included_pred, excluded_pred))

    iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    pr_at_ious = {k: Counter() for k in iou_thresholds}

    intersection_count = Counter()
    union_count = Counter()
    for frame_idx in eval_indices:
        #  calc scores
        eval_frames[frame_idx].score(
            intersection_count,
            union_count,
            pr_at_ious,
            method=volume_method)

    results_table = {'iou_per_obj': {}, 'pr_per_iou': {}}

    # FIXME determine how we want to combined IOU scores between object classes:
    # - combine volumes across all classes before ratio (vehicles dominate pedestrians)
    # - simple mean of per class ratio (current)
    # - weighted mean of per class ratio
    iou_sum = 0.0
    for k in intersection_count.keys():
        iou = intersection_count[k] / union_count[k] if union_count[k] else 0.
        results_table['iou_per_obj'][k] = float(iou)
        iou_sum += iou
    results_table['iou_per_obj']['All'] = \
        float(iou_sum / len(intersection_count)) if len(intersection_count) else 0.

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
    tracklet_score('tracklet_labels.xml', 'tracklet_labels_gt.xml', filter_indices_file=None,
                   exclude_indices_file=None, output_dir='./', volume_method='sphere')
