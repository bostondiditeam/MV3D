#!/usr/bin/env python

import rosbag, rospy

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from filterpy.kalman import KalmanFilter, MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.common import Q_discrete_white_noise
from parse_tracklet import parse_xml
# from sync import generate_frame_map
from generate_tracklet import Tracklet, TrackletCollection
from shapely.geometry import Polygon
import pandas as pd

import random, math
from bisect import bisect_left
import numpy as np
from numba import jit
from itertools import combinations
from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.cluster.hierarchy import linkage, fcluster


def normalize_angle(x):
    x = x % (2 * np.pi)    # force in range [0, 2 pi)
    if x > np.pi:          # move to [-pi, pi)
        x -= 2 * np.pi
    return x


def f_cv(x, dt):
    """ state transition function for a
    constant velocity model (vx,vy)"""
    # state x, vx, y, vy, z, yaw
    F = np.array([[1, dt, 0,  0, 0, 0],
                  [0,  1, 0,  0, 0, 0],
                  [0,  0, 1, dt, 0, 0],
                  [0,  0, 0,  1, 0, 0],
                  [0,  0, 0,  0, 1, 0],
                  [0,  0, 0,  0, 0, 1]])
    res = np.dot(F,x)
    return np.dot(F, x)

def h_lidar(x):
    return np.array([x[0], x[2], x[4], normalize_angle(x[5])])

def h_radar(x):
    offset = 0.5 #lidar-radar offset
    px,py,vx,vy = (x[0]-offset,x[2],x[1],x[3])
    r = (px**2 + py**2)
    if (r < 0.1) :
        px=py=0.0707
        r=0.1
    phi = math.atan2(-py,px) # phi positive clockwise but right is negative y
    radial_vel = (vx*px + vy*py)/r
    lat_vel = (py*vx - px*py)/r
    return np.array([r, phi, radial_vel, lat_vel])

def residual_radar(x,y):
    res = np.subtract(x,y)
    res[1] = (res[1] + np.pi) % (2*np.pi) - np.pi
    return res


def intersect_bbox_with_yaw(box_a, box_b):
    """
    A simplified calculation of 3d bounding box intersection.
    It is assumed that the bounding box is only rotated
    around Z axis (yaw) from an axis-aligned box.
    :param box_a, box_b: obstacle bounding boxes for comparison
    :return: intersection volume (float)
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
        return 0.

    # oriented XY overlap
    xy_poly_a = Polygon(zip(*box_a[0:2, 0:4]))
    xy_poly_b = Polygon(zip(*box_b[0:2, 0:4]))
    xy_intersection = xy_poly_a.intersection(xy_poly_b).area
    if xy_intersection == 0:
        return 0.

    return z_intersection * xy_intersection


def iou_old(bbox_a,bbox_b):
    dims_a, dims_b = bbox_a[4:], bbox_b[4:]
    box_a = get_bbox(bbox_a[:3], dims_a, bbox_a[3])
    box_b = get_bbox(bbox_b[:3], dims_b, bbox_b[3])
    vol_intersect = intersect_bbox_with_yaw(box_a, box_b)
    union = get_vol_box(dims_a) + get_vol_box(dims_b) - vol_intersect
    return vol_intersect / union

def eucl_dist(x,y):
    return math.sqrt((x[1]-x[0])**2 + (y[1]-y[0])**2)

def iou(bbox_a, bbox_b):
    xa,ya = bbox_a[:2]
    xb,yb = bbox_b[:2]
    return max(0, 1.-math.sqrt((xa-xb)**2 + (ya-yb)**2)/7)  # upto 7m overlap

def get_vol_box(dims):
        return dims[0] * dims[1] * dims[2]

def convert_x_to_bbox(x, dims):
    return np.concatenate((x[[0,2,4,5]], dims))


def lwh_to_box(l, w, h):
    box = np.array([
        [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
        [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2],
    ])
    return box

def get_bbox(centroid, dims, yaw):
        bbox = lwh_to_box(*dims)
        # calc 3D bound box in capture vehicle oriented coordinates
        rot_mat = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0]])
        oriented_bbox = np.dot(rot_mat, bbox) + np.tile(position, (8, 1)).T
        return oriented_bbox


class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, start_time):
        """
        Initialises a tracker using initial bounding box or radar measurement
        dt is the time between two measurements in seconds.
        """
        dt_lidar = 0.10  # default lidar dt
        dt_radar = 0.05  # default radar dt

        # sigma points
        sigmas = MerweScaledSigmaPoints(6, alpha=.001, beta=2., kappa=-2.)

        # constant vx,vy and constant z,yaw model
        self.ukf_lidar = UKF(dim_x=6, dim_z=4, fx=f_cv,
                             hx=h_lidar, dt=dt_lidar, points=sigmas)
        self.ukf_radar = UKF(dim_x=6, dim_z=4, fx=f_cv,
                             hx=h_radar, dt=dt_radar, points=sigmas, residual_z=residual_radar)

        # state process noise
        self.set_process_noise(dt_radar)

        # measurement noise
        std_x = std_y = 0.1  # detector error for x,y (in meters)
        std_z = 0.1  # detector error for z (in meters)
        std_yaw = 1  # error in yaw (in radians)
        std_r = 5  # radar range error (in meters)
        std_phi = 0.6  # radar angle error (in radians)
        std_vel_r = 0.01  # radial vel error (in m/s)
        std_vel_l = 2.  # lateral vel error (in m/s)

        self.ukf_lidar.R = np.diag([std_x ** 2, std_y ** 2, std_z ** 2, std_yaw ** 2])
        self.ukf_radar.R = np.diag([std_r ** 2, std_phi ** 2, std_vel_r ** 2, std_vel_l ** 2])

        self.x = np.zeros(6)  # state
        self.P = np.eye(6) * 0.5  # state covariance matrix
        self.x[[0, 2, 4, 5]] = bbox[:4]
        self.P[1, 1] = self.P[3, 3] = 100.
        self.ukf_lidar.x = self.ukf_radar.x = self.x
        self.ukf_lidar.P = self.ukf_radar.P = self.P

        self.dims = [4.359, 1.824, 1.466]  # l,w,h
        self.current_time = start_time
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def set_process_noise(self, dt):
        Q = np.eye(6)
        Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=2.2)  # x-accel
        Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=dt, var=2.2)  # y_accel
        Q[4, 4] = (1.5 * dt) ** 2  # z-vel
        Q[5, 5] = (0.6 * dt) ** 2  # yaw-rate
        self.ukf_lidar.Q = Q
        self.ukf_radar.Q = Q

    def update(self, time, measurement, sensor='lidar'):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        if sensor == 'radar':
            self.ukf_radar.update(measurement)
            self.x = self.ukf_lidar.x = self.ukf_radar.x
            self.P = self.ukf_lidar.P = self.ukf_radar.P
        elif sensor == 'lidar':
            self.ukf_lidar.update(measurement[:4])
            self.x = self.ukf_radar.x = self.ukf_lidar.x
            self.P = self.ukf_radar.P = self.ukf_lidar.P
        self.current_time = time

    def predict(self, time, tracking=True):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        # constraints

        # self.kf.x[4,0] = min(1.0, max(-0.4, self.kf.x[4,0])) # z constraint
        dt = (time - self.current_time) / 1000000000.
        self.current_time = time
        self.set_process_noise(dt)
        self.ukf_lidar.predict(dt)
        # sync radar and lidar states, covariance matrix and sigma points
        self.x = self.ukf_radar.x = self.ukf_lidar.x
        self.P = self.ukf_radar.P = self.ukf_lidar.P
        self.ukf_radar.sigmas_f = self.ukf_lidar.sigmas_f
        self.age += dt
        if (self.time_since_update > 0 and tracking):
            self.hit_streak = 0
        self.time_since_update += dt
        self.history.append(convert_x_to_bbox(self.x, self.dims))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.x, self.dims)

def associate_detections_to_trackers(detections, trackers, iou_threshold):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 7), dtype=int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    matched_indices = linear_assignment(-iou_matrix)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def merge_detections(dets, distance_threshold=2.):
    if len(dets) == 0:
        return dets
    if len(dets) == 1:
        return np.array(dets)
    merged_dets = []
    clusters = fcluster(linkage(dets[:, :2]), 2, criterion='distance')
    for c in np.unique(clusters):
        indices = np.where(clusters == c)[0]
        # if len(indices)==1 :
        #    x,y,z,h = dets[indices, 0:4]
        #    r = max(0.4, (dets[indices, 4] + dets[indices, 5])/2 )
        # else :
        # weights = 1- np.minimum(1, .5*(np.abs(dets[indices][:,4]-0.8)+np.abs(dets[indices][:,5]-0.8)))
        # x,y,z,h = np.average(dets[indices][:,0:4], axis=0, weights=weights)
        # r = max(0.4, (np.max(dets[indices][:,4])+ np.max(dets[indices][:,5]))/2 )
        # x_min,y_min = np.minimum(dets[indices][0:2], axis=0)[0]
        # x_max,y_max = np.maximum(dets[indices][0:2], axis=0)[0]
        # x0,y0 = (x_min+x_max)/2., (y_min+y_max)/2.
        merged_det = np.average(dets[indices], axis=0)
        # merged_dets[0:2] = [x0,y0]
        merged_dets.append(merged_det)
    return merged_dets

class Sort(object):
    def __init__(self, max_age=0.1, min_hits=3, iou_threshold=0.05, max_time_elapsed=0.1):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.iou_threshold = iou_threshold
        self.max_time_elapsed = max_time_elapsed

    def update(self, dets, time):
        """
        Params:
          dets - a numpy array of detections in the format [[x,y,z,yaw,l,w,h],[x,y,z,yaw,l,w,h],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 7))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict(time)
            trk[:] = pos
            if (np.any(np.isnan(pos))):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # merge detections
        merged_dets = merge_detections(dets)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(merged_dets,
                                                                                   trks, self.iou_threshold)

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if (t not in unmatched_trks):
                d = matched[np.where(matched[:, 1] == t)[0][0], 0]
                trk.update(time, dets[d, :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :], time)
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if ((trk.time_since_update < self.max_time_elapsed)
                and (trk.hit_streak >= self.min_hits - 1
                     or self.frame_count <= self.min_hits)):
                ret.append(
                    np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 8))

    def radar_update(self, timestamp, radar_tracks):
        # find the closest radar track
        for radar_trk in radar_tracks:
            r, phi = radar_trk[1:3]
            radar_pos = [r * math.cos(phi) + 3, -r * math.sin(phi)]
            for trk in self.trackers:
                d = trk.get_state()
                if eucl_dist(radar_pos, d[0:2]) < 5:
                    trk.predict(timestamp)
                    trk.update(timestamp, [r, phi, radar_trk[3], radar_trk[4]], sensor='radar')