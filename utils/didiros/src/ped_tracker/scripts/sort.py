#!/usr/bin/env python

from __future__ import print_function, division

import rosbag, rospy
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

import random, math
from itertools import combinations
from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.cluster.hierarchy import linkage, fcluster



def get_vol_cylinder(r,h):
    return np.pi * r**2 * h


def _lens(r, ro, d):
    td = (d**2 + r**2 - ro**2)/(2*d)
    return r**2 * math.acos(td/r) - td * math.sqrt(r**2 - td**2)


def intersect_cylinder(cyl_a, cyl_b):
    """
    3d cylinder intersection.
    :param cyl_a, cyl_b: cylinders for comparison
        packed as [x,y,z,r,h] where xyz points to center of axis aligned (h along z) cylinder
        Note : r and h are hard-coded in this implementation 
    :return: intersection volume (float)
    """
    xyz_a, xyz_b = cyl_a[0:3], cyl_b[0:3]
    r_a, r_b = cyl_a[3], cyl_b[3]
    h_a, h_b = cyl_a[4], cyl_b[4]

    # height (Z) overlap
    zh_a = [xyz_a[2] + h_a/2, xyz_a[2] - h_a/2]
    zh_b = [xyz_b[2] + h_b/2, xyz_b[2] - h_b/2]
    max_of_min = np.max([zh_a[1], zh_b[1]])
    min_of_max = np.min([zh_a[0], zh_b[0]])
    z_intersection = np.max([0, min_of_max - max_of_min])
    if z_intersection == 0:
        return 0.

    dist = np.linalg.norm(cyl_a[0:2] - cyl_b[0:2])
    if dist >= r_a + r_b:
        # cylinders do not overlap in any way
        return 0.
    elif dist <= abs(r_a - r_b):
        # one cylinder fully inside the other (includes coincident)
        # take volume of smaller sphere as intersection
        intersection = np.pi * min(r_a, r_b)**2 * z_intersection
    else:    
        circle_intersection = _lens(r_a, r_b, dist) + _lens(r_b, r_a, dist)
        intersection = circle_intersection * z_intersection
    return intersection


def iou(bb_test,bb_gt):
    vol_intersect = intersect_cylinder(bb_test, bb_gt)
    union = (get_vol_cylinder(*bb_test[3:]) + get_vol_cylinder(*bb_gt[3:]) 
            - vol_intersect)
    return vol_intersect / union 


def convert_bbox_to_z(bbox):
    return np.array(bbox[0][:3]).reshape((-1,1))


def convert_x_to_bbox(x, dims):
    return np.concatenate((x[[0,2,4],0], dims))




class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self,bbox, start_time):
        """
        Initialises a tracker using initial bounding box 
        dt is the time between two measurements in seconds.
        """
        #define constant velocity model
        self.kf = KalmanFilter(dim_x=5, dim_z=3)
        dt = 1./30 
        self.kf.F = np.array([[1, dt, 0,  0, 0], 
                              [0,  1, 0,  0, 0],
                              [0,  0, 1, dt, 0],
                              [0,  0, 0,  1, 0],
                              [0,  0, 0,  0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 1]])
        
        std_x, std_y, std_z = 2,2,0.8  #TODO : tune
        self.kf.R = np.diag([std_x**2, std_y**2, std_z**2])
        self.kf.P[1,1] = self.kf.P[3,3] = 1000. 
        #give high uncertainty to the unobservable initial velocities
        self.kf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=1.2) #x-accel
        self.kf.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=dt, var=1.2) #y_accel
        self.kf.Q[4,4] = (1.5*dt)**2 #z-vel
        

        self.kf.x[[0,2,4],0] = bbox[:3]
        self.dims = [0.4, 1.78]
        self.current_time = start_time
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self,bbox, time):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        self.current_time = time

    def predict(self, time, is_update=True):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        # constraints 
        
        #self.kf.x[4,0] = min(1.0, max(-0.4, self.kf.x[4,0])) # z constraint 
        dt = (time-self.current_time)/1000000000
        self.current_time = time
        self.kf.F = np.array([[1, dt, 0,  0, 0], 
                              [0,  1, 0,  0, 0],
                              [0,  0, 1, dt, 0],
                              [0,  0, 0,  1, 0],
                              [0,  0, 0,  0, 1]])
        self.kf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=1.2) #x-accel
        self.kf.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=dt, var=1.2) #y_accel
        self.kf.Q[4,4] = (1.5*dt)**2 #z-vel
        self.kf.predict()
        self.age += dt
        if(self.time_since_update>0 and is_update):
            self.hit_streak = 0
        self.time_since_update += dt
        self.history.append(convert_x_to_bbox(self.kf.x, self.dims))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x, self.dims)


def associate_detections_to_trackers(detections,trackers,iou_threshold):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
    iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

    for d,det in enumerate(detections):
        for t,trk in enumerate(trackers):
            iou_matrix[d,t] = iou(det,trk)
    matched_indices = linear_assignment(-iou_matrix)

    unmatched_detections = []
    for d,det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t,trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0],m[1]]<iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def merge_detections(dets, distance_threshold=2.):
    if len(dets)==0 :
        return dets
    if len(dets)==1 : 
        x,y,z,h = dets[0,0:4]
        r = max(0.4, (dets[0,4]+dets[0,5])/2) 
        return [np.array([x,y,z,r,h])]
    merged_dets = []
    clusters = fcluster(linkage(dets[:2]), 2, criterion='distance')
    for c in np.unique(clusters) :
        indices = np.where(clusters==c)[0]
        weights = 1- np.minimum(1, .5*(np.abs(dets[indices][:,4]-0.8)
                                       +np.abs(dets[indices][:,5]-0.8))) 
        x,y,z,h = np.average(dets[indices][:,0:4], axis=0, weights=weights)
        r = max(0.4, (np.max(dets[indices][:,4])+ np.max(dets[indices][:,5]))/2 )
        merged_dets.append(np.array([x,y,z,r,h]))      
    return merged_dets


class Sort(object):
    def __init__(self,max_age=0.1,min_hits=3, 
                 iou_threshold=0.05, max_time_elapsed=0.1):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.iou_threshold = iou_threshold
        self.max_time_elapsed = max_time_elapsed
        
    def update(self,dets,time):
        """
        Params:
          dets - a numpy array of detections in the format [[x,y,z,r,h],[x,y,z,r,h],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        #get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers),5))
        to_del = []
        for t,trk in enumerate(trks):
            pos = self.trackers[t].predict(time)
            trk[:] = pos
            if(np.any(np.isnan(pos))):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
            
        # merge detections
        merged_dets = merge_detections(dets)
        matched, unmatched_dets, unmatched_trks = \
        associate_detections_to_trackers(merged_dets,trks,self.iou_threshold)

        #update matched trackers with assigned detections
        for t,trk in enumerate(self.trackers):
            if(t not in unmatched_trks):
                d = matched[np.where(matched[:,1]==t)[0],0]
                trk.update(dets[d,:], time)

        #create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:], time) 
            self.trackers.append(trk)
        i = len(self.trackers)
        return self.good_tracks()

    def good_tracks(self): 
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if((trk.time_since_update < self.max_time_elapsed) 
                    and (trk.hit_streak >= self.min_hits-1 
                         or self.frame_count <= self.min_hits)):
                ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) 
            i -= 1
            #remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0,6))
