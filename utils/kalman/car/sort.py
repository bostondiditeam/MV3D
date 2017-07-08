#!/usr/bin/env python

import rosbag, rospy
import numpy as np
from parse_tracklet import parse_xml
from sync import generate_frame_map
from generate_tracklet import Tracklet, TrackletCollection
from shapely.geometry import Polygon
from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.cluster.hierarchy import linkage, fcluster
import pandas as pd

import random, math
from itertools import combinations
import ctypes

def eucl_dist(x,y):
    return math.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)

def iou(bbox_a, bbox_b):
    xa,ya = bbox_a[:2]
    xb,yb = bbox_b[:2]
    return max(0, 1.-math.sqrt((xa-xb)**2 + (ya-yb)**2)/5)  # upto 5m overlap  

def get_vol_box(dims):
        return dims[0] * dims[1] * dims[2]

def convert_x_to_bbox(x, dims):
    return np.concatenate((x.T[0][[0,1]], dims))


ukf = ctypes.cdll.LoadLibrary("./UKF_Python_to_C++/SharedLib.so")

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
        self.ukf = ukf.ClassA()
        self.x = np.zeros((5, 1), dtype = np.double)
        meas_lidar = np.ones((4, 1), dtype = np.double)
        meas_lidar[0] = start_time
        meas_lidar[1] = 0
        meas_lidar[2] = bbox[0] 
        meas_lidar[3] = bbox[1] 
        ukf.initializeStateVector(self.ukf, ctypes.c_void_p(meas_lidar.ctypes.data))
        self.x[0:2,0] = bbox[0:2]
        
        self.dims = [bbox[2], bbox[3], 4.359,1.824,1.466] # z,psi,l,w,h
        self.z_history = [bbox[2]]
        self.psi_history = [bbox[3]]
        self.current_time = start_time
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.radar_hits = 0
        self.hit_streak = 0
        self.age = 0
        
    def update(self, time, measurement, sensor='lidar'):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        if sensor == 'radar' :
            meas_radar = np.ones((5, 1), dtype = np.double)
            meas_radar[0] = time
            meas_radar[1] = 1
            meas_radar[2] = measurement[0] 
            meas_radar[3] = measurement[1] 
            meas_radar[4] = measurement[2]
            ukf.update(self.ukf, ctypes.c_void_p(meas_radar.ctypes.data), ctypes.c_void_p(self.x.ctypes.data))
            self.radar_hits += 1
        elif sensor=='lidar' :
            meas_lidar = np.ones((4, 1), dtype = np.double)
            meas_lidar[0] = time
            meas_lidar[1] = 0
            meas_lidar[2] = measurement[0] 
            meas_lidar[3] = measurement[1] 
            self.z_history.append(measurement[2])
            self.psi_history.append(measurement[3])
            if len(self.z_history)>9:
                self.z_history.pop(0)
                self.psi_history.pop(0)
            self.dims[0] = np.mean(self.z_history)
            self.dims[1] = np.mean(self.psi_history)
            ukf.update(self.ukf, ctypes.c_void_p(meas_lidar.ctypes.data), ctypes.c_void_p(self.x.ctypes.data))
        self.current_time = time

    def predict(self, time, tracking=True):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        # constraints 
        
        #self.kf.x[4,0] = min(1.0, max(-0.4, self.kf.x[4,0])) # z constraint 
        dt = (time-self.current_time)
        delta_t = dt * np.ones(1, dtype = np.double)
        ukf.predict(self.ukf, ctypes.c_void_p(self.x.ctypes.data), ctypes.c_void_p(delta_t.ctypes.data))
        dt = dt/1000000000.
        self.current_time = time
        self.age += dt
        if(self.time_since_update>0 and tracking):
            self.hit_streak = 0
        self.time_since_update += dt
        self.history.append(convert_x_to_bbox(self.x, self.dims))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.x, self.dims)



def associate_detections_to_trackers(detections,trackers,iou_threshold):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,7),dtype=int)
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
        return np.array(dets)
    merged_dets = []
    clusters = fcluster(linkage(dets[:,:2]), distance_threshold, criterion='distance')
    for c in np.unique(clusters) :
        indices = np.where(clusters==c)[0]
        #if len(indices)==1 :
        #    x,y,z,h = dets[indices, 0:4]
        #    r = max(0.4, (dets[indices, 4] + dets[indices, 5])/2 )
        #else : 
        #weights = 1- np.minimum(1, .5*(np.abs(dets[indices][:,4]-0.8)+np.abs(dets[indices][:,5]-0.8))) 
        #x,y,z,h = np.average(dets[indices][:,0:4], axis=0, weights=weights)
        #r = max(0.4, (np.max(dets[indices][:,4])+ np.max(dets[indices][:,5]))/2 )
        #x_min,y_min = np.minimum(dets[indices][0:2], axis=0)[0]
        #x_max,y_max = np.maximum(dets[indices][0:2], axis=0)[0]
        #x0,y0 = (x_min+x_max)/2., (y_min+y_max)/2.
        merged_det = np.average(dets[indices], axis=0)
        #merged_dets[0:2] = [x0,y0]
        merged_dets.append(merged_det)      
    return merged_dets


class Sort(object):
    def __init__(self,max_age=0.1,min_hits=3, iou_threshold=0.05, max_time_elapsed=0.1):
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
          dets - a numpy array of detections in the format [[x,y,z,yaw,l,w,h],[x,y,z,yaw,l,w,h],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        #get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers),7))
        to_del = []
        ret = []
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
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(merged_dets,
                                                                                   trks,self.iou_threshold)


        #update matched trackers with assigned detections
        for t,trk in enumerate(self.trackers):
            if(t not in unmatched_trks):
                d = matched[np.where(matched[:,1]==t)[0][0],0]
                trk.update(time, dets[d,:])

        #create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:], time) 
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if((trk.time_since_update < self.max_time_elapsed) 
                    and (trk.hit_streak >= self.min_hits-1 
                         or self.frame_count <= self.min_hits)):
                ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
            i -= 1
            r = math.sqrt(d[0]**2 + d[1]**2)
            #remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
            # check if consistent with radar
            elif (d[0]>0 and r>2 and trk.hits>3 and trk.radar_hits<0.2*trk.hits) :
                self.trackers.pop(i)
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0,8))
    
    def radar_update(self, timestamp, radar_tracks) :
        # find the closest radar track
        self.frame_count += 1
        for radar_trk in radar_tracks :
            r,phi = radar_trk[1:3]
            radar_pos = [(r+2)*math.cos(phi)+1.5, -(r+2)*math.sin(phi)]
            for trk in self.trackers :
                d = trk.get_state()
                if eucl_dist(radar_pos, d[0:2]) < 2 :
                    trk.predict(timestamp)
                    trk.update(timestamp, [r, phi, radar_trk[3], radar_trk[4]], sensor='radar')
            



car_tag = "02"
bag_file = '/media/prerit/Data/didi_data/ford/ford_'+car_tag+'/'+'ford'+car_tag+'.bag'
bag = rosbag.Bag(bag_file)
tracklet_file = '/media/prerit/Data/didi_data/ford/ford_'+car_tag+'/'+'ford'+car_tag+'.xml'
tracklets = parse_xml(tracklet_file)


bag = rosbag.Bag(bag_file)
print('Reading timestamps from bag ', bag_file)
n_stamps = bag.get_message_count(topic_filters=['/image_raw'])
timestamps, radar_timestamps = [],[]
radar_tracks = {}
for topic,msg,t in bag.read_messages(topics=['/image_raw','/radar/tracks']):
    if topic=='/image_raw':
        timestamps.append(t.to_nsec()) 
    else :
        radar_time = t.to_nsec()
        radar_timestamps.append(radar_time)
        radar_tracks[radar_time] = [] 
        for track in msg.tracks :
            if (track.status!=3) or (track.range>60) or (track.range<4):
                continue
            radar_tracks[radar_time].append([track.number, track.range, 
                                            track.angle/180*np.pi, track.rate, track.late_rate])
detections = {t:[] for t in timestamps}
for track in tracklets:
    stamp = timestamps[track.first_frame]
    detections[stamp].append(np.concatenate((track.trans[0],[track.rots[0][2]], track.size))) # (x,y,z,yaw, h,w,l)


cam_df = pd.DataFrame(index=timestamps, columns=['sensor'])
cam_df['sensor'] = 'C'
radar_df = pd.DataFrame(index=radar_timestamps, columns=['sensor'])
radar_df['sensor'] = 'R'
sensor_df = pd.merge(cam_df, radar_df, left_index=True, right_index=True, on='sensor', how='outer')
sensor_df

KalmanBoxTracker.count = 0
mot_tracker = Sort(max_age=3, min_hits=3, iou_threshold=0.1, max_time_elapsed=3)
collection = TrackletCollection()
min_dets = 5
min_radar_dets = 4

tracklets = {}
frame_count=0
track_count = {}
for timestamp,sensor in sensor_df.iterrows():
    print(timestamp, sensor['sensor'],frame_count, track_count)
    if sensor['sensor'] == 'R':
        #print('radar detections : ', radar_tracks[timestamp])
        mot_tracker.radar_update(timestamp, radar_tracks[timestamp])
    elif sensor['sensor'] == 'C':
        d = detections[timestamp]
        print('MV3D detections : ', d)
        tracks =  mot_tracker.update(np.array(d),timestamp)
        print("~~~~ tracks ~~~~~ ", tracks)
        for track in tracks :    
            trk_id = int(track[7])
            print("================ ID : ", trk_id)
            if trk_id not in track_count.keys() :
                track_count[trk_id] = 0
                continue
            else :
                track_count[trk_id] += 1 
            if track_count[trk_id] < min_dets :
                continue
            if track_count[trk_id] == min_dets :
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                tracklets[trk_id] = Tracklet(
                        object_type = 'Car', 
                        l = track[4], 
                        w = track[5], 
                        h = track[6], 
                        first_frame=frame_count)
            frame = dict(tx=track[0],
                     ty=track[1],
                     tz=track[2],
                     rx=0.,
                     ry=0.,
                     rz=track[3])
            tracklets[trk_id].poses.append(frame)
        frame_count += 1
print(tracklets)
for trk_id in tracklets.keys() :
    print(trk_id)
    collection.tracklets.append(tracklets[trk_id])
out_file = '/media/prerit/Data/didi_data/ford/ford_'+car_tag+'/'+'ford'+car_tag+'_corrected.xml'
collection.write_xml(out_file)

