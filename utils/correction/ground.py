#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
import rospy, rosbag, tf
from sensor_msgs.msg import PointCloud2, PointField
import cv2, cv_bridge
import numpy as np
import csv, sys, os, copy
from collections import defaultdict
import PyKDL as kd
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn import cluster
import argparse, pickle

from extract_lidar import *
from parse_tracklet import *
from sync import *


def sync_time(timestamp_map, velodyne_timestamps):
    camera_df = pd.DataFrame({'cam_index':timestamp_map})
    camera_df.index = pd.to_datetime(camera_df.index)
    velodyne_df = pd.DataFrame({'vel_index':velodyne_timestamps})
    velodyne_df.index = pd.to_datetime(velodyne_df.index)
    time_df = pd.merge(camera_df, velodyne_df, left_index=True, right_index=True, how='outer')
    time_df = time_df.interpolate(method='time', limit=100, limit_direction='both')
    time_df = time_df.round().astype(int)
    time_df.index = time_df.index.astype(np.int64)
    return time_df


def get_patch(points, corners):
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]
    reflectance = points[:,3]
    
    range_min = np.min(corners, axis=0)
    range_max = np.max(corners, axis=0)
    center = (range_max+range_min)/2
    size = np.abs(range_max - range_min)/2+0.3
    z_min = -2.5
    
    search_size = 5
    z_max = min(-1.2, max(-1.2, center[2]/2))
    
    # filter capture vehicle
    c_filt = np.logical_or(np.abs(x_points)>4.7/2, np.abs(y_points)>2.1/2)
    
    f_z = (z_points < z_max) & (z_points > z_min) 
    
    k_means = cluster.KMeans(n_clusters=3)
    
    # filter obstacle vehicle 
    indices = []
    
    while(len(indices)<50) :
        f_x = (np.abs(x_points-center[0]) > size[0]) & (np.abs(x_points-center[0]) < search_size)
        f_y = (np.abs(y_points-center[1]) > size[1]) & (np.abs(y_points-center[1]) < search_size)
        indices = np.argwhere(f_x & f_y & f_z & c_filt).flatten()
        search_size += 5
        
   
    xi_points = (x_points[indices]-center[0]+search_size).astype(np.int32)
    yi_points = (y_points[indices]-center[1]+search_size).astype(np.int32)
    zi_points = z_points[indices]
    k_means.fit(zi_points.reshape((-1,1)))
    labels, counts = np.unique(k_means.labels_ , return_counts=True)
    ground_cluster = labels[np.argmax(counts)]
    ground_height = k_means.cluster_centers_[ground_cluster][0]
    return ground_height


def height_map(bag_file, velodyne_timestamps, frame_map):
    df = pd.DataFrame(index=sorted(velodyne_timestamps.keys()), columns=['z_min'])
    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics='/velodyne_points'):
            timestamp = t.to_nsec()
            if (timestamp//1e8)%10 == 0 :
                print('currently on ', timestamp)
            arr = msg_to_arr(msg)
            lidar = np.array([[a[0], a[1], a[2], a[3]] for a in arr])
            frame_index = time_df.loc[[t.to_nsec()]].cam_index.values[0]
            for i, f in enumerate(frame_map[frame_index]):
                dims = f.size[::-1]
                obs_centroid = np.array(f.trans)
                R = tf.transformations.quaternion_matrix(f.rotq)
                corners = [0.5*np.array([i,j,k])*dims for i in [-1,1] 
                            for j in [-1,1] for k in [-1,1]]
                corners = [obs_centroid + R.dot(list(c)+[1])[:3] for c in corners]
                df.loc[timestamp] = [get_patch(lidar, corners)]
    return df

def smooth_profile(df, velodyne_timestamps, save_plot=None):
    t = sorted(np.array(velodyne_timestamps.keys()))
    t0 = min(t)
    t = (t-t0)/1e9
    extra = 0.2*t[-1]
    n = int(10*extra)
    
    df = df.sort_index()
    df.index = t
    df_pre = pd.concat([pd.DataFrame(df.iloc[:n].mean()).transpose()]*n)
    df_pre.index = np.linspace(-extra, -0.1, n)
    df_post = pd.concat([pd.DataFrame(df.iloc[-n:].mean()).transpose()]*n)
    df_post.index = np.linspace(t[-1]+0.1, t[-1]+extra, n)
    df_new = pd.concat([df_pre, df, df_post])
    z = np.polyfit(df_new.index, df_new['z_min'], 13)
    poly = np.poly1d(z)    
    
    tp = np.linspace(df_new.index[0], max(t)+extra, 100)
    plt.plot(df_new.index, df_new['z_min'])
    plt.plot(tp, poly(tp))
    if save_plot:
        plt.savefig(save_plot)    
    return poly


def save_data(poly, df_out, time_df, out_file):
    t = df_out.index
    t0= min(t)
    t = (t-t0)/1e9
    #valid_indices = np.argwhere(mask).flatten()
    valid_indices = list(range(len(df_out)))
    mask1 = np.in1d(time_df['vel_index'], valid_indices)
    mask2 = np.in1d(time_df[mask1].index, df_out.index)
    valid_times = time_df[mask1][mask2].index
    #for key in poly.keys():
    df_out.loc[valid_times, 'z_min'] = poly((valid_times-t0)/1e9)
    df_out.to_csv(out_file, index_label=False)


if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="generate ground correction")
    parser.add_argument('bag', type=str, nargs='?', default='', help='bag filename')
    parser.add_argument('tracklet', type=str, nargs='?', default='', help='tracklet filename')
    parser.add_argument('out', type=str, nargs='?', default='', help='output directory')
    parser.add_argument('--data', type=str, nargs='?', default='/media/prerit/Data/didi_data', 
                        help='data directory')
    args = parser.parse_args()


    data_dir = args.data
    bag_file = os.path.join(data_dir, args.bag)
    tracklet_file = os.path.join(data_dir, args.tracklet)
    out_dir = os.path.join(data_dir, args.out)
    assert os.path.isfile(bag_file), 'Bag file %s does not exist' % bag_file
    assert os.path.isfile(tracklet_file), 'Tracklet file %s does not exist' % tracklet_file

    bag_name = os.path.basename(bag_file).split('.')[0]
    save_dir=os.path.join(out_dir, bag_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    timestamp_pkl = os.path.join(save_dir, 'timestamp.pkl')
    if os.path.isfile(timestamp_pkl) :
        print('Loading data from ', timestamp_pkl)
        with open(timestamp_pkl, 'rb') as f:
            timestamp_map = pickle.load(f)
    else :
        print('Creating timestamp map')
        timestamp_map = extract_bag_timestamps(bag_file, topics=['/image_raw', '/velodyne_points'])
        print('Dumping data to pickle')
        with open(timestamp_pkl, 'wb') as f:
            pickle.dump(timestamp_map, f, pickle.HIGHEST_PROTOCOL)
    
    
    frame_map_pkl = os.path.join(save_dir, 'frame_map.pkl')
    if os.path.isfile(frame_map_pkl) :
        print('Loading data from ', frame_map_pkl)
        with open(frame_map_pkl, 'rb') as f:
            frame_map = pickle.load(f)
    else :
        print('Creating frame map')
        tracklets = parse_xml(tracklet_file)
        frame_map = generate_frame_map(tracklets)
        print('Dumping data to pickle')
        with open(frame_map_pkl, 'wb') as f:
            pickle.dump(frame_map, f, pickle.HIGHEST_PROTOCOL)

    time_df = sync_time(timestamp_map['/image_raw'], timestamp_map['/velodyne_points'])
    df = height_map(bag_file, timestamp_map['/velodyne_points'], frame_map)
    fig_file = os.path.join(save_dir, 'height_profile.png')
    poly = smooth_profile(df, timestamp_map['/velodyne_points'], save_plot=fig_file)
    df_out = pd.DataFrame(index=sorted(timestamp_map['/image_raw'].keys()), 
                          columns=['z_min'])
    out_file = os.path.join(out_dir, bag_name+'_ground.csv')
    save_data(poly, df_out, time_df, out_file)
