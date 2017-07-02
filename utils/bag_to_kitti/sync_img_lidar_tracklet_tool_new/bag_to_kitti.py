#! /usr/bin/python
""" Udacity Self-Driving Car Challenge Bag Processing
"""

from __future__ import print_function
from collections import defaultdict
import os
import sys
import math
import argparse
import functools
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PyKDL as kd

from .bag_topic_def import *
from .bag_utils import *
from .generate_tracklet import *
from scipy.spatial import kdtree
from scipy import stats
import shutil
import glob


# Bag message timestamp source
TS_SRC_PUB = 0
TS_SRC_REC = 1
TS_SRC_OBS_REC = 2

# Correction method
CORRECT_NONE = 0
CORRECT_PLANE = 1

CAP_RTK_FRONT_Z = .3323 + 1.2192
CAP_RTK_REAR_Z = .3323 + .8636

# From capture vehicle 'GPS FRONT' - 'LIDAR' in
# https://github.com/udacity/didi-competition/blob/master/mkz-description/mkz.urdf.xacro
FRONT_TO_LIDAR = [-1.0922, 0, -0.0508]

# For pedestrian capture, a different TF from mkz.urdf was used in capture. This must match
# so using that value here.
BASE_LINK_TO_LIDAR_PED = [1.9, 0., 1.6]

CAMERA_COLS = ["timestamp", "width", "height", "frame_id", "filename"]
RADAR_COLS = ["timestamp", "filename"]
GPS_COLS = ["timestamp", "lat", "long", "alt"]
POS_COLS = ["timestamp", "tx", "ty", "tz", "rx", "ry", "rz"]


def obs_name_from_topic(topic):
    return topic.split('/')[2]


def obs_prefix_from_topic(topic):
    words = topic.split('/')
    prefix = '_'.join(words[1:4])
    name = words[2]
    return prefix, name

def interpolate_to_camera(camera_df, other_dfs, filter_cols=[]):
    if not isinstance(other_dfs, list):
        other_dfs = [other_dfs]
    if not isinstance(camera_df.index, pd.DatetimeIndex):
        print('Error: Camera dataframe needs to be indexed by timestamp for interpolation')
        return pd.DataFrame()

    for o in other_dfs:
        o['timestamp'] = pd.to_datetime(o['timestamp'])
        o.set_index(['timestamp'], inplace=True)
        o.index.rename('index', inplace=True)

    merged = functools.reduce(lambda left, right: pd.merge(
        left, right, how='outer', left_index=True, right_index=True), [camera_df] + other_dfs)
    merged.interpolate(method='time', inplace=True, limit=100, limit_direction='both')

    filtered = merged.loc[camera_df.index]  # back to only camera rows
    filtered.fillna(0.0, inplace=True)
    filtered['timestamp'] = filtered.index.astype('int')  # add back original timestamp integer col
    if filter_cols:
        if not 'timestamp' in filter_cols:
            filter_cols += ['timestamp']
        filtered = filtered[filter_cols]

    return filtered

# if corresponding velodyne directory exists, calibrate them and save it in output directory.
def interp_lidar_to_camera(camera_index_df, lidar_indir, bs, dataset_outdir):
    # if corresponding velodyne directory exists, calibrate them and save it in output directory.
    lidar_cols = ["timestamp", "filename"]
    lidar_dict = defaultdict(list)

    # lidar_indir = os.listdir(lidar_indir)
    if bs.name in os.listdir(lidar_indir):
        lidar_dir = os.path.join(lidar_indir, bs.name, "velodyne_points")
        lidar_dir = os.path.join(lidar_indir, bs.name)
        lidar_outdir = get_outdir(dataset_outdir, "velodyne_points/data")
        # print("bag name is: ", bs.name)
        # print("lidar_dir is here: ", lidar_dir)
        # print("output file name is here: ", velodyne_dir)
        # generate timestamp file.

        txt_file = glob.glob(lidar_dir + "/*.txt")
        # print("txt_files are here: ", txt_file)
        lidar_maps = []
        for file_name in txt_file:
            with open(file_name) as f:
                a = f.read()
                a = np.uint64(a) + 1
                bin_name = file_name.split('/')[-1].split('_')[0]
                bin_name = int(bin_name)
                lidar_maps.append((a, bin_name))
        lidar_frame = pd.DataFrame(lidar_maps, columns=lidar_cols)
        lidar_frame = lidar_frame.sort_values(by=lidar_cols[0])

        lidar_frame_csv = lidar_frame.copy()
        # print(list(lidar_frame_csv))
        lidar_frame_csv['timestamp'] = lidar_frame_csv.timestamp.map(lambda x: '{:.0f}'.format(x))
        lidar_frame_csv.to_csv(lidar_outdir + '/../timestamp.csv', index=False)
        # compare the timestamp and copy and rename bin file here.
        # lidar_interp is a dataframe.

        lidar_interp = interpolate_to_camera(camera_index_df, lidar_frame, filter_cols=lidar_cols)

        lidar_interp['filename'] = lidar_interp['filename'].round().astype(int)
        lidar_interp = lidar_interp.astype(object)

        lidar_interp_csv = lidar_interp.copy()
        lidar_interp_csv['timestamp'] = lidar_interp_csv.timestamp.map(lambda x: str(x))
        lidar_interp_csv.to_csv(lidar_outdir + '/../map.csv', index=False)

        for index, row in lidar_interp.iterrows():
            source_lidar_file_path = os.path.join(lidar_dir, str(row['filename']) + '.bin')
            dest_lidar_file_path = os.path.join(lidar_outdir, str(row['timestamp']) + '.bin')

            # print("lidar outout here: ", dest_lidar_file_path)
            shutil.copy(source_lidar_file_path, dest_lidar_file_path)

        print("input lidar dir is here: ", lidar_dir)
        print("output lidar dir is here: ", lidar_outdir)


def camera2dict(timestamp, msg, write_results, camera_dict):
    camera_dict["timestamp"].append(timestamp)
    if write_results:
        camera_dict["width"].append(write_results['width'] if 'width' in write_results else msg.width)
        camera_dict['height'].append(write_results['height'] if 'height' in write_results else msg.height)
        camera_dict["frame_id"].append(msg.header.frame_id)
        camera_dict["filename"].append(write_results['filename'])


def gps2dict(timestamp, msg, gps_dict):
    gps_dict["timestamp"].append(timestamp)
    gps_dict["lat"].append(msg.latitude)
    gps_dict["long"].append(msg.longitude)
    gps_dict["alt"].append(msg.altitude)


def pose2dict(timestamp, msg, pose_dict):
    pose_dict["timestamp"].append(timestamp)
    pose_dict["tx"].append(msg.pose.position.x)
    pose_dict["ty"].append(msg.pose.position.y)
    pose_dict["tz"].append(msg.pose.position.z)
    rotq = kd.Rotation.Quaternion(
        msg.pose.orientation.x,
        msg.pose.orientation.y,
        msg.pose.orientation.z,
        msg.pose.orientation.w)
    rot_xyz = rotq.GetRPY()
    pose_dict["rx"].append(rot_xyz[0])
    pose_dict["ry"].append(rot_xyz[1])
    pose_dict["rz"].append(rot_xyz[2])

#ittwi:radar
def radar2dict(timestamp, filename, radar_dict):
    radar_dict["timestamp"].append(timestamp)
    radar_dict["filename"].append(filename)


def tf2dict(timestamp, tf, tf_dict):
    tf_dict["timestamp"].append(timestamp)
    tf_dict["tx"].append(tf.translation.x)
    tf_dict["ty"].append(tf.translation.y)
    tf_dict["tz"].append(tf.translation.z)
    rotq = kd.Rotation.Quaternion(
        tf.rotation.x,
        tf.rotation.y,
        tf.rotation.z,
        tf.rotation.w)
    rot_xyz = rotq.GetRPY()
    tf_dict["rx"].append(rot_xyz[0])
    tf_dict["ry"].append(rot_xyz[1])
    tf_dict["rz"].append(rot_xyz[2])


def imu2dict(timestamp, msg, imu_dict):
    imu_dict["timestamp"].append(timestamp)
    imu_dict["ax"].append(msg.linear_acceleration.x)
    imu_dict["ay"].append(msg.linear_acceleration.y)
    imu_dict["az"].append(msg.linear_acceleration.z)


def get_yaw(p1, p2):
    return math.atan2(p1[1] - p2[1], p1[0] - p2[0])


def dict_to_vect(di):
    return kd.Vector(di['tx'], di['ty'], di['tz'])


def list_to_vect(li):
    return kd.Vector(li[0], li[1], li[2])


def vect_to_dict3(v):
    return dict(tx=v[0], ty=v[1], tz=v[2])


def vect_to_dict6(v):
    if len(v) == 6:
        return dict(tx=v[0], ty=v[1], tz=v[2], rx=v[3], ry=v[4], rz=v[5])
    else:
        return dict(tx=v[0], ty=v[1], tz=v[2], rx=0, ry=0, rz=0)


def frame_to_dict(frame, yaw_only=False):
    r, p, y = frame.M.GetRPY()
    if yaw_only:
        return dict(tx=frame.p[0], ty=frame.p[1], tz=frame.p[2], rx=0., ry=0., rz=y)
    return dict(tx=frame.p[0], ty=frame.p[1], tz=frame.p[2], rx=r, ry=p, rz=y)


def dict_to_frame(di):
    return kd.Frame(
        kd.Rotation.RPY(di['rx'], di['ry'], di['rz']),
        kd.Vector(di['tx'], di['ty'], di['tz']))


def init_df(data_dict, cols, filename, outdir=''):
    df = pd.DataFrame(data=data_dict, columns=cols)
    if len(df.index) and filename:
        df.to_csv(os.path.join(outdir, filename), index=False)
    return df


def interpolate_df(input_dfs, index_df, filter_cols=[], filename='', outdir=''):
    if not isinstance(input_dfs, list):
        input_dfs = [input_dfs]
    if not isinstance(index_df.index, pd.DatetimeIndex):
        print('Error: Camera dataframe needs to be indexed by timestamp for interpolation')
        return pd.DataFrame()

    for i in input_dfs:
        if len(i.index) == 0:
            print('Warning: Empty dataframe passed to interpolate, skipping.')
            return pd.DataFrame()
        i['timestamp'] = pd.to_datetime(i['timestamp'])
        i.set_index(['timestamp'], inplace=True)
        i.index.rename('index', inplace=True)

    merged = functools.reduce(lambda left, right: pd.merge(
        left, right, how='outer', left_index=True, right_index=True), [index_df] + input_dfs)
    merged.interpolate(method='time', inplace=True, limit=100, limit_direction='both')

    filtered = merged.loc[index_df.index]  # back to only index' rows
    filtered.fillna(0.0, inplace=True)
    filtered['timestamp'] = filtered.index.astype('int')  # add back original timestamp integer col
    if filter_cols:
        if not 'timestamp' in filter_cols:
            filter_cols += ['timestamp']
        filtered = filtered[filter_cols]

    if len(filtered.index) and filename:
        filtered.to_csv(os.path.join(outdir, filename), header=True)
    return filtered


def obstacle_rtk_to_pose(
        cap_front,
        cap_rear,
        obs_front,
        obs_rear,
        obs_gps_to_centroid,
        front_to_velodyne,
        cap_yaw_err=0.,
        cap_pitch_err=0.):

    # calculate capture yaw in ENU frame and setup correction rotation
    cap_front_v = dict_to_vect(cap_front)
    cap_rear_v = dict_to_vect(cap_rear)
    cap_yaw = get_yaw(cap_front_v, cap_rear_v)
    cap_yaw += cap_yaw_err
    rot_cap = kd.Rotation.EulerZYX(-cap_yaw, -cap_pitch_err, 0)

    obs_rear_v = dict_to_vect(obs_rear)
    if obs_front:
        obs_front_v = dict_to_vect(obs_front)
        obs_yaw = get_yaw(obs_front_v, obs_rear_v)
        # use the front gps as the obstacle reference point if it exists as it's closers
        # to the centroid and mounting metadata seems more reliable
        cap_to_obs = obs_front_v - cap_front_v
    else:
        cap_to_obs = obs_rear_v - cap_front_v

    # transform capture car to obstacle vector into capture car velodyne lidar frame
    res = rot_cap * cap_to_obs
    res += list_to_vect(front_to_velodyne)

    # obs_gps_to_centroid is offset for front gps if it exists, otherwise rear
    obs_gps_to_centroid_v = list_to_vect(obs_gps_to_centroid)
    if obs_front:
        # if we have both front + rear RTK calculate an obstacle yaw and use it for centroid offset
        obs_rot_z = kd.Rotation.RotZ(obs_yaw - cap_yaw)
        centroid_offset = obs_rot_z * obs_gps_to_centroid_v
    else:
        # if no obstacle yaw calculation possible, treat rear RTK as centroid and offset in Z only
        obs_rot_z = kd.Rotation()
        centroid_offset = kd.Vector(0, 0, obs_gps_to_centroid_v[2])
    res += centroid_offset
    return frame_to_dict(kd.Frame(obs_rot_z, res), yaw_only=True)


def filter_outlier_points(points):
    kt = kdtree.KDTree(points)
    distances, i = kt.query(kt.data, k=9)
    z_distances = stats.zscore(np.mean(distances, axis=1))
    o_filter = abs(z_distances) < 1  # rather arbitrary
    return points[o_filter]


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    http://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
    """
    axis = np.asarray(axis)
    axis /= math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


def fit_plane(points, do_plot=True, dataset_outdir='', name='', debug=True):
    if debug:
        print('Processing %d points' % points.shape[0])

    centroid = np.mean(points, axis=0)
    if debug:
        print('centroid', centroid)
    points -= centroid

    _, _, v = np.linalg.svd(points)
    line = v[0] / np.linalg.norm(v[0])
    norm = v[-1] / np.linalg.norm(v[-1])
    norm *= np.sign(v[-1][-1])
    use_line = False
    if np.argmax(norm) != 2:
        print('Warning: Z component of plane normal is not largest, plane fit likely not optimal. Fitting line instead.')
        use_line = True
    if debug:
        print('line', line)
        print('norm', norm)

    if use_line:
        # find a rotation axis perpendicular to the fit line of the coords and
        # calculate rotation angle around that axis that levels fit line in z
        axis = np.cross(line, np.array([0, 0, 1.]))
        angle = line[2]
    else:
        # use plane normal to calculate a rotation axis and angle necessary
        # to level points in z
        z_cross_norm = np.cross(np.array([0, 0, 1.]), norm)
        angle = np.arcsin(np.linalg.norm(z_cross_norm))
        axis = z_cross_norm / np.linalg.norm(z_cross_norm)
    if debug:
        print('rotation', angle, axis)

    rot_m = rotation_matrix(axis, -angle)

    if do_plot:
        x_max, x_min = max(points[:, 0]), min(points[:, 0])
        y_max, y_min = max(points[:, 1]), min(points[:, 1])
        xy_max = max(x_max, y_max)
        xy_min = min(x_min, y_min)

        # compute normal of corrected points to visualize and verify
        points_rot = np.dot(rot_m, points.T).T
        _, _, vr = np.linalg.svd(points_rot)
        norm_rot = vr[-1] / np.linalg.norm(vr[-1])

        # build plane surface for original points and best fit plane for plotting
        # NOTE if line fit was used instead, this still just plots the plane fit
        d = np.array([0, 0, 0]).dot(norm)
        dr = np.array([0, 0, 0]).dot(norm_rot)
        xg, yg = np.meshgrid(
            range(int(x_min*1.3), int(math.ceil(x_max*1.3))),
            range(int(y_min*1.3), int(math.ceil(y_max*1.3))))
        zg = (d - norm[0] * xg - norm[1] * yg) * 1. / norm[2]
        zgr = (dr - norm_rot[0] * xg - norm_rot[1] * yg) * 1. / norm_rot[2]

        line_pts = line * np.mgrid[xy_min:xy_max:2j][:, np.newaxis]
        axis_pts = axis * np.mgrid[-xy_min:xy_min:2j][:, np.newaxis]
        norm_pts = norm * np.mgrid[-xy_min:xy_min:2j][:, np.newaxis]

        if False:
            points += centroid
            points_rot += centroid
            line_pts += centroid
            axis_pts += centroid
            norm_pts += centroid
            xg += int(centroid[0])
            yg += int(centroid[1])
            zg += centroid[2]
            zgr += centroid[2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*points.T)
        ax.scatter(*points_rot.T, c='y')
        ax.plot3D(*line_pts.T, c='r')
        ax.plot3D(*axis_pts.T, c='c')
        #ax.plot3D(*norm_pts.T, c='g')
        ax.plot_surface(xg, yg, zg, alpha=0.3)
        ax.plot_surface(xg, yg, zgr, alpha=0.3, color='y')
        angles = [0, 30, 60]
        elev = [0, 15, 30]
        for a in angles:
            for e in elev:
                ax.view_init(elev=e, azim=a)
                fig.savefig(os.path.join(dataset_outdir, '%s-%d-%d-plot.png' % (name, e, a)))
        plt.close(fig)
    return centroid, norm, rot_m


def extract_metadata(md, obs_name):
    md = next(x for x in md if x['obstacle_name'] == obs_name)
    if 'gps_l' in md:
        # make old rear RTK only obstacle metadata compatible with new
        md['rear_gps_l'] = md['gps_l']
        md['rear_gps_w'] = md['gps_w']
        md['rear_gps_h'] = md['gps_h']
    return md


def process_rtk_data(
        bagset,
        cap_data,
        obs_data,
        index_df,
        outdir,
        correct=CORRECT_NONE,
        yaw_err=0.,
        pitch_err=0.
):
    tracklets = []
    cap_rear_gps_df = init_df(cap_data['rear_gps'], GPS_COLS, 'cap_rear_gps.csv', outdir)
    cap_front_gps_df = init_df(cap_data['front_gps'], GPS_COLS, 'cap_front_gps.csv', outdir)
    cap_rear_rtk_df = init_df(cap_data['rear_rtk'], POS_COLS, 'cap_rear_rtk.csv', outdir)
    cap_front_rtk_df = init_df(cap_data['front_rtk'], POS_COLS, 'cap_front_rtk.csv', outdir)
    if not len(cap_rear_rtk_df.index):
        print('Error: No capture vehicle rear RTK entries exist.'
              ' Skipping bag %s.' % bagset.name)
        return tracklets
    if not len(cap_rear_rtk_df.index):
        print('Error: No capture vehicle front RTK entries exist.'
              ' Skipping bag %s.' % bagset.name)
        return tracklets

    rtk_z_offsets = [np.array([0., 0., CAP_RTK_FRONT_Z]), np.array([0., 0., CAP_RTK_REAR_Z])]
    if correct > 0:
        # Correction algorithm attempts to fit plane to rtk measurements across both capture rtk
        # units and all obstacles. We will subtract known RTK unit mounting heights first.
        cap_front_points = cap_front_rtk_df.as_matrix(columns=['tx', 'ty', 'tz']) - rtk_z_offsets[0]
        cap_rear_points = cap_rear_rtk_df.as_matrix(columns=['tx', 'ty', 'tz']) - rtk_z_offsets[1]
        point_arrays = [cap_front_points, cap_rear_points]
        filtered_point_arrays = [filter_outlier_points(cap_front_points), filter_outlier_points(cap_rear_points)]

    obs_rtk_dfs = {}
    for obs_name, obs_rtk_dict in obs_data.items():
        obs_front_rtk_df = init_df(obs_rtk_dict['front_rtk'], POS_COLS, '%s_front_rtk.csv' % obs_name, outdir)
        obs_rear_rtk_df = init_df(obs_rtk_dict['rear_rtk'], POS_COLS, '%s_rear_rtk.csv' % obs_name, outdir)
        if not len(obs_rear_rtk_df.index):
            print('Warning: No entries for obstacle %s in %s. Skipping.' % (obs_name, bagset.name))
            continue
        obs_rtk_dfs[obs_name] = {'rear': obs_rear_rtk_df}
        if len(obs_front_rtk_df.index):
            obs_rtk_dfs[obs_name]['front'] = obs_front_rtk_df
        if correct > 0:
            # Use obstacle metadata to determine rtk mounting height and subtract that height
            # from obstacle readings
            md = extract_metadata(bagset.metadata, obs_name)
            if not md:
                print('Error: No metadata found for %s, skipping obstacle.' % obs_name)
                continue
            if len(obs_front_rtk_df.index):
                obs_z_offset = np.array([0., 0., md['front_gps_h']])
                rtk_z_offsets.append(obs_z_offset)
                obs_front_points = obs_front_rtk_df.as_matrix(columns=['tx', 'ty', 'tz']) - obs_z_offset
                point_arrays.append(obs_front_points)
                filtered_point_arrays.append(filter_outlier_points(obs_front_points))
            obs_z_offset = np.array([0., 0., md['rear_gps_h']])
            rtk_z_offsets.append(obs_z_offset)
            obs_rear_points = obs_rear_rtk_df.as_matrix(columns=['tx', 'ty', 'tz']) - obs_z_offset
            point_arrays.append(obs_rear_points)
            filtered_point_arrays.append(filter_outlier_points(obs_rear_points))

    if correct == CORRECT_PLANE:
        points = np.array(np.concatenate(filtered_point_arrays))
        centroid, normal, rotation = fit_plane(
            points, do_plot=True, dataset_outdir=outdir, name=bagset.name)

        def apply_correction(p, z):
            p -= centroid
            p = np.dot(rotation, p.T).T
            c = np.concatenate([centroid[0:2], z[2:]])
            p += c
            return p

        corrected_points = [apply_correction(pa, z) for pa, z in zip(point_arrays, rtk_z_offsets)]
        cap_front_rtk_df.loc[:, ['tx', 'ty', 'tz']] = corrected_points[0]
        cap_rear_rtk_df.loc[:, ['tx', 'ty', 'tz']] = corrected_points[1]
        pts_idx = 2
        for obs_name in obs_rtk_dfs.keys():
            if 'front' in obs_rtk_dfs[obs_name]:
                obs_rtk_dfs[obs_name]['front'].loc[:, ['tx', 'ty', 'tz']] = corrected_points[pts_idx]
                pts_idx += 1
            obs_rtk_dfs[obs_name]['rear'].loc[:, ['tx', 'ty', 'tz']] = corrected_points[pts_idx]
            pts_idx += 1

    interpolate_df(
        cap_front_gps_df, index_df, GPS_COLS, 'cap_front_gps_interp.csv', outdir)
    interpolate_df(
        cap_rear_gps_df, index_df, GPS_COLS, 'cap_rear_gps_interp.csv', outdir)
    cap_front_rtk_interp = interpolate_df(
        cap_front_rtk_df, index_df, POS_COLS, 'cap_front_rtk_interp.csv', outdir)
    cap_rear_rtk_interp = interpolate_df(
        cap_rear_rtk_df, index_df, POS_COLS, 'cap_rear_rtk_interp.csv', outdir)

    if not obs_rtk_dfs:
        print('Warning: No obstacles or obstacle RTK data present. '
              'Skipping Tracklet generation for %s.' % bagset.name)
        return tracklets
    if not bagset.metadata:
        print('Error: No metadata found, metadata.csv file should be with .bag files.'
              'Skipping tracklet generation.')
        return tracklets

    cap_front_rtk_rec = cap_front_rtk_interp.to_dict(orient='records')
    cap_rear_rtk_rec = cap_rear_rtk_interp.to_dict(orient='records')
    for obs_name in obs_rtk_dfs.keys():
        obs_front_rec = {}
        if 'front' in obs_rtk_dfs[obs_name]:
            obs_front_interp = interpolate_df(
                obs_rtk_dfs[obs_name]['front'], index_df, POS_COLS, '%s_front_rtk_interpolated.csv' % obs_name, outdir)
            obs_front_rec = obs_front_interp.to_dict(orient='records')
        obs_rear_interp = interpolate_df(
            obs_rtk_dfs[obs_name]['rear'], index_df, POS_COLS, '%s_rear_rtk_interpolated.csv' % obs_name, outdir)
        obs_rear_rec = obs_rear_interp.to_dict(orient='records')

        # Plot obstacle and front/rear rtk paths in absolute RTK ENU coords
        fig = plt.figure()
        plt.plot(
            cap_front_rtk_interp['tx'].tolist(),
            cap_front_rtk_interp['ty'].tolist(),
            cap_rear_rtk_interp['tx'].tolist(),
            cap_rear_rtk_interp['ty'].tolist(),
            obs_rear_interp['tx'].tolist(),
            obs_rear_interp['ty'].tolist())
        if 'front' in obs_rtk_dfs[obs_name]:
            plt.plot(
                obs_front_interp['tx'].tolist(),
                obs_front_interp['ty'].tolist())
        fig.savefig(os.path.join(outdir, '%s-%s-plot.png' % (bagset.name, obs_name)))
        plt.close(fig)

        # Extract lwh and object type from CSV metadata mapping file
        md = extract_metadata(bagset.metadata, obs_name)

        obs_tracklet = Tracklet(
            object_type=md['object_type'], l=md['l'], w=md['w'], h=md['h'], first_frame=0)

        # NOTE these calculations are done in obstacle oriented coordinates. The LWH offsets from
        # metadata specify offsets from lower left, rear, ground corner of the vehicle. Where +ve is
        # along the respective length, width, height axis away from that point. They are converted to
        # velodyne/ROS compatible X,Y,Z where X +ve is forward, Y +ve is left, and Z +ve is up.
        lrg_to_centroid = [md['l'] / 2., -md['w'] / 2., md['h'] / 2.]
        if 'front' in obs_rtk_dfs[obs_name]:
            lrg_to_front_gps = [md['front_gps_l'], -md['front_gps_w'], md['front_gps_h']]
            gps_to_centroid = np.subtract(lrg_to_centroid, lrg_to_front_gps)
        else:
            lrg_to_rear_gps = [md['rear_gps_l'], -md['rear_gps_w'], md['rear_gps_h']]
            gps_to_centroid = np.subtract(lrg_to_centroid, lrg_to_rear_gps)

        # Convert ENU RTK coords of obstacle to capture vehicle body frame relative coordinates
        if obs_front_rec:
            rtk_coords = zip(cap_front_rtk_rec, cap_rear_rtk_rec, obs_front_rec, obs_rear_rec)
            obs_tracklet.poses = [obstacle_rtk_to_pose(
                c[0], c[1], c[2], c[3],
                gps_to_centroid, FRONT_TO_LIDAR, yaw_err, pitch_err) for c in rtk_coords]
        else:
            rtk_coords = zip(cap_front_rtk_rec, cap_rear_rtk_rec, obs_rear_rec)
            obs_tracklet.poses = [obstacle_rtk_to_pose(
                c[0], c[1], {}, c[2],
                gps_to_centroid, FRONT_TO_LIDAR, yaw_err, pitch_err) for c in rtk_coords]

        tracklets.append(obs_tracklet)
    return tracklets


def process_pose_data(
        bagset,
        cap_data,
        obs_data,
        index_df,
        outdir,
):
    tracklets = []
    cap_pose_df = init_df(cap_data['base_link_pose'], POS_COLS, 'cap_pose.csv', outdir)
    cap_pose_interp = interpolate_df(
        cap_pose_df, index_df, POS_COLS, 'cap_pose_interp.csv', outdir)
    cap_pose_rec = cap_pose_interp.to_dict(orient='records')

    for obs_name, obs_pose_dict in obs_data.items():
        obs_pose_df = init_df(obs_pose_dict['pose'], POS_COLS, 'obs_pose.csv', outdir)
        obs_pose_interp = interpolate_df(
            obs_pose_df, index_df, POS_COLS, 'obs_pose_interp.csv', outdir)
        obs_pose_rec = obs_pose_interp.to_dict(orient='records')

        # Plot obstacle and front/rear rtk paths in absolute RTK ENU coords
        fig = plt.figure()
        plt.plot(
            obs_pose_interp['tx'].tolist(),
            obs_pose_interp['ty'].tolist(),
            cap_pose_interp['tx'].tolist(),
            cap_pose_interp['ty'].tolist())
        fig.savefig(os.path.join(outdir, '%s-%s-plot.png' % (bagset.name, obs_name)))
        plt.close(fig)

        # FIXME hard coded metadata, only Pedestrians currently using pose capture and there is only one person
        md = {'object_type': 'Pedestrian', 'l': 0.8, 'w': 0.8, 'h': 1.708}
        base_link_to_lidar = BASE_LINK_TO_LIDAR_PED

        obs_tracklet = Tracklet(
            object_type=md['object_type'], l=md['l'], w=md['w'], h=md['h'], first_frame=0)

        def _calc_cap_to_obs(cap, obs):
            cap_frame = dict_to_frame(cap)
            obs_frame = dict_to_frame(obs)
            cap_to_obs = cap_frame.Inverse() * obs_frame
            cap_to_obs.p -= list_to_vect(base_link_to_lidar)
            cap_to_obs.p -= kd.Vector(0, 0, md['h'] / 2)
            return frame_to_dict(cap_to_obs, yaw_only=True)

        obs_tracklet.poses = [_calc_cap_to_obs(c[0], c[1]) for c in zip(cap_pose_rec, obs_pose_rec)]
        tracklets.append(obs_tracklet)
    return tracklets

def sync_lidar(camera_index_df, bs, lidar_indir, dataset_outdir):
    if lidar_indir is not None:
        interp_lidar_to_camera(camera_index_df, lidar_indir, bs, dataset_outdir)
    else:
        print('lidar data sync not enabled!!!')

def main():
    parser = argparse.ArgumentParser(description='Convert rosbag to images and csv.')
    parser.add_argument('-o', '--outdir', type=str, nargs='?', default='/output',
        help='Output folder')
    parser.add_argument('-i', '--indir', type=str, nargs='?', default='/data',
        help='Input folder where bagfiles are located')
    # revised by iitwi default from jpg -> png
    parser.add_argument('-f', '--img_format', type=str, nargs='?', default='png',
        help='Image encode format, png or jpg')
    parser.add_argument('-t', '--ts_src', type=str, nargs='?', default='pub',
        help="""Timestamp source. 'pub'=capture node publish time, 'rec'=receiver bag record time,
        'obs_rec'=record time for obstacles topics only, pub for others. Default='pub'""")
    parser.add_argument('-c', '--correct', type=str, nargs='?', default='',
        help="""Correction method. ''=no correction, 'plane'=fit plane to RTK coords and level. Default=''""")
    parser.add_argument('--yaw_err', type=float, nargs='?', default='0.0',
        help="""Amount in degrees to compensate for RTK based yaw measurement. Default=0.0'""")
    parser.add_argument('--pitch_err', type=float, nargs='?', default='0.0',
        help="""Amount in degrees to compensate for RTK based yaw measurement. Default=0.0.""")

    # added by 'in it to win it' (iitwi) team
    parser.add_argument('-pc', '--pointclouddir', type=str, nargs='?', default=None,
                        help='where the velodyne data located')


    parser.add_argument('-m', dest='msg_only', action='store_true', help='Messages only, no images')
    parser.add_argument('-u', dest='unique_paths', action='store_true', help='Unique bag output paths')
    # ittwi:radar
    parser.add_argument('-r', dest='include_radar', action='store_true', help='Include radar data')
    parser.set_defaults(include_radar=False)

    parser.set_defaults(msg_only=False)
    parser.set_defaults(unique_paths=False)
    parser.set_defaults(debug=False)
    args = parser.parse_args()

    img_format = args.img_format
    base_outdir = args.outdir
    indir = args.indir
    ts_src = TS_SRC_PUB
    if args.ts_src == 'rec':
        ts_src = TS_SRC_REC
    elif args.ts_src == 'obs_rec':
        ts_src = TS_SRC_OBS_REC
    correct = CORRECT_NONE
    if args.correct == 'plane':
        correct = CORRECT_PLANE
    yaw_err = args.yaw_err * np.pi / 180
    pitch_err = args.pitch_err * np.pi / 180
    msg_only = args.msg_only
    unique_paths = args.unique_paths
    image_bridge = ImageBridge()

    # added by iitwi team
    lidar_indir = args.pointclouddir

    include_images = False if msg_only else True

    #ittwi:radar
    include_radar = args.include_radar
    

    filter_topics = CAMERA_TOPICS + CAP_FRONT_RTK_TOPICS + CAP_REAR_RTK_TOPICS \
        + CAP_FRONT_GPS_TOPICS + CAP_REAR_GPS_TOPICS + RADAR_TOPICS

    # FIXME hard coded obstacles
    # The original intent was to scan bag info for obstacles and populate dynamically in combination
    # with metadata.csv. Since obstacle names were very static, and the obstacle topic root was not consistent
    # between data releases, that didn't happen.
    obstacle_topics = []

    # For obstacles tracked via RTK messages
    OBS_RTK_NAMES = ['obs1']
    OBS_FRONT_RTK_TOPICS = [OBJECTS_TOPIC_ROOT + '/' + x + '/front/gps/rtkfix' for x in OBS_RTK_NAMES]
    OBS_REAR_RTK_TOPICS = [OBJECTS_TOPIC_ROOT + '/' + x + '/rear/gps/rtkfix' for x in OBS_RTK_NAMES]
    obstacle_topics += OBS_FRONT_RTK_TOPICS
    obstacle_topics += OBS_REAR_RTK_TOPICS

    # For obstacles tracked via TF + pose messages
    OBS_POSE_TOPICS = ['/obstacle/ped/pose']  # not under same root as other obstacles for some reason
    obstacle_topics += OBS_POSE_TOPICS
    filter_topics += [TF_TOPIC]  # pose based obstacles rely on TF

    filter_topics += obstacle_topics

    bagsets = find_bagsets(indir, filter_topics=filter_topics, set_per_file=True, metadata_filename='metadata.csv')
    if not bagsets:
        print("No bags found in %s" % indir)
        exit(-1)

    for bs in bagsets:
        print("Processing set %s" % bs.name)
        sys.stdout.flush()

        cap_data = defaultdict(lambda: defaultdict(list))
        obs_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        outdir = os.path.join(base_outdir, bs.get_name(unique_paths))
        get_outdir(outdir)
        if include_images:
            camera_outdir = get_outdir(outdir, "image_02/data")
        #ittwi:radar
        if include_radar:
            radar_outdir = get_outdir(outdir, "radar/data")
        bs.write_infos(outdir)
        readers = bs.get_readers()
        stats_acc = defaultdict(int)

        def _process_msg(topic, msg, ts_recorded, stats):
            if topic == '/tf':
                timestamp = msg.transforms[0].header.stamp.to_nsec()
            else:
                timestamp = msg.header.stamp.to_nsec()  # default to publish timestamp in message header
            if ts_src == TS_SRC_REC:
                timestamp = ts_recorded.to_nsec()
            elif ts_src == TS_SRC_OBS_REC and topic in obstacle_topics:
                timestamp = ts_recorded.to_nsec()

            if topic in CAMERA_TOPICS:
                write_results = {}
                if include_images:
                    write_results = image_bridge.write_image(camera_outdir, msg, fmt=img_format)
                    write_results['filename'] = os.path.relpath(write_results['filename'], outdir)
                camera2dict(timestamp, msg, write_results, cap_data['camera'])
                stats['img_count'] += 1
                stats['msg_count'] += 1

            elif topic in RADAR_TOPICS:
                radar_filename=None
                if include_radar:
                    radar_filename = writeRadar(msg, radar_outdir)
                    radar_filename = os.path.relpath(radar_filename, outdir)
                radar2dict(timestamp, radar_filename, cap_data['radar'])
                stats['msg_count'] += 1
                

            elif topic in CAP_REAR_RTK_TOPICS:
                pose2dict(timestamp, msg.pose, cap_data['rear_rtk'])
                stats['msg_count'] += 1

            elif topic in CAP_FRONT_RTK_TOPICS:
                pose2dict(timestamp, msg.pose, cap_data['front_rtk'])
                stats['msg_count'] += 1

            elif topic in CAP_REAR_GPS_TOPICS:
                gps2dict(timestamp, msg, cap_data['rear_gps'])
                stats['msg_count'] += 1

            elif topic in CAP_FRONT_GPS_TOPICS:
                gps2dict(timestamp, msg, cap_data['front_gps'])
                stats['msg_count'] += 1

            elif topic in OBS_REAR_RTK_TOPICS:
                name = obs_name_from_topic(topic)
                pose2dict(timestamp, msg.pose, obs_data[name]['rear_rtk'])
                stats['msg_count'] += 1

            elif topic in OBS_FRONT_RTK_TOPICS:
                name = obs_name_from_topic(topic)
                pose2dict(timestamp, msg.pose, obs_data[name]['front_rtk'])
                stats['msg_count'] += 1

            elif topic == TF_TOPIC:
                for t in msg.transforms:
                    if t.child_frame_id == '/base_link':
                        tf2dict(timestamp, t.transform, cap_data['base_link_pose'])

            elif topic in OBS_POSE_TOPICS:
                name = obs_name_from_topic(topic)
                pose2dict(timestamp, msg, obs_data[name]['pose'])
                stats['msg_count'] += 1

            else:
                pass

        for reader in readers:
            last_img_log = 0
            last_msg_log = 0
            for result in reader.read_messages():
                _process_msg(*result, stats=stats_acc)
                if last_img_log != stats_acc['img_count'] and stats_acc['img_count'] % 1000 == 0:
                    print("%d images, processed..." % stats_acc['img_count'])
                    last_img_log = stats_acc['img_count']
                    sys.stdout.flush()
                if last_msg_log != stats_acc['msg_count'] and stats_acc['msg_count'] % 10000 == 0:
                    print("%d messages processed..." % stats_acc['msg_count'])
                    last_msg_log = stats_acc['msg_count']
                    sys.stdout.flush()

        print("Writing done. %d images, %d messages processed." %
              (stats_acc['img_count'], stats_acc['msg_count']))
        sys.stdout.flush()

        camera_df = pd.DataFrame(data=cap_data['camera'], columns=CAMERA_COLS)
        if include_images:
            camera_df.to_csv(os.path.join(outdir, 'cap_camera.csv'), index=False)

        #ittwi : radar
        radar_df = pd.DataFrame(data=cap_data['radar'], columns=RADAR_COLS)
        if include_radar:
            radar_df.to_csv(os.path.join(outdir, 'cap_radar.csv'), index=False)

        if len(camera_df['timestamp']):
            # Interpolate samples from all used sensors to camera frame timestamps
            camera_df['timestamp'] = pd.to_datetime(camera_df['timestamp'])
            camera_df.set_index(['timestamp'], inplace=True)
            camera_df.index.rename('index', inplace=True)
            camera_index_df = pd.DataFrame(index=camera_df.index)

            collection = TrackletCollection()

            if 'front_rtk' in cap_data and 'rear_rtk' in cap_data:
                tracklets = process_rtk_data(
                    bs, cap_data, obs_data, camera_index_df, outdir,
                    correct=correct, yaw_err=yaw_err, pitch_err=pitch_err)
                collection.tracklets += tracklets

            if 'base_link_pose' in cap_data:
                tracklets = process_pose_data(
                    bs, cap_data, obs_data, camera_index_df, outdir)
                collection.tracklets += tracklets

            if collection.tracklets:
                tracklet_path = os.path.join(outdir, 'tracklet_labels.xml')
                collection.write_xml(tracklet_path)

            # added by iitwi team
            sync_lidar(camera_index_df, bs, lidar_indir, outdir)
        else:
            print('Warning: No camera image times were found. '
                  'Skipping sensor interpolation and Tracklet generation.')


if __name__ == '__main__':
    main()
