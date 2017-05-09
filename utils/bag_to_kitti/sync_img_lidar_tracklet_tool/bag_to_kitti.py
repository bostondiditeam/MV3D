#! /usr/bin/python
""" Udacity Self-Driving Car Challenge Bag Processing
"""
from __future__ import print_function

from cv_bridge import CvBridge, CvBridgeError
from collections import defaultdict
import os
import sys
import cv2
import math
import imghdr
import argparse
import functools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PyKDL as kd
import glob
import shutil
from numpy import linalg as LA

from bag_topic_def import *
from bag_utils import *
from generate_tracklet import *


def get_outdir(base_dir, name=''):
    outdir = os.path.join(base_dir, name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def obs_prefix_from_topic(topic):
    words = topic.split('/')
    start, end = (1, 4) if topic.startswith(OBJECTS_TOPIC_ROOT) else (1, 3)
    prefix = '_'.join(words[start:end])
    name = words[2] if topic.startswith(OBJECTS_TOPIC_ROOT) else words[1]
    return prefix, name


def check_format(data):
    img_fmt = imghdr.what(None, h=data)
    return 'jpg' if img_fmt == 'jpeg' else img_fmt


def write_image(bridge, outdir, msg, fmt='png'):
    results = {}
    image_filename = os.path.join(outdir, str(msg.header.stamp.to_nsec()) + '.' + fmt)
    try:
        if hasattr(msg, 'format') and 'compressed' in msg.format:
            buf = np.ndarray(shape=(1, len(msg.data)), dtype=np.uint8, buffer=msg.data)
            cv_image = cv2.imdecode(buf, cv2.IMREAD_ANYCOLOR)
            if cv_image.shape[2] != 3:
                print("Invalid image %s" % image_filename)
                return results
            results['height'] = cv_image.shape[0]
            results['width'] = cv_image.shape[1]
            # Avoid re-encoding if we don't have to
            if check_format(msg.data) == fmt:
                buf.tofile(image_filename)
            else:
                cv2.imwrite(image_filename, cv_image)
        else:
            cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
            cv2.imwrite(image_filename, cv_image)
    except CvBridgeError as e:
        print(e)
    results['filename'] = image_filename
    return results


def camera2dict(msg, write_results, camera_dict):
    camera_dict["timestamp"].append(msg.header.stamp.to_nsec())
    if write_results:
        camera_dict["width"].append(write_results['width'] if 'width' in write_results else msg.width)
        camera_dict['height'].append(write_results['height'] if 'height' in write_results else msg.height)
        camera_dict["frame_id"].append(msg.header.frame_id)
        camera_dict["filename"].append(write_results['filename'])


def gps2dict(msg, gps_dict):
    gps_dict["timestamp"].append(msg.header.stamp.to_nsec())
    gps_dict["lat"].append(msg.latitude)
    gps_dict["long"].append(msg.longitude)
    gps_dict["alt"].append(msg.altitude)



def rtk2dict(msg, rtk_dict):
    rtk_dict["timestamp"].append(msg.header.stamp.to_nsec())
    rtk_dict["tx"].append(msg.pose.pose.position.x)
    rtk_dict["ty"].append(msg.pose.pose.position.y)
    rtk_dict["tz"].append(msg.pose.pose.position.z)
    rotq = kd.Rotation.Quaternion(
        msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w)
    rot_xyz = rotq.GetRPY()
    rtk_dict["rx"].append(0.0) #rot_xyz[0]
    rtk_dict["ry"].append(0.0) #rot_xyz[1]
    rtk_dict["rz"].append(rot_xyz[2])


def imu2dict(msg, imu_dict):
    imu_dict["timestamp"].append(msg.header.stamp.to_nsec())
    imu_dict["ax"].append(msg.linear_acceleration.x)
    imu_dict["ay"].append(msg.linear_acceleration.y)
    imu_dict["az"].append(msg.linear_acceleration.z)


def get_yaw(p1, p2):
    return math.atan2(p1[1] - p2[1], p1[0] - p2[0])

def get_pitch(p1,p2):
    p1=np.array([p1[0],p1[1],p1[2]])
    p2=np.array([p2[0],p2[1],p2[2]])
    delta_heigth=p1[2] - p2[2]-0.3556 # 0.3556 = front_z-rear_z,refrence tf.launch
    return -np.arcsin(delta_heigth/LA.norm(p1-p2))


def dict_to_vect(di):
    return kd.Vector(di['tx'], di['ty'], di['tz'])


def list_to_vect(li):
    return kd.Vector(li[0], li[1], li[2])


def frame_to_dict(frame):
    r, p, y = frame.M.GetRPY()
    return dict(tx=frame.p[0], ty=frame.p[1], tz=frame.p[2], rx=r, ry=p, rz=y)


def get_obstacle_pos(
        front,
        rear,
        obstacle,
        velodyne_to_front,
        gps_to_centroid):
    front_v = dict_to_vect(front)
    rear_v = dict_to_vect(rear)
    obs_v = dict_to_vect(obstacle)

    yaw = get_yaw(front_v, rear_v)
    rot_z = kd.Rotation.RotZ(-yaw)

    pitch = get_pitch(front_v, rear_v)
    rot_y = kd.Rotation.RotY(-pitch)

    diff = obs_v - front_v
    res = rot_y * rot_z * diff
    res += list_to_vect(velodyne_to_front)

    # FIXME the gps_to_centroid offset of the obstacle should be rotated by
    # the obstacle's yaw. Unfortunately the obstacle's pose is unknown at this
    # point so we will assume obstacle is axis aligned with capture vehicle
    # for now.
    res += list_to_vect(gps_to_centroid)

    return frame_to_dict(kd.Frame(kd.Rotation(), res))


def interpolate_to_camera(camera_df, other_dfs, filter_cols=[]):
    if not isinstance(other_dfs, list):
        other_dfs = [other_dfs]
    if not isinstance(camera_df.index, pd.DatetimeIndex):
        print('Error: Camera dataframe needs to be indexed by timestamp for interpolation')
        return pd.DataFrame()

    for o in other_dfs:
        # print("datatype here: ", o['timestamp'].dtype)
        o['timestamp'] = pd.to_datetime(o['timestamp'])
        o.set_index(['timestamp'], inplace=True)
        o.index.rename('index', inplace=True)
    # print("camera_df is here: ", [camera_df])
    # print("other_df is here: ", other_dfs)
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


def estimate_obstacle_poses(
    cap_front_rtk,
    #cap_front_gps_offset,
    cap_rear_rtk,
    #cap_rear_gps_offset,
    obs_rear_rtk,
    obs_rear_gps_offset,  # offset along [l, w, h] dim of car, in obstacle relative coords
):
    # offsets are all [l, w, h] lists (or tuples)
    assert(len(obs_rear_gps_offset) == 3)
    # all coordinate records should be interpolated to same sample base at this point
    assert len(cap_front_rtk) == len(cap_rear_rtk) == len(obs_rear_rtk)

    velo_to_front = [-1.0922, 0, -0.0508]
    rtk_coords = zip(cap_front_rtk, cap_rear_rtk, obs_rear_rtk)
    output_poses = [
        get_obstacle_pos(c[0], c[1], c[2], velo_to_front, obs_rear_gps_offset) for c in rtk_coords]

    return output_poses


def check_oneof_topics_present(topic_map, name, topics):
    if not isinstance(topics, list):
        topics = [topics]
    if not any(t in topic_map for t in topics):
        print('Error: One of %s must exist in bag, skipping bag %s.' % (topics, name))
        return False
    return True

def syc_rtk_timestamp_to_camera(obs_rtk,camera):
    for k in obs_rtk:
        if len(obs_rtk[k]['timestamp']) != 0:
            offset = obs_rtk[k]['timestamp'][0] - camera['timestamp'][0]
            obs_rtk[k]['timestamp'] = [t - offset for t in obs_rtk[k]['timestamp']]

def main():
    parser = argparse.ArgumentParser(description='Convert rosbag to images and csv.')
    parser.add_argument('-o', '--outdir', type=str, nargs='?', default='/output',
        help='Output folder')
    parser.add_argument('-i', '--indir', type=str, nargs='?', default='/data',
        help='Input folder where bagfiles are located')
    parser.add_argument('-pc', '--pointclouddir', type=str, nargs='?', default='/data',
                        help='where the velodyne data located')
    parser.add_argument('-f', '--img_format', type=str, nargs='?', default='png',
        help='Image encode format, png or jpg')
    parser.add_argument('-m', dest='msg_only', action='store_true', help='Messages only, no images')
    parser.add_argument('-nolidar', dest='nolidar', action='store_true', help='No lidar data syc')
    parser.add_argument('-d', dest='debug', action='store_true', help='Debug print enable')
    parser.set_defaults(msg_only=False)
    parser.set_defaults(debug=False)
    args = parser.parse_args()

    img_format = args.img_format
    base_outdir = args.outdir
    indir = args.indir
    lidar_indir = args.pointclouddir

    msg_only = args.msg_only
    debug_print = args.debug

    nolidar=args.nolidar

    bridge = CvBridge()

    include_images = False if msg_only else True

    filter_topics = CAMERA_TOPICS + CAP_FRONT_RTK_TOPICS + CAP_REAR_RTK_TOPICS \
        + CAP_FRONT_GPS_TOPICS + CAP_REAR_GPS_TOPICS

    # For bag sets that may have missing metadata.csv file
    default_metadata = [{
        'obstacle_name': 'obs1',
        'object_type': 'Car',
        'gps_l': 2.032,
        'gps_w': 1.4478,
        'gps_h': 1.6256,
        'l': 4.2418,
        'w': 1.4478,
        'h': 1.5748,
    }]

    #FIXME scan from bag info in /obstacles/ topic path
    OBSTACLES = ['obs1']
    OBSTACLE_RTK_TOPICS = [OBJECTS_TOPIC_ROOT + '/' + x + '/rear/gps/rtkfix' for x in OBSTACLES]
    filter_topics += OBSTACLE_RTK_TOPICS
    print("Filter topics are here: ", filter_topics)

    bagsets = find_bagsets(indir, filter_topics=filter_topics, set_per_file=True, metadata_filename='metadata.csv')
    print("bagsets path here: ", bagsets)
    if not bagsets:
        print("No bags found in %s" % indir)
        exit(-1)

    for bs in bagsets:
        print("Processing set %s" % bs.name)
        sys.stdout.flush()

        if not check_oneof_topics_present(bs.topic_map, bs.name, CAP_FRONT_RTK_TOPICS):
            continue
        if not check_oneof_topics_present(bs.topic_map, bs.name, CAP_REAR_RTK_TOPICS):
            continue

        camera_cols = ["timestamp", "width", "height", "frame_id", "filename"]
        camera_dict = defaultdict(list)

        lidar_cols = ["timestamp", "filename"]
        lidar_dict = defaultdict(list)

        gps_cols = ["timestamp", "lat", "long", "alt"]
        cap_rear_gps_dict = defaultdict(list)
        cap_front_gps_dict = defaultdict(list)

        rtk_cols = ["timestamp", "tx", "ty", "tz", "rx", "ry", "rz"]
        cap_rear_rtk_dict = defaultdict(list)
        cap_front_rtk_dict = defaultdict(list)

        # For the obstacles, keep track of rtk values for each one in a dictionary (key == topic)
        obstacle_rtk_dicts = {k: defaultdict(list) for k in OBSTACLE_RTK_TOPICS}

        dataset_outdir = os.path.join(base_outdir, "%s" % bs.name)
        get_outdir(dataset_outdir)
        if include_images:
            camera_outdir = get_outdir(dataset_outdir, "image_02/data")
        # todo change it into if for production
        lidar_outdir = get_outdir(dataset_outdir, "velodyne_points/data")
        bs.write_infos(dataset_outdir)
        readers = bs.get_readers()
        stats_acc = defaultdict(int)

        def _process_msg(topic, msg, stats):
            timestamp = msg.header.stamp.to_nsec()
            if topic in CAMERA_TOPICS:
                if debug_print:
                    print("%s_camera %d" % (topic[1], timestamp))

                write_results = {}
                if include_images:
                    write_results = write_image(bridge, camera_outdir, msg, fmt=img_format)
                    write_results['filename'] = os.path.relpath(write_results['filename'], dataset_outdir)
                camera2dict(msg, write_results, camera_dict)
                stats['img_count'] += 1
                stats['msg_count'] += 1

            elif topic in CAP_REAR_RTK_TOPICS:
                rtk2dict(msg, cap_rear_rtk_dict)
                stats['msg_count'] += 1

            elif topic in CAP_FRONT_RTK_TOPICS:
                rtk2dict(msg, cap_front_rtk_dict)
                stats['msg_count'] += 1

            elif topic in CAP_REAR_GPS_TOPICS:
                gps2dict(msg, cap_rear_gps_dict)
                stats['msg_count'] += 1

            elif topic in CAP_FRONT_GPS_TOPICS:
                gps2dict(msg, cap_front_gps_dict)
                stats['msg_count'] += 1

            elif topic in OBSTACLE_RTK_TOPICS:
                rtk2dict(msg, obstacle_rtk_dicts[topic])
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

        #todo : remove it after udacity fixed this bug
        syc_rtk_timestamp_to_camera(obstacle_rtk_dicts, camera_dict)

        # take all lidar cloud point txt here and construct a dataframe
        velodyne_dir = os.path.join(dataset_outdir, 'velodyne_points')

        def read_txt_mapping():
            txt_path = velodyne_dir + "/*.txt"
            velodyne_txt_list = glob.glob(velodyne_dir)



        camera_df = pd.DataFrame(data=camera_dict, columns=camera_cols)
        cap_rear_gps_df = pd.DataFrame(data=cap_rear_gps_dict, columns=gps_cols)
        cap_front_gps_df = pd.DataFrame(data=cap_front_gps_dict, columns=gps_cols)
        cap_rear_rtk_df = pd.DataFrame(data=cap_rear_rtk_dict, columns=rtk_cols)
        if not len(cap_rear_rtk_df.index):
            print('Error: No capture vehicle rear RTK entries exist.'
                  'Skipping bag %s.' % bs.name)
            continue
        cap_front_rtk_df = pd.DataFrame(data=cap_front_rtk_dict, columns=rtk_cols)
        if not len(cap_rear_rtk_df.index):
            print('Error: No capture vehicle front RTK entries exist.'
                  'Skipping bag %s.' % bs.name)
            continue

        if include_images:
            camera_df.to_csv(os.path.join(dataset_outdir, 'capture_vehicle_camera.csv'), index=False)
        cap_rear_gps_df.to_csv(os.path.join(dataset_outdir, 'capture_vehicle_rear_gps.csv'), index=False)
        cap_front_gps_df.to_csv(os.path.join(dataset_outdir, 'capture_vehicle_front_gps.csv'), index=False)
        cap_rear_rtk_df.to_csv(os.path.join(dataset_outdir, 'capture_vehicle_rear_rtk.csv'), index=False)
        cap_front_rtk_df.to_csv(os.path.join(dataset_outdir, 'capture_vehicle_front_rtk.csv'), index=False)

        obs_rtk_df_dict = {}
        for obs_topic, obs_rtk_dict in obstacle_rtk_dicts.items():
            obs_prefix, obs_name = obs_prefix_from_topic(obs_topic)
            obs_rtk_df = pd.DataFrame(data=obs_rtk_dict, columns=rtk_cols)
            if not len(obs_rtk_df.index):
                print('Warning: No entries for obstacle %s in %s. Skipping.' % (obs_name, bs.name))
                continue
            obs_rtk_df.to_csv(os.path.join(dataset_outdir, '%s_rtk.csv' % obs_prefix), index=False)
            obs_rtk_df_dict[obs_topic] = obs_rtk_df

        if len(camera_dict['timestamp']):
            # Interpolate samples from all used sensors to camera frame timestamps
            camera_df['timestamp'] = pd.to_datetime(camera_df['timestamp'])
            print("the camera dataframe is here: ", camera_df)
            camera_df.set_index(['timestamp'], inplace=True)
            camera_df.index.rename('index', inplace=True)

            camera_index_df = pd.DataFrame(index=camera_df.index)

            cap_rear_gps_interp = interpolate_to_camera(camera_index_df, cap_rear_gps_df, filter_cols=gps_cols)
            cap_rear_gps_interp.to_csv(
                os.path.join(dataset_outdir, 'capture_vehicle_rear_gps_interp.csv'), header=True)

            cap_front_gps_interp = interpolate_to_camera(camera_index_df, cap_front_gps_df, filter_cols=gps_cols)
            cap_front_gps_interp.to_csv(
                os.path.join(dataset_outdir, 'capture_vehicle_front_gps_interp.csv'), header=True)

            cap_rear_rtk_interp = interpolate_to_camera(camera_index_df, cap_rear_rtk_df, filter_cols=rtk_cols)
            cap_rear_rtk_interp.to_csv(
                os.path.join(dataset_outdir, 'capture_vehicle_rear_rtk_interp.csv'), header=True)
            cap_rear_rtk_interp_rec = cap_rear_rtk_interp.to_dict(orient='records')

            cap_front_rtk_interp = interpolate_to_camera(camera_index_df, cap_front_rtk_df, filter_cols=rtk_cols)
            cap_front_rtk_interp.to_csv(
                os.path.join(dataset_outdir, 'capture_vehicle_front_rtk_interp.csv'), header=True)
            cap_front_rtk_interp_rec = cap_front_rtk_interp.to_dict(orient='records')

            if nolidar==False:
                # if corresponding velodyne directory exists, calibrate them and save it in output directory.
                print("I'm here")
                # lidar_indir = os.listdir(lidar_indir)
                if bs.name in os.listdir(lidar_indir):
                    lidar_dir = os.path.join(lidar_indir, bs.name, "velodyne_points")
                    lidar_dir = os.path.join(lidar_indir, bs.name)
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
                        source_lidar_file_path = os.path.join(lidar_dir, str(row['filename'])+'.bin')
                        dest_lidar_file_path = os.path.join(lidar_outdir, str(row['timestamp'])+'.bin')

                        # print("lidar outout here: ", dest_lidar_file_path)
                        shutil.copy(source_lidar_file_path, dest_lidar_file_path)


                    print("input lidar dir is here: ", lidar_dir)
                    print("output lidar dir is here: ", lidar_outdir)
            else:
                print('lidar data sycn disable!!!')

            if not obs_rtk_df_dict:
                print('Warning: No obstacles or obstacle RTK data present. '
                      'Skipping Tracklet generation for %s.' % bs.name)
                continue

            collection = TrackletCollection()
            for obs_topic in obstacle_rtk_dicts.keys():
                obs_rtk_df = obs_rtk_df_dict[obs_topic]
                obs_interp = interpolate_to_camera(camera_index_df, obs_rtk_df, filter_cols=rtk_cols)
                obs_prefix, obs_name = obs_prefix_from_topic(obs_topic)
                obs_interp.to_csv(
                    os.path.join(dataset_outdir, '%s_rtk_interpolated.csv' % obs_prefix), header=True)

                # Plot obstacle and front/rear rtk paths in absolute RTK ENU coords
                fig = plt.figure()
                plt.plot(
                    obs_interp['tx'].tolist(),
                    obs_interp['ty'].tolist(),
                    cap_front_rtk_interp['tx'].tolist(),
                    cap_front_rtk_interp['ty'].tolist(),
                    cap_rear_rtk_interp['tx'].tolist(),
                    cap_rear_rtk_interp['ty'].tolist())
                fig.savefig(os.path.join(dataset_outdir, '%s-%s-plot.png' % (bs.name, obs_name)))
                plt.close(fig)

                # Extract lwh and object type from CSV metadata mapping file
                md = bs.metadata if bs.metadata else default_metadata
                if not bs.metadata:
                    print('Warning: Default metadata used, metadata.csv file should be with .bag files.')
                for x in md:
                    if x['obstacle_name'] == obs_name:
                        mdr = x

                obs_tracklet = Tracklet(
                    object_type=mdr['object_type'], l=mdr['l'], w=mdr['w'], h=mdr['h'], first_frame=0)

                # NOTE these calculations are done in obstacle oriented coordinates. The LWH offsets from
                # metadata specify offsets from lower left, rear, ground corner of the vehicle. Where +ve is
                # along the respective length, width, height axis away from that point. They are converted to
                # velodyne/ROS compatible X,Y,Z where X +ve is forward, Y +ve is left, and Z +ve is up.
                lrg_to_gps = [mdr['gps_l'], -mdr['gps_w'], mdr['gps_h']]
                lrg_to_centroid = [mdr['l'] / 2., -mdr['w'] / 2., mdr['h'] / 2.]
                gps_to_centroid = np.subtract(lrg_to_centroid, lrg_to_gps)

                # Convert NED RTK coords of obstacle to capture vehicle body frame relative coordinates
                obs_tracklet.poses = estimate_obstacle_poses(
                    cap_front_rtk=cap_front_rtk_interp_rec,
                    #cap_front_gps_offset=[0.0, 0.0, 0.0],
                    cap_rear_rtk=cap_rear_rtk_interp_rec,
                    #cap_rear_gps_offset=[0.0, 0.0, 0.0],
                    obs_rear_rtk=obs_interp.to_dict(orient='records'),
                    obs_rear_gps_offset=gps_to_centroid,
                )

                collection.tracklets.append(obs_tracklet)
                # end for obs_topic loop

            tracklet_path = os.path.join(dataset_outdir, 'tracklet_labels.xml')
            collection.write_xml(tracklet_path)
        else:
            print('Warning: No camera image times were found. '
                  'Skipping sensor interpolation and Tracklet generation.')

if __name__ == '__main__':
    main()
