import os
import glob
import numpy as np
from collections import defaultdict
import shutil
from ..bag_to_kitti import *


def get_pitch_local(p1,p2):
    p1=np.array([p1[0],p1[1],p1[2]])
    p2=np.array([p2[0],p2[1],p2[2]])
    delta_heigth=p1[2] - p2[2]-0.3556 # 0.3556 = front_z-rear_z,refrence tf.launch
    return -np.arcsin(delta_heigth/LA.norm(p1-p2))

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

def get_obstacle_pos_local(
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

    pitch = get_pitch_local(front_v, rear_v)
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


def estimate_obstacle_poses_local(
    cap_front_rtk,
    cap_rear_rtk,
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
        get_obstacle_pos_local(c[0], c[1], c[2], velo_to_front, obs_rear_gps_offset) for c in rtk_coords]

    return output_poses

# def choose_udacity_or_local_correction(local_correction):
#     if local_correction:
#         return estimate_obstacle_poses_local
#     else:
#         return estimate_obstacle_poses


def syc_rtk_timestamp_to_camera(obs_rtk,camera):
    for k in obs_rtk:
        if len(obs_rtk[k]['timestamp']) != 0:
            offset = obs_rtk[k]['timestamp'][0] - camera['timestamp'][0]
            obs_rtk[k]['timestamp'] = [t - offset for t in obs_rtk[k]['timestamp']]

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


def sync_lidar(camera_index_df, bs, lidar_indir, dataset_outdir):
    if lidar_indir is not None:
        interp_lidar_to_camera(camera_index_df, lidar_indir, bs, dataset_outdir)
    else:
        print('lidar data sync not enabled!!!')