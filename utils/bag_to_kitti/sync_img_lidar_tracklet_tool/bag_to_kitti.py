import os
import sys
user_path = os.environ['PYTHONPATH']
#print('python path here: ', user_path)
sys.path.append(os.path.join(sys.path[0],"../../external_modules/didi-competition"))
CORRECT_BASIC = None

from tracklets.python.bag_to_kitti import *
# import patches has to be under 'from tracklet.python.bag_to_kitti import *' since there are function overwritten
from patches.funcs import *

def main():
    parser = argparse.ArgumentParser(description='Convert rosbag to images and csv.')
    # added by Udacity
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
        help="""Correction method. ''=no correction, 'plane'=fit plane to RTK coords and level.
        Default=''""")
    parser.add_argument('-m', dest='msg_only', action='store_true', help='Messages only, no images')
    parser.add_argument('-d', dest='debug', action='store_true', help='Debug print enable')


    # added by 'in it to win it' (iitwi) team
    parser.add_argument('-pc', '--pointclouddir', type=str, nargs='?', default=None,
                        help='where the velodyne data located')
    parser.add_argument('-l', '--local_correction', type=str, nargs='?', default=False,
                        help='if set to True, local correction will be used for tracklet contents, otherwise use '
                             '--correction provided by Udacity')


    parser.set_defaults(msg_only=False)
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
    elif args.correct == 'basic':
        correct = CORRECT_BASIC
    msg_only = args.msg_only
    debug_print = args.debug
    local_correction = args.local_correction

    # added by iitwi team
    lidar_indir = args.pointclouddir

    bridge = CvBridge()

    include_images = False if msg_only else True

    filter_topics = CAMERA_TOPICS + CAP_FRONT_RTK_TOPICS + CAP_REAR_RTK_TOPICS \
        + CAP_FRONT_GPS_TOPICS + CAP_REAR_GPS_TOPICS

    #FIXME scan from bag info in /obstacles/ topic path
    OBSTACLES = ['obs1']
    OBSTACLE_RTK_TOPICS = [OBJECTS_TOPIC_ROOT + '/' + x + '/rear/gps/rtkfix' for x in OBSTACLES]
    filter_topics += OBSTACLE_RTK_TOPICS

    bagsets = find_bagsets(indir, filter_topics=filter_topics, set_per_file=True, metadata_filename='metadata.csv')
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

        gps_cols = ["timestamp", "lat", "long", "alt"]
        cap_rear_gps_dict = defaultdict(list)
        cap_front_gps_dict = defaultdict(list)

        rtk_cols = ["timestamp", "tx", "ty", "tz", "rx", "ry", "rz"]
        cap_rear_rtk_dict = defaultdict(list)
        cap_front_rtk_dict = defaultdict(list)

        # For the obstacles, keep track of rtk values for each one in a dictionary (key == topic)
        obstacle_rtk_dicts = {k: defaultdict(list) for k in OBSTACLE_RTK_TOPICS}


        # added by iitwi team.
        # todo : remove it after udacity fixed this bug
        syc_rtk_timestamp_to_camera(obstacle_rtk_dicts, camera_dict)


        dataset_outdir = os.path.join(base_outdir, "%s" % bs.name)
        get_outdir(dataset_outdir)
        if include_images:
            # changed by iitwi.
            camera_outdir = get_outdir(dataset_outdir, "image_02/data")
        bs.write_infos(dataset_outdir)
        readers = bs.get_readers()
        stats_acc = defaultdict(int)

        def _process_msg(topic, msg, ts_recorded, stats):
            timestamp = msg.header.stamp.to_nsec()  # default to publish timestamp in message header
            if ts_src == TS_SRC_REC:
                timestamp = ts_recorded.to_nsec()
            elif ts_src == TS_SRC_OBS_REC and topic in OBSTACLE_RTK_TOPICS:
                timestamp = ts_recorded.to_nsec()

            if topic in CAMERA_TOPICS:
                if debug_print:
                    print("%s_camera %d" % (topic[1], timestamp))

                write_results = {}
                if include_images:
                    write_results = write_image(bridge, camera_outdir, msg, fmt=img_format)
                    write_results['filename'] = os.path.relpath(write_results['filename'], dataset_outdir)
                camera2dict(timestamp, msg, write_results, camera_dict)
                stats['img_count'] += 1
                stats['msg_count'] += 1

            elif topic in CAP_REAR_RTK_TOPICS:
                rtk2dict(timestamp, msg, cap_rear_rtk_dict)
                stats['msg_count'] += 1

            elif topic in CAP_FRONT_RTK_TOPICS:
                rtk2dict(timestamp, msg, cap_front_rtk_dict)
                stats['msg_count'] += 1

            elif topic in CAP_REAR_GPS_TOPICS:
                gps2dict(timestamp, msg, cap_rear_gps_dict)
                stats['msg_count'] += 1

            elif topic in CAP_FRONT_GPS_TOPICS:
                gps2dict(timestamp, msg, cap_front_gps_dict)
                stats['msg_count'] += 1

            elif topic in OBSTACLE_RTK_TOPICS:
                rtk2dict(timestamp, msg, obstacle_rtk_dicts[topic])
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

        camera_df = pd.DataFrame(data=camera_dict, columns=camera_cols)
        cap_rear_gps_df = pd.DataFrame(data=cap_rear_gps_dict, columns=gps_cols)
        cap_front_gps_df = pd.DataFrame(data=cap_front_gps_dict, columns=gps_cols)
        cap_rear_rtk_df = pd.DataFrame(data=cap_rear_rtk_dict, columns=rtk_cols)
        if not len(cap_rear_rtk_df.index):
            print('Error: No capture vehicle rear RTK entries exist.'
                  'Skipping bag %s.' % bag.name)
            continue
        cap_front_rtk_df = pd.DataFrame(data=cap_front_rtk_dict, columns=rtk_cols)
        if not len(cap_rear_rtk_df.index):
            print('Error: No capture vehicle front RTK entries exist.'
                  'Skipping bag %s.' % bag.name)
            continue

        if include_images:
            camera_df.to_csv(os.path.join(dataset_outdir, 'capture_vehicle_camera.csv'), index=False)
        cap_rear_gps_df.to_csv(os.path.join(dataset_outdir, 'capture_vehicle_rear_gps.csv'), index=False)
        cap_front_gps_df.to_csv(os.path.join(dataset_outdir, 'capture_vehicle_front_gps.csv'), index=False)
        cap_rear_rtk_df.to_csv(os.path.join(dataset_outdir, 'capture_vehicle_rear_rtk.csv'), index=False)
        cap_front_rtk_df.to_csv(os.path.join(dataset_outdir, 'capture_vehicle_front_rtk.csv'), index=False)

        rtk_z_offsets = [
            np.array([0., 0., CAP_RTK_FRONT_Z]),
            np.array([0., 0., CAP_RTK_REAR_Z])]
        if correct > 0:
            # Correction algorithm attempts to fit plane to rtk measurements across both capture rtk
            # units and all obstacles. We will subtract known RTK unit mounting heights first.
            cap_front_points = cap_front_rtk_df.as_matrix(columns=['tx', 'ty', 'tz']) - rtk_z_offsets[0]
            cap_rear_points = cap_rear_rtk_df.as_matrix(columns=['tx', 'ty', 'tz']) - rtk_z_offsets[1]
            point_arrays = [cap_front_points, cap_rear_points]
            filtered_point_arrays = [filter_outliers(cap_front_points), filter_outliers(cap_rear_points)]

        obs_rtk_df_dict = {}
        for obs_topic, obs_rtk_dict in obstacle_rtk_dicts.items():
            obs_prefix, obs_name = obs_prefix_from_topic(obs_topic)
            obs_rtk_df = pd.DataFrame(data=obs_rtk_dict, columns=rtk_cols)
            if not len(obs_rtk_df.index):
                print('Warning: No entries for obstacle %s in %s. Skipping.' % (obs_name, bs.name))
                continue
            obs_rtk_df.to_csv(os.path.join(dataset_outdir, '%s_rtk.csv' % obs_prefix), index=False)
            obs_rtk_df_dict[obs_topic] = obs_rtk_df
            if correct > 0:
                # Use obstacle metadata to determine rtk mounting height and subtract that height
                # from obstacle readings
                md = next(x for x in bs.metadata if x['obstacle_name'] == obs_name)
                if not md:
                    print('Error: No metadata found for %s, skipping obstacle.' % obs_name)
                    continue
                obs_z_offset = np.array([0., 0., md['gps_h']])
                rtk_z_offsets.append(obs_z_offset)
                obs_points = obs_rtk_df.as_matrix(columns=['tx', 'ty', 'tz']) - obs_z_offset
                point_arrays.append(obs_points)
                filtered_point_arrays.append(filter_outliers(obs_points))

        if correct == CORRECT_PLANE:
            points = np.array(np.concatenate(filtered_point_arrays))
            centroid, normal, rotation = fit_plane(
                points, do_plot=True, dataset_outdir=dataset_outdir, name=bs.name)

            def apply_correction(p, z):
                p -= centroid
                p = np.dot(rotation, p.T).T
                c = np.concatenate([centroid[0:2], z[2:]])
                p += c
                return p

            corrected_points = [apply_correction(pa, z) for pa, z in zip(point_arrays, rtk_z_offsets)]
            cap_front_rtk_df.loc[:, ['tx', 'ty', 'tz']] = corrected_points[0]
            cap_rear_rtk_df.loc[:, ['tx', 'ty', 'tz']] = corrected_points[1]
            for i, o in enumerate(obstacle_rtk_dicts.items()):
                obs_rtk_df_dict[o[0]].loc[:, ['tx', 'ty', 'tz']] = corrected_points[2 + i]

        if len(camera_dict['timestamp']):
            # Interpolate samples from all used sensors to camera frame timestamps
            camera_df['timestamp'] = pd.to_datetime(camera_df['timestamp'])
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

            # added by iitwi team
            sync_lidar(camera_index_df, bs, lidar_indir, dataset_outdir)


            if not obs_rtk_df_dict:
                print('Warning: No obstacles or obstacle RTK data present. '
                      'Skipping Tracklet generation for %s.' % bs.name)
                continue
            if not bs.metadata:
                print('Error: No metadata found, metadata.csv file should be with .bag files.'
                      'Skipping tracklet generation.')
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
                md = next(x for x in bs.metadata if x['obstacle_name'] == obs_name)

                obs_tracklet = Tracklet(
                    object_type=md['object_type'], l=md['l'], w=md['w'], h=md['h'], first_frame=0)

                # NOTE these calculations are done in obstacle oriented coordinates. The LWH offsets from
                # metadata specify offsets from lower left, rear, ground corner of the vehicle. Where +ve is
                # along the respective length, width, height axis away from that point. They are converted to
                # velodyne/ROS compatible X,Y,Z where X +ve is forward, Y +ve is left, and Z +ve is up.
                lrg_to_gps = [md['front_gps_l'], -md['front_gps_w'], md['front_gps_h']]
                lrg_to_centroid = [md['l'] / 2., -md['w'] / 2., md['h'] / 2.]
                gps_to_centroid = np.subtract(lrg_to_centroid, lrg_to_gps)

                # added by iitwi team
                # change estimate position in tracklet file, use udacity provided function or local function.
                estimate_obstacle_poses = choose_udacity_or_local_correction(local_correction)

                # Convert NED RTK coords of obstacle to capture vehicle body frame relative coordinates
                obs_tracklet.poses = estimate_obstacle_poses(
                    cap_front_rtk=cap_front_rtk_interp_rec,
                    cap_rear_rtk=cap_rear_rtk_interp_rec,
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
