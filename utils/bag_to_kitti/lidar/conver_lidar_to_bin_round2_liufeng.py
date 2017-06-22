import glob
import os
import time

if __name__ == '__main__':
    # os.system('source ./devel/setup.zsh')
    unsynced_dir="/ext2/round2_data/unsynced"
    os.system('pkill -f rosmaster')
    os.system('mkdir ./lidar_data')
    time.sleep(0.5)
    os.system('roscore &')
    time.sleep(0.5)
    os.system('roslaunch velodyne_pointcloud 32e_points.launch &')
    time.sleep(0.5)

    # conver all
    # round2_data_dir = "/ext/Data/Round_2/release/car/training"
    # round2_data_dir = "/ext/Data/Round_2/release/car"
    round2_data_dir = "/ext/Data/Round_2/release/"
    dirs = glob.glob(os.path.join(round2_data_dir, '*'))
    folder_names_1=[os.path.basename(dir) for dir in dirs]
    for n1 in folder_names_1:
        bags_path=glob.glob(os.path.join(round2_data_dir,n1,'*_train.bag'))
        print("bags_path : ", bags_path)
        for bag_path in bags_path:
            print("bag_path : ", bag_path)
            os.system('rm -rf ./lidar_data/*')
            bag_name=os.path.basename(bag_path).split('.bag')[0]
            os.system('rosrun lidar lidar_node &')
            os.system('rosbag play {}'.format(bag_path))
            os.system('pkill -f lidar_node')
            # mv_to_dir = os.path.join(unsynced_dir, 'training', n1, bag_name)
            # mv_to_dir = os.path.join(unsynced_dir, 'testing', n1, bag_name)
            mv_to_dir = os.path.join(unsynced_dir,'ped',n1,bag_name)
            os.system( 'mkdir %s -p' % (mv_to_dir))
            os.system('mv ./lidar_data/* {}'.format(mv_to_dir))
    os.system('rm -rf ./lidar_data')
    print('-------------')
    print('\n\nConvert lidar data done !!')