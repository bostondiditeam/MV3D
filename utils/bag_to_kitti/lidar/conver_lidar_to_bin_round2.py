import glob
import os
import time
import subprocess

if __name__ == '__main__':
    # os.system('source ./devel/setup.zsh')
    home_dir = '/home/stu'
    unsynced_dir=os.path.join(home_dir,'round12_data/unsynced')
    os.system('pkill -f rosmaster')
    os.system('mkdir ./lidar_data')
    time.sleep(0.5)
    os.system('roscore &')
    time.sleep(0.5)
    os.system('roslaunch velodyne_pointcloud 32e_points.launch &')
    time.sleep(0.5)

    # conver all
    if 1:
        convert_fail_list =[]
        round2_data_dir = os.path.join(home_dir,'/home/stu/d_3t/bag_split')
        dirs = glob.glob(os.path.join(round2_data_dir, '*'))
        folder_names_1=[os.path.basename(dir) for dir in dirs]
        print('Found:\n{}'.format(folder_names_1))
        if len(folder_names_1) == 0: exit(-1)
        for n1 in folder_names_1:
            bags_path=glob.glob(os.path.join(round2_data_dir,n1,'*.bag'))
            print('Found bag:\n{}'.format(bags_path))
            for bag_path in bags_path:
                os.system('rm -rf ./lidar_data/*')
                bag_name=os.path.basename(bag_path).split('.bag')[0]
                exit_code = subprocess.call('rosrun lidar lidar_node &',shell=True)
                if exit_code != 0:
                    convert_fail_list.append(bag_path)
                    time.sleep(5)
                    continue

                exit_code = subprocess.call('rosbag play {}'.format(bag_path),shell=True)
                if exit_code != 0:
                    convert_fail_list.append(bag_path)
                    time.sleep(5)
                    continue

                os.system('pkill -f lidar_node')
                mv_to_dir=os.path.join(unsynced_dir,'data3_split',n1,bag_name)
                os.system( 'mkdir %s -p' % (mv_to_dir))
                exit_code = subprocess.call('mv ./lidar_data/* {}'.format(mv_to_dir),shell=True)
                if exit_code != 0:
                    convert_fail_list.append(bag_path)
                    time.sleep(5)
                    continue

        print('-------------')
        print('\n\nConvert lidar data done !!')
        print('convert_fail_list:\n{}'.format(convert_fail_list))

    # test
    if 0:
        # 1  2  3  README.md  Round1Test
        folder_names_1 = ['Round1Test']
        for n1 in folder_names_1:
            bags_path = glob.glob(home_dir + '/competition_data/didi_dataset/dataset_2/Data/' + n1 + '/*.bag')
            for bag_path in bags_path:
                os.system('rm -rf ./lidar_data/*')
                bag_name = os.path.basename(bag_path).split('.bag')[0]
                os.system('rosrun lidar lidar_node &')
                os.system('rosbag play {}'.format(bag_path))
                os.system('pkill -f lidar_node')
                mv_to_dir = os.path.join('./output', unsynced_dir, 'data2', n1, bag_name)
                os.makedirs(mv_to_dir, exist_ok=True)
                os.system('mv ./lidar_data/* {}'.format(mv_to_dir))

        print('-------------')
        print('\n\nConvert lidar data done !!')