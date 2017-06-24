import glob
import os
import time

if __name__ == '__main__':
    # os.system('source ./devel/setup.zsh')
    home_dir='/home/stu'
    unsynced_dir='round12_data/unsynced'
    os.system('pkill -f rosmaster')
    os.makedirs('./lidar_data',exist_ok=True)
    time.sleep(0.5)
    os.system('roscore &')
    time.sleep(0.5)
    os.system('roslaunch velodyne_pointcloud 32e_points.launch &')
    time.sleep(0.5)

    # conver all
    if 1:
        #1  2  3  README.md  Round1Test
        folder_names_1=['1','2','3','Round1Test']
        for n1 in folder_names_1:
            bags_path=glob.glob(home_dir+'/competition_data/didi_dataset/dataset_2/Data/'+n1+'/*.bag')
            for bag_path in bags_path:
                os.system('rm -rf ./lidar_data/*')
                bag_name=os.path.basename(bag_path).split('.bag')[0]
                os.system('rosrun lidar lidar_node &')
                os.system('rosbag play {}'.format(bag_path))
                os.system('pkill -f lidar_node')
                mv_to_dir=os.path.join(home_dir,unsynced_dir,'data2',n1,bag_name)
                os.makedirs(mv_to_dir,exist_ok=True)
                os.system('mv ./lidar_data/* {}'.format(mv_to_dir))

        print('-------------')
        print('\n\nConvert lidar data done !!')

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