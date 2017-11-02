import argparse
import os
import glob
os.environ["DISPLAY"] = ":0"

def main():
    parser = argparse.ArgumentParser(description='Play ROS bag')
    parser.add_argument('-g', '--group', type=str, nargs='?',
                        help='bag group')
    parser.add_argument('-f', '--file', type=str, nargs='?',
        help='*.bag', default='')
    parser.add_argument('-r', '--rate', type=float, nargs='?',
                        help='*.bag', default=1.0)

    args = parser.parse_args()
    group = args.group
    file = args.file
    rate = args.rate
    # a = glob.glob(group+'/*.bag')[0].split('/')[1].split('.')[0]
    # print('a value is here: ', a)
    os.chdir('/home/stu/catkin_ws/src/projection')
    os.system('roslaunch launch/tracklet_ped_test_2_tracklets.launch bag_group:={} bag:={} rate:={}'.format(group, file, rate))
    print(file)

if __name__ == '__main__':
    main()

print('pass')
