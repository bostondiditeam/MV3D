# setup
- ```catkin_make```
- ```rosrun lidar lidar_node```
- ```rosrun nodelet nodelet standalone velodyne_pointcloud/CloudNodelet```
- ```rosbag play -l name-of-file.bag``` The "-l" keeps the bag file playing on repeat so that you can keep working on your algorithm without having to mess with the data playback.
