# Usage 
As titled, the 2 tools converts didi rosbag data into synced frames(in camera timeline).

# Logic to use them.
- First run tools under ./lidar directory(It's written is C++ read readme.md under that directory on how to use it).
It will take lidar data from ros bag and convert it into .bin files(a bin file for each frame).
- After run ./lidar tool, run python script under ./sync_img_lidar_tracklet_tool/bag_to_kitti.py. It gets other data
(image, gps, rtk etc) from ROS bag, combine lidar data generated from previous tool, sync them to camera timeline, 
then you get same data format as KITTI raw data. 

# Dependency
- ROS
- Python2.7 


# How to run the tools
## For tool under ./lidar, change directory to ./lidar by cd ./lidar:
### setup
- ```catkin_make```
- ```source ./devel/setup.zsh```  # or setup.bash
- ```sudo apt-get install ros-indigo-velodyne```
- ```python2 conver_lidar_to_bin.py```   

## For bag_to_kitti.py tool, first `cd` to `MV3D/utils/bag_to_kitti` and run: (update data paths)
```
 python2 -m sync_img_lidar_tracklet_tool.bag_to_kitti  \
      -i /home/didi/didi_dataset/dataset_2/Data/1  \
      -o  /home/didi/didi_dataset/dataset_2/output \
      -pc /home/didi/sync_img_lidar/output/
```
      
Read some reference from here [https://github.com/udacity/didi-competition/tree/master/tracklets]


/home/didi/sync_img_lidar/output/ is the directory hold lidar output of ./lidar tool. 


# Todos
- A python wrapper which can combine ./lidar tools into ./sync_img_lidar_tracklet_tool, then user can just run python
 script to get KITTI raw data format from didi rosbags.  
 
 
