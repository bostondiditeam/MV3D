# Usage 
As titled, the 2 tools converts didi rosbag data into synced frames(in camera timeline).

# Logic to use them.
- First run tools under ./lidar directory(It's written is C++ read readme.md under that directory on how to use it).
It will take lidar data from ros bag and convert it into .bin files(a bin file for each frame).

- After run tools under ./lidar directory, you should already get lidar .bin files extracted from ROS bags. Then 
`source MV3D/pythonpath.sh`, it will add external modules(package to be extracted) into current pythonpath, then you 
should be able to use code from MV3D/external_modules/didi-competition for bag_to_kitti.py script. After that, run 
python script under ./sync_img_lidar_tracklet_tool/bag_to_kitti.py. (You can refer to code inside ./sample_to_cov_all_car_data
.sh for standard syntax). It gets other data (image, gps, rtk etc) from ROS bag, combine lidar data generated from previous tool, sync them to camera timeline, 
then you get same data format as KITTI raw data. 

# Dependency
- ROS
- Python2.7 


# How to run the tools
## For tool under ./lidar, change directory to ./lidar by `cd ./lidar`:
### setup
- ```catkin_make```
- ```source ./devel/setup.zsh```  # or setup.bash
- ```sudo apt-get install ros-indigo-velodyne```
- ```python2 conver_lidar_to_bin.py```   

## For bag_to_kitti.py tool, first `source pythonpath.sh` under MV3D/ directory, then `cd MV3D/utils/bag_to_kitti` and run: (update data paths)
```
 python2 -m sync_img_lidar_tracklet_tool.bag_to_kitti  \
      -i /home/didi/didi_dataset/dataset_2/Data/1  \
      -o  /home/didi/didi_dataset/dataset_2/output \
      -pc /home/didi/sync_img_lidar/output/
      -t obs_rec 
      -c plane
    (Please note for path after -i -o and -pc you should change it according to your own environment.
    and /home/didi/sync_img_lidar/output/ is the directory hold lidar output of ./lidar tool. )
```
      
Read some reference from here [https://github.com/udacity/didi-competition/tree/master/tracklets]


# Todos
- A python wrapper which can combine ./lidar tools into ./sync_img_lidar_tracklet_tool, then user can just run python
 script to get KITTI raw data format from didi rosbags.  
 
 
