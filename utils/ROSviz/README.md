## Bounding box visualization for DiDi 

### Required ROS packages
* rviz
* velodyne

### Instructions

* Download the project to the `src` of your catkin workspace. For example,
```
cp -r projection ~/catkin_ws/src/
```

* Make sure all python files in source directory (`~/catkin_ws/src/projection/scripts/*.py`) are executable. 
Use the command `chmod +x` for this step.

* Build the catkin package followed by `source ~/catkin_ws/devel/setup.bash`. 
This last step has to be performed for every new terminal.

* To play the rosbag with bounding box visualization on camera, 
```
roslaunch launch/projection.launch bag:=<absolute_path_of_bag_file> rate:=0.1
```
Deafult rate is 0.1, so this option can be skipped. You may get an error `ROS time moved backwards` which can be safely ignored. This error is caused because the rosbag is played in loop. 

* This should open rviz window which looks like this :
![](demo/demo.png) 
The interactive markers will allow you to rotate and translate with respect to origin (of velodyne frame) as well as to rotate bounding box with respect to obstacle vehicle. 

* As you manually interact with the markers, projection on camera plane will be automatically modified. 

* To view the final transformations, use 
```
rostopic echo /obstacle_marker/feedback 
```
If `marker_name` is  `capture vehicle`, it corresponds to rotation/translation about velodyne frame. If `marker_name` is `obstacle vehicle`, it corresponds to rotation about obstacle vehicle centroid. Record the `pose` which can then be used for training. 

