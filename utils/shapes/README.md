# Function
This script plots a simple static box and publish to ROS as a Marker topic `/visualization_marker` which can be vizualized using RViz with other toics, e.g. lidar cloud points.

# Reference 
This is adapted from the [RViz tutorials](http://wiki.ros.org/rviz/Tutorials/Markers%3A%20Basic%20Shapes).


# Usage
## To run

```source devel/setup.bash```

```rosrun using_markers basic_shapes```

You should see a warning message when the topic is live:
```Please subscribe to Marker topic /visualization_marker```

In RViz, add topic (+ sign) `/visualization_marker`

## To alter box dimensions
To change the dimension and coordinates of the static both, edit `src/basic_shapes.cpp` in `using_markers`. 

```make clean; make```

## To compile from scratch

Setup ROS workspace.

```
mkdir -p shapes/src
cd shapes
catkin_make
source devel/setup.bash
cd src
```

Then in `shapes/src`

```catkin_create_pkg using_markers roscpp visualization_msgs```


Copy the basic_shapes.cpp to `using_markers\src`


Edit the `CMakeLists.txt` file in `using_markers\`. Add the following lines:

```
add_executable(basic_shapes src/basic_shapes.cpp)
target_link_libraries(basic_shapes ${catkin_LIBRARIES})
```

Then compile the executable,
```make clean; make```
