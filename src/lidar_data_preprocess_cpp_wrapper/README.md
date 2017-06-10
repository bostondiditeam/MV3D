# Description

This is lidar point cloud data preprocessing written in C++ which can be invoked in Python script by given lidar data file path and return 3 Python preprocessed list : density_map_list, intensity_map_list and height_maps_list.

[density_map_list, intensity_map_list, height_maps_list] = lidar_preprocess_ext.lidar_preprocess(b_lidar_data_src_path)

For detailed usage example, please see "python_call_cpp_preprocess_test.py"

## Requirements:

--> CMake 2.8 or higher

--> PCL 1.2 or higher

--> Python 2.7

## To compile the code:

--> Run cmake .

--> Run make

--> liblidar_preprocess.so and lidar_preprocess_ext.so will be generated(For UBUNTU)

## To test preprocess module, run the python script file as follows:

--> python2 python_call_cpp_preprocess_test.py   (For UBUNTU)

## To specify lidar data source file path :

Modify lidar_data_src_path in python_call_cpp_preprocess_test.py line 12 to 15
