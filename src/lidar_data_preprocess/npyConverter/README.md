# npy_converter

This is an application to convert Velodyne LiDAR frames stored as .bin files into birds eye view maps required by the MV3D network. The maps for each frame are saved as one 3 dimensional npy array. Currently, each frame produces 8 400x400 maps (6 height maps, 1 density map, 1 intensity map). The dimensions of each npy array are therefore 400x400x8. These parameters can be changed in npy_converter.cpp.

Build Instructions:

First install the cnpy library from https://github.com/rogersce/cnpy
Then compile npy_converter.cpp with the CMakeLists.txt file provided in this repository.
Steps:
--> mkdir build
--> cd build
--> cmake ..
--> make

Before running the executable, manually create a folder named "npy_arrays" inside the build folder. This is the folder where the npy arrays will be saved. Without this folder, a segmentation fault will occur during run time.
