# README

## Requirements
- Python 2.7 or Python 3
- numpy
- matplotlib Python library (optional for visualization)
- time (optional for processing time evaluation)

The LiDAR data pre-processing pipeline is contained in LidarPreprocess.c.

## To compile LidarPreprocess.c:
make

## To run SampleProgram.py:
python SampleProgram.py

## Comments:
- After running make, a shared object named LidarPreprocess.so will be created.
- The shared object is called by the code in SampleProgram.py.
- Change lidar sensing range and resolution (TOP_X_MIN, TOP_X_MAX, ...) to LidarPreprocess.so in SampleProgram.py.
- Change the filepath (lidar_data_src_path) to LidarPreprocess.so in SampleProgram.py.
