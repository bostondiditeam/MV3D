# README

This is c wrapper python module for generating top view (and front view) maps in MV3D.

## Requirements
- Python 2.7 or Python 3
- numpy
- matplotlib Python library (optional for visualization)
- time (optional for processing time evaluation)

## Module 1 : Lidar Top Maps Generation

### To compile LidarTopPreprocess.c  (generate LidarTopPreprocess.so)
make

### To run ExampleUsage_Top.py
python ExampleUsage_Top.py

### Comments:
- After running make, shared object named LidarTopPreprocess.so will be created.
- The shared objects are called by the code in ExampleUsage_Top.py.
- Change lidar top view sensing range and resolution (TOP_X_MIN, TOP_X_MAX, ...) to LidarTopPreprocess.so in ExampleUsage_Top.py line 5~13.
- Change numpy array to be loaded into 'raw' to LidarTopPreprocess.so in ExampleUsage_Top.py line 38.

## Module 2 : Lidar Top And Front Maps Generation

### To compile LidarTopAndFrontPreprocess.c  (generate LidarTopAndFrontPreprocess.so)
make

### To run ExampleUsage_TopAndFront.py
python ExampleUsage_TopAndFront.py

### Comments:
- After running make, shared object named LidarTopAndFrontPreprocess.so will be created.
- The shared objects are called by the code in ExampleUsage_TopAndFront.py.
- Change lidar top view and front view sensing range and resolution to LidarTopAndFrontPreprocess.so in ExampleUsage_TopAndFront.py line 5~19.
- Change numpy array to be loaded into 'raw' to LidarTopAndFrontPreprocess.so in ExampleUsage_TopAndFront.py line 54.
- Note only points inside (TOP_X_MIN, TOP_X_MAX, TOP_Y_MIN, ...) will be processed to generate front view maps. 


