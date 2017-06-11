import ctypes
import numpy as np
import matplotlib.pyplot as plt
import time

x_MIN = 0.0
x_MAX = 40.0
y_MIN =-20.0
y_MAX = 20.0
z_MIN = -0.4
z_MAX = 2
x_DIVISION = 0.1
y_DIVISION = 0.1
z_DIVISION = 0.4          #was 0.2 originally

X_SIZE = (int)((x_MAX-x_MIN)//x_DIVISION)+1   #400
Y_SIZE = (int)((y_MAX-y_MIN)//y_DIVISION)+1   #400
Z_SIZE = (int)((z_MAX-z_MIN)//z_DIVISION)+1    # 6

print('Image Size X_SIZE, Y_SIZE, Z_SIZE : '+ str(X_SIZE)+ ", " + str(Y_SIZE) + ", " +str(Z_SIZE))
print ('LiDAR data pre-processing starting...')

# initialize an np 3D array with 1's
indata = np.ones((X_SIZE, Y_SIZE, Z_SIZE+2), dtype = np.double)

# IMPORTANT: CHANGE THE FILE PATH TO THE .so FILE
# create a handle to LidarPreprocess.c
SharedLib = ctypes.cdll.LoadLibrary('./LidarPreprocess.so')

lidar_data_src_dir = "../raw/kitti/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/"

tStart = time.time()
for frameNum in range(0,3):
	lidar_data_src_path = lidar_data_src_dir + str(frameNum).zfill(10) + ".bin"
	b_lidar_data_src_path = lidar_data_src_path.encode('utf-8')
	# call the C function to create top view maps
	# The np array indata will be edited by createTopViewMaps to populate it with the 8 top view maps 
	SharedLib.createTopViewMaps(ctypes.c_void_p(indata.ctypes.data), ctypes.c_char_p(b_lidar_data_src_path), ctypes.c_float(x_MIN), ctypes.c_float(x_MAX), ctypes.c_float(y_MIN), ctypes.c_float(y_MAX), ctypes.c_float(z_MIN), ctypes.c_float(z_MAX), ctypes.c_float(x_DIVISION), ctypes.c_float(y_DIVISION), ctypes.c_float(z_DIVISION)  )

	# At this point, the pre-processed current frame is stored in the variable indata which is a 400x400x8 array.

	# Pass indata to the rest of the MV3D pipeline.

	# Code to visualize the 8 top view maps (optional)

	#plt.figure()
	#for i in range(8):
	#	plt.subplot(2, 4, i+1)
	#	plt.imshow(indata[:,:,i])
	#	plt.gray()
	#plt.show()

print ('LiDAR data pre-processing complete for', frameNum + 1, 'frames')
tEnd = time.time()
print("It takes %f sec" % (tEnd-tStart))



