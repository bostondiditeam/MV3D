import ctypes
import numpy as np
import matplotlib.pyplot as plt
import math
#import time

TOP_X_MIN =-45
TOP_X_MAX =45
TOP_Y_MIN =-10
TOP_Y_MAX =10
TOP_Z_MIN =-3
TOP_Z_MAX =1.0
TOP_X_DIVISION = 0.2
TOP_Y_DIVISION = 0.2
TOP_Z_DIVISION = 0.5

Xn = int((TOP_X_MAX-TOP_X_MIN)//TOP_X_DIVISION)+1    
Yn = int((TOP_Y_MAX-TOP_Y_MIN)//TOP_Y_DIVISION)+1
Zn = int((TOP_Z_MAX-TOP_Z_MIN)/TOP_Z_DIVISION)
height  = Xn
width   = Yn
channel = Zn + 2

print('Feature Maps Size (height, width, channel) is ('+ str(height)+ ", " + str(width) + ", " +str(channel)+")" )
print ('LiDAR data pre-processing starting...')

# initialize an np 3D array with 1's
top = np.ones((height, width, channel), dtype = np.double)

# create a handle to LidarPreprocess.c
SharedLib = ctypes.cdll.LoadLibrary('./LidarPreprocess.so')

# CHANGE LIDAR DATA DIR HERE !!!!
lidar_data_src_dir = "../../raw/kitti/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/"

#tStart = time.time()
for frameNum in range(0,1):    # CHANGE LIDAR DATA FRAME NUMBER HERE !!!! 
	lidar_data_src_path = lidar_data_src_dir + str(frameNum).zfill(10) + ".bin"

	# OR OVERWRITE lidar_data_src_path TO SPICIFY THE PATH OF LIDAR DATA FILE !!!!
	lidar_data_src_path = "0000000002.bin"

	b_lidar_data_src_path = lidar_data_src_path.encode('utf-8')
	# call the C function to create top view maps
	# The np array indata will be edited by createTopViewMaps to populate it with the 8 top view maps 
	SharedLib.createTopViewMaps(ctypes.c_void_p(top.ctypes.data), ctypes.c_char_p(b_lidar_data_src_path), ctypes.c_float(TOP_X_MIN), 
								ctypes.c_float(TOP_X_MAX), ctypes.c_float(TOP_Y_MIN), ctypes.c_float(TOP_Z_MAX), ctypes.c_float(TOP_Z_MIN), 
								ctypes.c_float(TOP_Z_MAX), ctypes.c_float(TOP_X_DIVISION), ctypes.c_float(TOP_Y_DIVISION), ctypes.c_float(TOP_Z_DIVISION), 
								ctypes.c_int(Xn), ctypes.c_int(Yn), ctypes.c_int(Zn)  )	

	# flip image to match the original preprocess module result (data.py)
	top = np.flipud(np.fliplr(top))

	# Example code to visualize image for one lidar frame (optional)
	# col 1~8 image : height maps  
	# col 9 image : intensity map
	# col 10 image : density map
	#plt.figure()
	#for i in range(10):
	#	plt.subplot(1, 10, i+1)
	#	plt.imshow(top[:,:,i])
	#plt.show()

	# Example code to visualize all images for one lidar frame (optional)
	#row = int(pow((Zn+2),0.5))
	#col = int(math.ceil((Zn+2)/row))
	#if (int(row * col) != Zn+2 ):
	#	col = col+1
	#plt.figure()
	#for i in range(Zn):
	#	plt.subplot(row, col, i+1)
	#	plt.imshow(top[:,:,i])
	#plt.subplot(row, col, Zn+1)
	#plt.imshow(top[:,:,Zn])
	#plt.title("density map")
	#plt.subplot(row, col, Zn+2)
	#plt.imshow(top[:,:,Zn+1])
	#plt.title("intensity map")
	#plt.show()

print ('LiDAR data pre-processing complete for', frameNum + 1, 'frames')

#tEnd = time.time()
#print("It takes %f sec" % (tEnd-tStart))



