import ctypes
import numpy as np
import matplotlib.pyplot as plt

rows = 400
cols = 400
height = 8

print 'LiDAR data pre-processing starting...'

# initialize an np 3D array with 1's
indata = np.ones((rows, cols, height), dtype = np.double)

# IMPORTANT: CHANGE THE FILE PATH TO THE .so FILE
# create a handle to LidarPreprocess.c
SharedLib = ctypes.cdll.LoadLibrary('/home/afrah/Desktop/Lidar_Preprocess/LidarPreprocess.so')

for frameNum in range(108):

	# call the C function to create top view maps
	# The np array indata will be edited by createTopViewMaps to populate it with the 8 top view maps 
	SharedLib.createTopViewMaps(ctypes.c_void_p(indata.ctypes.data), frameNum)

	# At this point, the pre-processed current frame is stored in the variable indata which is a 400x400x8 array.

	# Pass indata to the rest of the MV3D pipeline.

	# Code to visualize the 8 top view maps (optional)

	# plt.figure()

	# for i in range(8):
		
	# 	plt.subplot(2, 4, i+1)
	# 	plt.imshow(indata[:,:,i])
	# 	plt.gray()

	# plt.show()

print 'LiDAR data pre-processing complete for', frameNum + 1, 'frames'
