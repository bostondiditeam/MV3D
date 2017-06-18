import numpy as np
import math
# -------------------- 1. SETTING PARAMETERS HERE !!! ----------------------------
# initial setting for  Kitti/2011_09_26/2011_09_26_drive_0005_sync/velodyne_points/data/0000000004.bin
TOP_X_MIN =0  
TOP_X_MAX =40
TOP_Y_MIN =-20  
TOP_Y_MAX =20
TOP_Z_MIN =-2
TOP_Z_MAX = 1.0
TOP_X_DIVISION = 0.1
TOP_Y_DIVISION = 0.1
TOP_Z_DIVISION = 0.4
FRONT_PHI_MIN = -20
FRONT_PHI_MAX = -FRONT_PHI_MIN	#DONT'T CHANGE THIS !
FRONT_THETA_MIN = -45
FRONT_THETA_MAX = -FRONT_THETA_MIN	# DONT'CHANGE THIS !
FRONT_PHI_DIVISION = 0.1
FRONT_THETA_DIVISION = 0.2

# setting for DiDi from xuefung
#TOP_X_MIN =-45   #-15
#TOP_X_MAX =45	#45
#TOP_Y_MIN =-10  #-30
#TOP_Y_MAX =10	#30
#TOP_Z_MIN =-3
#TOP_Z_MAX =1.0
#TOP_X_DIVISION = 0.2
#TOP_Y_DIVISION = 0.2
#TOP_Z_DIVISION = 0.5
#FRONT_PHI_MIN = -20
#FRONT_PHI_MAX = -FRONT_PHI_MIN
#FRONT_THETA_MIN = -45
#FRONT_THETA_MAX = -FRONT_THETA_MIN
#FRONT_PHI_DIVISION = 0.1
#FRONT_THETA_DIVISION = 0.2


# Calculate map size and pack parameters for top view and front view map (DON'T CHANGE THIS !)
Xn = math.floor((TOP_X_MAX-TOP_X_MIN)/TOP_X_DIVISION)  
Yn = math.floor((TOP_Y_MAX-TOP_Y_MIN)/TOP_Y_DIVISION)   
Zn = math.floor((TOP_Z_MAX-TOP_Z_MIN)/TOP_Z_DIVISION)   
Rn = math.floor((FRONT_PHI_MAX-FRONT_PHI_MIN)/FRONT_PHI_DIVISION)  #deg/deg = grid #   vertical grid|
Cn = math.floor((FRONT_THETA_MAX-FRONT_THETA_MIN)/FRONT_THETA_DIVISION) #deg/deg = grid #  horizontal grid-
Fn = 3 

top_paras = (TOP_X_MIN, TOP_X_MAX, TOP_Y_MIN, TOP_Y_MAX, TOP_Z_MIN, TOP_Z_MAX, TOP_X_DIVISION, TOP_Y_DIVISION, TOP_Z_DIVISION, Xn, Yn, Zn)
front_paras = (FRONT_PHI_MIN, FRONT_PHI_MAX, FRONT_THETA_MAX, FRONT_THETA_MAX, FRONT_PHI_DIVISION, FRONT_THETA_DIVISION, Rn, Cn, Fn)
#------------------------------------------------------------------------------


#------------------- 2. SET SOURCE RAW DATA TO BE PROCESSED --------------------
# load lidar raw data  (presumed raw data dimension : num x 4)    
raw = np.load("raw_kitti_2011_09_26_0005_0000000004.npy")
num = raw.shape[0]  # DON'T CHANGE THIS !
# num : number of points in one lidar frame
# 4 : total channels of single point (x, y, z, intensity)


# top view, front view data structure for saving processed maps (initialized with 1's)
top_flip = np.ones((Xn, Yn, Zn+2), dtype = np.double) 	# DON'T CHANGE THIS !
# top-view maps : Zn height maps + 1 intensity map + 1 density map
# top-view map size : Xn * Yn

front_flip = np.ones((Rn, Cn, Fn), dtype = np.double)  # DON'T CHANGE THIS !
# front-view maps : 1 height map + 1 distance map + 1 intensity map
# front-view map size : Rn * Cn
# ------------------------------------------------------------------------------


#------------------- 3. CREATE TOP VIEW AND FRONT VIEW MAPS --------------------
#import time
#tStart = time.time()

from createTopAndFrontMaps import *
createTopAndFrontMaps(raw, num, top_flip, front_flip, top_paras, front_paras, './LidarTopAndFrontPreprocess.so')
top = np.flipud(np.fliplr(top_flip))
front = np.flipud(np.fliplr(front_flip))

#tEnd = time.time()
#print("It takes %f sec" % (tEnd-tStart))
# ------------------------------------------------------------------------------


#------------------- 4. SHOW TOP VIEW AND FRONT VIEW MAPS (optional) -----------
import matplotlib.pyplot as plt
# SHOW TOP VIEW MAPS (optional)
map_num = len(top[0][0])
plt.figure()
for i in range(map_num):
	plt.subplot(1, map_num, i+1)
	plt.imshow(top[:,:,i])
	plt.gray()
plt.show()

# SHOW FRONT VIEW MAPS (optional)
plt.imshow(front[:,:,0])
plt.gray()
plt.show()
plt.imshow(front[:,:,1])
plt.gray()
plt.show()
plt.imshow(front[:,:,2])
plt.gray()
plt.show()
# ------------------------------------------------------------------------------




