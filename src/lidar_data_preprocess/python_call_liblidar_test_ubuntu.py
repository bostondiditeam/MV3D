from ctypes import *

#load the shared object file
#liblidar = CDLL('./liblidar.dylib')	#OSX
liblidar = CDLL('./liblidar.so')	#LINUX

#CHANGE HERE !!! -----------------
#Change lidar data source dir, lidar top view image saved destination dir, lidar front view image saved destination dir 
lidar_data_src_dir = "./raw/kitti/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/"
top_image_dst_dir = "./preprocessed/kitti/top_image/"
front_image_dst_dir = "./preprocessed/kitti/front_image/"
delay = "0" #ms,  DISPLAY DELAY TIME
##--------------------------------
argc = "5" # DO NOT CHANGE

# create byte objects from the strings
b_argc = argc.encode('utf-8')
b_lidar_data_src_dir = lidar_data_src_dir.encode('utf-8')
b_top_image_dst_dir = top_image_dst_dir.encode('utf-8')
b_front_image_dst_dir = front_image_dst_dir.encode('utf-8')
b_delay = delay.encode('utf-8')

def call_liblidar(L):
    arr = (c_char_p * len(L))()
    arr[:] = L
    liblidar.main(len(L), arr)

call_liblidar( [b_argc , b_lidar_data_src_dir, b_top_image_dst_dir, b_front_image_dst_dir, b_delay])




