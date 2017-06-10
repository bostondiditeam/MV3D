from ctypes import *
import lidar_preprocess_ext
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def show_image (img,title ="no title"):
    plt.figure()
    plt.imshow(img)
    plt.title(title)
    plt.show()

for frame_count in range(0,3):    # CHANGE LIDAR FRAME TO BE PREPROCESSED HERE 
	#---------CHANGE LIDAR SOURCE DIRECTORY HERE -------------------
	lidar_data_src_dir = "./raw/kitti/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/"
	lidar_data_src_path = lidar_data_src_dir + str(frame_count).zfill(10) + ".bin"
	#------------------------------------------------------------------------


	print("Preprocessing file : "+lidar_data_src_path)

	b_lidar_data_src_path = lidar_data_src_path.encode('utf-8')
	lidar_preprocess_ext.argtypes = [c_char_p]

	[density_map_list, intensity_map_list, height_maps_list] =lidar_preprocess_ext.lidar_preprocess(b_lidar_data_src_path)


	# SHOW DENSITY MAP
	print("-------")
	[y, x, color_depth] = [len(density_map_list), len(density_map_list[0]), len(density_map_list[0][0])]
	print("Density map size : "+ str(y) + " * " + str(x) + " * " + str(color_depth) )
	title = "Density Map" + " (frame:"  + str(frame_count).zfill(10) + ")" 
	#show_image(density_map_list , title)
	
	# SHOW INTENSITY MAP
	[y, x, color_depth] = [len(intensity_map_list), len(intensity_map_list[0]), len(intensity_map_list[0][0])]
	print("Intensity map size : "+ str(y) + " * " + str(x) + " * " + str(color_depth) )
	title = "Intensity Map" + " (frame:"  + str(frame_count).zfill(10) + ")" 
	#show_image(intensity_map_list, title)

	# SHOW HEIGHT MAPS
	[z, y, x, color_depth] = [len(height_maps_list), len(height_maps_list[0]), len(height_maps_list[0][0]), len(height_maps_list[0][0][0])]
	print("Height maps size : " + str(z) + " * " + str(y) + " * " + str(x) + " * " + str(color_depth) ) 
	for layer in range(0, len(height_maps_list)):
		title = "Height Map, layer no. " + str(layer) + " (frame:"  + str(frame_count).zfill(10) + ")" 
		#show_image(height_maps_list[layer], title)



