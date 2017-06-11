#include <stdio.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h> 
#include <string.h>
#include <cstring>
#include <unistd.h>
#include <sstream>
#include <iomanip>  
#include <vector>
#include <math.h>
#include <iostream>
#include <cstdlib>
#include <iostream>
#include <fstream>    
#include <string>
#include <vector>

using namespace std;

extern "C"
{	
	typedef struct PointT {
		float x;
		float y;
		float z;
		float intensity;
	} PointT;


	const float x_MIN = 0.0;
	const float x_MAX = 40.0;
	const float y_MIN =-20.0;
	const float y_MAX = 20.0;
	const float z_MIN = -0.4; // -0.4	////TODO : to be determined ....
	const float z_MAX = 2; // 2
	const float x_DIVISION = 0.1;
	const float y_DIVISION = 0.1;
	const float z_DIVISION = 0.4;	// was 0.2 originally

	int X_SIZE = (int)((x_MAX-x_MIN)/x_DIVISION);	// X_SIZE is 400
	int Y_SIZE = (int)((y_MAX-y_MIN)/y_DIVISION);   // Y_SIZE is 400
	int Z_SIZE = (int)((z_MAX-z_MIN)/z_DIVISION);   // Z_SIZE is 6

	inline int getX(float x){
		return (int)((x-x_MIN)/x_DIVISION);
	}

	inline int getY(float y){
		return (int)((y-y_MIN)/y_DIVISION);
	}

	inline int getZ(float z){
		return (int)((z-z_MIN)/z_DIVISION);
	}

    void createTopViewMaps(const void * indatav, int frame_counter)
    {	
	    double * data_cube = (double *) indatav;

		// load point cloud
		FILE *fp;
	  	
	  	// IMPORTANT: CHANGE THIS FILE PATH TO THE .bin VELODYNE FILES
		string velo_dir  = "/home/afrah/Desktop/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/";
	
	    int32_t num = 1000000;
	    float *data = (float*)malloc(num*sizeof(float));

	    float *px = data+0;
		float *py = data+1;
		float *pz = data+2;
		float *pr = data+3;

		ostringstream velo_filename;
		velo_filename << setfill('0') << setw(10) << frame_counter << ".bin";

		string velo_path = velo_dir + velo_filename.str();

		const char* x = velo_path.c_str();
		fp = fopen (x, "rb");

		if(fp == NULL){
			cout << x << " not found. Ensure that the file path is correct." << endl;
		}

		num = fread(data,sizeof(float),num,fp)/4;

		//3D grid box index
		int X = 0;	
		int Y = 0;
		int Z = 0;

		//height features X_SIZE * Y_SIZE * Z_SIZE
		vector<vector<vector<float> > > height_maps;

		//max height X_SIZE * Y_SIZE * 1  (used to record current highest cell for X,Y while parsing data)
		vector<vector<float> > max_height_map;

		//density feature X_SIZE * Y_SIZE * 1
		vector<vector<float> > density_map;

		//intensity feature X_SIZE * Y_SIZE * 1
		vector<vector<float > > intensity_map;

		height_maps.resize(X_SIZE);
		max_height_map.resize(X_SIZE);
		density_map.resize(X_SIZE);
		intensity_map.resize(X_SIZE);

		for (int i=0; i<X_SIZE; i++)
		{
			height_maps[i].resize(Y_SIZE);
			max_height_map[i].resize(Y_SIZE);
			density_map[i].resize(Y_SIZE);
			intensity_map[i].resize(Y_SIZE);

			for (int j=0; j<Y_SIZE; j++)
			{
				height_maps[i][j].resize(Z_SIZE);
				
				//initialization for height_maps
				for (int k=0; k<Z_SIZE; k++)
					height_maps[i][j][k] = 0;	//value stored inside always >= 0 (relative to z_MIN, unit : m)
			
				//initialization for density_map, max_height_map, intensity_map
				density_map[i][j] = 0;	//value stored inside always >= 0, usually < 1 ( log(count#+1)/log(64), no unit )
				max_height_map[i][j] = 0;	//value stored inside always >= 0  (relative to z_MIN, unit : m)
				intensity_map[i][j] = 0;	//value stored inside always >= 0 && <=255 (range=0~255, no unit)
			}
		}

		//allocate point cloud for temporally data visualization (only used for validation)
		vector<std::vector<PointT> > height_cloud_vec;

		// Initialize the vector of vectors with a dummy. The dummy is later deleted.
		for(int i = 0; i < Z_SIZE; i++){
			std::vector<PointT> dummy;
			PointT dummy_member;
			dummy.push_back(dummy_member);
			height_cloud_vec.push_back(dummy);
		}

		vector<PointT> intensity_cloud;
		vector<PointT> density_cloud;

		for (int32_t i=0; i<num; i++) {

			PointT point;
		    point.x = *px;
		    point.y = *py;
		    point.z = *pz;
			point.intensity = (*pr) * 255;	//TODO : check if original Kitti data normalized between 0 and 1 ?

			X = getX(point.x);
			Y = getY(point.y);
			Z = getZ(point.z);
		
			//For every point in each cloud, only select points inside a predefined 3D grid box
			if (X >= 0 && Y>= 0 && Z >=0 && X < X_SIZE && Y < Y_SIZE && Z < Z_SIZE)
			{
				//For every point in predefined 3D grid box.....
				if ((point.z - z_MIN) > height_maps[X][Y][Z])
				{	
					height_maps[X][Y][Z] = point.z - z_MIN;
					
					//Save to point cloud for visualization -----				
					PointT grid_point;
					grid_point.x = X;
					grid_point.y = Y;
					grid_point.z = 0;
					grid_point.intensity = point.z - z_MIN;
					// height_cloud_vec[Z].push_back(grid_point);

					height_cloud_vec.at(Z).push_back(grid_point);
				}
			
				if ((point.z - z_MIN) > max_height_map[X][Y])
				{
					max_height_map[X][Y] = point.z - z_MIN;
					intensity_map[X][Y] = point.intensity;
					density_map[X][Y]++;	// update count#, need to be normalized afterwards

					//Save to point cloud for visualization -----
					PointT grid_point;
					grid_point.x = X;
					grid_point.y = Y;
					grid_point.z = 0;
					grid_point.intensity = point.intensity;
					intensity_cloud.push_back(grid_point);
				
					grid_point.intensity = density_map[X][Y];
					density_cloud.push_back(grid_point);
				}
			}
		    px+=4; py+=4; pz+=4; pr+=4;
		}

		// erase the dummy vector added at the start
		for(int i = 0; i < Z_SIZE; i++){
			height_cloud_vec.at(i).erase(height_cloud_vec.at(i).begin());
		}

		// normalization
		for (unsigned int i=0; i < density_cloud.size(); i++){

			float val = log10(density_cloud.at(i).intensity+1)/log10(64);

			if(val < 1)	
            	density_cloud.at(i).intensity = val;
            else
            	density_cloud.at(i).intensity = 1;
		}

		int data_cube_size = X_SIZE * Y_SIZE * (Z_SIZE + 2);
		// float data_cube[data_cube_size];	// dimensions 400x400x8

		// initialize the data cube to be all zeros
		for(int i = 0; i < data_cube_size; i++){
			data_cube[i] = 0;
		}
		
		// Treat the height maps first
		vector<PointT> *cloud_demo;

		for (int k = 0; k<Z_SIZE; k++){
			// *cloud_demo = height_cloud_vec[k];
			cloud_demo = &height_cloud_vec.at(k);
		  	int cloud_size = cloud_demo->size();
			int plane_idx = k;	// because this is the kth plane (starting from zero)

			for(int i = 0; i < cloud_size; i++){
				int x_coord = (int)cloud_demo->at(i).y;
				int y_coord = (int)cloud_demo->at(i).x;

				// array_idx equals 400*8*y + 8*x + plane_idx where plane_idx goes from 0 to 5 (for the 6 height maps)
				// The formula for array_idx below was empirically calculated to fit the exact order in which the elements
				// of a 3D array are stored in memory. Elements are read starting with the top left corner of the bottom plane
				// the 3D array, then moving up in the z direction, then across (x) and then below (y)
				int array_idx = X_SIZE * (Z_SIZE + 2) * y_coord + (Z_SIZE + 2) * x_coord + plane_idx;
				float value = cloud_demo->at(i).intensity;

				data_cube[array_idx] = value;
			}
		}

		// Treat the density map
		cloud_demo = &density_cloud;
		int cloud_size = cloud_demo->size();
		int plane_idx = Z_SIZE;	// because this is the 6th plane (starting from zero)

		for(int i = 0; i < cloud_size; i++){
			int x_coord = (int)cloud_demo->at(i).y;
			int y_coord = (int)cloud_demo->at(i).x;

			// array_idx equals 400*8*x + 8*y + 6 where 6 is the 6th (starting at 0) 400x400 plane
			int array_idx = X_SIZE * (Z_SIZE + 2) * y_coord + (Z_SIZE + 2) * x_coord + plane_idx;
			float value = cloud_demo->at(i).intensity;

			data_cube[array_idx] = value;
		}
		
		// Treat the intensity map
		cloud_demo = &intensity_cloud;
		cloud_size = cloud_demo->size();
		plane_idx = Z_SIZE + 1;	// because this is the 7th plane (starting from zero)

		for(int i = 0; i < cloud_size; i++){
			int x_coord = (int)cloud_demo->at(i).y;
			int y_coord = (int)cloud_demo->at(i).x;

			// array_idx equals 400*8*x + 8*y + 7 where 7 is the 7th (starting at 0) 400x400 plane
			int array_idx = X_SIZE * (Z_SIZE + 2) * y_coord + (Z_SIZE + 2) * x_coord + plane_idx;
			float value = cloud_demo->at(i).intensity;

			data_cube[array_idx] = value;
		}

		fclose(fp);

		delete data;
		
		// cout << "C++ part exited. LiDAR data pre-processing completed" << endl;
    }
}