#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h> 
#include <string.h>
#include <cstring>
#include <unistd.h>
#include <sstream>
#include <iomanip>  

#include <vector>

#include <pcl/visualization/cloud_viewer.h>

#include <pcl/console/time.h>
#include <boost/chrono.hpp>

#include <iostream>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>

using namespace std;

typedef pcl::PointXYZI PointT;
struct CloudBoundary
{
	float x_min;
	float x_max;
	float y_min;
	float y_max;
	float z_min;
	float z_max;
};
typedef struct CloudBoundary CloudBoundaryT;

void showPoint(PointT& p)
{
	std::cout<<"("<< p.x<<" ,"<<p.y<<" ,"<<p.z<<" ,"<<p.intensity<<")"<<std::endl;
}

void showCloudBoundary (CloudBoundaryT& cb)
{
	std::cout<< "x_min = " <<cb.x_min <<", x_max = "<<cb.x_max<<std::endl;
	std::cout<< "y_min = " <<cb.y_min <<", y_max = "<<cb.y_max<<std::endl;
	std::cout<< "z_min = " <<cb.z_min <<", z_max = "<<cb.z_max<<std::endl;
}

void updateCloudBoundary(CloudBoundaryT& cb, PointT& p)
{
	if (p.x < cb.x_min)
		cb.x_min = p.x;
	if (p.x > cb.x_max)
		cb.x_max = p.x;
	if (p.y < cb.y_min)
		cb.y_min = p.y;
	if (p.y > cb.y_max)
		cb.y_max = p.y;
	if (p.z < cb.z_min)
		cb.z_min = p.z;
	if (p.z > cb.z_max)
		cb.z_max = p.z;
}


int delay = 1000; // delay between successive LiDAR frames in ms

const float x_MIN = 0.0;
const float x_MAX = 40.0;
const float y_MIN =-20.0;
const float y_MAX = 20.0;
const float z_MIN = -2.0;
const float z_MAX = 0.4;
const float x_DIVISION = 0.1;
const float y_DIVISION = 0.1;
const float z_DIVISION = 0.2;

int X_SIZE = (int)((x_MAX-x_MIN)/x_DIVISION)+1;
int Y_SIZE = (int)((y_MAX-y_MIN)/y_DIVISION)+1;
int Z_SIZE = (int)((z_MAX-z_MIN)/z_DIVISION)+1;

int getX(float x)
{
	return (int)((x-x_MIN)/x_DIVISION);
}

int getY(float y)
{
	return (int)((y-y_MIN)/y_DIVISION);
}

int getZ(float z)
{
	return (int)((z-z_MIN)/z_DIVISION);
}


int main()
{
	boost::shared_ptr<pcl::PointCloud<PointT> > cloud (new pcl::PointCloud<PointT>);
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("PCL Cloud"));
	pcl::visualization::PointCloudColorHandlerGenericField<PointT> handler ("intensity"); 

	pcl::console::TicToc tt;

	// load point cloud
	FILE *fp;
	int frame_counter = 0;
  
	string velo_dir  = "2011_09_26_drive_0001_sync/velodyne_points/data/";

	while (!viewer->wasStopped ())
	{ 
	    std::cerr << "LiDAR Data Load Start ...\n", tt.tic ();

	    int32_t num = 1000000;
	    float *data = (float*)malloc(num*sizeof(float));

	    float *px = data+0;
		float *py = data+1;
		float *pz = data+2;
		float *pr = data+3;

		ostringstream velo_filename;
		velo_filename << setfill('0') << setw(10) << frame_counter << ".bin";
		frame_counter++;

		string velo_path = velo_dir + velo_filename.str();

		const char* x = velo_path.c_str();
		fp = fopen (x, "rb");

		if(fp == NULL){
		  cout << x << " not found. Ensure that the file path is correct." << endl;
		  return 0;
		}

		num = fread(data,sizeof(float),num,fp)/4;

		CloudBoundaryT cloud_boundary = {0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0};
		//showCloudBoundary(cloud_boundary);

		//3D grid box index
		int X = 0;	
		int Y = 0;
		int Z = 0;

		//height features X_SIZE * Y_SIZE * Z_SIZE
		std::vector<std::vector<std::vector<float> > > height_maps;

		//max height X_SIZE * Y_SIZE * 1  (used to record current highest cell for X,Y while parsing data)
		std::vector<std::vector<float> > max_height_map;

		//density feature X_SIZE * Y_SIZE * 1
		std::vector<std::vector<float> > density_map;

		//intensity feature X_SIZE * Y_SIZE * 1
		std::vector<std::vector<float > > intensity_map;

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
					height_maps[i][j][k] = -100;
			
				//initialization for density_map, max_height_map, intensity_map
				density_map[i][j] = 0;
				max_height_map[i][j] = -100;
				intensity_map[i][j] = -100;
			}
		}

	
		// use point cloud as data structure to store feature map for visualization
		// need to be stored into image later ...
		boost::shared_ptr<pcl::PointCloud<PointT> > height_cloud (new pcl::PointCloud<PointT>);
		boost::shared_ptr<pcl::PointCloud<PointT> > intensity_cloud (new pcl::PointCloud<PointT>);
		boost::shared_ptr<pcl::PointCloud<PointT> > density_cloud (new pcl::PointCloud<PointT>);

		for (int32_t i=0; i<num; i++) {

			PointT point;
		    point.x = *px;
		    point.y = *py;
		    point.z = *pz;
			point.intensity = *pr;

			X = getX(point.x);
			Y = getY(point.y);
			Z = getZ(point.z);
		
			//For every point in each cloud, only select points inside a predefined 3D grid box
			if (X >= 0 && Y>= 0 && Z >=0 && X < X_SIZE && Y < Y_SIZE && Z < Z_SIZE)
			{
				//showPoint(point);
				//std::cout<< X <<","<<Y<<","<<Z<<std::endl;
				//updateCloudBoundary(cloud_boundary, point);
			
				//For every point in predefined 3D grid box.....
				if (point.z > height_maps[X][Y][Z])
				{	
					//std::cout<<X<<","<<Y<<","<<Z<<std::endl;
					height_maps[X][Y][Z] = point.z;
				
					if (Z==5)	//NHERE only choose one height layer!!!!
					{
						PointT grid_point;
						grid_point.x = X;
						grid_point.y = Y;
						grid_point.z = 0;
						grid_point.intensity = point.z;
						height_cloud->push_back(grid_point);
					}				
				}
			
				if (point.z > max_height_map[X][Y])
				{
					max_height_map[X][Y] = point.z;
					intensity_map[X][Y] = point.intensity;
					density_map[X][Y]++;	// update count, need to be normalized afterwards

					//std::cout<<point.z <<","<<max_height_map[X][Y]<<","<<density_map[X][Y]<<std::endl;

					PointT grid_point;
					grid_point.x = X;
					grid_point.y = Y;
					grid_point.z = 0;
					grid_point.intensity = point.intensity;
					intensity_cloud->points.push_back(grid_point);
				
					grid_point.intensity = density_map[X][Y];
					density_cloud->points.push_back(grid_point);
				
				}
		
		    	cloud->points.push_back(point);
			}
		    px+=4; py+=4; pz+=4; pr+=4;
		}

		//normalize density map
		for (int X=0; X<X_SIZE; X++)
			for (int Y=0; Y<Y_SIZE; Y++)
				density_map[X][Y] = log(density_map[X][Y]+1)/log(64);
				
		boost::shared_ptr<pcl::PointCloud<PointT> > cloud_demo (new pcl::PointCloud<PointT>);

		std::cerr << "LiDAR Preprocess Done : "<< tt.toc ()<<" ms \n"; 

		//*cloud_demo=*height_cloud;
		//*cloud_demo=*density_cloud;
		*cloud_demo=*intensity_cloud;
		//*cloud_demo=*cloud;

	 	std::cout<<cloud_demo->size()<<std::endl;

	  	handler.setInputCloud (cloud_demo);
		if (!viewer->updatePointCloud (cloud_demo, handler, "demo"))
			viewer->addPointCloud (cloud_demo, handler, "demo");
		viewer->spinOnce (delay);

	
		fclose(fp);

		cloud->points.clear();

		delete data;
	}

	return 0;  
}

