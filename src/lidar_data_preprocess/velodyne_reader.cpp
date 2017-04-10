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

#include <iostream>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv ;

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
const float z_MIN = -2.0;	////TODO : to be determined ....
const float z_MAX = 0.4;
const float x_DIVISION = 0.1;
const float y_DIVISION = 0.1;
const float z_DIVISION = 0.4;

const float delta_PHI = 0.1;//0.4; //vertical resolution (deg)
const float delta_THETA = 0.2;//0.08; //horizontal resolution	(deg)		
const float RAD_TO_DEG = 180/3.141596;

int X_SIZE = (int)((x_MAX-x_MIN)/x_DIVISION)+1;	//meter/meter = grid #
int Y_SIZE = (int)((y_MAX-y_MIN)/y_DIVISION)+1;	//meter/meter = grid #
int Z_SIZE = (int)((z_MAX-z_MIN)/z_DIVISION)+1;	//meter/meter = grid #

int C_SIZE = (int)(190 / delta_THETA);	// deg/deg = grid #   horizontal -
int R_SIZE = (int)(50 / delta_PHI);	// deg/deg = grid #   vertical |

int FV_CENTER_C = C_SIZE/2;	//-
int FV_CENTER_R = R_SIZE/2;	//|


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

int getC(float x, float y, float delta_THETA)
{
	return (int)(atan2(y, x) * RAD_TO_DEG / delta_THETA);
}

int getR(float x, float y, float z, float delta_PHI)
{
	return (int)(- atan2(z, sqrt(pow(x,2)+pow(y,2))) * RAD_TO_DEG / delta_PHI );
	// note "-" is used for conversion between LiDAR and Camera coordinate (LiDAR +Z = Camera -R)
}


int main()
{
	pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
	pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("PCL Cloud"));
	pcl::visualization::PointCloudColorHandlerGenericField<PointT> handler ("intensity"); 

	pcl::console::TicToc tt;

	// load point cloud
	FILE *fp;
	int frame_counter = 0;
  
	string velo_dir  = "2011_09_26_drive_0001_sync/velodyne_points/data/";

	while (!viewer->wasStopped ())
	{ 
	    std::cerr << "=== LiDAR Data Load Start ===\n", tt.tic ();

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
					height_maps[i][j][k] = 0;	//value stored inside always >= 0 (relative to z_MIN, unit : m)
			
				//initialization for density_map, max_height_map, intensity_map
				density_map[i][j] = 0;	//value stored inside always >= 0, usually < 1 ( log(count#+1)/log(64), no unit )
				max_height_map[i][j] = 0;	//value stored inside always >= 0  (relative to z_MIN, unit : m)
				intensity_map[i][j] = 0;	//value stored inside always >= 0 && <=255 (range=0~255, no unit)

			}
		}

		//Allocate point cloud for temporally data visualization (only used for validation)
		std::vector<pcl::PointCloud<PointT> > height_cloud_vec;
		height_cloud_vec.resize(Z_SIZE);
		pcl::PointCloud<PointT>::Ptr intensity_cloud (new pcl::PointCloud<PointT>);
		pcl::PointCloud<PointT>::Ptr density_cloud (new pcl::PointCloud<PointT>);

		//Allocate top view feature images (intensity feature, density feature, height features)
		cv::Mat TV_intensity_image(X_SIZE,Y_SIZE, CV_8UC3, Scalar(0,0,0));	//BGR
		cv::Mat TV_density_image(X_SIZE,Y_SIZE, CV_8UC3, Scalar(0,0,0));	//BGR
		std::vector< cv::Mat > TV_height_images;
		cv::Mat TV_height_image(X_SIZE,Y_SIZE, CV_8UC3, Scalar(0,0,0));	//BGR		
		for (int k=0; k<Z_SIZE; k++)
			TV_height_images.push_back(TV_height_image.clone());	//Note : clone() to prevent reference of single image in OpenCV

		//Allocate front view feature images (height, distance, intensity)
		cv::Mat FV_height_image(R_SIZE,C_SIZE, CV_8UC3, Scalar(0,0,0));	//BGR		
		cv::Mat FV_distance_image(R_SIZE,C_SIZE, CV_8UC3, Scalar(0,0,0));	//BGR		
		cv::Mat FV_intensity_image(R_SIZE,C_SIZE, CV_8UC3, Scalar(0,0,0));	//BGR

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
				//showPoint(point);
				//std::cout<< X <<","<<Y<<","<<Z<<std::endl;
				//updateCloudBoundary(cloud_boundary, point);

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
					height_cloud_vec[Z].push_back(grid_point);
				}
			
				if ((point.z - z_MIN) > max_height_map[X][Y])
				{
					max_height_map[X][Y] = point.z - z_MIN;
					intensity_map[X][Y] = point.intensity;
					density_map[X][Y]++;	// update count#, need to be normalized afterwards

					//std::cout<<point.z <<","<<max_height_map[X][Y]<<","<<density_map[X][Y]<<std::endl;

					//Save to point cloud for visualization -----
					PointT grid_point;
					grid_point.x = X;
					grid_point.y = Y;
					grid_point.z = 0;
					grid_point.intensity = point.intensity;
					intensity_cloud->points.push_back(grid_point);
				
					grid_point.intensity = density_map[X][Y];
					density_cloud->points.push_back(grid_point);

				}

				int R = getR(point.x, point.y, point.z, delta_PHI);
				int C = getC(point.x, point.y, delta_THETA);

				if ((R+FV_CENTER_R) <= R_SIZE && (C+FV_CENTER_C) <= C_SIZE && (R+FV_CENTER_R) >=0 && (C+FV_CENTER_C) >=0)
				{
					//Front view - height feature
					FV_height_image.at<cv::Vec3b> (R+FV_CENTER_R,C+FV_CENTER_C)[0] = (int)((point.z-z_MIN)/(z_MAX-z_MIN) *255);
					FV_height_image.at<cv::Vec3b> (R+FV_CENTER_R,C+FV_CENTER_C)[1] = (int)((point.z-z_MIN)/(z_MAX-z_MIN) *255);
					FV_height_image.at<cv::Vec3b> (R+FV_CENTER_R,C+FV_CENTER_C)[2] = (int)((point.z-z_MIN)/(z_MAX-z_MIN) *255);
					//Front view - distance feature
					FV_distance_image.at<cv::Vec3b> (R+FV_CENTER_R,C+FV_CENTER_C)[0] = (int)(sqrt(pow(point.x,2)+pow(point.y,2))/120 *255);
					FV_distance_image.at<cv::Vec3b> (R+FV_CENTER_R,C+FV_CENTER_C)[1] = (int)(sqrt(pow(point.x,2)+pow(point.y,2))/120 *255);
					FV_distance_image.at<cv::Vec3b> (R+FV_CENTER_R,C+FV_CENTER_C)[2] = (int)(sqrt(pow(point.x,2)+pow(point.y,2))/120 *255);
					//Front view - intensity feature
					FV_intensity_image.at<cv::Vec3b> (R+FV_CENTER_R,C+FV_CENTER_C)[0] = (int)((point.intensity-0)/255 *255);
					FV_intensity_image.at<cv::Vec3b> (R+FV_CENTER_R,C+FV_CENTER_C)[1] = (int)((point.intensity-0)/255 *255);
					FV_intensity_image.at<cv::Vec3b> (R+FV_CENTER_R,C+FV_CENTER_C)[2] = (int)((point.intensity-0)/255 *255);
				}
				else
				{
					std::cout<< "R_SIZE (Vertical) :"<<R_SIZE <<",C_SIZE (Horizontal) :" << C_SIZE<<std::endl;
					std::cout<< "FV_CENTER_R:"<<FV_CENTER_R  <<",FV_CENTER_C:" << FV_CENTER_R <<std::endl;
					std::cout<< "R:"<< R <<",C:" << C <<std::endl;
					std::cout<< "R+FV_CENTER_R:"<<R+FV_CENTER_R <<",C+FV_CENTER_C:" << C+FV_CENTER_C <<std::endl;
					std::cout<< "Data is out of image range"<<std::endl;
				}
				
		    	cloud->points.push_back(point);
			}
		    px+=4; py+=4; pz+=4; pr+=4;
		}

		cv::imshow("Front View - Height feature image", FV_height_image) ;
		cv::imshow("Front View - Distance feature image", FV_distance_image) ;
		cv::imshow("Front View - Intensity feature image", FV_intensity_image) ;
		//cv::waitKey(delay) ;
		//cv::destroyWindow("Front View - Height feature image");
		//cv::destroyWindow("Front View - Distance feature image");
		//cv::destroyWindow("Front View - Intensity feature image");

		//Save front view (FV) feature images
		ostringstream str_frame_id;
		str_frame_id << frame_counter ;

		string full_str = "seg/front_image/front_height_image_";
		full_str=full_str + str_frame_id.str() + ".png";
		imwrite(full_str.c_str(),FV_height_image);
		full_str = "seg/front_image/front_distance_image_";
		full_str=full_str + str_frame_id.str() + ".png";
		imwrite(full_str.c_str(),FV_distance_image);
		full_str = "seg/front_image/front_intensity_image_";
		full_str=full_str + str_frame_id.str() + ".png";
		imwrite(full_str.c_str(),FV_intensity_image);


		//normalize density map & normalized for top view images to be saved
		//TODO : check LiDAR vs. camera coord. corresponding...
		for (int X=0; X<X_SIZE; X++)
			for (int Y=0; Y<Y_SIZE; Y++)
			{
				density_map[X][Y] = log(density_map[X][Y]+1)/log(64);

				TV_density_image.at<cv::Vec3b> (X,Y)[0] = (int)((density_map[X][Y]-0)/1 *255);
				TV_density_image.at<cv::Vec3b> (X,Y)[1] = (int)((density_map[X][Y]-0)/1 *255);
				TV_density_image.at<cv::Vec3b> (X,Y)[2] = (int)((density_map[X][Y]-0)/1 *255);

				TV_intensity_image.at<cv::Vec3b> (X,Y)[0] = (int)((intensity_map[X][Y]-0)/255 *255);
				TV_intensity_image.at<cv::Vec3b> (X,Y)[1] = (int)((intensity_map[X][Y]-0)/255 *255);
				TV_intensity_image.at<cv::Vec3b> (X,Y)[2] = (int)((intensity_map[X][Y]-0)/255 *255);
			}	

		//normalize density cloud 
		for (int i=0; i < density_cloud->size(); i++)
			density_cloud->at(i).intensity = log(density_cloud->at(i).intensity+1)/log(64);


		//Show image ---
		cv::imshow("Top View - Density feature image", TV_density_image) ;
		cv::imshow("Top View - Intensity feature image", TV_intensity_image) ;
    	//cv::waitKey(delay) ;
		//cv::destroyWindow("Top View - Density feature image");
    	//cv::destroyWindow("Top View - Intensity feature image");


		////Save top view (TV) feature images
		
		full_str = "seg/top_image/top_intensity_image_";
		full_str=full_str + str_frame_id.str() + ".png";
		imwrite(full_str.c_str(),TV_intensity_image);

		full_str = "seg/top_image/top_density_image_";
		full_str=full_str + str_frame_id.str() + ".png";
		imwrite(full_str.c_str(),TV_density_image);		

		for (int Z=0; Z<Z_SIZE; Z++)
		{
			for (int X=0; X<X_SIZE; X++)
			{
				for (int Y=0; Y<Y_SIZE; Y++)
				{
					TV_height_images[Z].at<cv::Vec3b> (X,Y)[0] =  (int)(height_maps[X][Y][Z]/(z_MAX-z_MIN) *255);
					TV_height_images[Z].at<cv::Vec3b> (X,Y)[1] =  (int)(height_maps[X][Y][Z]/(z_MAX-z_MIN) *255);
					TV_height_images[Z].at<cv::Vec3b> (X,Y)[2] =  (int)(height_maps[X][Y][Z]/(z_MAX-z_MIN) *255);
				}
			}
			cv::imshow("Top View - Height feature image", TV_height_images[Z]) ;
			cv::waitKey(delay/2) ;
			//cv::destroyWindow("Top View - Height feature image");

			ostringstream str_height_layer_id;
			str_height_layer_id << Z ;
			full_str = "seg/top_image/top_height_image_";
			full_str=full_str + str_frame_id.str() + "-" + str_height_layer_id.str() +".png";
			imwrite(full_str.c_str(),TV_height_images[Z]);
		}

		//release image data
		FV_height_image.release();
		FV_distance_image.release();
		FV_intensity_image.release();

		TV_density_image.release();
    	TV_intensity_image.release();
    	for (int Z=0; Z<Z_SIZE; Z++)
    		TV_height_images[Z].release();

/*
		float min_value = 0;
		float max_value = 255;
		//Get MIN and MAX (dynamic min and max ...)
		for (int X=0; X<X_SIZE; X++)
		{
			for (int Y=0; Y<Y_SIZE; Y++)
			{
				if (density_map[X][Y] > max_value)
					max_value = density_map[X][Y];
				if (density_map[X][Y] < min_value)
					min_value = density_map[X][Y];
				if (intensity_map[X][Y] > max_value)
					max_value = intensity_map[X][Y];
				if (intensity_map[X][Y] < min_value)
					min_value = intensity_map[X][Y];
				for (int Z=0; Z<Z_SIZE; Z++)
				{
					if (height_maps[X][Y][Z] > max_value)
						max_value = height_maps[X][Y][Z];
					if (height_maps[X][Y][Z] < min_value)
						min_value = height_maps[X][Y][Z];
				}	
			}
		}
		std::cout <<"MAX : "<<max_value<<"MIN : "<<min_value<<std::endl;
*/

		std::cerr << "=== LiDAR Preprocess Done "<< tt.toc ()<<" ms === \n"; 

		pcl::PointCloud<PointT>::Ptr cloud_demo (new pcl::PointCloud<PointT>);
		viewer->addCoordinateSystem(1.0);
		std::cout <<"Frame # : "<< frame_counter <<std::endl;
		std::cout <<"Show height_map ... "<< std::endl;
		for (int k = 0; k<Z_SIZE; k++)
		{
			*cloud_demo=height_cloud_vec[k];
			std::cout<< "- Layer " << k << std::endl;
	 		//std::cout<<cloud_demo->size()<<std::endl;

		  	handler.setInputCloud (cloud_demo);
			if (!viewer->updatePointCloud (cloud_demo, handler, "demo"))
				viewer->addPointCloud (cloud_demo, handler, "demo");
//			viewer->spinOnce (delay/2);
		}

		std::cout <<"Show density map ... "<< std::endl;
		*cloud_demo=*density_cloud;
		handler.setInputCloud (cloud_demo);
		if (!viewer->updatePointCloud (cloud_demo, handler, "demo"))
			viewer->addPointCloud (cloud_demo, handler, "demo");
//		viewer->spinOnce (delay * 2);

		std::cout <<"Show intensity map ... "<< std::endl;
		*cloud_demo=*intensity_cloud;
		handler.setInputCloud (cloud_demo);
		if (!viewer->updatePointCloud (cloud_demo, handler, "demo"))
			viewer->addPointCloud (cloud_demo, handler, "demo");
//		viewer->spinOnce (delay * 2);

		//update frame counter
		frame_counter++;		

		fclose(fp);

		cloud->points.clear();

		delete data;
	}

	return 0;  
}

