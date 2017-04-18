// ROS Point Cloud DEM Generation
// MacCallister Higgins

#include <cmath>
#include <vector>
#include <float.h>
#include <stdio.h>
#include <math.h>
#include <sstream>
#include <iostream>
#include <fstream>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <ros/ros.h>
#include <ros/console.h>
#include <sensor_msgs/PointCloud2.h>

#define IMAGE_HEIGHT 701
#define IMAGE_WIDTH 801
#define BIN 0.1

#define NO_IMAGE_SHOW
#define NO_IMAGE_WRITE

using namespace cv;
using namespace std;

// Global Publishers/Subscribers
ros::Subscriber subPointCloud;
ros::Publisher pubPointCloud;
ros::Subscriber subVelodynePackets;

//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud <pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_grid(new pcl::PointCloud<pcl::PointXYZI>);
sensor_msgs::PointCloud2 output;

double heightArray[IMAGE_HEIGHT][IMAGE_WIDTH];
double intensityArray[IMAGE_HEIGHT][IMAGE_WIDTH] = {0};

cv::Mat *heightmap;
vector<int> compression_params;

int fnameCounter;
double lowest;

// map meters to 0->255
int map_m2i(double val)
{
    return (int)round((val + 3.0) / 6.0 * 255);
}

// map meters to index
// returns 0 if not in range, 1 if in range and row/column are set
int map_pc2rc(double x, double y, int *row, int *column)
{
    // Find x -> row mapping
    *row = (int)round(floor(((((IMAGE_HEIGHT * BIN) / 2.0) - x) / (IMAGE_HEIGHT * BIN)) *
                            IMAGE_HEIGHT)); // obviously can be simplified, but leaving for debug

    // Find y -> column mapping
    *column = (int)round(floor(((((IMAGE_WIDTH * BIN) / 2.0) - y) / (IMAGE_WIDTH * BIN)) * IMAGE_WIDTH));

    // Return success
    return 1;
}

// map index to meters
// returns 0 if not in range, 1 if in range and x/y are set
int map_rc2pc(double *x, double *y, int row, int column)
{
    // Check if falls within range
    if (row >= 0 && row < IMAGE_HEIGHT && column >= 0 && column < IMAGE_WIDTH)
    {
        // Find row -> x mapping
        *x = (double)(BIN * -1.0 * (row - (IMAGE_HEIGHT / 2.0))); // this one is simplified

        // column -> y mapping
        *y = (double)(BIN * -1.0 * (column - (IMAGE_WIDTH / 2.0)));

        // Return success
        return 1;
    }

    return 0;
}

int save_lidar_pointcloud(const std::string path, pcl::PointCloud<pcl::PointXYZI>::Ptr cloud)
{
    ofstream myfile;
    myfile.open(path.c_str(), ios::out | ios::binary);

    if (myfile.is_open() == false)
    {
        return -1;
    }

    for (int j = 0; j < cloud->points.size(); j++)
    {
        float data[4]={(float)(cloud->points[j].x) ,(float)(cloud->points[j].y) ,(float)(cloud->points[j].z) ,
                        (float)(cloud->points[j].intensity / 255.0)};
        myfile.write((char *)data,sizeof(data));
    }
    myfile.close();
    return 0;
}

int save_timestamp(const std::string path, uint64_t timestamp)
{
    ofstream myfile;
    myfile.open(path.c_str(), ios::out);
    if (myfile.is_open() == false)
    {
        return -1;
    }
    myfile << timestamp;
    myfile.close();
    return 0;
}

// main generation function
void DEM(const sensor_msgs::PointCloud2ConstPtr &pointCloudMsg)
{
    ROS_DEBUG("Point Cloud Received");

    // clear cloud and height map array
    lowest = FLT_MAX;
    for (int i = 0; i < IMAGE_HEIGHT; ++i)
    {
        for (int j = 0; j < IMAGE_WIDTH; ++j)
        {
            heightArray[i][j] = (double)(-FLT_MAX);
        }
    }

    // Convert from ROS message to PCL point cloud
    pcl::fromROSMsg(*pointCloudMsg, *cloud);

    // save lidar data
    char filepath[100];
    snprintf(filepath, 100, "lidar_data/%d.bin", fnameCounter);

    int ret = save_lidar_pointcloud(filepath, cloud);
    if (ret != 0)
    {
        printf("save_lidar_pointcloud error: %d \n", ret);
        exit(-1);
    }

    snprintf(filepath, 100, "lidar_data/%d_timestamp.txt", fnameCounter);
    ret = save_timestamp(filepath, pointCloudMsg->header.stamp.toNSec());
    if (ret != 0)
    {
        printf("save_timestamp error: %d \n", ret);
        exit(-1);
    }

    // //todo: this test print,delete it
    // static float min = 900;
    // static float max = 0;
    // for (int j = 0; j < cloud->points.size(); j++)
    // {
    //     max = max > cloud->points[j].intensity ? max : cloud->points[j].intensity / 1.0f;
    //     min = min < cloud->points[j].intensity ? min : cloud->points[j].intensity / 1.0f;
    //     // printf("%f %f %f %f\n",cloud->points[j].x,cloud->points[j].y,cloud->points[j].z,cloud->points[j].intensity);
    //     // printf("size= %lu\n",cloud->points.size());
    // }
    // printf("max=%f min=%f\n", max, min);

#ifndef NO_IMAGE_WRITE
    // Populate the DEM grid by looping through every point
    int row, column;
    for (size_t j = 0; j < cloud->points.size(); ++j)
    {
        // If the point is within the image size bounds
        if (map_pc2rc(cloud->points[j].x, cloud->points[j].y, &row, &column) == 1 && row >= 0 && row < IMAGE_HEIGHT &&
            column >= 0 && column < IMAGE_WIDTH)
        {
            if (cloud->points[j].z > heightArray[row][column])
            {
                heightArray[row][column] = cloud->points[j].z;
            }

            //get max intensity
            if (cloud->points[j].intensity > intensityArray[row][column])
            {
                intensityArray[row][column] = cloud->points[j].intensity;
            }
            // Keep track of lowest point in cloud for flood fill
            else if (cloud->points[j].z < lowest)
            {
                lowest = cloud->points[j].z;
            }
        }
    }

    // Create "point cloud" and opencv image to be published for visualization
    int index = 0;
    double x, y;
    for (int i = 0; i < IMAGE_HEIGHT; ++i)
    {
        for (int j = 0; j < IMAGE_WIDTH; ++j)
        {
            // Add point to cloud
            (void)map_rc2pc(&x, &y, i, j);
            cloud_grid->points[index].x = x;
            cloud_grid->points[index].y = y;
            cloud_grid->points[index].z = heightArray[i][j];
            ++index;

            // Add point to image
            cv::Vec3b &pixel = heightmap->at<cv::Vec3b>(i, j);
            if (heightArray[i][j] > -FLT_MAX)
            {
                pixel[0] = 0;
                pixel[1] = 0;
                pixel[2] = map_m2i(heightArray[i][j]);
            }
            else
            {
                pixel[0] = 0;
                pixel[1] = 0;
                pixel[2] = 0; //map_m2i(lowest);
            }
        }
    }

    // Save image to disk
    char filename[100];
    snprintf(filename, 100, "images/image_%d.png", fnameCounter);
    cv::imwrite(filename, *heightmap, compression_params);
#endif

#ifndef NO_IMAGE_SHOW
    // Display iamge
    cv::imshow("Height Map", *heightmap);
#endif

    ++fnameCounter;

    // // Output height map to point cloud for python node to parse to PNG
    // pcl::toROSMsg(*cloud_grid, output);
    // output.header.stamp = ros::Time::now();
    // output.header.frame_id = "velodyne";
    // pubPointCloud.publish(output);
}

int main(int argc, char **argv)
{
    ROS_INFO("Starting LIDAR Node");
    ros::init(argc, argv, "lidar_node");
    ros::NodeHandle nh;

    // Setup output cloud
    cloud_grid->width = IMAGE_WIDTH;
    cloud_grid->height = IMAGE_HEIGHT;
    cloud_grid->points.resize(cloud_grid->width * cloud_grid->height);

    // Setup image
    cv::Mat map(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
    heightmap = &map;

#ifndef NO_IMAGE_SHOW
    cvNamedWindow("Height Map", CV_WINDOW_AUTOSIZE);
    cvStartWindowThread();
    cv::imshow("Height Map", *heightmap);
#endif

    // Setup Image Output Parameters
    fnameCounter = 0;
    lowest = FLT_MAX;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);

    // Setup indicies in point clouds
    /*
    int index = 0;
    double x, y;
    for(int i = 0; i < IMAGE_HEIGHT; ++i){
    for(int j = 0; j < IMAGE_WIDTH; ++j){
      index = i * j;
      (void)map_rc2pc(&x, &y, i, j);
      cloud_grid->points[index].x = x;
      cloud_grid->points[index].y = y;
      cloud_grid->points[index].z = (-FLT_MAX);
      // Temp storage
      heightArray[i][j] = (-FLT_MAX);
      }
    }
    */

    subPointCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 2, DEM);
    pubPointCloud = nh.advertise<sensor_msgs::PointCloud2>("/heightmap/pointcloud", 1);

    //    subVelodynePackets = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_packets", 2, DEM);
    ros::spin();

    return 0;
}
