cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(lidar_image_generator)

find_package(PCL 1.7 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(VTK REQUIRED)

include_directories(${PCL_INCLUDE_DIRS} ${VTK_INCLUDE_DIRS} ${OpenCV_DIR} )  
link_directories(${PCL_LIBRARY_DIRS} ${VTK_LIBRARY_DIRS} )
add_definitions(${PCL_DEFINITIONS})

add_executable (lidar_image_generator lidar_image_generator.cpp)

add_library(lidar SHARED lidar_image_generator.cpp )

target_link_libraries (lidar ${PCL_LIBRARIES}  ${VTK_LIBRARY_DIRS} ${OpenCV_LIBS}) 

target_link_libraries (lidar_image_generator ${PCL_LIBRARIES}  ${VTK_LIBRARY_DIRS} ${OpenCV_LIBS}) 

