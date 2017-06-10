cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(lidar_prorpocess)

find_package(PCL 1.7 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(VTK REQUIRED)

FIND_PACKAGE(PythonLibs 2.7 REQUIRED)
FIND_PACKAGE(Boost COMPONENTS python)

include_directories(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${VTK_INCLUDE_DIRS} ${OpenCV_DIR} )  
link_directories(${PCL_LIBRARY_DIRS} ${VTK_LIBRARY_DIRS} )
add_definitions(${PCL_DEFINITIONS})

#add_executable (lidar_preprocess lidar_preprocess.cpp)

add_library(lidar_preprocess SHARED lidar_preprocess.cpp )
add_library(lidar_preprocess_ext SHARED lidar_preprocess_ext.cpp )

target_link_libraries (lidar_preprocess_ext lidar_preprocess ${PYTHON_LIBRARIES} ${Boost_LIBRARIES} ${PCL_LIBRARIES}  ${VTK_LIBRARY_DIRS} ${OpenCV_LIBS}) 
set_target_properties(lidar_preprocess_ext PROPERTIES PREFIX "")

#target_link_libraries (lidar_preprocess ${PCL_LIBRARIES}  ${VTK_LIBRARY_DIRS} ${OpenCV_LIBS}) 

