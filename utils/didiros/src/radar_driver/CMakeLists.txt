cmake_minimum_required(VERSION 2.8.3)
project(radar_driver)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED
	rospy
	std_msgs
	message_generation
)

## Generate messages in the 'msg' folder
add_message_files(
 FILES
 RadarTracks.msg
 Track.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
   DEPENDENCIES
   std_msgs  # Or other packages containing msgs
)
