cmake_minimum_required(VERSION 3.8)
project(instance_segmentation_utils)

set(CMAKE_BUILD_TYPE "Debug") 
set(CMAKE_CXX_FLAGS_RELEASE "-O3 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 ${CMAKE_CXX_FLAGS}")

## Dependencies
find_package(ament_cmake REQUIRED)
find_package(PCL REQUIRED)
set(AMENT_DEPS
    rclcpp
    tf2
	tf2_ros
	tf2_geometry_msgs
	sensor_msgs
	segmentation_msgs
	image_geometry
    visualization_msgs
	cv_bridge
    pcl_conversions
    )
foreach(Dependency IN ITEMS ${AMENT_DEPS})
  find_package(${Dependency} REQUIRED)
endforeach()

find_package(OpenCV REQUIRED)

## TARGETS
include_directories(include ${PCL_INCLUDE_DIRS})

add_executable(projectTo3D src/projectTo3D.cpp)
ament_target_dependencies(projectTo3D ${AMENT_DEPS})
target_link_libraries(projectTo3D ${OpenCV_LIBS})

## INSTALL
install(
	TARGETS projectTo3D
	DESTINATION lib/${PROJECT_NAME}
)
ament_package()
