CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)

PROJECT(clf_2d_gpu_detect)

SET(ENV{ROS_LANG_DISABLE} "genjava" "genlisp")

FIND_PACKAGE(CUDA)
FIND_PACKAGE(Boost REQUIRED COMPONENTS system program_options)
FIND_PACKAGE(catkin REQUIRED COMPONENTS roscpp std_msgs sensor_msgs image_transport cv_bridge)
FIND_PACKAGE(OpenCV ${OpenCV_MIN_VERSION} REQUIRED PATHS ${CMAKE_INSTALL_PREFIX} NO_DEFAULT_PATH)

MESSAGE(STATUS "catkin_LIBRARIES: " ${catkin_LIBRARIES})

SET(OpenCV_MIN_VERSION "3.2.0")
SET(ROS_VERSION "kinetic")

IF(OpenCV_FOUND)
        IF(${OpenCV_VERSION} VERSION_LESS ${OpenCV_MIN_VERSION} AND (SUPPORT_VIDEO OR SUPPORT_IMAGE) )
                MESSAGE(FATAL_ERROR "OpenCV version (${OpenCV_VERSION} is present, while ${OpenCV_MIN_VERSION}+ is required")
        ENDIF()

        MESSAGE(STATUS "OpenCV_VERSION: " ${OpenCV_VERSION})
        MESSAGE(STATUS "OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS})
        MESSAGE(STATUS "OpenCV_LIBRARIES: " ${OpenCV_LIBRARIES})
        MESSAGE(STATUS "OpenCV_LIBRARY_DIRS: " ${OpenCV_LIBRARY_DIRS})
ELSE()
        MESSAGE(FATAL_ERROR "OpenCV not found.")
ENDIF()

IF(OpenCV_CUDA_VERSION)
	MESSAGE("Found OpenCV_CUDA_VERSION: ${OpenCV_CUDA_VERSION}")
ELSE()
	MESSAGE(FATAL_ERROR "OpenCV_CUDA_VERSION NOT FOUND")
ENDIF()

SET(ROS_BUILD_TYPE RELEASE)
SET(CMAKE_BUILD_TYPE RELEASE)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

# This is a hack :( ROS also provides an OPENCV3 installation
# _without_ CUDA support. Somehow, catkin enforces the ROS CV
# installation and thus, thise target is linked against the
# 'wrong' CV. Set CMAKE_PREFIX_PATH and LD_LIBRARY_PATH doesn't
# help. If someone finds a nicer solution, let me know.

SET(CUSTOM_ROS_LIBS /opt/ros/${ROS_VERSION}/lib/libroscpp.so
                    /opt/ros/${ROS_VERSION}/lib/libroscpp.so
                    /opt/ros/${ROS_VERSION}/lib/libcv_bridge.so
                    /opt/ros/${ROS_VERSION}/lib/librosconsole.so
                    /opt/ros/${ROS_VERSION}/lib/libroscpp_serialization.so
                    /opt/ros/${ROS_VERSION}/lib/librostime.so
	                /opt/ros/${ROS_VERSION}/lib/libmessage_filters.so
                    /opt/ros/${ROS_VERSION}/lib/libclass_loader.so
                    /opt/ros/${ROS_VERSION}/lib/libroslib.so
                    /opt/ros/${ROS_VERSION}/lib/libxmlrpcpp.so
                    /opt/ros/${ROS_VERSION}/lib/libcpp_common.so
                    /opt/ros/${ROS_VERSION}/lib/librosconsole_log4cxx.so
                    /opt/ros/${ROS_VERSION}/lib/librosconsole_backend_interface.so
                    /opt/ros/${ROS_VERSION}/lib/librospack.so
                    /opt/ros/${ROS_VERSION}/lib/libimage_transport.so
)

MESSAGE(STATUS "CUSTOM_ROS_LIBS ": ${CUSTOM_ROS_LIBS})

CATKIN_PACKAGE(
  INCLUDE_DIRS include
  CATKIN_DEPENDS message_runtime
)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS} include)

FILE(GLOB CLF_HEADER_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} include/*.hpp)

ADD_EXECUTABLE(clf_2d_detect_ros
               src/clf_2d_detect.cpp
               src/ros_grabber.cpp
               src/main.cpp
               ${CLF_HEADER_LIST}
               ${INCLUDE_DIRS}

)

ADD_EXECUTABLE(clf_2d_collect
               src/clf_2d_collect.cpp
               src/ros_grabber.cpp
               ${CLF_HEADER_LIST}
               ${INCLUDE_DIRS}

)

# ADD_EXECUTABLE(clf_2d_faces_ros
#                src/clf_2d_faces.cpp
#                src/ros_grabber.cpp
#                ${CLF_HEADER_LIST}
#                ${INCLUDE_DIRS}
#
# )

# NO ${catkin_LIBRARIES}, see note above !!

TARGET_LINK_LIBRARIES(clf_2d_detect_ros ${OpenCV_LIBS} ${CUSTOM_ROS_LIBS} ${Boost_LIBRARIES})
TARGET_LINK_LIBRARIES(clf_2d_collect ${OpenCV_LIBS} ${CUSTOM_ROS_LIBS} ${Boost_LIBRARIES})
# TARGET_LINK_LIBRARIES(clf_2d_faces_ros ${OpenCV_LIBS} ${CUSTOM_ROS_LIBS} ${Boost_LIBRARIES})

INSTALL(TARGETS clf_2d_detect_ros RUNTIME DESTINATION bin)
INSTALL(TARGETS clf_2d_collect RUNTIME DESTINATION bin)
# INSTALL(TARGETS clf_2d_faces_ros RUNTIME DESTINATION bin)

INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/data" DESTINATION ${CMAKE_INSTALL_PREFIX}/share/clf_2d_gpu_detect)