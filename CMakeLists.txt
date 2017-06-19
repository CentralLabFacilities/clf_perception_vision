CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)
PROJECT(clf_2d_gpu_detect)

FIND_PACKAGE(CUDA)

FIND_PACKAGE(Boost REQUIRED COMPONENTS system program_options)

SET(ENV{ROS_LANG_DISABLE} "genjava" "genlisp")

FIND_PACKAGE(catkin COMPONENTS roscpp std_msgs sensor_msgs image_transport cv_bridge)

SET(OpenCV_MIN_VERSION "3.2.0")

FIND_PACKAGE( OpenCV ${OpenCV_MIN_VERSION} REQUIRED COMPONENTS core features2d imgproc objdetect video highgui flann)

IF(OpenCV_FOUND)
        IF(${OpenCV_VERSION} VERSION_LESS ${OpenCV_MIN_VERSION} AND (SUPPORT_VIDEO OR SUPPORT_IMAGE) )
                MESSAGE(FATAL_ERROR "OpenCV version (${OpenCV_VERSION} is present, while ${OpenCV_MIN_VERSION}+ is required")
        ENDIF()

        INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

        MESSAGE("Found OpenCV ${OpenCV_VERSION} libraries ${OpenCV_LIBRARIES} and headers ${OpenCV_INCLUDE_DIRS}")
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

#ADD_EXECUTABLE(clf_2d_faces_ros
#               src/clf_2d_faces.cpp
#               src/ros_grabber.cpp
#               ${CLF_HEADER_LIST}
#               ${INCLUDE_DIRS}
#
#)

TARGET_LINK_LIBRARIES(clf_2d_detect_ros ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})
#TARGET_LINK_LIBRARIES(clf_2d_faces_ros ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})

SET_PROPERTY(TARGET clf_2d_detect_ros PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
#SET_PROPERTY(TARGET clf_2d_faces_ros PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
SET_TARGET_PROPERTIES(clf_2d_detect_ros PROPERTIES COMPILE_FLAGS -DCMAKE_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}")
#SET_TARGET_PROPERTIES(clf_2d_faces_ros PROPERTIES COMPILE_FLAGS -DCMAKE_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}")

INSTALL(TARGETS clf_2d_detect_ros RUNTIME DESTINATION bin)
#INSTALL(TARGETS clf_2d_faces_ros RUNTIME DESTINATION bin)

INSTALL(PROGRAMS "${CMAKE_CURRENT_SOURCE_DIR}/tools/clf_2d_collect" DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/data" DESTINATION ${CMAKE_INSTALL_PREFIX}/share/clf_2d_gpu_detect)
