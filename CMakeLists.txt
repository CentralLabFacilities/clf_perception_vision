### Maintainer: flier[@techfak.uni-bielefeld.de] ###

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)

PROJECT(clf_perception_vision)

SET(OpenCV_MIN_VERSION "3.3.1")

SET(ROS_VERSION "kinetic")
MESSAGE(STATUS "BUILD FOR ROS: " ${ROS_VERSION})

SET(DEFAULT_CXX_FLAGS "" CACHE STRING "-O2 -march=native -Wno-deprecated-declarations")
SET(CMAKE_CXX_FLAGS "${DEFAULT_CXX_FLAGS} ${CMAKE_CXX_FLAGS} -std=gnu++11 -Wno-deprecated-declarations")
MESSAGE(STATUS "CXX_FLAGS: " ${CMAKE_CXX_FLAGS})

SET(ROS_BUILD_TYPE Release)
SET(CMAKE_BUILD_TYPE Release)

FIND_PACKAGE(dlib REQUIRED)
FIND_PACKAGE(Caffe REQUIRED)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(Boost REQUIRED COMPONENTS system program_options)
FIND_PACKAGE(catkin REQUIRED COMPONENTS tf
                                        roscpp
                                        nodelet
                                        std_msgs
                                        cv_bridge
                                        sensor_msgs
                                        geometry_msgs
                                        message_filters
                                        image_transport
                                        visualization_msgs
					                    clf_perception_vision_msgs
)
FIND_PACKAGE(OpenCV REQUIRED COMPONENTS highgui
                                        core
                                        imgproc
                                        features2d
                                        xfeatures2d
                                        imgcodecs
                                        calib3d
                                        cudafeatures2d
                                        cudaarithm
                                        cudabgsegm
                                        cudafilters
                                        cudaimgproc
                                        cudalegacy
                                        cudaobjdetect
                                        cudaoptflow
                                        cudastereo
                                        cudawarping
                                        cudacodec
)

ADD_DEFINITIONS(${Caffe_DEFINITIONS})

IF(OpenCV_FOUND)
        IF(${OpenCV_VERSION} VERSION_LESS ${OpenCV_MIN_VERSION} AND (SUPPORT_VIDEO OR SUPPORT_IMAGE) )
                MESSAGE(FATAL_ERROR "OpenCV version (${OpenCV_VERSION} is present, while ${OpenCV_MIN_VERSION}+ is required")
        ENDIF()

        MESSAGE(STATUS "OpenCV_VERSION: " ${OpenCV_VERSION})
        # MESSAGE(STATUS "OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS})
        # MESSAGE(STATUS "OpenCV_LIBRARIES: " ${OpenCV_LIBS})
ELSE()
        MESSAGE(FATAL_ERROR "OpenCV NOT FOUND")
ENDIF()

IF(OpenCV_CUDA_VERSION)
	MESSAGE("Found OpenCV_CUDA_VERSION: ${OpenCV_CUDA_VERSION}")
ELSE()
	MESSAGE(FATAL_ERROR "OpenCV_CUDA_VERSION NOT FOUND")
ENDIF()

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

SET(CUSTOM_CV_LIBS  ${CMAKE_INSTALL_PREFIX}/lib/libopencv_imgcodecs.so
                    ${CMAKE_INSTALL_PREFIX}/lib/libopencv_face.so
                    ${CMAKE_INSTALL_PREFIX}/lib/libopencv_imgproc.so
                    ${CMAKE_INSTALL_PREFIX}/lib/libopencv_core.so
                    ${CMAKE_INSTALL_PREFIX}/lib/libopencv_highgui.so
                    ${CMAKE_INSTALL_PREFIX}/lib/libopencv_xfeatures2d.so
                    ${CMAKE_INSTALL_PREFIX}/lib/libopencv_features2d.so
                    ${CMAKE_INSTALL_PREFIX}/lib/libopencv_calib3d.so
                    ${CMAKE_INSTALL_PREFIX}/lib/libopencv_objdetect.so
)

SET(CUSTOM_CUDA_LIBS  ${CMAKE_INSTALL_PREFIX}/lib/libopencv_cudaarithm.so
                      ${CMAKE_INSTALL_PREFIX}/lib/libopencv_cudabgsegm.so
                      ${CMAKE_INSTALL_PREFIX}/lib/libopencv_cudacodec.so
                      ${CMAKE_INSTALL_PREFIX}/lib/libopencv_cudafeatures2d.so
                      ${CMAKE_INSTALL_PREFIX}/lib/libopencv_cudafilters.so
                      ${CMAKE_INSTALL_PREFIX}/lib/libopencv_cudaimgproc.so
                      ${CMAKE_INSTALL_PREFIX}/lib/libopencv_cudalegacy.so
                      ${CMAKE_INSTALL_PREFIX}/lib/libopencv_cudaobjdetect.so
                      ${CMAKE_INSTALL_PREFIX}/lib/libopencv_cudaoptflow.so
                      ${CMAKE_INSTALL_PREFIX}/lib/libopencv_cudastereo.so
                      ${CMAKE_INSTALL_PREFIX}/lib/libopencv_cudawarping.so
)

# MESSAGE(STATUS "CUSTOM_ROS_LIBS: " ${CUSTOM_ROS_LIBS})
# MESSAGE(STATUS "CUSTOM_CV_LIBS: " ${CUSTOM_CV_LIBS})
# MESSAGE(STATUS "CUSTOM_CUDA_LIBS: "${CUSTOM_CUDA_LIBS})

CATKIN_PACKAGE(INCLUDE_DIRS include
               # LIBRARIES clf_perception_nodelets
               CATKIN_DEPENDS tf
                              roscpp
                              nodelet
                              std_msgs
                              cv_bridge
                              sensor_msgs
                              geometry_msgs
                              message_filters
                              message_runtime
                              image_transport
                              clf_perception_vision_msgs
        DEPENDS dlib
                       CUDA
                       Caffe
                       Boost
                       OpenCV
)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS}
                    ${Boost_INCLUDE_DIRS}
                    ${catkin_INCLUDE_DIRS}
                    ${Caffe_INCLUDE_DIRS}
                    ${dlib_INCLUDE_DIRS}
                    ${catkin_INCLUDE_DIRS}
                    include)

ADD_EXECUTABLE(clf_perception_surb
               src/ros_grabber.cpp
               src/clf_perception_surb.cpp
               src/clf_perception_surb_main.cpp
               ${INCLUDE_DIRS}
)

ADD_EXECUTABLE(clf_perception_interactive_surb
               src/ros_grabber.cpp
               src/clf_perception_interactive_surb.cpp
               src/clf_perception_interactive_surb_main.cpp
               ${INCLUDE_DIRS}
)

ADD_EXECUTABLE(clf_perception_genage
               src/ros_grabber.cpp
               src/dlib_detection.cpp
               src/caffee_classification.cpp
               src/clf_perception_gender_age_main.cpp
               ${INCLUDE_DIRS}
)

ADD_EXECUTABLE(clf_perception_collect_data
               src/clf_perception_collect_data.cpp
               src/ros_grabber.cpp
               ${INCLUDE_DIRS}
)

ADD_EXECUTABLE(clf_perception_depth_lookup_persons
               src/clf_perception_depth_lookup_persons.cpp
               ${INCLUDE_DIRS}
)

# ADD_EXECUTABLE(clf_perception_depth_lookup_objects
#               src/clf_perception_depth_lookup_objects.cpp
#               ${INCLUDE_DIRS}
# )

# Nodelets #
# ADD_LIBRARY(clf_perception_nodelets
#            src/clf_perception_depth_lookup_nodelet.cpp
#            src/clf_perception_gender_age_nodelet.cpp
#            src/ros_grabber.cpp
#            src/dlib_detection.cpp
#            src/caffee_classification.cpp
# )

# Nodelets #
# TARGET_LINK_LIBRARIES(clf_perception_nodelets ${catkin_LIBRARIES}
#                                               ${CUSTOM_ROS_LIBS}
#                                               ${Caffe_LIBRARIES}
#                                               ${CUSTOM_CV_LIBS}
#                                               ${dlib_LIBRARIES}
#                                               ${Boost_LIBRARIES})

# IF(catkin_EXPORTED_LIBRARIES)
#  ADD_DEPENDENCIES(clf_perception_nodelets ${catkin_EXPORTED_LIBRARIES})
# ENDIF()

TARGET_LINK_LIBRARIES(clf_perception_surb ${CUSTOM_CUDA_LIBS} ${CUSTOM_CV_LIBS} ${CUSTOM_ROS_LIBS} ${Boost_LIBRARIES})
TARGET_LINK_LIBRARIES(clf_perception_interactive_surb ${CUSTOM_CUDA_LIBS} ${CUSTOM_CV_LIBS} ${CUSTOM_ROS_LIBS} ${Boost_LIBRARIES})
TARGET_LINK_LIBRARIES(clf_perception_genage ${Caffe_LIBRARIES} ${CUSTOM_CV_LIBS} ${CUSTOM_ROS_LIBS} ${Boost_LIBRARIES} ${dlib_LIBRARIES})
TARGET_LINK_LIBRARIES(clf_perception_depth_lookup_persons ${catkin_LIBRARIES} ${CUSTOM_CV_LIBS} ${Boost_LIBRARIES})
TARGET_LINK_LIBRARIES(clf_perception_collect_data ${CUSTOM_CV_LIBS} ${CUSTOM_ROS_LIBS} ${Boost_LIBRARIES})

# Nodelets #
# INSTALL(TARGETS clf_perception_nodelets ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION} LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION} RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

INSTALL(TARGETS clf_perception_surb RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
INSTALL(TARGETS clf_perception_interactive_surb RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
INSTALL(TARGETS clf_perception_genage RUNTIME DESTINATION RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
INSTALL(TARGETS clf_perception_depth_lookup_persons RUNTIME DESTINATION RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
INSTALL(TARGETS clf_perception_collect_data RUNTIME DESTINATION RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/data" DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/launch" DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

# Nodelets #
# INSTALL(FILES clf_perception_nodelets.xml DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

CATKIN_INSTALL_PYTHON(PROGRAMS src/python/clf_perception_box2people.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
CATKIN_INSTALL_PYTHON(PROGRAMS src/python/clf_perception_people2map.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
CATKIN_INSTALL_PYTHON(PROGRAMS src/python/clf_perception_box2objects.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
CATKIN_INSTALL_PYTHON(PROGRAMS src/python/clf_perception_vision_send_isurf.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
