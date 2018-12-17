### Maintainer: flier[@techfak.uni-bielefeld.de] ###

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)

PROJECT(clf_perception_vision)

SET(OpenCV_MIN_VERSION "3.3.1")

SET(ROS_VERSION "kinetic")
MESSAGE(STATUS "BUILD FOR ROS: " ${ROS_VERSION})

SET(DEFAULT_CXX_FLAGS "" CACHE STRING "-Wno-deprecated-declarations")
SET(CMAKE_CXX_FLAGS "${DEFAULT_CXX_FLAGS} ${CMAKE_CXX_FLAGS} -std=gnu++11 -Wno-deprecated-declarations")
MESSAGE(STATUS "CXX_FLAGS: " ${CMAKE_CXX_FLAGS})

SET(ROS_BUILD_TYPE Release)
SET(CMAKE_BUILD_TYPE Release)

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
                                        bayes_people_tracker_msgs
)
FIND_PACKAGE(OpenCV REQUIRED COMPONENTS highgui
                                        core
                                        imgproc
                                        features2d
                                        xfeatures2d
                                        imgcodecs
                                        calib3d
)

ADD_DEFINITIONS(${Caffe_DEFINITIONS})

IF(OpenCV_FOUND)
        IF(${OpenCV_VERSION} VERSION_LESS ${OpenCV_MIN_VERSION} AND (SUPPORT_VIDEO OR SUPPORT_IMAGE) )
                MESSAGE(FATAL_ERROR "OpenCV version (${OpenCV_VERSION} is present, while ${OpenCV_MIN_VERSION}+ is required")
        ENDIF()

        MESSAGE(STATUS "OpenCV_VERSION: " ${OpenCV_VERSION})
ELSE()
        MESSAGE(FATAL_ERROR "OpenCV NOT FOUND")
ENDIF()

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
                              bayes_people_tracker_msgs
               DEPENDS OpenCV
)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS}
                    ${Boost_INCLUDE_DIRS}
                    ${catkin_INCLUDE_DIRS}
                    ${catkin_INCLUDE_DIRS}
                    include)

ADD_EXECUTABLE(clf_perception_depth_lookup_persons
               src/clf_perception_depth_lookup_persons.cpp
               ${INCLUDE_DIRS}
)

ADD_EXECUTABLE(clf_perception_depth_lookup_objects
              src/clf_perception_depth_lookup_objects.cpp
              ${INCLUDE_DIRS}
)


TARGET_LINK_LIBRARIES(clf_perception_depth_lookup_persons ${catkin_LIBRARIES} ${CUSTOM_CV_LIBS} ${Boost_LIBRARIES})

INSTALL(TARGETS clf_perception_depth_lookup_persons RUNTIME DESTINATION RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/data" DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/launch" DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

CATKIN_INSTALL_PYTHON(PROGRAMS src/python/clf_perception_box2people.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
CATKIN_INSTALL_PYTHON(PROGRAMS src/python/clf_perception_people2map.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
CATKIN_INSTALL_PYTHON(PROGRAMS src/python/clf_perception_box2objects.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
